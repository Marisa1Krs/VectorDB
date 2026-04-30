/**
 * @file SemanticIndexer.cpp
 * @brief SemanticIndexer 实现 — 基于 BERT 句向量的语义搜索引擎（SQLite 持久化）
 *
 * 实现细节：
 * 1. 构造函数：初始化 BERT 引擎 + 打开 SQLite 数据库 + 创建表结构
 * 2. buildIndex：解析 XML → BERT 编码 → SQLite 事务写入
 * 3. find：BERT 编码查询 → SQLite 全表扫描 → 点积排序 → TopK
 * 4. suggest：BERT 编码查询 → 逐标题编码 → 语义相似度排序 → 推荐
 * 5. 辅助工具：HTML 清洗、数据库统计、单例管理
 */

#include "SemanticIndexer.h"

#include <set>
#include <unordered_map>
#include <cmath>

// ============================================================
// 构造函数 —— 初始化引擎 + 打开 SQLite 数据库
// ============================================================

SemanticIndexer::SemanticIndexer(const string& model_path, const string& vocab_path, Configer& conf)
    : engine_(model_path, vocab_path)
    , _conf(conf)
    , db_(nullptr)
{
    // 获取 SQLite 数据库路径（从配置）
    _dbPath = _conf.getConfigMap()["sqliteDbPath"];
    if (_dbPath.empty()) {
        _dbPath = "semantic_index.db";  // 默认路径
    }

    // ---- 打开 SQLite 数据库 ----
    int rc = sqlite3_open_v2(
        _dbPath.c_str(),
        &db_,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
        nullptr
    );
    if (rc != SQLITE_OK) {
        LOG_ERROR("SemanticIndexer: 打开数据库失败: %s, rc=%d",
                  sqlite3_errmsg(db_), rc);
        return;
    }
    LOG_INFO("SemanticIndexer: 打开数据库 %s", _dbPath.c_str());

    // 设置忙等待超时（5秒），避免因 stale WAL 文件导致 SQLITE_BUSY
    sqlite3_busy_timeout(db_, 5000);

    // ---- 创建表结构（含 title_embedding，用于 suggest 的 ANN） ----
    const char* createTableSQL =
        "CREATE TABLE IF NOT EXISTS docs ("
        "  id              INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  title           TEXT    NOT NULL DEFAULT '',"
        "  url             TEXT    NOT NULL DEFAULT '',"
        "  content         TEXT    NOT NULL DEFAULT '',"
        "  embedding       BLOB   NULL,"
        "  title_embedding BLOB   NULL"
        ");";
    char* errMsg = nullptr;
    rc = sqlite3_exec(db_, createTableSQL, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        LOG_ERROR("SemanticIndexer: 创建表失败: %s", errMsg);
        sqlite3_free(errMsg);
    } else {
        LOG_INFO("SemanticIndexer: 表结构就绪，embedding_dim=%zu, max_seq_len=%zu",
                 engine_.dim(), engine_.max_seq_len());
    }

    // ---- 兼容旧数据库：添加 title_embedding 列（如已存在则忽略） ----
    // 旧数据库没有 title_embedding 列，需要 ALTER TABLE 添加
    const char* alterSQL = "ALTER TABLE docs ADD COLUMN title_embedding BLOB NULL;";
    rc = sqlite3_exec(db_, alterSQL, nullptr, nullptr, nullptr);
    if (rc == SQLITE_OK) {
        LOG_INFO("SemanticIndexer: 已为旧数据库添加 title_embedding 列");
    } else {
        // 列已存在的错误可忽略（SQLITE_ERROR 表示列已存在）
        if (rc != SQLITE_ERROR) {
            LOG_WARN("SemanticIndexer: ALTER TABLE 添加 title_embedding 列: rc=%d", rc);
        }
    }

    // 检查是否已有数据
    checkDbStats();
}

// ============================================================
// 析构函数
// ============================================================

SemanticIndexer::~SemanticIndexer() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
        LOG_INFO("SemanticIndexer: 数据库已关闭");
    }
}

// ============================================================
// 单例模式
// ============================================================

void SemanticIndexer::init(const string& model_path, const string& vocab_path, Configer& conf) {
    if (_ptr == nullptr) {
        _ptr = new SemanticIndexer(model_path, vocab_path, conf);
    }
}

SemanticIndexer* SemanticIndexer::getPtr() {
    return _ptr;
}

// ============================================================
// 建索引 —— 从 XML 解析 → BERT 编码 → SQLite 存储
// ============================================================

void SemanticIndexer::buildIndex() {
    string xmlDir = _conf.getConfigMap()["xmlPath"];
    LOG_INFO("SemanticIndexer 开始构建索引，xmlDir=%s", xmlDir.c_str());

    // ---- 1. 扫描 XML 文件 ----
    std::vector<std::string> files;
    DIR* fileDir = opendir(xmlDir.c_str());
    if (!fileDir) {
        LOG_ERROR("SemanticIndexer: 无法打开目录 %s", xmlDir.c_str());
        return;
    }
    struct dirent* enterDir;
    while ((enterDir = readdir(fileDir)) != nullptr) {
        if (strcmp(enterDir->d_name, ".") == 0 || strcmp(enterDir->d_name, "..") == 0)
            continue;
        if (enterDir->d_type == DT_REG) {
            files.push_back(xmlDir + "/" + string(enterDir->d_name));
        }
    }
    closedir(fileDir);
    LOG_INFO("SemanticIndexer: 找到 %zu 个 XML 文件", files.size());

    // ---- 2. 开始事务 ----
    sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);

    // ---- 3. 预编译 INSERT 语句 ----
    // embedding 列存储 512 个 float32（2048 字节）的 BLOB
    // title_embedding 也存储 512 个 float32，用于 suggest 的 ANN 匹配
    const char* insertSQL =
        "INSERT INTO docs (title, url, content, embedding, title_embedding) "
        "VALUES (?, ?, ?, ?, ?);";
    sqlite3_stmt* insertStmt = nullptr;
    if (sqlite3_prepare_v2(db_, insertSQL, -1, &insertStmt, nullptr) != SQLITE_OK) {
        LOG_ERROR("SemanticIndexer: 预编译 INSERT 失败: %s", sqlite3_errmsg(db_));
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return;
    }

    // ---- 4. 解析每个 XML 文件 ----
    int totalDocs = 0;
    for (const auto& filePath : files) {
        tinyxml2::XMLDocument doc;
        tinyxml2::XMLError eResult = doc.LoadFile(filePath.c_str());
        if (eResult != tinyxml2::XML_SUCCESS) {
            LOG_ERROR("SemanticIndexer: 加载 XML 失败: %s", filePath.c_str());
            continue;
        }

        tinyxml2::XMLElement* root = doc.FirstChildElement("rss");
        if (!root) continue;
        tinyxml2::XMLElement* channel = root->FirstChildElement("channel");
        if (!channel) continue;

        // 安全的 GetText 包装
        auto getSafeText = [](tinyxml2::XMLElement* elem) -> string {
            if (!elem) return "";
            const char* text = elem->GetText();
            return text ? string(text) : "";
        };

        for (auto* item = channel->FirstChildElement("item");
             item != nullptr; item = item->NextSiblingElement("item"))
        {
            string title   = getSafeText(item->FirstChildElement("title"));
            string url     = getSafeText(item->FirstChildElement("link"));
            string content = getSafeText(item->FirstChildElement("description"));

            // 如果 description 为空，尝试 content 字段
            if (content.empty()) {
                content = getSafeText(item->FirstChildElement("content"));
            }

            if (content.empty()) continue;

            // 清洗 HTML
            string cleanText = stripHtml(content);

            // ---- 5. 编码为句向量 ----
            std::vector<float> emb = engine_.encode(cleanText);
            // 额外编码标题，用于 suggest 的 ANN 语义匹配
            std::vector<float> titleEmb;
            if (!title.empty()) {
                titleEmb = engine_.encode(title);
            }

            // ---- 6. 写入 SQLite ----
            sqlite3_bind_text(insertStmt, 1, title.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(insertStmt, 2, url.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(insertStmt, 3, cleanText.c_str(), -1, SQLITE_TRANSIENT);
            // embedding 作为 BLOB：512 float32 = 2048 bytes
            sqlite3_bind_blob(insertStmt, 4, emb.data(),
                              static_cast<int>(emb.size() * sizeof(float)),
                              SQLITE_TRANSIENT);
            // title_embedding 作为 BLOB（用于 suggest）
            if (!titleEmb.empty()) {
                sqlite3_bind_blob(insertStmt, 5, titleEmb.data(),
                                  static_cast<int>(titleEmb.size() * sizeof(float)),
                                  SQLITE_TRANSIENT);
            } else {
                sqlite3_bind_null(insertStmt, 5);
            }

            // 执行 INSERT，遇到 SQLITE_BUSY 时重试（最多 10 次）
            int rc;
            int retries = 0;
            do {
                rc = sqlite3_step(insertStmt);
                if (rc == SQLITE_BUSY) {
                    // sqlite3_busy_timeout 已设置，但保险起见增加重试计数
                    retries++;
                    if (retries >= 10) {
                        LOG_ERROR("SemanticIndexer: INSERT 连续忙等待 %d 次后放弃: %s",
                                  retries, sqlite3_errmsg(db_));
                        break;
                    }
                    // 等待 100ms 后重试
                    struct timespec ts = {0, 100 * 1000 * 1000};
                    nanosleep(&ts, nullptr);
                }
            } while (rc == SQLITE_BUSY);
            if (rc != SQLITE_DONE && rc != SQLITE_BUSY) {
                LOG_ERROR("SemanticIndexer: INSERT 失败: %s (rc=%d)", sqlite3_errmsg(db_), rc);
            }
            sqlite3_reset(insertStmt);

            totalDocs++;
            if (totalDocs % 100 == 0) {
                LOG_INFO("SemanticIndexer: 已编码 %d 篇文档...", totalDocs);
            }
        }
    }

    // ---- 7. 提交事务 ----
    sqlite3_finalize(insertStmt);
    sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr);

    LOG_INFO("SemanticIndexer: 索引构建完成，共 %d 篇文档写入 SQLite，每篇维度 %zu",
             totalDocs, engine_.dim());

    // 输出数据库统计
    checkDbStats();
}

// ============================================================
// 语义搜索
// ============================================================

json SemanticIndexer::find(const string& query, int topK) {
    json ans = json::array();

    if (!db_) {
        LOG_ERROR("SemanticIndexer: 数据库未打开");
        return ans;
    }

    // ---- 1. 对查询编码 ----
    std::vector<float> queryEmb = engine_.encode(query);

    // ---- 2. 从 SQLite 读取所有文档 ----
    const char* selectSQL =
        "SELECT id, title, url, content, embedding FROM docs;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, selectSQL, -1, &stmt, nullptr) != SQLITE_OK) {
        LOG_ERROR("SemanticIndexer: 查询失败: %s", sqlite3_errmsg(db_));
        return ans;
    }

    // 逐行读取并计算相似度
    std::vector<std::pair<float, DocRecord>> scoredDocs;
    scoredDocs.reserve(10000);  // 预分配

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        DocRecord doc;
        doc.docId   = sqlite3_column_int(stmt, 0);
        doc.title   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        doc.url     = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        doc.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));

        // 读取 embedding BLOB
        const void* blob = sqlite3_column_blob(stmt, 4);
        int nBytes = sqlite3_column_bytes(stmt, 4);
        int nFloats = nBytes / static_cast<int>(sizeof(float));
        doc.embedding.resize(nFloats);
        if (blob && nFloats > 0) {
            std::memcpy(doc.embedding.data(), blob, static_cast<size_t>(nBytes));
        }

        // ---- 3. 计算点积（余弦相似度） ----
        float dot = 0.0f;
        size_t dim = std::min(queryEmb.size(), doc.embedding.size());
        for (size_t j = 0; j < dim; ++j) {
            dot += queryEmb[j] * doc.embedding[j];
        }

        scoredDocs.push_back({dot, std::move(doc)});
    }
    sqlite3_finalize(stmt);

    if (scoredDocs.empty()) {
        LOG_WARN("SemanticIndexer: 数据库为空，请先调用 buildIndex()");
        return ans;
    }

    // ---- 4. 排序（取 TopK） ----
    std::partial_sort(
        scoredDocs.begin(),
        scoredDocs.begin() + std::min(topK, static_cast<int>(scoredDocs.size())),
        scoredDocs.end(),
        [](const std::pair<float, DocRecord>& a,
           const std::pair<float, DocRecord>& b) {
            return a.first > b.first;
        });

    // ---- 5. 构建 JSON 结果 ----
    int resultCount = std::min(topK, static_cast<int>(scoredDocs.size()));
    for (int i = 0; i < resultCount; ++i) {
        const auto& doc = scoredDocs[i].second;
        json item;
        item["title"]      = doc.title;
        item["url"]        = doc.url;
        item["content"]    = doc.content;
        item["similarity"] = scoredDocs[i].first;  // 余弦相似度值
        item["docId"]      = doc.docId;
        ans.push_back(std::move(item));
    }

    LOG_INFO("SemanticIndexer: 查询 '%s' 返回 %d 条结果，top1 相似度=%f",
             query.c_str(), resultCount,
             resultCount > 0 ? scoredDocs[0].first : 0.0);

    return ans;
}

// ============================================================
// 词语推荐（基于 BERT 语义相似度 + ANN 最近邻）
// ============================================================
// 算法说明：
//   1. BERT 编码查询 → 查询向量（512维，L2归一化）
//   2. SQLite 全表扫描 → 对每个文档的内容嵌入做点积（余弦相似度）
//      → 这就是 ANN（近似最近邻）搜索
//   3. 取 topN（N=50）个最相似文档
//   4. 从这些文档标题中提取词语（按标点/空格/中英文边界切分）
//   5. 每个词语的得分 = 它所在文档的语义得分的加权和
//   6. 按得分排序，返回前 topK 个词语

// 静态辅助函数：检测 Unicode 字符是否为 CJK 中日韩统一表意文字
static bool _suggest_isCJK(unsigned char c1, unsigned char c2, unsigned char c3) {
    // UTF-8 3字节编码的 CJK 统一表意文字范围 U+4E00..U+9FFF
    // 编码为 0xE4 0xB8 0x80 .. 0xE9 0xBF 0xBF
    if (c1 >= 0xE4 && c1 <= 0xE9) {
        if (c1 == 0xE4 && c2 < 0xB8) return false; // < U+4E00
        if (c1 == 0xE9 && c2 > 0xBF) return false; // > U+9FFF
        if (c1 == 0xE9 && c2 == 0xBF && c3 > 0xBF) return false;
        return true;
    }
    return false;
}

// 静态辅助函数：从标题中提取有意义的词语
// 使用简单的边界检测（标点/空格/中英文切换）替代 jieba 分词
static std::vector<std::string> _suggest_extractWords(const std::string& title) {
    std::vector<std::string> words;
    std::string seg;
    bool hasCJK = false;
    bool hasAlpha = false;

    size_t i = 0;
    while (i < title.size()) {
        unsigned char c = static_cast<unsigned char>(title[i]);

        // ASCII 字符
        if (c < 0x80) {
            if (std::isalnum(c) || c == '-' || c == '_') {
                // 字母/数字/连字符 — 追加到当前段
                seg += static_cast<char>(c);
                hasAlpha = hasAlpha || std::isalpha(c);
                ++i;
            } else {
                // 非字母数字 — 结束当前段
                if (!seg.empty()) {
                    if (seg.size() >= 2 || (!hasCJK && seg.size() >= 1)) {
                        words.push_back(seg);
                    }
                    seg.clear();
                    hasCJK = false;
                    hasAlpha = false;
                }
                ++i;
            }
        }
        // 3字节 UTF-8（CJK 及其他）
        else if ((c & 0xF0) == 0xE0 && i + 2 < title.size()) {
            unsigned char c2 = static_cast<unsigned char>(title[i+1]);
            unsigned char c3 = static_cast<unsigned char>(title[i+2]);
            if (_suggest_isCJK(c, c2, c3)) {
                // CJK 字符 — 追加到当前段
                seg += static_cast<char>(c);
                seg += static_cast<char>(c2);
                seg += static_cast<char>(c3);
                hasCJK = true;
                i += 3;
            } else {
                // 非 CJK 的 3字节字符 — 结束当前段
                if (!seg.empty()) {
                    if (seg.size() >= 2 || (!hasCJK && seg.size() >= 1)) {
                        words.push_back(seg);
                    }
                    seg.clear();
                    hasCJK = false;
                    hasAlpha = false;
                }
                i += 3;
            }
        }
        // 多字节非 CJK（如 emoji 等）
        else {
            if (!seg.empty()) {
                if (seg.size() >= 2 || (!hasCJK && seg.size() >= 1)) {
                    words.push_back(seg);
                }
                seg.clear();
                hasCJK = false;
                hasAlpha = false;
            }
            if ((c & 0xE0) == 0xC0) i += 2;       // 2字节 UTF-8
            else if ((c & 0xF0) == 0xE0) i += 3;   // 3字节 UTF-8
            else if ((c & 0xF8) == 0xF0) i += 4;   // 4字节 UTF-8
            else ++i;
        }
    }

    // 处理最后一段
    if (!seg.empty()) {
        if (seg.size() >= 2 || (!hasCJK && seg.size() >= 1)) {
            words.push_back(seg);
        }
    }

    return words;
}

json SemanticIndexer::suggest(const string& query, int topK) {
    json ans = json::array();
    if (!db_ || query.empty()) {
        LOG_WARN("SemanticIndexer::suggest: db_=null 或 query 为空, query='%s'", query.c_str());
        return ans;
    }

    LOG_INFO("SemanticIndexer::suggest: 收到查询='%s', topK=%d", query.c_str(), topK);

    // ---- 1. 编码查询（BERT 推理） ----
    std::vector<float> queryEmb = engine_.encode(query);
    if (queryEmb.empty()) {
        LOG_ERROR("SemanticIndexer::suggest: 查询编码失败, query='%s'", query.c_str());
        return ans;
    }
    LOG_INFO("SemanticIndexer::suggest: 查询编码完成, dim=%zu", queryEmb.size());

    // ---- 2. ANN 搜索：全表扫描 + 点积（余弦相似度） ----
    // 使用 title_embedding（标题嵌入）做语义匹配，确保"词语-标题"对齐
    // 这是最简单的 ANN 实现（暴力搜索），对于 4000+ 文档数量级完全够用
    constexpr int ANN_TOP_N = 50;
    const char* sql = "SELECT title, title_embedding FROM docs WHERE title_embedding IS NOT NULL;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        LOG_ERROR("SemanticIndexer::suggest: 查询失败: %s", sqlite3_errmsg(db_));
        return ans;
    }

    // 存储 (语义得分, 标题) 对
    std::vector<std::pair<float, string>> scoredTitles;
    int rowCount = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        rowCount++;
        const char* title = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (!title) continue;
        string titleStr(title);
        if (titleStr.empty()) continue;

        const void* blob = sqlite3_column_blob(stmt, 1);
        int nBytes = sqlite3_column_bytes(stmt, 1);
        int nFloats = nBytes / static_cast<int>(sizeof(float));
        if (!blob || nFloats <= 0) continue;

        const float* embData = static_cast<const float*>(blob);
        float dot = 0.0f;
        size_t dim = std::min(queryEmb.size(), static_cast<size_t>(nFloats));
        for (size_t j = 0; j < dim; ++j) {
            dot += queryEmb[j] * embData[j];
        }
        scoredTitles.push_back({dot, std::move(titleStr)});
    }
    sqlite3_finalize(stmt);

    LOG_INFO("SemanticIndexer::suggest: ANN 扫描 %d 行", rowCount);

    if (scoredTitles.empty()) {
        LOG_WARN("SemanticIndexer::suggest: 无有效评分结果");
        return ans;
    }

    // ---- 3. 按语义得分排序，取 topN ----
    std::partial_sort(scoredTitles.begin(),
                      scoredTitles.begin() + std::min(ANN_TOP_N, (int)scoredTitles.size()),
                      scoredTitles.end(),
                      [](const std::pair<float, string>& a, const std::pair<float, string>& b) {
                          return a.first > b.first;
                      });

    int topN = std::min(ANN_TOP_N, (int)scoredTitles.size());
    LOG_INFO("SemanticIndexer::suggest: ANN top%d 最高分=%f, 最低分=%f",
             topN, scoredTitles[0].first, scoredTitles[topN-1].first);

    // ---- 4. 从 topN 标题中提取词语，按语义得分加权 ----
    // word → {总得分, 出现次数}
    std::unordered_map<string, std::pair<float, int>> wordStats;
    for (int i = 0; i < topN; ++i) {
        const auto& scored = scoredTitles[i];
        float docScore = scored.first;
        std::vector<std::string> extracted = _suggest_extractWords(scored.second);

        for (const auto& word : extracted) {
            auto& stat = wordStats[word];
            stat.first += docScore;  // 累加语义得分
            stat.second++;           // 累加出现次数
        }
    }

    LOG_INFO("SemanticIndexer::suggest: 提取到 %zu 个候选词语", wordStats.size());

    if (wordStats.empty()) {
        LOG_WARN("SemanticIndexer::suggest: 未能从标题中提取到词语");
        return ans;
    }

    // ---- 5. 按 (加权得分 * log(1+频率)) 排序 ----
    std::vector<std::pair<float, string>> scoredWords;
    for (const auto& w : wordStats) {
        // 综合得分 = 语义得分和 * log(1+出现次数)
        float combined = w.second.first * std::log(1.0f + w.second.second);
        scoredWords.push_back({combined, w.first});
    }

    std::sort(scoredWords.begin(), scoredWords.end(),
              [](const std::pair<float, string>& a, const std::pair<float, string>& b) {
                  return a.first > b.first;
              });

    // ---- 6. 返回 topK 词语 ----
    int count = 0;
    for (const auto& sw : scoredWords) {
        if (count >= topK) break;
        ans.push_back(sw.second);
        count++;
    }

    LOG_INFO("SemanticIndexer::suggest: 返回 %d 个词语推荐, query='%s'",
             count, query.c_str());
    if (count > 0) {
        LOG_INFO("SemanticIndexer::suggest: top1='%s', 综合得分=%f, 语义得分=%f, 出现%d次",
                 ans[0].get<string>().c_str(), scoredWords[0].first,
                 wordStats[scoredWords[0].second].first,
                 wordStats[scoredWords[0].second].second);
    }

    return ans;
}

// ============================================================
// 文档总数
// ============================================================

size_t SemanticIndexer::size() {
    if (!db_) return 0;
    const char* countSQL = "SELECT COUNT(*) FROM docs;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, countSQL, -1, &stmt, nullptr) != SQLITE_OK) {
        return 0;
    }
    size_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = static_cast<size_t>(sqlite3_column_int(stmt, 0));
    }
    sqlite3_finalize(stmt);
    return count;
}

// ============================================================
// 清除数据
// ============================================================

void SemanticIndexer::clear() {
    if (!db_) return;
    char* errMsg = nullptr;
    int rc = sqlite3_exec(db_, "DELETE FROM docs;", nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        LOG_ERROR("SemanticIndexer: 清空数据失败: %s", errMsg);
        sqlite3_free(errMsg);
    } else {
        LOG_INFO("SemanticIndexer: 已清空所有数据");
    }
}

// ============================================================
// 数据库统计
// ============================================================

void SemanticIndexer::checkDbStats() {
    if (!db_) return;
    const char* sql = "SELECT COUNT(*), "
        "SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM docs;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return;
    }
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        int total = sqlite3_column_int(stmt, 0);
        int withEmb = sqlite3_column_int(stmt, 1);
        LOG_INFO("SemanticIndexer: 数据库统计 — 文档数=%d, 含embedding=%d",
                 total, withEmb);
    }
    sqlite3_finalize(stmt);
}

// ============================================================
// HTML 清洗工具函数
// ============================================================

string SemanticIndexer::decodeHtmlEntities(const string& input) {
    string result;
    size_t i = 0;
    while (i < input.size()) {
        if (input[i] == '&') {
            size_t semicolon = input.find(';', i);
            if (semicolon != string::npos && semicolon - i <= 10) {
                string entity = input.substr(i, semicolon - i + 1);
                // 检查常见 HTML 实体
                if (entity.size() >= 4 && entity[1] == 'l' && entity[2] == 't' && entity[3] == ';') {
                    result += '<'; i = semicolon + 1; continue;
                }
                if (entity.size() >= 4 && entity[1] == 'g' && entity[2] == 't' && entity[3] == ';') {
                    result += '>'; i = semicolon + 1; continue;
                }
                if (entity.size() >= 5 && entity[1] == 'a' && entity[2] == 'm' && entity[3] == 'p' && entity[4] == ';') {
                    result += '&'; i = semicolon + 1; continue;
                }
                if (entity.size() >= 6 && entity[1] == 'q' && entity[2] == 'u' && entity[3] == 'o' && entity[4] == 't' && entity[5] == ';') {
                    result += '"'; i = semicolon + 1; continue;
                }
                if (entity.size() >= 6 && entity[1] == 'a' && entity[2] == 'p' && entity[3] == 'o' && entity[4] == 's' && entity[5] == ';') {
                    result += '\''; i = semicolon + 1; continue;
                }
                if (entity.size() >= 6 && entity[1] == 'n' && entity[2] == 'b' && entity[3] == 's' && entity[4] == 'p' && entity[5] == ';') {
                    result += ' '; i = semicolon + 1; continue;
                }
                // 数字实体跳过
                if (entity.size() > 3 && entity[1] == '#') {
                    i = semicolon + 1;
                    continue;
                }
            }
        }
        result += input[i];
        ++i;
    }
    return result;
}

string SemanticIndexer::stripHtml(const string& input) {
    string result;
    bool inTag = false;
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '<') {
            inTag = true;
            continue;
        }
        if (input[i] == '>') {
            inTag = false;
            if (!result.empty() && result.back() != ' ') {
                result += ' ';
            }
            continue;
        }
        if (!inTag) {
            result += input[i];
        }
    }
    return decodeHtmlEntities(result);
}
