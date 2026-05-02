/**
 * @file SemanticIndexer.cpp
 * @brief 语义搜索引擎的实现
 *
 * 主要功能：
 * 1. 建库：读 XML 文章 → 转成向量 → 存 SQLite → 训练 IVF 分堆
 * 2. 搜索：查询词转向量 → IVF 找最近的堆 → 堆里文章算分排序
 * 3. 推荐：jieba 字形匹配(50%) + BGE 语义匹配(50%) → 综合排序
 */

#include "SemanticIndexer.h"

#include <set>
#include <unordered_map>
#include <cmath>

// ============================================================
// LfuCache 实现 — 低频淘汰缓存
// ============================================================

std::optional<json> LfuCache::get(const std::string& key) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return std::nullopt;  // 没命中
    }
    it->second.second++;       // 命中，访问频率 +1
    return it->second.first;   // 返回缓存的结果
}

void LfuCache::put(const std::string& key, json value) {
    std::lock_guard<std::mutex> lock(mtx_);

    // 如果 key 已存在，直接覆盖 + 频率 +1
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        it->second.first = std::move(value);
        it->second.second++;
        return;
    }

    // 缓存满了，淘汰一个
    if (entries_.size() >= capacity_) {
        evictOne();
    }

    // 插入新条目（频率初始为 1）
    entries_[key] = {std::move(value), 1};
    order_.push_back(key);
}

void LfuCache::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    entries_.clear();
    order_.clear();
}

size_t LfuCache::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return entries_.size();
}

void LfuCache::evictOne() {
    if (entries_.empty()) return;

    // 策略：扫描所有条目，找频率最低的
    // 如果频率一样，淘汰最早插入的（order_ 靠前的）
    size_t minFreq = SIZE_MAX;
    std::string evictKey;
    size_t minOrder = SIZE_MAX;

    size_t idx = 0;
    for (const auto& key : order_) {
        auto it = entries_.find(key);
        if (it == entries_.end()) continue;  // 不应该发生，但安全起见跳过

        if (it->second.second < minFreq ||
            (it->second.second == minFreq && idx < minOrder)) {
            minFreq = it->second.second;
            evictKey = it->first;
            minOrder = idx;
        }
        idx++;
    }

    if (!evictKey.empty()) {
        entries_.erase(evictKey);
        // 从 order_ 里也移除
        for (auto oit = order_.begin(); oit != order_.end(); ++oit) {
            if (*oit == evictKey) {
                order_.erase(oit);
                break;
            }
        }
    }
}

// ============================================================
// 构造函数 —— 初始化引擎 + 打开 SQLite 数据库 + 加载 IVF
// ============================================================

SemanticIndexer::SemanticIndexer(const string& model_path, const string& vocab_path, Configer& conf)
    : engine_(model_path, vocab_path)
    , _conf(conf)
    , db_(nullptr)
    , dictProducer_(_conf.getConfigMap()["jiebaDictPath"])
    , ivfIndex_(engine_.dim())  // IVF 也用 BERT 的维度（512 维）
{
    // 读数据库路径，如果没有就默认放当前目录
    _dbPath = _conf.getConfigMap()["sqliteDbPath"];
    if (_dbPath.empty()) {
        _dbPath = "semantic_index.db";
    }

    // 读 IVF 索引文件路径
    ivfIndexPath_ = _conf.getConfigMap()["ivfIndexPath"];
    if (ivfIndexPath_.empty()) {
        ivfIndexPath_ = "data/ivf_index.bin";
    }

    // ---- 打开（或创建）SQLite 数据库 ----
    // 主连接：用于建表、写入、IVF 训练等写操作
    int rc = sqlite3_open_v2(
        _dbPath.c_str(),
        &db_,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,  // 读写，没有就创建
        nullptr
    );
    if (rc != SQLITE_OK) {
        LOG_ERROR("SemanticIndexer: 打开数据库失败: %s, rc=%d",
                  sqlite3_errmsg(db_), rc);
        return;
    }
    LOG_INFO("SemanticIndexer: 打开数据库 %s", _dbPath.c_str());

    // 如果数据库正忙，最多等 5 秒
    sqlite3_busy_timeout(db_, 5000);

    // ---- 启用 WAL 模式（Write-Ahead Logging） ----
    // WAL 模式允许：多个线程同时读（不互斥），写入不阻塞读取
    // 没有 WAL 的话，一个线程在读时其他线程只能干等
    {
        char* errMsg = nullptr;
        rc = sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK) {
            LOG_WARN("SemanticIndexer: 启用 WAL 模式失败: %s", errMsg ? errMsg : "unknown");
            sqlite3_free(errMsg);
        } else {
            LOG_INFO("SemanticIndexer: 已启用 WAL 模式，支持多线程并发读取");
        }
    }

    // ---- 建表：存文章和向量 ----
    const char* createTableSQL =
        "CREATE TABLE IF NOT EXISTS docs ("
        "  id              INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  title           TEXT    NOT NULL DEFAULT '',"
        "  url             TEXT    NOT NULL DEFAULT '',"
        "  content         TEXT    NOT NULL DEFAULT '',"
        "  embedding       BLOB   NULL,"
        "  title_embedding BLOB   NULL,"
        "  cluster_id      INTEGER DEFAULT -1"
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

    // ---- 兼容旧数据库：如果之前没有 title_embedding 列，加上 ----
    const char* alterTitleSQL = "ALTER TABLE docs ADD COLUMN title_embedding BLOB NULL;";
    rc = sqlite3_exec(db_, alterTitleSQL, nullptr, nullptr, nullptr);
    if (rc == SQLITE_OK) {
        LOG_INFO("SemanticIndexer: 已为旧数据库添加 title_embedding 列");
    } else {
        if (rc != SQLITE_ERROR) {
            LOG_WARN("SemanticIndexer: ALTER TABLE title_embedding: rc=%d", rc);
        }
    }

    // ---- 兼容旧数据库：如果之前没有 cluster_id 列，加上 ----
    const char* alterClusterSQL = "ALTER TABLE docs ADD COLUMN cluster_id INTEGER DEFAULT -1;";
    rc = sqlite3_exec(db_, alterClusterSQL, nullptr, nullptr, nullptr);
    if (rc == SQLITE_OK) {
        LOG_INFO("SemanticIndexer: 已为旧数据库添加 cluster_id 列");
    } else {
        if (rc != SQLITE_ERROR) {
            LOG_WARN("SemanticIndexer: ALTER TABLE cluster_id: rc=%d", rc);
        }
    }

    // ---- 看看有没有之前训练好的 IVF 文件，有就直接加载 ----
    std::ifstream ivfFile(ivfIndexPath_, std::ios::binary);
    if (ivfFile.good()) {
        ivfFile.close();
        ivfIndex_.load(ivfIndexPath_);
        if (ivfIndex_.size() > 0) {
            ivfTrained_ = true;  // 标记 IVF 已可用
            LOG_INFO("SemanticIndexer: 已加载现有 IVF 索引 (K=%d, ntotal=%d, dim=%d)",
                     ivfIndex_.getK(), ivfIndex_.size(), ivfIndex_.getDim());
        } else {
            LOG_WARN("SemanticIndexer: IVF 索引文件存在但为空: %s", ivfIndexPath_.c_str());
        }
    } else {
        LOG_INFO("SemanticIndexer: 未找到现有 IVF 索引，先用全表扫描模式（等 buildIndex 后会自动训练）");
    }

    // 看看数据库里有没有数据
    checkDbStats();

    LOG_INFO("SemanticIndexer: DictProducer 词典加载完成, 共 %zu 个词条",
             dictProducer_.size());
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
// 建库：读 XML 文章 → BERT 转向量 → 存 SQLite → 训练 IVF 分堆
// ============================================================

void SemanticIndexer::buildIndex() {
    string xmlDir = _conf.getConfigMap()["xmlPath"];
    LOG_INFO("SemanticIndexer 开始构建索引，xmlDir=%s", xmlDir.c_str());

    // ---- 1. 扫描 xmlPath 目录下有哪些 XML 文件 ----
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

    // ---- 7. 全部写完，提交事务 ----
    sqlite3_finalize(insertStmt);
    sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr);

    LOG_INFO("SemanticIndexer: 索引构建完成，共 %d 篇文档写入 SQLite，每篇维度 %zu",
             totalDocs, engine_.dim());

    checkDbStats();

    // ============================================================
    // 8. 训练 IVF 分堆（K-Means 聚类）
    //    以后搜索时就不用全部文章都看了
    // ============================================================
    if (totalDocs <= 0) {
        LOG_WARN("SemanticIndexer: 没有文档，跳过 IVF 训练");
        return;
    }

    // ---- 8a. 从 SQLite 读出所有文章的向量 ----
    LOG_INFO("SemanticIndexer: 开始训练 IVF 分堆 (K-Means, %d 篇文档)", totalDocs);
    const char* readEmbSQL = "SELECT id, embedding FROM docs ORDER BY id;";
    sqlite3_stmt* embStmt = nullptr;
    if (sqlite3_prepare_v2(db_, readEmbSQL, -1, &embStmt, nullptr) != SQLITE_OK) {
        LOG_ERROR("SemanticIndexer: 读取向量失败: %s", sqlite3_errmsg(db_));
        return;
    }

    size_t dim = engine_.dim();
    std::vector<float> allEmbs(totalDocs * dim);  // 所有向量拼成一个大数组
    std::vector<int> docIds(totalDocs);            // 每篇文章的 ID
    int rowIdx = 0;

    while (sqlite3_step(embStmt) == SQLITE_ROW) {
        docIds[rowIdx] = sqlite3_column_int(embStmt, 0);  // 记下文章 ID
        const void* blob = sqlite3_column_blob(embStmt, 1);
        int nBytes = sqlite3_column_bytes(embStmt, 1);
        int nFloats = nBytes / static_cast<int>(sizeof(float));
        if (blob && nFloats > 0) {
            size_t copySize = std::min(static_cast<size_t>(nFloats), dim);
            std::memcpy(&allEmbs[rowIdx * dim], blob, copySize * sizeof(float));
        }
        ++rowIdx;
    }
    sqlite3_finalize(embStmt);

    if (rowIdx != totalDocs) {
        LOG_WARN("SemanticIndexer: 读到的文档数 %d 和预期 %d 不一样", rowIdx, totalDocs);
        totalDocs = rowIdx;
    }

    // ---- 8b. 决定分几堆（K 值） ----
    // 经验法则：大约 √N 堆，最多不超过总数的一半
    int K = std::max(10, static_cast<int>(std::sqrt(static_cast<double>(totalDocs))));
    K = std::min(K, totalDocs / 2);
    if (K < 2) K = 2;

    LOG_INFO("SemanticIndexer: IVF K=%d (sqrt(%d)=%d), nprobe=%d, dim=%zu",
             K, totalDocs, (int)std::sqrt(totalDocs), ivfNprobe_, dim);

    // ---- 8c. 开始 K-Means 训练（把向量分成 K 堆） ----
    ivfIndex_ = IVFIndex(static_cast<int>(dim));
    auto assignments = ivfIndex_.train(allEmbs.data(), totalDocs, K);

    if (assignments.empty()) {
        LOG_ERROR("SemanticIndexer: IVF 训练失败了，后面只能用全表扫描");
        return;
    }
    LOG_INFO("SemanticIndexer: IVF 训练完成，K=%d, 共 %d 个向量", K, totalDocs);

    // ---- 8d. 在数据库里记下每篇文章属于哪个堆 ----
    const char* updateSQL = "UPDATE docs SET cluster_id = ? WHERE id = ?;";
    sqlite3_stmt* updateStmt = nullptr;
    if (sqlite3_prepare_v2(db_, updateSQL, -1, &updateStmt, nullptr) != SQLITE_OK) {
        LOG_ERROR("SemanticIndexer: 预编译 UPDATE cluster_id 失败: %s", sqlite3_errmsg(db_));
        return;
    }

    sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);
    for (int i = 0; i < totalDocs; ++i) {
        sqlite3_bind_int(updateStmt, 1, assignments[i]);  // 堆编号
        sqlite3_bind_int(updateStmt, 2, docIds[i]);        // 文章 ID
        int rc = sqlite3_step(updateStmt);
        if (rc != SQLITE_DONE) {
            LOG_ERROR("SemanticIndexer: 更新 docId=%d cluster_id=%d 失败: rc=%d",
                      docIds[i], assignments[i], rc);
        }
        sqlite3_reset(updateStmt);
    }
    sqlite3_finalize(updateStmt);
    sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr);

    LOG_INFO("SemanticIndexer: 已为 %d 篇文档记录 cluster_id", totalDocs);

    // ---- 8e. 把分堆结果存到文件，下次启动直接读 ----
    ivfIndex_.save(ivfIndexPath_);
    ivfTrained_ = true;  // 标记 IVF 已可用，以后搜索就用加速模式

    LOG_INFO("SemanticIndexer: IVF 索引已保存到 %s (K=%d, ntotal=%d, dim=%d)",
             ivfIndexPath_.c_str(), ivfIndex_.getK(), ivfIndex_.size(), ivfIndex_.getDim());
}

// ============================================================
// 语义搜索：输入查询词，返回最像的 topK 篇文章
// ============================================================

json SemanticIndexer::find(const string& query, int topK) {
    json ans = json::array();

    if (!db_) {
        LOG_ERROR("SemanticIndexer: 数据库没打开");
        return ans;
    }

    // ---- 0. 查 LFU 缓存：如果之前搜过同样的词，直接返回 ----
    {
        auto cached = searchCache_.get(query);
        if (cached.has_value()) {
            LOG_INFO("SemanticIndexer::find: LFU 缓存命中，直接返回 query='%s'", query.c_str());
            return cached.value();
        }
    }

    // ---- 1. 把查询词也转成向量 ----
    std::vector<float> queryEmb = engine_.encode(query);

    // ---- 2. 决定怎么查：有 IVF 就只看最近几个堆，没有就全表扫描 ----
    std::string selectSQL;
    sqlite3_stmt* stmt = nullptr;

    if (ivfTrained_) {
        // ---- 模式A（IVF 加速）：先找最近的几个堆，只看堆里的文章 ----
        auto nearestClusters = ivfIndex_.searchCentroids(queryEmb.data(), ivfNprobe_);

        if (nearestClusters.empty()) {
            LOG_ERROR("SemanticIndexer: IVF 搜索没找到任何堆");
            return ans;
        }

        // 拼 SQL：WHERE cluster_id IN (3, 7, 12, ...)
        // 意思是"只看第 3、7、12 这些堆里的文章"
        std::string clusterList;
        for (size_t i = 0; i < nearestClusters.size(); ++i) {
            if (i > 0) clusterList += ",";
            clusterList += std::to_string(nearestClusters[i].first);
        }

        selectSQL = "SELECT id, title, url, content, embedding FROM docs"
                    " WHERE cluster_id IN (" + clusterList + ");";

        LOG_INFO("SemanticIndexer: IVF 加速，只看 %zu 个堆 (%s)",
                 nearestClusters.size(), clusterList.c_str());
    } else {
        // ---- 模式B（全表扫描）：没有 IVF，所有文章都看一遍（慢但全） ----
        selectSQL = "SELECT id, title, url, content, embedding FROM docs;";
        LOG_INFO("SemanticIndexer: 全表扫描模式（没有 IVF 索引）");
    }

    if (sqlite3_prepare_v2(db_, selectSQL.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        LOG_ERROR("SemanticIndexer: SQL 查询失败: %s", sqlite3_errmsg(db_));
        return ans;
    }

    // 逐条读出来，算每篇跟查询词的"像不像"（点积 = 余弦相似度）
    std::vector<std::pair<float, DocRecord>> scoredDocs;
    scoredDocs.reserve(ivfTrained_ ? 5000 : 10000);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        DocRecord doc;
        doc.docId   = sqlite3_column_int(stmt, 0);
        doc.title   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        doc.url     = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        doc.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));

        // 读出存好的向量（2048 字节 = 512 个 float）
        const void* blob = sqlite3_column_blob(stmt, 4);
        int nBytes = sqlite3_column_bytes(stmt, 4);
        int nFloats = nBytes / static_cast<int>(sizeof(float));
        doc.embedding.resize(nFloats);
        if (blob && nFloats > 0) {
            std::memcpy(doc.embedding.data(), blob, static_cast<size_t>(nBytes));
        }

        // ---- 3. 算相似度：点积 = 余弦相似度 ----
        // 因为所有向量都归一化了，直接点积就行
        float dot = 0.0f;
        size_t dim = std::min(queryEmb.size(), doc.embedding.size());
        for (size_t j = 0; j < dim; ++j) {
            dot += queryEmb[j] * doc.embedding[j];  // 每个维度乘起来再加
        }

        scoredDocs.push_back({dot, std::move(doc)});
    }
    sqlite3_finalize(stmt);

    if (scoredDocs.empty()) {
        LOG_WARN("SemanticIndexer: 库里没有文章或堆里没东西，请先跑 buildIndex()");
        return ans;
    }

    // ---- 4. 按相似度从高到低排序，取前 topK 个 ----
    std::partial_sort(
        scoredDocs.begin(),
        scoredDocs.begin() + std::min(topK, static_cast<int>(scoredDocs.size())),
        scoredDocs.end(),
        [](const std::pair<float, DocRecord>& a,
           const std::pair<float, DocRecord>& b) {
            return a.first > b.first;  // 分数高的在前
        });

    // ---- 5. 组装成 JSON 返回 ----
    int resultCount = std::min(topK, static_cast<int>(scoredDocs.size()));
    for (int i = 0; i < resultCount; ++i) {
        const auto& doc = scoredDocs[i].second;
        json item;
        item["title"]      = doc.title;
        item["url"]        = doc.url;
        item["content"]    = doc.content;
        item["similarity"] = scoredDocs[i].first;  // 相似度分数
        item["docId"]      = doc.docId;
        ans.push_back(std::move(item));
    }

    LOG_INFO("SemanticIndexer: 搜索 '%s' 返回 %d 条结果 (模式=%s)，最像的分数=%f",
             query.c_str(), resultCount,
             ivfTrained_ ? "IVF加速" : "全表扫描",
             resultCount > 0 ? scoredDocs[0].first : 0.0);

    // ---- 6. 把结果写入 LFU 缓存，下次同样的查询直接秒回 ----
    searchCache_.put(query, ans);

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

    // ---- 查 LFU 缓存：同样的推荐词直接返回 ----
    {
        auto cached = suggestCache_.get(query);
        if (cached.has_value()) {
            LOG_INFO("SemanticIndexer::suggest: LFU 缓存命中 query='%s'，跳过 BERT+SQLite", query.c_str());
            return cached.value();
        }
    }

    LOG_INFO("SemanticIndexer::suggest: 收到查询='%s', topK=%d (jieba50%% + BGE50%%)",
             query.c_str(), topK);

    // =============================================================
    // BGE 语义引擎 — BERT 编码 → ANN → 语义词提取 → 归一化
    // =============================================================

    // 先声明 BGE 结果容器（会在不同代码路径中使用）
    std::vector<std::pair<float, string>> bgeScored;
    float bgeMaxScore = 0.0f;

    // ---- 编码查询 ----
    std::vector<float> queryEmb = engine_.encode(query);
    if (queryEmb.empty()) {
        LOG_ERROR("SemanticIndexer::suggest: BGE 查询编码失败, query='%s'", query.c_str());
    } else {
        LOG_INFO("SemanticIndexer::suggest: BGE 查询编码完成, dim=%zu", queryEmb.size());

        // ---- ANN 搜索：全表扫描 + 点积 ----
        constexpr int ANN_TOP_N = 50;
        const char* sql = "SELECT title, title_embedding FROM docs WHERE title_embedding IS NOT NULL;";
        sqlite3_stmt* stmt = nullptr;
        bool annOk = true;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            LOG_ERROR("SemanticIndexer::suggest: ANN 查询失败: %s", sqlite3_errmsg(db_));
            annOk = false;
        }

        std::vector<std::pair<float, string>> scoredTitles;
        if (annOk) {
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
            LOG_INFO("SemanticIndexer::suggest: BGE ANN 扫描 %d 行", rowCount);

            if (scoredTitles.empty()) {
                LOG_WARN("SemanticIndexer::suggest: BGE ANN 无有效评分结果");
            }
        }

        // ---- 按语义得分排序，取 topN ----
        if (!scoredTitles.empty()) {
            std::partial_sort(scoredTitles.begin(),
                              scoredTitles.begin() + std::min(ANN_TOP_N, (int)scoredTitles.size()),
                              scoredTitles.end(),
                              [](const std::pair<float, string>& a, const std::pair<float, string>& b) {
                                  return a.first > b.first;
                              });

            int topN = std::min(ANN_TOP_N, (int)scoredTitles.size());
            LOG_INFO("SemanticIndexer::suggest: BGE top%d 最高分=%f, 最低分=%f",
                     topN, scoredTitles[0].first, scoredTitles[topN-1].first);

            // ---- 从 topN 标题中提取词语，按语义得分加权 ----
            std::unordered_map<string, std::pair<float, int>> bgeWordStats;
            for (int i = 0; i < topN; ++i) {
                const auto& scored = scoredTitles[i];
                float docScore = scored.first;
                std::vector<std::string> extracted = _suggest_extractWords(scored.second);

                for (const auto& word : extracted) {
                    auto& stat = bgeWordStats[word];
                    stat.first += docScore;
                    stat.second++;
                }
            }

            LOG_INFO("SemanticIndexer::suggest: BGE 提取到 %zu 个候选词语", bgeWordStats.size());

            // ---- 计算 BGE 最终得分 = 语义得分和 × log(1+频率) ----
            for (const auto& w : bgeWordStats) {
                float score = w.second.first * std::log(1.0f + w.second.second);
                if (score > bgeMaxScore) bgeMaxScore = score;
                bgeScored.push_back({score, w.first});
            }
        }
    }

    // =============================================================
    // Jieba 引擎 — 编辑距离模糊匹配（降准匹配）
    // =============================================================

    auto jiebaResults = dictProducer_.find(query, topK * 3);
    LOG_INFO("SemanticIndexer::suggest: Jieba 编辑距离匹配到 %zu 个候选词",
             jiebaResults.size());

    // =============================================================
    // 合并评分 — jieba(50%) + BGE(50%)
    // =============================================================

    // ---- 构建 jieba 分数映射 ----
    std::unordered_map<string, float> jiebaScoreMap;
    for (const auto& jr : jiebaResults) {
        jiebaScoreMap[jr.first] = jr.second;
    }

    // ---- 合并分数 ----
    std::unordered_map<string, float> combinedScores;

    // 遍历 jieba 结果（保证 jieba 词全被覆盖）
    for (const auto& jr : jiebaResults) {
        float scoreJ = jr.second;                          // jieba 分 [0, 1]
        float scoreB = 0.0f;                               // 默认 BGE 分
        if (!bgeScored.empty() && bgeMaxScore > 0.0f) {
            for (const auto& bs : bgeScored) {
                if (bs.second == jr.first) {
                    scoreB = bs.first / bgeMaxScore;       // 归一化 [0, 1]
                    break;
                }
            }
        }
        combinedScores[jr.first] = 0.5f * scoreJ + 0.5f * scoreB;
    }

    // 遍历 BGE 结果中未被 jieba 覆盖的词
    for (const auto& bs : bgeScored) {
        if (jiebaScoreMap.find(bs.second) == jiebaScoreMap.end()) {
            float scoreB = (bgeMaxScore > 0.0f) ? (bs.first / bgeMaxScore) : 0.0f;
            combinedScores[bs.second] = 0.5f * scoreB;     // jieba 分 = 0
        }
    }

    // ---- 按综合评分排序 ----
    std::vector<std::pair<float, string>> finalScored;
    for (const auto& cs : combinedScores) {
        finalScored.push_back({cs.second, cs.first});
    }
    std::sort(finalScored.begin(), finalScored.end(),
              [](const std::pair<float, string>& a, const std::pair<float, string>& b) {
                  return a.first > b.first;
              });

    // ---- 返回 topK ----
    int count = 0;
    for (const auto& fs : finalScored) {
        if (count >= topK) break;
        ans.push_back(fs.second);
        count++;
    }

    LOG_INFO("SemanticIndexer::suggest: 返回 %d 个词语推荐 (jieba50%% + BGE50%%), query='%s'",
             count, query.c_str());
    if (count > 0 && !finalScored.empty()) {
        float top1J = 0.0f, top1B = 0.0f;
        auto it = jiebaScoreMap.find(finalScored[0].second);
        if (it != jiebaScoreMap.end()) top1J = it->second;
        if (!bgeScored.empty() && bgeMaxScore > 0.0f) {
            for (const auto& bs : bgeScored) {
                if (bs.second == finalScored[0].second) {
                    top1B = bs.first / bgeMaxScore;
                    break;
                }
            }
        }
        LOG_INFO("SemanticIndexer::suggest: top1='%s', 综合=%f, jieba=%f, bge=%f",
                 ans[0].get<string>().c_str(), finalScored[0].first, top1J, top1B);
    }

    // ---- 写入 LFU 缓存，下次同样的词直接秒回 ----
    suggestCache_.put(query, ans);

    return ans;
}

// ============================================================
// jieba 降准匹配 — 编辑距离模糊匹配（用于 /search 的"您是不是想找"）
// ============================================================

json SemanticIndexer::jiebaSuggest(const string& query, int topK) const {
    json ans = json::array();
    auto results = dictProducer_.find(query, topK);
    for (const auto& r : results) {
        ans.push_back(r.first);
    }
    LOG_INFO("SemanticIndexer::jiebaSuggest: query='%s', 匹配到 %zu 个词",
             query.c_str(), results.size());
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
