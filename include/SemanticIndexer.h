#ifndef SEMANTIC_INDEXER_HPP
#define SEMANTIC_INDEXER_HPP

/**
 * @file SemanticIndexer.hpp
 * @brief 语义搜索引擎，把文章转成向量存到 SQLite，用 IVF 加速搜索
 *
 * 简单说就是：
 *   1. 文章 → BERT 模型 → 一串数字（叫"向量"，512 维）
 *   2. 把向量存到 SQLite 数据库里
 *   3. 搜的时候：查询词也转成向量 → 跟库里的向量比谁更"像"
 *   4. 用 IVF 分堆搜索，不用全部文章都看一遍
 *
 * 为什么用 SQLite 而不是文件：
 *   - 字段分开存（标题/链接/内容/向量），好管理
 *   - 写入时不会崩坏数据（原子写入）
 *   - 可以按条件过滤（比如只看某个分类）
 *   - 不需要额外装数据库软件，SQLite 是嵌入式的
 *
 * 使用示例：
 * @code
 *   SemanticIndexer::init("model/model.onnx", "model/tokenizer.json", conf);
 *   SemanticIndexer::getPtr()->buildIndex();    // 建库：读 XML → BERT 编码 → 存 SQLite → 训练 IVF
 *   json results = SemanticIndexer::getPtr()->find("癌症治疗", 5);  // 搜：编码 → IVF 找堆 → 算分排序
 * @endcode
 */

#include "InferEngine.h"
#include "IVFIndex.h"
#include "DictProducer.h"
#include "Configer.h"
#include "sqlite3.h"
#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "mylog.h"

using json = nlohmann::json;

// ============================================================
// LfuCache — 低频淘汰缓存（Least Frequently Used）
// ============================================================

/**
 * @brief 简单的 LFU 缓存，用于缓存搜索/推荐结果
 *
 * 工作原理：
 *   1. 每个缓存的条目都记录"被访问了多少次"
 *   2. 查缓存时，命中的条目频率 +1
 *   3. 缓存满了要放新条目时，踢掉访问频率最低的那个
 *   4. 如果频率一样，踢掉最早放进去的
 *
 * 为什么要用 LFU：
 *   - 搜索场景中，高频词（如"癌症"、"糖尿病"）会被反复查
 *   - LFU 会保留这些热词的结果，省掉每次都要 BERT 推理 + SQLite 扫描
 *   - 低频词（用户随便搜的）自然被淘汰，不影响热词缓存
 *
 * 线程安全：内部用 std::mutex 保护，多线程同时调用没问题
 */
struct LfuCache {
    /// @brief 构造
    /// @param cap 最大能缓存多少个条目（默认 1000）
    explicit LfuCache(size_t cap = 1000) : capacity_(cap) {}

    /**
     * @brief 从缓存中取结果
     * @param key  查询词（如"癌症"）
     * @return 如果命中，返回缓存的 JSON 结果；没命中返回空 optional
     *
     * 命中后会把访问频率 +1（LFU 的核心逻辑）
     */
    std::optional<json> get(const std::string& key);

    /**
     * @brief 把结果放进缓存
     * @param key    查询词
     * @param value  要缓存的结果（JSON）
     *
     * 如果缓存已满，会淘汰访问频率最低的条目。
     * 如果 key 已经存在，覆盖旧值并把频率 +1
     */
    void put(const std::string& key, json value);

    /// 清空缓存
    void clear();

    /// 当前缓存条目数
    size_t size() const;

    /// 最大缓存容量
    size_t capacity() const { return capacity_; }

private:
    size_t capacity_;                                    ///< 最大条目数
    mutable std::mutex mtx_;                             ///< 线程安全锁
    std::unordered_map<std::string,
                       std::pair<json, size_t>> entries_; ///< key → {结果, 访问频率}
    std::list<std::string> order_;                       ///< 插入顺序（同频率时淘汰最老的）

    /// 淘汰频率最低的一个条目
    void evictOne();
};

// ============================================================
// SemanticIndexer — 语义搜索引擎
// ============================================================

/**
 * @brief 负责：解析文章 → 转成向量 → 存 SQLite → 搜索时用 IVF 加速
 *
 * SQLite 里存的表结构：
 * @code
 *   CREATE TABLE IF NOT EXISTS docs (
 *       id        INTEGER PRIMARY KEY AUTOINCREMENT,
 *       title     TEXT,           -- 文章标题
 *       url       TEXT,           -- 文章链接
 *       content   TEXT,           -- 文章内容
 *       embedding BLOB,           -- 文章内容向量（512个float32 = 2048字节）
 *       title_embedding BLOB,     -- 标题向量（用于词语推荐）
 *       cluster_id INTEGER        -- 文章属于 IVF 的哪个组（-1 表示未分组）
 *   );
 * @endcode
 */
class SemanticIndexer {
public:
    /// 文档向量记录（从 SQLite 读取后暂存内存用于计算）
    struct DocRecord {
        int     docId;      ///< 文档 ID（SQLite 自增主键）
        string  title;      ///< 文档标题
        string  url;        ///< 文档 URL
        string  content;    ///< 文档纯文本内容（已清洗 HTML）
        std::vector<float> embedding;  ///< 512 维句向量（已 L2 归一化）
    };

public:
    /**
     * @brief 构造函数
     * @param model_path   BERT 模型文件路径（如 "model/model.onnx"）
     * @param vocab_path   分词器配置路径（如 "model/tokenizer.json"）
     * @param conf         配置器（从这里读各种路径设置）
     *
     * 自动做三件事：
     *   1. 加载 BERT 模型（准备把文字转成向量）
     *   2. 打开 SQLite 数据库（没有就新建一个）
     *   3. 创建 docs 表（存文章和向量）
     */
    SemanticIndexer(const string& model_path, const string& vocab_path, Configer& conf);

    /**
     * @brief 析构函数，关闭 SQLite 数据库
     */
    ~SemanticIndexer();

    // 禁止拷贝
    SemanticIndexer(const SemanticIndexer&) = delete;
    SemanticIndexer& operator=(const SemanticIndexer&) = delete;

    // ============================================================
    // 单例模式
    // ============================================================

    /**
     * @brief 初始化单例（整个程序只一个搜索引擎实例）
     * @param model_path   BERT 模型路径
     * @param vocab_path   分词器配置路径
     * @param conf         配置器
     */
    static void init(const string& model_path, const string& vocab_path, Configer& conf);

    /// 获取单例指针
    static SemanticIndexer* getPtr();

    // ============================================================
    // 建索引 —— 从 XML 解析 → BERT 编码 → SQLite 存储
    // ============================================================

    /**
     * @brief 建库：读 XML 文章 → BERT 转成向量 → 存 SQLite → 训练 IVF 分堆
     *
     * 具体步骤：
     *   1. 扫描 xmlPath 目录下所有 XML 文件
     *   2. 解析每个 RSS 条目（标题 + 内容 + 链接）
     *   3. 去掉 HTML 标签
     *   4. BERT 模型把文章转成 512 维向量
     *   5. 批量写入 SQLite 数据库
     *   6. 用 K-Means 算法把向量分成 K 堆（IVF 训练）
     *   7. 每篇文章记下它在哪个堆里
     *   8. 保存训练好的分组信息到文件
     */
    void buildIndex();

    // ============================================================
    // 语义搜索 —— BERT 编码查询 → SQLite 读取 → 向量检索
    // ============================================================

    /**
     * @brief 语义搜索：输入查询词，返回最像的 topK 篇文章
     * @param query  用户输入的查询词（UTF-8 编码）
     * @param topK   返回前几条结果（默认 10 条）
     * @return json  搜索结果数组
     *
     * 搜索逻辑（有两种模式，自动选择）：
     *
     * 【模式一：IVF 加速】（有 IVF 文件时）
     *   1. BERT 把查询词转成 512 维向量
     *   2. 看查询向量离哪几个"堆"最近（找最近的 nprobe 个组）
     *   3. 只看这几个堆里的文章，不扫描全部
     *   4. 算每篇文章向量和查询向量的"像不像"（点积 = 余弦相似度）
     *   5. 按相似度从高到低排，返回前 topK 个
     *
     * 【模式二：全表扫描】（没有 IVF 文件时，兜底）
     *   - 所有文章一一比对，虽然慢但保证能找到
     *
     * 性能对比（假设一万篇文章）：
     *   - 全表扫：看 10000 篇
     *   - IVF 加速：先看 100 个组找最近 → 再看最近 5 个组里的 ~500 篇
     *   - 大约快 16 倍
     */
    json find(const string& query, int topK = 10);

    /**
     * @brief 词语推荐：查一个词，推荐跟它相关的词
     * @param query  用户输入的关键词（UTF-8）
     * @param topK   返回前几个推荐词（默认 10）
     * @return json  推荐词列表（纯文本数组）
     *
     * 推荐逻辑（双引擎评分，各占一半）：
     *   1. jieba 引擎（50%）：在词典里找字形相近的词（编辑距离）
     *      比如搜"搜索引"能找到"搜索引擎"
     *   2. BGE 引擎（50%）：BERT 找语义相近的标题里的词
     *      比如搜"电脑"能找到"计算机"
     *   3. 综合分 = jieba 分 × 0.5 + BGE 分 × 0.5
     *   4. 按综合分排序，取前 topK 个
     */
    json suggest(const string& query, int topK = 10);

    /**
     * @brief jieba 模糊匹配：搜错了也能给你纠正
     * @param query  用户输入的内容
     * @param topK   返回前几个匹配词
     * @return json  匹配到的词列表
     *
     * 比如用户输入"搜索引"，能匹配到"搜索引擎"。
     * 底层用的是编辑距离算法（Levenshtein）。
     */
    json jiebaSuggest(const string& query, int topK = 5) const;

    /// 获取向量维度
    size_t dim() const { return engine_.dim(); }

    /// 获取文档总数（从 SQLite 查询）
    size_t size();

    /**
     * @brief 清除数据库中的所有数据
     * 用于重新构建索引时，清空旧的索引数据
     */
    void clear();

private:
    // ============================================================
    // 看看数据库里有多少数据
    // ============================================================
    void checkDbStats();

    // ============================================================
    // HTML 清洗工具函数
    // ============================================================

    /**
     * @brief 把 HTML 里的特殊符号转回正常字符
     *
     * 比如 "<" 转成 "<"，"&" 转成 "&"
     * 因为 XML 文章里有大量 HTML 编码，不转的话全是乱码
     */
    static string decodeHtmlEntities(const string& input);

    /**
     * @brief 去掉 HTML 标签，只留纯文字
     *
     * 原理很简单：看到 '<' 就忽略，直到看到 '>' 才恢复。
     * 这样 "<p>你好</p>" 就变成了 "你好"
     */
    static string stripHtml(const string& input);

    // ============================================================
    // 成员变量
    // ============================================================

    BertInferEngine engine_;        ///< BERT 模型引擎（负责把文字转向量）
    Configer& _conf;                ///< 配置器（读配置文件用）
    DictProducer dictProducer_;     ///< jieba 词典匹配器（编辑距离模糊匹配）
    sqlite3* db_;                   ///< SQLite 数据库连接
    string _dbPath;                 ///< 数据库文件路径

    // ---- IVF 分堆搜索相关 ----
    IVFIndex ivfIndex_;             ///< IVF 分堆索引（K-Means 聚类）
    string ivfIndexPath_;           ///<  IVF 索引文件存哪
    bool ivfTrained_ = false;       ///< IVF 训练好了没（有文件就算训练好）
    int ivfNprobe_ = 5;            ///< 搜索时看最近的几个堆（默认 5）

    // ---- LFU 缓存（提升 QPS） ----
    LfuCache searchCache_;          ///< 语义搜索结果缓存（find），容量 500
    LfuCache suggestCache_;         ///< 词语推荐结果缓存（suggest），容量 500

    inline static SemanticIndexer* _ptr = nullptr;  ///< 单例指针（整个程序只有一份）
};

#endif // SEMANTIC_INDEXER_HPP
