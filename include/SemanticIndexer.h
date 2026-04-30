#ifndef SEMANTIC_INDEXER_HPP
#define SEMANTIC_INDEXER_HPP

/**
 * @file SemanticIndexer.hpp
 * @brief 基于 BERT 句向量的语义搜索引擎（声明，SQLite 持久化）
 *
 * ============================================================
 * 架构变更说明（对比原 PageLibPreprocessor 倒排索引）
 * ============================================================
 *
 * 【旧架构】（已删除）
 *   XML → jieba 分词 → TF-IDF → 倒排索引 (_invertIndex)
 *   → 写入 newripepage.dat（原始文件 I/O）
 *   → 查询时 jieba 分词 → 集合交集 → 余弦相似度
 *
 * 【新架构】
 *   XML → BERT 编码 → 512维句向量 → SQLite 数据库（BLOB 存储）
 *   → 查询时 BERT 编码 → SQLite 读取全部向量 → 点积排序 → TopK
 *
 * 【核心原理】
 *   1. BERT 模型将文本映射到 512 维语义向量空间
 *   2. 语义相似的文本在向量空间中距离更近
 *   3. 使用点积（等价于余弦相似度，因已 L2 归一化）度量相似度
 *   4. SQLite 替代原始文件 I/O，提供结构化存储和快速查询
 *
 * 【为何用 SQLite 替代文件 I/O】
 *   1. 结构化存储：字段分离（id/title/url/content/embedding）
 *   2. 原子写入：避免文件写入崩溃导致数据损坏
 *   3. 可查询：支持按条件过滤（未来可按分类、时间等查询）
 *   4. 跨平台：SQLite 是嵌入式数据库，无需服务进程
 *   5. 序列化无痛：BLOB 类型直接存储二进制向量
 *
 * 使用示例：
 * @code
 *   SemanticIndexer::init("model/model.onnx", "model/tokenizer.json", conf);
 *   SemanticIndexer::getPtr()->buildIndex();        // 解析 XML → BERT 编码 → SQLite
 *   json results = SemanticIndexer::getPtr()->find("癌症治疗", 5);  // 语义搜索
 * @endcode
 *
 * 依赖：
 *   - InferEngine.h（BERT 推理引擎）
 *   - DictProducer.h（jieba 词典编辑距离匹配）
 *   - tinyxml2（解析 XML 文档库）
 *   - sqlite3（嵌入式数据库，C API）
 *   - Configer（读取配置文件）
 */

#include "InferEngine.h"
#include "DictProducer.h"
#include "Configer.h"
#include "sqlite3.h"
#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

#include "json.hpp"
#include "mylog.h"

using json = nlohmann::json;

// ============================================================
// SemanticIndexer —— 基于 SQLite 持久化的语义搜索引擎
// ============================================================

/**
 * @brief 语义搜索引擎，使用 BERT 模型 + SQLite 数据库
 *
 * 替代原有的三件套：
 *   PageLibPreprocessor（倒排索引）+
 *   DictProducer（词典构建）+
 *   jieba（中文分词）
 *
 * 全部统一为一个 SemanticIndexer：
 *   文档解析 → BERT 编码 → SQLite 存储 → 向量检索
 *
 * SQLite 表结构：
 * @code
 *   CREATE TABLE IF NOT EXISTS docs (
 *       id        INTEGER PRIMARY KEY AUTOINCREMENT,
 *       title     TEXT,
 *       url       TEXT,
 *       content   TEXT,
 *       embedding BLOB     -- 512 float32 = 2048 bytes
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
     * @brief 构造语义搜索引擎
     * @param model_path   ONNX 模型路径（如 "model/model.onnx"）
     * @param vocab_path   tokenizer.json 路径（如 "model/tokenizer.json"）
     * @param conf         配置器（用于读取 xmlPath, sqliteDbPath 等参数）
     *
     * 构造函数自动：
     *   1. 初始化 BERT 推理引擎
     *   2. 打开 SQLite 数据库（若不存在则创建）
     *   3. 创建 docs 表结构
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
     * @brief 初始化语义搜索引擎单例
     * @param model_path   ONNX 模型路径
     * @param vocab_path   tokenizer.json 路径
     * @param conf         配置器
     */
    static void init(const string& model_path, const string& vocab_path, Configer& conf);

    /// 获取单例指针
    static SemanticIndexer* getPtr();

    // ============================================================
    // 建索引 —— 从 XML 解析 → BERT 编码 → SQLite 存储
    // ============================================================

    /**
     * @brief 从 XML 文件构建语义索引（持久化到 SQLite）
     *
     * 处理流程：
     *   1. 扫描 xmlPath 目录下所有 XML 文件
     *   2. 解析每个 RSS item（title + description + link）
     *   3. 清洗 HTML 标签，解码 HTML 实体
     *   4. 使用 BertInferEngine 编码为 512 维句向量
     *   5. 将 title/url/content/embedding 写入 SQLite docs 表
     *
     * 底层原理：
     *   - BERT 模型的 [CLS] 位置输出经过池化后作为句向量
     *   - 句向量经过 L2 归一化，使点积 = 余弦相似度
     *   - embedding 以 BLOB 格式存储（512 float32 = 2048 bytes）
     *   - SQLite 事务确保批量写入的原子性和性能
     */
    void buildIndex();

    // ============================================================
    // 语义搜索 —— BERT 编码查询 → SQLite 读取 → 向量检索
    // ============================================================

    /**
     * @brief 执行语义搜索
     * @param query  用户查询字符串（UTF-8）
     * @param topK   返回前 K 个结果（默认 10）
     * @return json  搜索结果（格式兼容原 PageLibPreprocessor::find()）
     *
     * 搜索流程：
     *   1. 使用 BERT 对查询文本编码为 512 维向量
     *   2. 从 SQLite 读取所有文档的 title/url/content/embedding
     *   3. 计算查询向量与所有文档向量的点积（余弦相似度）
     *   4. 按相似度降序排序
     *   5. 返回 TopK 结果
     *
     * 底层原理：
     *   - 由于所有向量已 L2 归一化，点积 = 余弦相似度
     *   - 余弦相似度衡量两个向量方向的接近程度
     *   - 相比 TF-IDF 关键词匹配，语义搜索能召回"同义词"和"相关概念"
     *
     * 性能说明：
     *   - 每次查询需要从 SQLite 读取所有 embedding（全表扫描）
     *   - 适用于文档数量在数万级别的场景
     *   - 百万级以上需要 ANN（近似最近邻）索引如 FAISS
     */
    json find(const string& query, int topK = 10);

    /**
     * @brief 词语推荐（jieba 50% + BGE 50% 综合评分）
     * @param query  用户输入的部分查询（UTF-8）
     * @param topK   返回前 K 个推荐词（默认 10）
     * @return json  推荐词语数组（JSON 字符串数组）
     *
     * 实现原理（双引擎融合）：
     *   1. jieba 引擎（50%）：遍历词典，Levenshtein 编辑距离 → 相似度 score_j
     *   2. BGE 引擎（50%）：BERT 编码 → SQLite title_embedding ANN → 语义词提取 → score_b
     *   3. 综合评分 = 0.5 × score_j + 0.5 × score_b
     *   4. 按综合评分排序，返回 topK
     *
     * 设计思路：
     *   - jieba：保证词汇级别的"长得像"（纠错、补全）
     *   - BGE：保证语义级别的"意思像"（同义词、相关概念）
     *   - 50/50 平衡，避免纯语义推荐太泛或纯编辑距离太死板
     */
    json suggest(const string& query, int topK = 10);

    /**
     * @brief jieba 降准匹配（编辑距离模糊匹配）
     * @param query  查询字符串
     * @param topK   返回前 K 个
     * @return json  模糊匹配的词数组（JSON 字符串数组）
     *
     * 用于 /search 端点的"您是不是想找"提示。
     * 底层调用 DictProducer::find()，基于 Levenshtein 编辑距离。
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
    // 数据库统计
    // ============================================================

    /// 检查数据库中的文档数量和其他统计信息
    void checkDbStats();

    // ============================================================
    // HTML 清洗工具函数
    // ============================================================

    /**
     * @brief 解码常用的 HTML 实体
     *
     * 将 "<", ">", "&", """, "'", "&nbsp;" 等
     * HTML 实体替换为对应的字符。这些工具函数从原 PageLibPreprocessor
     * 迁移至此，替代了 jieba 分词和 TF-IDF 处理。
     */
    static string decodeHtmlEntities(const string& input);

    /**
     * @brief 去除 HTML 标签，保留纯文本
     *
     * 简单的状态机：遇到 '<' 进入 tag 模式，遇到 '>' 退出。
     * 标签结束后添加一个空格分隔文本。
     */
    static string stripHtml(const string& input);

    // ============================================================
    // 成员变量
    // ============================================================

    BertInferEngine engine_;        ///< BERT 推理引擎（封装 ONNX Runtime）
    Configer& _conf;                ///< 配置器引用（必须在 dictProducer_ 之前声明）
    DictProducer dictProducer_;     ///< jieba 词典编辑距离匹配器（构造依赖 _conf）
    sqlite3* db_;                   ///< SQLite 数据库连接指针
    string _dbPath;                 ///< SQLite 数据库文件路径

    inline static SemanticIndexer* _ptr = nullptr;  ///< 单例指针（C++17 inline static）
};

#endif // SEMANTIC_INDEXER_HPP
