#ifndef DICT_PRODUCER_H
#define DICT_PRODUCER_H

/**
 * @file DictProducer.h
 * @brief 词典构建器 — 加载 jieba 词典，提供编辑距离模糊匹配
 *
 * 用于 /suggest 和 /search 端点的降准匹配层。
 * 加载 dict/jieba.dict.utf8 格式的词典（word freq pos_tag），
 * 通过编辑距离（Levenshtein）计算候选词的相似度分数。
 *
 * 【与 BERT 的协同】
 *   - suggest() 最终评分 = 50% jieba 编辑距离 + 50% BGE 语义相似度
 *   - 确保纯词汇相似（输入法纠错）和语义相似（概念关联）的平衡
 */

#include <string>
#include <vector>
#include <utility>

class DictProducer {
public:
    /**
     * @brief 构造词典，加载词典文件
     * @param dictPath 词典文件路径（如 "dict/jieba.dict.utf8"）
     *
     * 词典格式：每行 "word freq pos_tag"
     * 例如：搜索引擎 5 n
     * 仅提取 word 部分，跳过 freq 和 pos_tag
     */
    explicit DictProducer(const std::string& dictPath);

    /**
     * @brief 基于编辑距离的模糊匹配（降准匹配）
     * @param query  查询字符串（UTF-8）
     * @param topK   返回前 K 个最相似的词
     * @return vector<pair<word, score>> 按相似度降序排列
     *
     * 算法流程：
     *   1. 首字符过滤：仅匹配 query[0] 相同的词（大幅减少计算量）
     *   2. 长度过滤：词长差距超过 queryLen/2+1 的跳过
     *   3. Levenshtein 编辑距离（两行 DP 优化）
     *   4. 相似度 = 1 - editDist / max(len1, len2)
     *   5. 过滤 score > 0.3 的结果
     */
    std::vector<std::pair<std::string, float>> find(const std::string& query, int topK = 10) const;

    /// 获取词典大小
    size_t size() const { return _words.size(); }

    /// 获取所有词典词条（用于遍历）
    const std::vector<std::string>& words() const { return _words; }

private:
    /**
     * @brief Levenshtein 编辑距离（两行 DP 优化）
     * @param s1 字符串1
     * @param s2 字符串2
     * @return 编辑距离（插入/删除/替换次数）
     *
     * 使用两行交替的 DP 数组，空间复杂度 O(min(m,n))
     * 时间复杂度 O(m*n)
     */
    static size_t _editDistance(const std::string& s1, const std::string& s2);

    std::vector<std::string> _words;  ///< 词典中的所有词（UTF-8 编码）
};

#endif // DICT_PRODUCER_H
