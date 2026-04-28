#ifndef WORDPIECE_TOKENIZER_HPP
#define WORDPIECE_TOKENIZER_HPP

/**
 * @file WordPieceTokenizer.hpp
 * @brief 基于 BERT WordPiece 算法的 Header-Only 分词器库（纯 UTF-8 实现）
 *
 * 该库实现了完整的 BERT WordPiece 分词流程，包括：
 * - 中文汉字自动加空格（pad_chinese_chars）
 * - 标点符号分割（run_split_on_punctuation）
 * - 特殊标记保护（基于 Trie 树的 Splitter）
 * - WordPiece 子词切分（wordpiece_tokenize）
 * - 自动添加 [CLS] / [SEP] 标记
 *
 * 所有字符串均为 UTF-8 编码的 std::string，无平台相关的宽字符依赖。
 *
 * 依赖：
 *   - nlohmann/json.hpp（解析 tokenizer.json 配置文件）
 *   - ICU (libicuuc) 库（Unicode 字符类型判断）
 *
 * 使用方式：
 * @code
 *   WordPieceTokenizer tokenizer("model/tokenizer.json");
 *   auto ids = tokenizer.tokenize_full("你好世界 Hello World!");
 * @endcode
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "json/json.hpp"
#include <unicode/uchar.h>

// ============================================================================
// 前置声明与类型别名
// ============================================================================

using json = nlohmann::json;

// ============================================================================
// UTF-8 编解码工具
// ============================================================================

/**
 * @brief 获取 UTF-8 字符的字节长度（根据首字节判断）
 * @param lead  UTF-8 字符的首字节
 * @return 字符的字节数（1-4），若为非法的 continuation byte 则返回 1
 */
inline size_t utf8_char_length(unsigned char lead)
{
    if (lead < 0x80) return 1;
    if (lead < 0xC0) return 1;  // continuation byte → 视为单字节
    if (lead < 0xE0) return 2;
    if (lead < 0xF0) return 3;
    return 4;
}

/**
 * @brief 从 UTF-8 字符串的指定位置解码一个 Unicode 码点
 *
 * 解码后自动移动 pos 指向下一个字符的起始位置。
 *
 * @param str  UTF-8 编码的字符串
 * @param pos  当前读取位置（输入输出参数，解码后推进）
 * @return 解码后的 Unicode 码点（UChar32）
 */
inline UChar32 utf8_decode(const std::string& str, size_t& pos)
{
    unsigned char lead = static_cast<unsigned char>(str[pos]);
    size_t len = utf8_char_length(lead);
    UChar32 cp = 0;
    switch (len) {
        case 1:
            cp = lead;
            break;
        case 2:
            cp = ((lead & 0x1F) << 6) |
                 (static_cast<unsigned char>(str[pos + 1]) & 0x3F);
            break;
        case 3:
            cp = ((lead & 0x0F) << 12) |
                 ((static_cast<unsigned char>(str[pos + 1]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(str[pos + 2]) & 0x3F);
            break;
        case 4:
            cp = ((lead & 0x07) << 18) |
                 ((static_cast<unsigned char>(str[pos + 1]) & 0x3F) << 12) |
                 ((static_cast<unsigned char>(str[pos + 2]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(str[pos + 3]) & 0x3F);
            break;
    }
    pos += len;
    return cp;
}

/**
 * @brief 将一个 Unicode 码点编码为 UTF-8 字符串
 * @param cp  Unicode 码点
 * @return UTF-8 编码的字符串（1-4 字节）
 */
inline std::string utf8_encode(UChar32 cp)
{
    std::string result;
    if (cp < 0x80) {
        result += static_cast<char>(cp);
    } else if (cp < 0x800) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}

// ============================================================================
// 辅助工具函数
// ============================================================================

/**
 * @brief 字符分类谓词模板，用于判断字符是否**不**属于给定的 ctype 分类
 *
 * 配合 std::find_if 使用，可过滤掉不属于指定分类的字符（如空格）。
 * 仅适用于单字节 ASCII 字符。
 *
 * @tparam mask  ctype_base 中的分类掩码（如 std::ctype_base::space）
 */
template <std::ctype_base::mask mask>
class IsNot
{
    std::locale myLocale;
    std::ctype<char> const* myCType;

public:
    explicit IsNot(std::locale const& l = std::locale())
        : myLocale(l)
        , myCType(&std::use_facet<std::ctype<char>>(l))
    {
    }

    /**
     * @brief 判断字符是否不属于指定分类
     * @param ch  待判断字符
     * @return true 如果字符不属于 mask 分类
     */
    bool operator()(char ch) const
    {
        return !myCType->is(mask, ch);
    }
};

typedef IsNot<std::ctype_base::space> IsNotSpace;

/**
 * @brief 去除字符串首尾的空白字符（ASCII 空白）
 * @param original  原始字符串
 * @return 去除首尾空白后的新字符串
 */
inline std::string trim(std::string const& original)
{
    std::string::const_iterator right =
        std::find_if(original.rbegin(), original.rend(), IsNotSpace()).base();
    std::string::const_iterator left =
        std::find_if(original.begin(), right, IsNotSpace());
    return std::string(left, right);
}

/**
 * @brief 以空白字符分割 UTF-8 字符串，返回单词列表
 *
 * 使用 std::stringstream 的 >> 操作符按空白切分（仅 ASCII 空白符）。
 *
 * @param input  输入的 UTF-8 字符串
 * @return 分割后的字符串向量
 */
inline std::vector<std::string> split(const std::string& input)
{
    std::stringstream stream(input);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

/**
 * @brief 判断 Unicode 字符是否为标点符号（使用 ICU 库的 u_charType）
 * @param charCode  Unicode 码点（UChar32）
 * @return true 如果是标点符号
 *
 * 覆盖以下标点分类：
 *   U_DASH_PUNCTUATION, U_START_PUNCTUATION, U_END_PUNCTUATION,
 *   U_CONNECTOR_PUNCTUATION, U_OTHER_PUNCTUATION,
 *   U_INITIAL_PUNCTUATION, U_FINAL_PUNCTUATION
 */
inline bool isPunctuation(UChar32 charCode)
{
    UCharCategory category = static_cast<UCharCategory>(u_charType(charCode));

    switch (category) {
        case U_DASH_PUNCTUATION:
        case U_START_PUNCTUATION:
        case U_END_PUNCTUATION:
        case U_CONNECTOR_PUNCTUATION:
        case U_OTHER_PUNCTUATION:
        case U_INITIAL_PUNCTUATION:
        case U_FINAL_PUNCTUATION:
            return true;
        default:
            return false;
    }
}

/**
 * @brief 综合判断字符是否为标点符号
 *
 * 同时检查 ASCII 标点范围（33-47, 58-64, 91-96, 123-126）
 * 以及 Unicode 标点（通过 isPunctuation 调用 ICU）。
 *
 * @param c  Unicode 码点
 * @return true 如果是标点符号
 */
inline bool _is_punctuation(UChar32 c)
{
    if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
        (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) {
        return true;
    }
    if (isPunctuation(c)) {
        return true;
    }
    return false;
}

/**
 * @brief 判断字符是否为 CJK 汉字
 *
 * 覆盖所有 CJK 统一表意文字区块：
 *   - CJK Unified Ideographs (0x4E00–0x9FFF)
 *   - Extension A (0x3400–0x4DBF)
 *   - Extension B (0x20000–0x2A6DF)
 *   - Extension C (0x2A700–0x2B73F)
 *   - Extension D (0x2B740–0x2B81F)
 *   - Extension E (0x2B820–0x2CEAF)
 *   - Compatibility Ideographs (0xF900–0xFAFF)
 *   - Compatibility Supplement (0x2F800–0x2FA1F)
 *
 * @param c  Unicode 码点
 * @return true 如果是 CJK 汉字
 */
inline bool _is_chinese_char(UChar32 c)
{
    if ((c >= 0x4E00 && c <= 0x9FFF) ||       // CJK Unified Ideographs
        (c >= 0x3400 && c <= 0x4DBF) ||       // Extension A
        (c >= 0x20000 && c <= 0x2A6DF) ||     // Extension B
        (c >= 0x2A700 && c <= 0x2B73F) ||     // Extension C
        (c >= 0x2B740 && c <= 0x2B81F) ||     // Extension D
        (c >= 0x2B820 && c <= 0x2CEAF) ||     // Extension E
        (c >= 0xF900 && c <= 0xFAFF) ||       // CJK Compatibility Ideographs
        (c >= 0x2F800 && c <= 0x2FA1F)) {     // Compatibility Supplement
        return true;
    }
    return false;
}

/**
 * @brief 在中文汉字前后添加空格（BERT 标准预处理步骤）
 *
 * 将每个中文字符用空格包裹，使其在后续 whitespace tokenize 时
 * 被当作独立的 token 处理。这是 BERT 论文中描述的标准做法。
 *
 * 输入和输出均为 UTF-8 编码。
 *
 * @param text  输入的 UTF-8 字符串
 * @return 每个汉字前后都加上了空格的 UTF-8 字符串
 */
inline std::string pad_chinese_chars(const std::string& text)
{
    std::string result;
    size_t pos = 0;
    while (pos < text.length()) {
        size_t char_start = pos;
        UChar32 cp = utf8_decode(text, pos);
        if (_is_chinese_char(cp)) {
            result += ' ';
            result += text.substr(char_start, pos - char_start);
            result += ' ';
        } else {
            result += text.substr(char_start, pos - char_start);
        }
    }
    return result;
}

/**
 * @brief 对标点符号进行分割（BERT BasicTokenizer 步骤）
 *
 * 将连续的标点符号作为独立 token 切分出来。
 * 如果文本是一个特殊标记（在 special_tokens 中）且 split_specials 为 false，
 * 则直接返回原文本而不进行分割。
 *
 * 输入和输出均为 UTF-8 编码。
 *
 * @param text            待分割的 UTF-8 字符串
 * @param split_specials  是否分割特殊标记
 * @param special_tokens  特殊标记列表（[CLS], [SEP], [UNK], [PAD], [MASK] 等）
 * @return 分割后的字符串向量
 */
inline std::vector<std::string> run_split_on_punctuation(
    const std::string& text,
    bool split_specials,
    const std::vector<std::string>& special_tokens)
{
    // 如果是特殊标记且不允许拆分，则整体返回
    if (!split_specials &&
        std::find(special_tokens.begin(), special_tokens.end(), text) != special_tokens.end()) {
        return std::vector<std::string>{text};
    }

    size_t i = 0;
    bool start_new_word = true;
    std::vector<std::string> output;

    while (i < text.length()) {
        size_t char_start = i;
        UChar32 cp = utf8_decode(text, i);
        std::string cur_char = text.substr(char_start, i - char_start);

        if (_is_punctuation(cp)) {
            // 标点符号作为独立 token
            output.push_back(cur_char);
            start_new_word = true;
        } else {
            if (start_new_word) {
                output.push_back(std::string());
            }
            start_new_word = false;
            output.back() += cur_char;
        }
    }

    return output;
}

// ============================================================================
// Trie 树 — 用于高效多模式串匹配（特殊标记分隔）
// ============================================================================

/**
 * @brief Trie 树节点
 *
 * 用于实现多模式串匹配的分层前缀树。每个节点存储子节点映射、
 * 是否为某个模式的结尾以及对应的完整分隔符字符串。
 *
 * @note 子节点键为 char（单字节），适用于 ASCII 特殊标记的匹配。
 *       若特殊标记包含非 ASCII 字符，需要修改为按码点匹配。
 */
class TrieNode {
public:
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    bool is_end;
    std::string delimiter;

    TrieNode()
        : is_end(false)
    {
    }
};

/**
 * @brief 基于 Trie 树的多分隔符文本分割器
 *
 * 与原始 HuggingFace 实现一致，能够同时匹配多个分隔符（如特殊标记），
 * 并在匹配处分隔文本。常用于在 tokenize 过程中保留 [CLS]、[SEP] 等
 * 特殊标记的完整性。
 *
 * @note 分隔符仅支持 ASCII 字符（如 [CLS], [SEP], [UNK] 等）。
 */
class Splitter {
private:
    std::unique_ptr<TrieNode> root;

    /**
     * @brief 向 Trie 树中插入一个分隔符模式
     * @param str  待插入的分隔符字符串
     */
    void insert(const std::string& str)
    {
        TrieNode* current = root.get();
        for (char ch : str) {
            if (!current->children[ch]) {
                current->children[ch] = std::unique_ptr<TrieNode>(new TrieNode());
            }
            current = current->children[ch].get();
        }
        current->is_end = true;
        current->delimiter = str;
    }

public:
    /**
     * @brief 构造函数
     * @param delimiters  所有待匹配的分隔符列表
     */
    explicit Splitter(const std::vector<std::string>& delimiters)
    {
        root = std::unique_ptr<TrieNode>(new TrieNode());
        for (const auto& delimiter : delimiters) {
            insert(delimiter);
        }
    }

    /**
     * @brief 使用已注册的分隔符分割输入文本
     *
     * 采用最长匹配原则（maximum munch），在每次匹配到分隔符时切分。
     * 分隔符本身**会**作为独立片段出现在输出结果中（由后续步骤处理）。
     *
     * @param input  待分割的 UTF-8 字符串
     * @return 分割后的文本片段列表
     */
    std::vector<std::string> split(const std::string& input)
    {
        std::vector<std::string> result;
        size_t start = 0;

        while (start < input.length()) {
            // 尝试从当前位置查找最长的分隔符匹配
            size_t best_match_length = 0;
            std::string matched_delimiter;

            TrieNode* current = root.get();
            size_t pos = start;

            while (pos < input.length() && current->children.count(input[pos])) {
                current = current->children[input[pos]].get();
                pos++;
                if (current->is_end) {
                    best_match_length = pos - start;
                    matched_delimiter = current->delimiter;
                }
            }

            if (best_match_length > 0) {
                // 将匹配到的分隔符本身也作为一段输出
                result.push_back(input.substr(start, best_match_length));
                start += best_match_length;
            } else {
                // 当前位置无分隔符匹配：向后找到下一个可能的分隔符起始位置
                size_t next_pos = start + 1;
                bool found_next = false;

                while (next_pos < input.length()) {
                    if (root->children.count(input[next_pos])) {
                        found_next = true;
                        break;
                    }
                    next_pos++;
                }

                // 提取从 start 到下一个候选位置（或末尾）的文本
                result.push_back(
                    input.substr(start,
                                 (found_next ? next_pos - start : std::string::npos)));
                start = next_pos;
            }
        }

        return result;
    }
};

// ============================================================================
// WordPieceTokenizer 主类
// ============================================================================

/**
 * @brief BERT WordPiece 分词器
 *
 * 完整实现 HuggingFace Transformers 中 BertTokenizer 的 WordPiece 算法流程：
 *
 * 1. pad_chinese_chars   — 中文字符周围加空格
 * 2. whitespace split    — 按空白切分
 * 3. Splitter.split      — 保护特殊标记不分割
 * 4. run_split_on_punctuation — 标点符号分割
 * 5. wordpiece_tokenize  — 贪心最长匹配子词切分（添加 ## 前缀）
 * 6. 自动添加 [CLS] 和 [SEP]
 *
 * 所有字符串均为 UTF-8 编码的 std::string。
 * 配置文件格式与 HuggingFace tokenizer.json 完全兼容。
 */
class WordPieceTokenizer {
private:
    json jsonObj;                           ///< 解析后的完整配置 JSON 对象
    json vocab;                             ///< 词汇表（model.vocab）
    size_t max_input_chars_per_word;        ///< 单词最大字符数，超出则映射为 UNK
    std::string unk_token;                  ///< UNK 标记字符串
    std::vector<std::string> special_tokens; ///< 特殊标记列表

public:
    /**
     * @brief 从 HuggingFace tokenizer.json 配置文件构造分词器
     * @param config_path  tokenizer.json 的文件路径
     *
     * 构造函数会自动读取以下配置项：
     *   - model.vocab                   ：词汇表（word -> id 映射）
     *   - model.max_input_chars_per_word ：单词最大字符数
     *   - model.unk_token               ：UNK 标记字符串
     *   - added_tokens[].special        ：特殊标记列表
     */
    explicit WordPieceTokenizer(const std::string& config_path)
    {
        std::ifstream file(config_path);
        file >> jsonObj;
        vocab = jsonObj["model"]["vocab"];
        max_input_chars_per_word = jsonObj["model"]["max_input_chars_per_word"];
        unk_token = jsonObj["model"]["unk_token"].get<std::string>();

        // 收集所有 special = true 的 added token 作为特殊标记
        for (auto& item : jsonObj["added_tokens"]) {
            if (item["special"]) {
                special_tokens.push_back(item["content"].get<std::string>());
            }
        }
    }

    /**
     * @brief 获取单词在词汇表中的索引
     * @param word  待查询的字符串
     * @return 词汇表中的 id（int），如果未找到则返回 -1
     */
    int get_word_index(const std::string& word)
    {
        if (vocab.find(word) != vocab.end()) {
            return vocab[word];
        } else {
            return -1;
        }
    }

    /**
     * @brief 执行完整的分词流程，返回 token ID 序列
     *
     * 流程：
     *   1. pad_chinese_chars          — 汉字周围加空格
     *   2. whitespace split            — 按空白切分
     *   3. Splitter（特殊标记保护）    — 基于 Trie 的分隔
     *   4. run_split_on_punctuation    — 标点分割
     *   5. wordpiece_tokenize          — WordPiece 子词切分
     *   6. 自动添加 [CLS]（id=101）和 [SEP]（id=102）
     *
     * @param input_text     输入的 UTF-8 文本
     * @param split_specials 是否允许分割特殊标记（默认 false）
     * @return token ID 序列，第一个元素为 [CLS]，最后一个为 [SEP]
     */
    std::vector<size_t> tokenize_full(const std::string& input_text,
                                      bool split_specials = false)
    {
        // Step 1: 对中文字符添加空格
        std::string padded_text = pad_chinese_chars(input_text);

        // Step 2: 空白切分
        std::vector<std::string> tokens = split(padded_text);

        // Step 3: 使用 Trie Splitter 保护特殊标记不被拆分
        Splitter splitter(special_tokens);

        std::vector<std::string> special_word_tokenized;
        for (size_t i = 0; i < tokens.size(); i++) {
            auto split_by_special = splitter.split(tokens[i]);
            special_word_tokenized.insert(special_word_tokenized.end(),
                                          split_by_special.begin(),
                                          split_by_special.end());
        }

        // Step 4: 标点符号分割
        std::vector<std::string> basic_tokenized;
        for (size_t i = 0; i < special_word_tokenized.size(); i++) {
            auto splitted_by_punc = run_split_on_punctuation(
                special_word_tokenized[i], split_specials, special_tokens);
            basic_tokenized.insert(basic_tokenized.end(),
                                   splitted_by_punc.begin(),
                                   splitted_by_punc.end());
        }

        // Step 5: WordPiece 子词切分
        std::vector<std::string> wordpiece_tokenized;
        for (size_t i = 0; i < basic_tokenized.size(); i++) {
            auto splitted_by_wordpiece = wordpiece_tokenize(basic_tokenized[i]);
            wordpiece_tokenized.insert(wordpiece_tokenized.end(),
                                       splitted_by_wordpiece.begin(),
                                       splitted_by_wordpiece.end());
        }

        // Step 6: 添加 [CLS] 和 [SEP]
        std::vector<size_t> tokenized_ids;
        tokenized_ids.push_back(get_word_index("[CLS]"));
        std::vector<size_t> seq_ids = convert_tokens_to_ids(wordpiece_tokenized);
        tokenized_ids.insert(tokenized_ids.end(), seq_ids.begin(), seq_ids.end());
        tokenized_ids.push_back(get_word_index("[SEP]"));

        return tokenized_ids;
    }

    /**
     * @brief WordPiece 子词切分（核心算法）
     *
     * 对每个单词执行基于词汇表的贪心最长匹配切分：
     *   - 如果单词长度 > max_input_chars_per_word，直接返回 UNK
     *   - 从左到右查找词汇表中匹配的最长子串
     *   - 除首个子词外，所有子词添加 "##" 前缀（表示是前一词的续接）
     *   - 如果某个字符无法被任何子词覆盖，标记为 bad 并整体返回 UNK
     *
     * @param input_text  待切分的单个单词（UTF-8）
     * @return 切分后的子词列表
     */
    std::vector<std::string> wordpiece_tokenize(const std::string& input_text)
    {
        std::vector<std::string> tokens = split(input_text);
        std::vector<std::string> output_tokens;

        for (size_t i = 0; i < tokens.size(); i++) {
            auto& tok = tokens[i];
            if (tok.length() > max_input_chars_per_word) {
                output_tokens.push_back(unk_token);
                continue;
            }

            bool is_bad = false;
            size_t start = 0;
            std::vector<std::string> sub_tokens;

            while (start < tok.length()) {
                size_t end = tok.length();
                std::string cur_substr;

                // 贪心：从最长到最短依次尝试匹配
                while (start < end) {
                    std::string substr = tok.substr(start, end - start);
                    if (start > 0) {
                        substr = "##" + substr;  // 续接前缀
                    }
                    int idx = get_word_index(substr);
                    if (idx != -1) {
                        cur_substr = substr;
                        break;
                    }
                    end--;
                }

                if (cur_substr.empty()) {
                    is_bad = true;
                    break;
                }
                sub_tokens.push_back(cur_substr);
                start = end;
            }

            if (is_bad) {
                output_tokens.push_back(unk_token);
            } else {
                output_tokens.insert(output_tokens.end(),
                                     sub_tokens.begin(),
                                     sub_tokens.end());
            }
        }
        return output_tokens;
    }

    /**
     * @brief 将 token 序列转换为对应的 ID 序列
     * @param input_seq  token 字符串向量
     * @return token ID 向量
     */
    std::vector<size_t> convert_tokens_to_ids(const std::vector<std::string>& input_seq)
    {
        std::vector<size_t> output_ids;
        for (size_t i = 0; i < input_seq.size(); i++) {
            output_ids.push_back(get_word_index(input_seq[i]));
        }
        return output_ids;
    }
};

#endif  // WORDPIECE_TOKENIZER_HPP
