#ifndef INFER_ENGINE_HPP
#define INFER_ENGINE_HPP

/**
 * @file InferEngine.hpp
 * @brief 基于 ONNX Runtime C API 的 BERT 模型推理引擎（声明）
 *
 * 使用 ONNX Runtime 1.25.1 的 C API（onnxruntime_c_api.h）完成模型加载和推理。
 * 注意：ONNX Runtime 1.25.1 的 C API 头文件中使用了 noexcept 作为函数指针类型的一部分
 * （C++17 特性），因此本项目需要使用 C++17 编译。
 *
 * 本文件提供：
 * 1. BertInferEngine —— 加载 model.onnx，执行 BERT 推理，返回 512 维句向量
 * 2. 自动处理 tokenize、padding、mean pooling、L2 归一化
 *
 * 依赖：
 *   - onnxruntime_c_api.h（位于 model/onnxruntime_lib/include/）
 *   - libonnxruntime.so（位于 model/onnxruntime_lib/lib/）
 *   - WordPieceTokenizer.hpp
 */

#include <onnxruntime_c_api.h>
#include "WordPieceTokenizer.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// 常量定义
// ============================================================

/// BERT 模型最大序列长度（来自 config.json max_position_embeddings）
static const size_t BERT_MAX_SEQ_LEN = 512;

/// BERT 隐藏层维度（来自 config.json hidden_size）
static const size_t BERT_HIDDEN_SIZE = 512;

// ============================================================
// BertInferEngine —— BERT 模型推理引擎
// ============================================================

/**
 * @brief BERT 模型推理引擎
 *
 * 封装 ONNX Runtime 的完整推理流程：
 * 1. 加载 model.onnx
 * 2. 接收文本 → WordPieceTokenizer 分词 → Padding
 * 3. 构建 input_ids / attention_mask / token_type_ids 三个张量
 * 4. 执行 ONNX Runtime 推理
 * 5. 从 last_hidden_state 提取句向量（mean pooling + L2 归一化）
 *
 * 使用示例：
 * @code
 *   BertInferEngine engine("model/model.onnx", "model/tokenizer.json");
 *   std::vector<float> emb = engine.encode("你好世界");
 *   // emb.size() == 512
 * @endcode
 *
 * @note 必须使用 C++17 或更高版本编译（ONNX Runtime 1.25.1 要求）
 *
 * @note 线程安全说明：
 *   本类使用单例共享的 OrtSession，不是线程安全的。
 *   多线程同时调用 encode() 会导致崩溃，因为 OrtSession::Run() 内部有状态。
 *   目前搜索引擎使用单线程处理请求（线程池队列本质上是串行处理的），
 *   如果有并发需求，请将 OrtSession 改为 thread_local 每线程独立加载。
 */
class BertInferEngine {
public:
    /**
     * @brief 构造推理引擎
     * @param model_path    ONNX 模型文件路径（如 "model/model.onnx"）
     * @param vocab_path    tokenizer.json 路径（由 WordPieceTokenizer 使用）
     *
     * @throws std::runtime_error 如果模型加载或 ORT 初始化失败
     */
    BertInferEngine(const std::string& model_path, const std::string& vocab_path);

    /// 析构函数 —— 自动通过 unique_ptr 释放所有 ORT 资源
    ~BertInferEngine() = default;

    // 禁止拷贝
    BertInferEngine(const BertInferEngine&) = delete;
    BertInferEngine& operator=(const BertInferEngine&) = delete;

    // ============================================================
    // 公有接口
    // ============================================================

    /**
     * @brief 将文本编码为 512 维句向量
     * @param text  输入文本（UTF-8 编码）
     * @return std::vector<float>  L2 归一化后的 512 维句向量
     *
     * 流程：Tokenizer → Padding → BERT 推理 → Mean Pooling → L2 归一化
     */
    std::vector<float> encode(const std::string& text);

    /**
     * @brief 批量编码多段文本
     * @param texts  输入文本数组
     * @return std::vector<std::vector<float>>  每个文本对应的 512 维句向量
     */
    std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts);

    /// 获取 embedding 维度
    size_t dim() const { return BERT_HIDDEN_SIZE; }

    /// 获取最大序列长度
    size_t max_seq_len() const { return BERT_MAX_SEQ_LEN; }

private:
    // ============================================================
    // 内部类：ONNX Runtime 资源 RAII 包装
    // ============================================================

    /// 用于 OrtEnv 的删除器
    struct EnvDeleter {
        const OrtApi* api = nullptr;
        void operator()(OrtEnv* p) const {
            if (p && api) api->ReleaseEnv(p);
        }
    };

    /// 用于 OrtSessionOptions 的删除器
    struct SessionOptionsDeleter {
        const OrtApi* api = nullptr;
        void operator()(OrtSessionOptions* p) const {
            if (p && api) api->ReleaseSessionOptions(p);
        }
    };

    /// 用于 OrtSession 的删除器
    struct SessionDeleter {
        const OrtApi* api = nullptr;
        void operator()(OrtSession* p) const {
            if (p && api) api->ReleaseSession(p);
        }
    };

    /// 用于 OrtMemoryInfo 的删除器
    struct MemoryInfoDeleter {
        const OrtApi* api = nullptr;
        void operator()(OrtMemoryInfo* p) const {
            if (p && api) api->ReleaseMemoryInfo(p);
        }
    };

    /// 用于 OrtValue 的删除器
    struct ValueDeleter {
        const OrtApi* api = nullptr;
        void operator()(OrtValue* p) const {
            if (p && api) api->ReleaseValue(p);
        }
    };

    using EnvPtr             = std::unique_ptr<OrtEnv,             EnvDeleter>;
    using SessionOptionsPtr  = std::unique_ptr<OrtSessionOptions,  SessionOptionsDeleter>;
    using SessionPtr         = std::unique_ptr<OrtSession,         SessionDeleter>;
    using MemoryInfoPtr      = std::unique_ptr<OrtMemoryInfo,      MemoryInfoDeleter>;
    using ValuePtr           = std::unique_ptr<OrtValue,           ValueDeleter>;

    // ============================================================
    // 内部方法
    // ============================================================

    /**
     * @brief 检查 OrtStatus，失败时抛出异常
     */
    static void check_status(OrtStatus* status, const std::string& msg);

    /**
     * @brief 对单条文本执行完整推理，返回归一化句向量
     */
    std::vector<float> encode_single(const std::string& text);

    // ============================================================
    // 成员变量
    // ============================================================

    const OrtApi* api_ = nullptr;                      ///< ONNX Runtime C API 函数表
    EnvPtr            env_{nullptr, {nullptr}};         ///< ORT 环境
    SessionOptionsPtr sess_opts_{nullptr, {nullptr}};  ///< Session 选项
    MemoryInfoPtr     mem_info_{nullptr,  {nullptr}};  ///< CPU 内存信息
    SessionPtr        session_{nullptr, {nullptr}};     ///< 全局共享的 ONNX Session（单线程使用）

    WordPieceTokenizer tokenizer_;                     ///< WordPiece 分词器

    const char* input_names_[3]  = {};                 ///< 输入张量名称
    const char* output_names_[1] = {};                 ///< 输出张量名称

    // 输入缓冲区（单线程共享，多线程调用需要外部加锁）
    std::vector<int64_t> input_ids_buf_;   ///< input_ids 缓冲区
    std::vector<int64_t> attn_mask_buf_;   ///< attention_mask 缓冲区
    std::vector<int64_t> token_type_buf_;  ///< token_type_ids 缓冲区
};

#endif // INFER_ENGINE_HPP
