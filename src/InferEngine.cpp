/**
 * @file InferEngine.cpp
 * @brief BertInferEngine 实现 — BERT 模型推理引擎
 *
 * 实现细节：
 * 1. ONNX Runtime C API 初始化与模型加载
 * 2. WordPieceTokenizer 分词 → Padding → 张量构建
 * 3. ONNX Runtime 推理执行
 * 4. Mean Pooling + L2 归一化提取句向量
 */

#include "InferEngine.h"

// ============================================================
// 构造函数 —— 初始化 ONNX Runtime 并加载模型
// ============================================================

BertInferEngine::BertInferEngine(const std::string& model_path, const std::string& vocab_path)
    : tokenizer_(vocab_path)
{
    // ---- 1. 获取 OrtApi 函数表 ----
    api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api_) {
        throw std::runtime_error("BertInferEngine: 无法获取 ONNX Runtime API (ORT_API_VERSION="
                                 + std::to_string(ORT_API_VERSION) + ")");
    }

    // ---- 2. 创建 OrtEnv ----
    OrtEnv* env_raw = nullptr;
    check_status(
        api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bert_infer", &env_raw),
        "创建 OrtEnv 失败");
    env_.reset(env_raw);
    env_.get_deleter().api = api_;

    // ---- 3. 创建 SessionOptions ----
    OrtSessionOptions* sess_opts_raw = nullptr;
    check_status(
        api_->CreateSessionOptions(&sess_opts_raw),
        "创建 SessionOptions 失败");
    sess_opts_.reset(sess_opts_raw);
    sess_opts_.get_deleter().api = api_;

    // 设置图优化级别为全部优化
    check_status(
        api_->SetSessionGraphOptimizationLevel(sess_opts_raw, ORT_ENABLE_ALL),
        "设置图优化级别失败");

    // 设置线程数：当前 2 核 CPU，intra_op=2 充分利用并行计算
    // intra_op 控制单个算子（如矩阵乘法）内部的线程数
    // inter_op 控制不同算子之间的并行（顺序依赖的算子无法并行）
    check_status(
        api_->SetIntraOpNumThreads(sess_opts_raw, 2),
        "设置 intra_op_num_threads 失败");
    check_status(
        api_->SetInterOpNumThreads(sess_opts_raw, 1),
        "设置 inter_op_num_threads 失败");

    // ---- 4. 创建 CPU MemoryInfo（用于创建输入张量） ----
    OrtMemoryInfo* mem_info_raw = nullptr;
    check_status(
        api_->CreateMemoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault, &mem_info_raw),
        "创建 MemoryInfo 失败");
    mem_info_.reset(mem_info_raw);
    mem_info_.get_deleter().api = api_;

    // ---- 5. 加载 ONNX 模型，创建主 Session ----
    OrtSession* session_raw = nullptr;
    check_status(
        api_->CreateSession(env_.get(), model_path.c_str(), sess_opts_.get(), &session_raw),
        "加载模型失败: " + model_path);
    session_.reset(session_raw);
    session_.get_deleter().api = api_;

    // ---- 6. 获取输入输出张量名称 ----
    OrtAllocator* allocator = nullptr;
    check_status(
        api_->GetAllocatorWithDefaultOptions(&allocator),
        "获取默认分配器失败");

    char* in0 = nullptr, *in1 = nullptr, *in2 = nullptr, *out0 = nullptr;
    check_status(api_->SessionGetInputName(session_.get(), 0, allocator, &in0), "获取输入名称[0]失败");
    check_status(api_->SessionGetInputName(session_.get(), 1, allocator, &in1), "获取输入名称[1]失败");
    check_status(api_->SessionGetInputName(session_.get(), 2, allocator, &in2), "获取输入名称[2]失败");
    check_status(api_->SessionGetOutputName(session_.get(), 0, allocator, &out0), "获取输出名称失败");
    input_names_[0] = in0;
    input_names_[1] = in1;
    input_names_[2] = in2;
    output_names_[0] = out0;

    // ---- 7. 预分配输入缓冲区 ----
    input_ids_buf_.assign(BERT_MAX_SEQ_LEN, static_cast<int64_t>(0));
    attn_mask_buf_.assign(BERT_MAX_SEQ_LEN, static_cast<int64_t>(0));
    token_type_buf_.assign(BERT_MAX_SEQ_LEN, static_cast<int64_t>(0));
}

// ============================================================
// 公有接口实现
// ============================================================

std::vector<float> BertInferEngine::encode(const std::string& text) {
    return encode_single(text);
}

std::vector<std::vector<float>> BertInferEngine::encode_batch(
    const std::vector<std::string>& texts)
{
    std::vector<std::vector<float>> results;
    results.reserve(texts.size());
    for (const auto& text : texts) {
        results.push_back(encode_single(text));
    }
    return results;
}

// ============================================================
// 内部方法实现
// ============================================================

void BertInferEngine::check_status(OrtStatus* status, const std::string& msg) {
    if (status) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        const char* err = api->GetErrorMessage(status);
        std::string full = msg + ": " + (err ? err : "unknown error");
        api->ReleaseStatus(status);
        throw std::runtime_error(full);
    }
}

std::vector<float> BertInferEngine::encode_single(const std::string& text) {
    // ---- 1. 分词 ----
    std::vector<size_t> token_ids = tokenizer_.tokenize_full(text);

    // ---- 2. 截断到 max_len ----
    size_t actual_len = token_ids.size();
    if (actual_len > BERT_MAX_SEQ_LEN) {
        actual_len = BERT_MAX_SEQ_LEN;
        token_ids.resize(BERT_MAX_SEQ_LEN - 1);
        token_ids.push_back(102);  // [SEP] id
    }

    // ---- 3. 填充缓冲区（共享的，单线程使用） ----
    std::fill(input_ids_buf_.begin(),  input_ids_buf_.end(),  static_cast<int64_t>(0));
    std::fill(attn_mask_buf_.begin(),  attn_mask_buf_.end(),  static_cast<int64_t>(0));
    std::fill(token_type_buf_.begin(), token_type_buf_.end(), static_cast<int64_t>(0));

    for (size_t i = 0; i < actual_len; ++i) {
        input_ids_buf_[i]  = static_cast<int64_t>(token_ids[i]);
        attn_mask_buf_[i]  = 1;
    }

    // ---- 4. 创建输入张量 OrtValue ----
    int64_t shape[2] = {1, static_cast<int64_t>(BERT_MAX_SEQ_LEN)};
    size_t tensor_bytes = BERT_MAX_SEQ_LEN * sizeof(int64_t);

    OrtValue* in_ids_raw   = nullptr;
    OrtValue* in_mask_raw  = nullptr;
    OrtValue* in_type_raw  = nullptr;

    check_status(
        api_->CreateTensorWithDataAsOrtValue(mem_info_.get(), input_ids_buf_.data(),
            tensor_bytes, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in_ids_raw),
        "创建 input_ids 张量失败");
    ValuePtr in_ids(in_ids_raw,  ValueDeleter{api_});

    check_status(
        api_->CreateTensorWithDataAsOrtValue(mem_info_.get(), attn_mask_buf_.data(),
            tensor_bytes, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in_mask_raw),
        "创建 attention_mask 张量失败");
    ValuePtr in_mask(in_mask_raw, ValueDeleter{api_});

    check_status(
        api_->CreateTensorWithDataAsOrtValue(mem_info_.get(), token_type_buf_.data(),
            tensor_bytes, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in_type_raw),
        "创建 token_type_ids 张量失败");
    ValuePtr in_type(in_type_raw, ValueDeleter{api_});

    // ---- 5. 准备输入输出数组 ----
    const OrtValue* input_tensors[3] = { in_ids.get(), in_mask.get(), in_type.get() };

    OrtValue* output_val_raw = nullptr;
    check_status(
        api_->Run(session_.get(), nullptr,
            input_names_, input_tensors, 3,
            output_names_, 1,
            &output_val_raw),
        "模型推理失败");
    ValuePtr output_val(output_val_raw, ValueDeleter{api_});

    // ---- 6. 提取 last_hidden_state 数据 ----
    float* output_data = nullptr;
    check_status(
        api_->GetTensorMutableData(output_val.get(), reinterpret_cast<void**>(&output_data)),
        "获取输出数据失败");

    // ---- 7. Mean Pooling（加权平均，排除 padding） ----
    // 优化：只循环 actual_len 而非 BERT_MAX_SEQ_LEN
    // 对短查询（如 "癌症" = 2 tokens），可减少 96% 的循环次数
    std::vector<float> embedding(BERT_HIDDEN_SIZE, 0.0f);
    float weight_sum = 0.0f;

    for (size_t i = 0; i < actual_len; ++i) {
        weight_sum += 1.0f;
        for (size_t j = 0; j < BERT_HIDDEN_SIZE; ++j) {
            embedding[j] += output_data[i * BERT_HIDDEN_SIZE + j];
        }
    }

    if (weight_sum > 0.0f) {
        float inv = 1.0f / weight_sum;
        for (size_t j = 0; j < BERT_HIDDEN_SIZE; ++j) {
            embedding[j] *= inv;
        }
    }

    // ---- 8. L2 归一化 ----
    float norm = 0.0f;
    for (size_t j = 0; j < BERT_HIDDEN_SIZE; ++j) {
        norm += embedding[j] * embedding[j];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-12f) {
        float inv = 1.0f / norm;
        for (size_t j = 0; j < BERT_HIDDEN_SIZE; ++j) {
            embedding[j] *= inv;
        }
    }

    return embedding;
}
