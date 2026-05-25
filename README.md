# 语义搜索引擎 (BERT + SQLite + jieba 双引擎)

基于 C++17 实现的**语义搜索引擎**，使用 **BERT 模型**进行稠密向量编码，通过**向量点积（余弦相似度）** 进行语义检索。使用 **SQLite** 持久化存储文档和 embedding。网络层采用多 Reactor 模型 + 线程池处理并发请求。

**双引擎词语推荐**: [`/suggest`](#-词语推荐) 端点整合了 **jieba 编辑距离(50%)** + **BGE 语义向量(50%)** 的综合评分，提供更准确的词语联想。

---

## 目录

- [系统架构](#系统架构)
- [项目目录结构](#项目目录结构)
- [环境要求](#环境要求)
- [构建与运行](#构建与运行)
- [配置文件](#配置文件)
- [协议与 API](#协议与-api)
- [客户端使用](#客户端使用)
- [核心模块说明](#核心模块说明)
- [数据流](#数据流)
- [技术原理](#技术原理)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    客户端 (cli/client)                        │
│        HTTP GET 请求 → JSON 响应（HTTP/1.1）                  │
└──────────────────────┬──────────────────────────────────────┘
                         │ TCP (IPv4)
┌──────────────────────▼──────────────────────────────────────┐
│                    服务端 (src/)                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐      │
│  │  TcpServer   │  │  threadpool  │  │   EventLoop     │      │
│  │ (多 Reactor) │─▶│  (线程池)    │  │ (主+子事件循环)  │      │
│  └──────┬───────┘  └──────┬───────┘  └────────────────┘      │
│         │                 │                                   │
│  ┌──────▼─────────────────▼───────┐                          │
│  │    onMessage() HTTP 路由       │                          │
│  │  recvHttp() → parseHttpGet()   │                          │
│  │  /search  → SemanticIndexer     │                          │
│  │  /semantic → SemanticIndexer   │                          │
│  │  /suggest → SemanticIndexer     │                          │
│  └─────────────────────────────────┘                          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │            SemanticIndexer (核心模块)                    │  │
│  │  ┌──────────────────┐  ┌────────────────────┐         │  │
│  │  │ BertInferEngine   │  │     SQLite3 DB    │         │  │
│  │  │ (ONNX Runtime)    │  │ docs表(id,title,  │         │  │
│  │  │ model.onnx +      │  │ url,content,      │         │  │
│  │  │ tokenizer.json    │  │ embedding BLOB,   │         │  │
│  │  └──────────────────┘  │ title_embedding,   │         │  │
│  │  ┌──────────────────┐  │ cluster_id)        │         │  │
│  │  │ IVFIndex          │  └────────────────────┘         │  │
│  │  │ K-Means 聚类      │                                 │  │
│  │  │ √ 训练: K-Means++ │                                 │  │
│  │  │ √ 搜索: IVF nprobe│                                 │  │
│  │  │ √ 持久化: 二进制   │                                 │  │
│  │  └──────────────────┘                                 │  │
│  │  ┌──────────────────┐                                 │  │
│  │  │ DictProducer      │                                 │  │
│  │  │ (jieba.dict.utf8) │                                 │  │
│  │  │ Levenshtein 编辑   │                                 │  │
│  │  │ 距离 → 降准匹配    │                                 │  │
│  │  └──────────────────┘                                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────────┐              │
│  │  WordPieceTokenizer│ │       Configer        │              │
│  │  (BERT 子词分词)    │ │   (配置解析器)        │              │
│  └──────────────────┘  └──────────────────────┘              │
└───────────────────────────────────────────────────────────────┘
```

### 数据流架构图

```
                          ┌──────────────┐
                          │  HTTP 请求    │
                          │  GET /xxx?q=  │
                          └──────┬───────┘
                                 │ parseHttpGet()
                                 ▼
                          ┌──────────────┐
                          │  路由分发      │
                          │  onMessage()  │
                          └──────┬───────┘
                                 │
              ┌──────────────────┼─────────────────────┐
              │                  │                      │
              ▼                  ▼                      ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  /search         │  │  /suggest        │  │  / (帮助)        │
   │  /semantic       │  │                  │  │                  │
   └──────┬───────────┘  └──────┬───────────┘  └──────────────────┘
          │                     │
          │ SemanticIndexer     │ SemanticIndexer::suggest()
          │ ::find()            │
          ▼                     ▼
 ┌──────────────────┐  ┌─────────────────────────────────────┐
 │ BERT 编码层       │  │  ┌─────────────────────────────┐   │
 │ query → 512维向量  │  │  │ BGE 引擎 (BERT)             │   │
 └────────┬─────────┘  │  │  query → 512维向量            │   │
          │            │  │  SELECT title_embedding       │   │
          ▼            │  │  dot product → top50          │   │
 ┌──────────────────┐  │  │  提取词语 → 归一化 score_bge  │   │
 │ IVF + SQLite 检索   │  │  └─────────────────────────────┘   │
 │ ① searchCentroids  │  │              +                     │
 │ ② cluster_id IN () │  │  ┌─────────────────────────────┐   │
 │ ③ dot product      │  │  │ jieba 引擎 (DictProducer)   │   │
 │ ④ sort Top-10      │  │  │  load jieba.dict.utf8        │   │
 └────────┬─────────┘  │  │  首字 + 长度过滤             │   │
          │            │  │  Levenshtein 编辑距离        │   │
          ▼            │  │  → score_jieba ∈ [0,1]      │   │
 ┌──────────────────┐  │  └─────────────────────────────┘   │
 │ JSON 裸数组       │  │              +                     │
 │ [{title, url,    │  │  ┌─────────────────────────────┐   │
 │   content,score}] │  │  │ 融合评分                      │   │
 └──────────────────┘  │  │  final = 0.5×score_jieba     │   │
                       │  │        + 0.5×score_bge       │   │
                       │  │  sort Top-10                 │   │
                       │  └─────────────────────────────┘   │
                       └──────────┬─────────────────────────┘
                                  │
                                  ▼
                       ┌──────────────────┐
                       │ JSON 文本数组     │
                       │ ["词1","词2",...] │
                       └──────────────────┘
```

---

## 项目目录结构

```
search-engine/
├── cli/                          # 客户端
│   ├── client.cpp                # TCP 客户端实现
│   ├── html/cli.html             # Web 客户端（前端页面）
│   └── CMakeLists.txt
├── config/
│   └── serch.conf                # 主配置文件（所有可调参数）
├── data/
│   ├── semantic_index.db         # SQLite 数据库（自动创建，存储文档+向量）
│   └── yuliao/
│       ├── xml/                  # RSS/XML 原始语料（分频道）
│       └── stop_words_zh.txt     # 中文停用词表
├── dict/                         # jieba 词典文件
│   └── jieba.dict.utf8           # jieba 词典（35 万词条）
├── model/                        # BERT 模型文件
│   ├── model.onnx                # ONNX 格式 BERT 模型
│   ├── tokenizer.json            # HuggingFace Tokenizer 配置
│   ├── vocab.txt                 # 词表（备用）
│   ├── config.json               # 模型配置
│   └── onnxruntime_lib/          # ONNX Runtime 预编译库
│       ├── include/              # C/C++ API 头文件
│       └── lib/                  # .so 动态库文件
├── include/                      # 头文件（声明）
│   ├── Acceptor.h                # 连接接收器
│   ├── Configer.h                # 配置解析器
│   ├── DictProducer.h            # jieba 词典加载 + 编辑距离模糊匹配
│   │                            # - 加载 jieba.dict.utf8（35万词条）
│   │                            # - find() 首字过滤 + Levenshtein 编辑距离
│   ├── EventLoop.h               # 事件循环
│   ├── InetAddress.h             # IP 地址封装
│   ├── InferEngine.h             # BERT 推理引擎
│   │                            # - BertInferEngine 类：ONNX Runtime 封装
│   │                            # - encode_single() / encode_batch()
│   ├── IVFIndex.h                # IVF 倒排文件索引（K-Means 聚类加速）
│   │                            # - K-Means++ 初始化（固定种子 42）
│   │                            # - 迭代训练（最多 50 轮）直到收敛
│   │                            # - 空簇处理（分裂噪声扰动）
│   │                            # - nprobe 搜索（查询最近 nprobe 个簇）
│   │                            # - 二进制持久化（load/save）
│   ├── mylog.h                   # 日志系统
│   ├── PageLib.h                 # 网页库
│   ├── SemanticIndexer.h         # 语义索引器
│   │                            # - SQLite 存储文档+embedding
│   │                            # - buildIndex() / find() / suggest() / size()
│   │                            # - suggest(): jieba(50%) + BGE(50%) 双引擎
│   │                            # - find(): IVF 加速（K-Means 聚类过滤）
│   ├── sqlite3.h                 # SQLite3 C API 头文件（amalgamation）
│   ├── Socket.h / SockIO.h       # Socket 封装
│   ├── Task.h                    # 任务封装
│   ├── taskQueue.h               # 线程安全任务队列
│   ├── TcpConnetion.h            # TCP 连接封装
│   ├── TcpServer.h               # TCP 服务器（Reactor）
│   ├── threadpool.h              # 线程池
│   ├── WordPieceTokenizer.hpp    # BERT 子词分词器（header-only）
│   │                            # - 加载 tokenizer.json
│   │                            # - tokenize() / convert_tokens_to_ids()
│   └── json/                     # JSON 库（nlohmann）
├── src/                          # 源文件（实现）
│   ├── main.cpp                  # 入口，HTTP 路由，初始化+启动
│   ├── Configer.cpp              # 配置解析实现
│   ├── DictProducer.cpp          # DictProducer 实现
│   ├── InferEngine.cpp           # BertInferEngine 实现
│   ├── IVFIndex.cpp              # IVFIndex 实现（K-Means 训练+搜索+持久化）
│   ├── SemanticIndexer.cpp       # SemanticIndexer 实现
│   ├── TcpServer.cpp             # 服务器网络层
│   ├── TcpConnetion.cpp          # 连接管理
│   ├── threadpool.cpp            # 线程池实现
│   └── ...                       # 其他网络组件
├── log/                          # 运行时日志输出
├── build/                        # 编译输出目录
├── CMakeLists.txt                # 顶级 CMake 配置
└── README.md                     # 本文件
```

---

## 环境要求

| 组件 | 版本 |
|------|------|
| 操作系统 | Linux (Ubuntu 18.04+) |
| 编译器 | GCC 7.0+（需要 C++17 支持） |
| CMake | 3.10+ |
| tinyxml2 | 用于 XML 解析（编译时链接） |
| SQLite3 | 运行时 libsqlite3.so（动态链接） |
| ONNX Runtime | 预编译版 1.25.1（已包含在 `model/onnxruntime_lib/`） |

### 安装依赖

```bash
# Ubuntu / Debian
sudo apt-get install build-essential cmake libtinyxml2-dev libsqlite3-dev
```

> 如果没有 root 权限，可以使用系统自带的 `libsqlite3.so.0` 运行时库，并提供 `sqlite3.h` 头文件（已包含在 `include/sqlite3.h`）。

---

## 构建与运行

### 1. 编译

```bash
cd search-engine
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

编译产物：
- `build/server` — 服务端可执行程序
- `build/cli/client` — 客户端可执行程序

### 2. 配置

编辑 [`config/serch.conf`](config/serch.conf) 确保以下路径正确：

```ini
modelPath:  /home/marisa/code1/VectorDB/model/model.onnx   # BERT ONNX 模型
vocabPath:  /home/marisa/code1/VectorDB/model/tokenizer.json # Tokenizer 配置
sqliteDbPath: /home/marisa/code1/VectorDB/data/semantic_index.db  # SQLite 数据库
xmlPath:    /home/marisa/code1/search-engine/data/yuliao/xml  # XML 语料目录
```

**首次运行前需要确保：**
- BERT 模型文件 [`model.onnx`](model/model.onnx) 和 [`tokenizer.json`](model/tokenizer.json) 存在
- XML 语料库在 [`xmlPath`](config/serch.conf) 指向的目录下

### 3. 运行服务端

```bash
cd build
LD_LIBRARY_PATH=../model/onnxruntime_lib/lib ./server
```

> 需设置 `LD_LIBRARY_PATH` 包含 ONNX Runtime 动态库路径。

启动过程：
1. 加载配置 & 初始化日志
2. **启动 TCP 服务器 & 线程池**（此时 `g_serverReady = false`，客户端连接会被拒绝并提示"正在初始化"）
3. 初始化 BERT 推理引擎（加载 model.onnx 到 ONNX Runtime）
4. 打开/创建 SQLite 数据库，检查 `docs` 表中是否有数据
5. **无数据时**：遍历 XML 语料 → BERT 编码为 512 维向量 → 写入 SQLite
6. **已有数据时**：跳过建库，直接提供服务
7. `g_serverReady = true`，开始正常处理客户端请求

### 4. 运行客户端

```bash
cd build
./cli/client <服务器IP> <端口>
# 默认连接 192.168.159.129:8080
# 示例：./cli/client 127.0.0.1 8080
```

---

## 配置文件

配置文件 [`config/serch.conf`](config/serch.conf) 采用 `key:value` 格式，支持 `#` 注释（行首注释和内联注释）。

### 完整配置项

```ini
# ==================== 路径配置 ====================
xmlPath:           # XML 语料库目录
modelPath:         # BERT ONNX 模型文件路径
vocabPath:         # HuggingFace tokenizer.json 路径
sqliteDbPath:      # SQLite 数据库文件路径
jiebaDictPath:     # jieba 词典文件路径（dict/jieba.dict.utf8）
logPath:           # 日志文件路径
ivfIndexPath:      # IVF 索引文件路径（二进制持久化，自动创建）

# ==================== 网络配置 ====================
serverIp:          # 监听 IP 地址
serverPort:        # 监听端口号
subEventLoopCount: # 子 EventLoop 数量（多 Reactor 模型）

# ==================== 线程池配置 ====================
threadPoolThreadCount: # 工作线程数
threadPoolQueueSize:   # 任务队列最大长度

# ==================== 日志配置 ====================
logBufferSize:     # 日志缓冲区大小（条目数）
logLevel:          # 日志级别：0=DEBUG 1=INFO 2=WARN 3=ERROR
```

---

## 协议与 API

### 通信协议

- **传输层**: TCP
- **应用层**: HTTP/1.1
- **编码**: UTF-8（URL 中的非 ASCII 字符需 Percent-Encoding）
- **请求格式**: HTTP GET 请求
- **响应格式**: HTTP 响应，`Content-Type: application/json; charset=utf-8`

### API 端点

#### 1. 语义搜索

**端点**: `GET /search?q=<关键词>` 或 `GET /semantic?q=<关键词>`

两个端点功能完全相同，均使用 BERT 语义搜索。

**请求示例**:

```
GET /search?q=%E7%99%8C%E7%97%87 HTTP/1.1
Host: 127.0.0.1:8080
Connection: keep-alive
Accept: application/json
```

**响应示例**:

```json
[
    {
        "title": "文章标题",
        "url": "原文链接",
        "content": "摘要文本...",
        "score": 0.8923
    },
    {
        "title": "另一篇相关文章",
        "url": "https://example.com/article2",
        "content": "摘要文本...",
        "score": 0.7641
    }
]
```

**实现**: [`SemanticIndexer::find()`](include/SemanticIndexer.h)

**搜索模式**（根据是否已训练 IVF 自动选择）：

| 模式 | 条件 | 性能 | 说明 |
|------|------|------|------|
| **IVF 加速** | IVF 索引文件存在 | O(K×dim + N×nprobe/K×dim) | K-Means 聚类过滤，仅扫描 nprobe 个最近簇 |
| **全表扫描（降级）** | IVF 未训练或无文件 | O(N×dim) | 暴力遍历所有文档（首次建库回退） |

搜索流程（IVF 加速模式）：
1. 用 BERT 将查询编码为 512 维向量
2. 调用 [`IVFIndex::searchCentroids()`](include/IVFIndex.h) 找到最近的 nprobe 个簇
3. 仅从这些簇中 `SELECT ... WHERE cluster_id IN (...)` 读取文档
4. 计算查询向量与各文档向量的点积（余弦相似度）
5. 按相似度降序排列，返回 Top-10

---

#### 2. 词语推荐

**端点**: `GET /suggest?q=<关键词>`

返回相关的词语推荐列表，使用 **双引擎综合评分**。

**请求示例**:

```
GET /suggest?q=%E6%90%9C%E7%B4%A2%E5%BC%95 HTTP/1.1
Host: 127.0.0.1:8080
Connection: keep-alive
Accept: application/json
```

**响应示例**:

```json
["搜索引擎", "搜索引擎优化", "搜索引擎营销", "搜索技术", "搜索算法"]
```

**双引擎评分机制**:

| 引擎 | 方法 | 分数占比 | 说明 |
|------|------|---------|------|
| **jieba 编辑距离** | [`DictProducer::find()`](include/DictProducer.h) | 50% | 从 jieba 词典（35 万词条）中找首字母匹配、长度相近的词，计算 Levenshtein 编辑距离 → 相似度 |
| **BGE 语义向量 + IVF** | [`SemanticIndexer::suggest()`](include/SemanticIndexer.h) | 50% | BERT 编码查询 → IVF 聚类加速（只查最近 N 堆）→ 两阶段 SQL（先 id+embedding 评分，再取 title 提取词语）→ 归一化 |

**综合评分**: `final_score = 0.5 × score_jieba + 0.5 × score_bge`

最终按综合评分降序排列，返回 Top-10 推荐词。

**性能优化说明**（v2.0）：
- **IVF 加速**：suggest 的语义搜索现在与 find() 共享 IVF 聚类索引，只扫描距离最近的 `nprobe × 2` 个堆中的文档，而非全表扫描。对于 10,000 篇文档，扫描量从 10,000 行降至约 250 行 — **加速约 40 倍**。
- **两阶段查询**：第一阶段只加载 `id + title_embedding`（BLOB）计算点积评分；第二阶段仅对排序后的前 50 篇读取 `title` 字符串。SQLite I/O 减少约 80%。
- **LFU 缓存**：每次 recommend 结果缓存 500 条，重复查询直接回。

---

#### 3. 根路径（帮助信息）

**端点**: `GET /`

返回可用 API 端点的 JSON 描述。

---

### 状态码

| 状态码 | 含义 |
|--------|------|
| 200 OK | 请求成功 |
| 503 Service Unavailable | 服务器正在初始化（BERT 模型加载或索引构建中） |

### 查询示例

| `curl` 命令 | 描述 |
|-------------|------|
| `curl "http://127.0.0.1:8080/search?q=慢性病"` | 语义搜索 |
| `curl "http://127.0.0.1:8080/semantic?q=癌症治疗"` | 语义搜索（同功能） |
| `curl "http://127.0.0.1:8080/suggest?q=搜索"` | 词语推荐（双引擎） |
| `curl "http://127.0.0.1:8080/"` | 查看 API 帮助 |

> **语义搜索 vs 关键词搜索**：搜索"癌症"时也能找到"肿瘤"相关的文档，因为 BERT 将两者编码到了语义空间的相近位置。
>
> **词语推荐**：/suggest 返回 jieba(50%) + BGE(50%) 综合评分的推荐词，支持模糊匹配（如输入"搜索引"可得到"搜索引擎"）。

---

## 客户端使用

客户端提供交互式命令行界面，基于 HTTP 协议与服务端通信。

```bash
./cli/client 192.168.159.129 8080
```

### 交互命令

| 命令 | 功能 |
|------|------|
| `<关键词>` | 词语推荐（双引擎），发送 `GET /suggest?q=<关键词>` |
| `/search <关键词>` | 语义搜索，发送 `GET /search?q=<关键词>` |
| `/semantic <关键词>` | 语义搜索（同功能），发送 `GET /semantic?q=<关键词>` |
| `/qps [关键词]` | 执行 20 秒 QPS 压测（HTTP，默认关键词"搜索引擎"） |
| `/help` 或 `/h` | 显示帮助 |
| `/quit` 或 `/exit` 或 `/q` | 退出客户端 |

### 输出格式

响应以格式化 JSON 输出，外壳为 `===` 分隔线。

---

## 核心模块说明

### [`SemanticIndexer`](include/SemanticIndexer.h) — 语义索引器（核心模块）

- **实现文件**: [`include/SemanticIndexer.h`](include/SemanticIndexer.h)（声明） + [`src/SemanticIndexer.cpp`](src/SemanticIndexer.cpp)（实现）
- **数据存储**: 使用 SQLite 数据库存储文档和 768 维稠密向量
- **`buildIndex()`**: 遍历 XML 语料 → BERT 编码 → 事务批量写入 SQLite → **训练 IVF K-Means 聚类 → 更新 cluster_id → 持久化 IVF 文件**
- **`find()`**: BERT 编码查询 → **IVF 加速**（searchCentroids → cluster_id 过滤 → 向量点积 → TopK）或全表扫描降级
- **`suggest()`**: **双引擎综合评分** — jieba 编辑距离(50%) + BGE 语义向量(50%) → **IVF 加速 ANN**（与 find() 共享聚类索引）+ **两阶段 SQL**（先 id+embedding 评分，再取 title 提取词语）→ 按综合分 TopK 返回
- **内部包含 [`DictProducer`](include/DictProducer.h)**：加载 jieba 词典，提供 Levenshtein 编辑距离匹配
- **内部包含 [`IVFIndex`](include/IVFIndex.h)**：K-Means 聚类加速向量检索
- **单例模式**: 通过 `init()` + `getPtr()` 全局访问

### [`IVFIndex`](include/IVFIndex.h) — IVF 分堆搜索（K-Means 聚类加速）

- **实现文件**: [`include/IVFIndex.h`](include/IVFIndex.h)（声明） + [`src/IVFIndex.cpp`](src/IVFIndex.cpp)（实现）
- **核心思路**: 把所有文章向量分成 K 堆，搜的时候只看最近的几堆，不用全看

**简单说就是"分堆找东西"**：

假设有一万篇文章，每篇都有一个 768 维的向量。搜的时候如果一篇篇比，太慢了。

**K-Means 分堆过程**:
1. **挑几个"组长"**：先随便挑 K 个点当"组长"（第一个随机挑，后面的尽量挑离已有点远的，这样组更分散）
2. **组员归队**：每篇文章去找离它最近的组长，加入这个组
3. **重选组长**：每个组算一个新组长（组内所有文章的平均位置）
4. **反复调整**：重复第 2-3 步，直到没人换组为止（最多 50 轮）
5. **特殊情况**：如果某个组没人了，从隔壁组复制一个队长加点随机抖动

**搜索时（nprobe）**：
- 查询词转成向量
- 先算跟 K 个组长的距离，找出最近的几个组（默认 5 个）
- 只看这几个组里的文章，不用全部扫描
- 原来要看 N 篇 → 现在只看 N × nprobe / K 篇，快很多

**存到文件**：
- 训练好的分组信息可以存到文件里，下次启动直接读
- 文件内容：[维度 | 组数 | 总数 | 组长坐标 | 每篇文章的归属]
- 文件路径在 [`config/serch.conf`](config/serch.conf) 的 `ivfIndexPath` 里配置

### [`BertInferEngine`](include/InferEngine.h) — BERT 推理引擎

- **实现文件**: [`include/InferEngine.h`](include/InferEngine.h)（声明） + [`src/InferEngine.cpp`](src/InferEngine.cpp)（实现）
- 封装 **ONNX Runtime C++ API**，加载 `model.onnx` 进行 CPU 推理
- **`encode(text)`**: 接收字符串，返回 768 维 L2 归一化向量
- **`encode_batch(texts)`**: 批量编码
- 内部使用 [`WordPieceTokenizer`](include/WordPieceTokenizer.hpp) 将文本转为 token IDs

### [`DictProducer`](include/DictProducer.h) — jieba 词典加载 + 编辑距离模糊匹配

- **实现文件**: [`include/DictProducer.h`](include/DictProducer.h)（声明） + [`src/DictProducer.cpp`](src/DictProducer.cpp)（实现）
- 加载 `dict/jieba.dict.utf8`（35 万词条），解析 `word freq pos_tag` 格式
- **`find(query, topK)`**: 首字过滤 + 长度过滤 → Levenshtein 编辑距离 → 相似度排序
- **编辑距离算法**: 两行 DP 优化，O(m×n) 时间，O(min(m,n)) 空间
- 为 `/suggest` 提供 jieba 编辑距离评分（50% 权重）

### [`WordPieceTokenizer`](include/WordPieceTokenizer.hpp) — BERT 子词分词器

- 从 `tokenizer.json`（HuggingFace 格式）加载词表和分词配置
- 实现 BERT 标准的 WordPiece 分词算法
- 自动添加 `[CLS]` / `[SEP]` 特殊标记
- 支持中文字符自动加空格（中文不需要分词，按字切分）

### [`Configer`](include/Configer.h) — 配置解析器

- 读取 `serch.conf`，每行 `key:value` 格式
- 跳过 `#` 开头的注释行，支持内联注释
- 提供 `getConfigMap()` 获取键值对映射

### SQLite 集成

- 使用 [`include/sqlite3.h`](include/sqlite3.h)（官方 amalgamation 头文件）和系统 `libsqlite3.so.0` 运行时库
- 建表语句：`CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, url TEXT, content TEXT, embedding BLOB, title_embedding BLOB)`
- `embedding` 列：768 维 content 向量（3072 字节 BLOB），用于 `/search` 文档检索
- `title_embedding` 列：768 维 title 向量（3072 字节 BLOB），用于 `/suggest` 词语推荐
- 使用事务（BEGIN/COMMIT）批量写入提升性能
- `sqlite3_busy_timeout(5000)` + 写入重试循环避免 SQLITE_BUSY

### 网络层

| 组件 | 功能 |
|------|------|
| [`TcpServer`](include/TcpServer.h) | 多 Reactor 模型服务器，组合 Acceptor + 主 EventLoop + EventLoopGroup |
| [`Acceptor`](include/Acceptor.h) | 监听 socket，接受新连接 |
| [`EventLoop`](include/EventLoop.h) | epoll 事件循环，管理 IO 事件；支持主/子两种模式 |
| [`EventLoopGroup`](include/EventLoopGroup.h) | 子 EventLoop 管理器，通过轮询（Round-Robin）分配新连接到子 Reactor |
| [`TcpConnetion`](include/TcpConnetion.h) | 封装 TCP 连接，处理读写；支持 HTTP 请求读取（`recvHttp()`） |
| [`threadpool`](include/threadpool.h) | 线程池，异步处理业务逻辑 |

### [`mylog`](include/mylog.h) — 日志系统

- 双缓冲异步日志，生产者-消费者模型
- 支持 5 个级别：DEBUG / INFO / WARN / ERROR / FATAL
- 通过宏 `LOG_DEBUG` / `LOG_INFO` / `LOG_WARN` / `LOG_ERROR` 调用

---

## 数据流

### 启动流程

```
main()
  ├─ Configer(confPath)            ← 读取配置文件
  ├─ mylog::init()                 ← 初始化日志
  │
  ├─ TcpServer()                   ← 创建服务器（主 EventLoop + 子 EventLoopGroup）
  ├─ threadpool::start()           ← 启动线程池
  ├─ server->start() (后台线程)     ← 启动 TCP 服务器（主 EventLoop 开始 accept）
  │   ├─ 主 EventLoop accept 新连接
  │   ├─ 轮询（Round-Robin）分配给子 EventLoop
  │   └─ 子 EventLoop 处理 IO 读写
  │
  ├─ [此时 g_serverReady=false]     ← 拒绝连接，提示"正在初始化"
  │
  ├─ SemanticIndexer::init(...)    ← 初始化 BERT 引擎 + SQLite + jieba 词典
  │   ├─ BertInferEngine(model_path, vocab_path)
  │   │   ├─ 加载 ONNX 模型到 Ort::Session
  │   │   └─ 加载 tokenizer.json 初始化 WordPieceTokenizer
  │   ├─ sqlite3_open_v2(db_path)  ← 打开/创建数据库
  │   └─ DictProducer(jiebaDictPath) ← 加载 jieba 词典（35万词条）
  │
  ├─ SemanticIndexer::size() == 0?  ← 检查是否已有数据
  │   └── 是 → buildIndex()
  │       ├─ 遍历 XML 文件
  │       ├─ stripHtml() 清理 HTML
  │       ├─ BertInferEngine::encode_single() → 512维向量（content）
  │       ├─ BertInferEngine::encode_single() → 512维向量（title）← title_embedding
  │       ├─ BEGIN TRANSACTION
  │       ├─ 逐条 INSERT INTO docs VALUES(?,?,?,?,?,?)
  │       ├─ COMMIT
  │       │
  │       ├─ (IVF 训练) ← 数据写入完成后自动训练
  │       │   ├─ 读取所有 embedding 到内存
  │       │   ├─ K = max(10, √N)   ← 自动选择簇数
  │       │   ├─ IVFIndex::train(data, N, K)  ← K-Means++ 训练
  │       │   ├─ UPDATE docs SET cluster_id = ?  ← 更新聚类分配
  │       │   └─ IVFIndex::save(ivfIndexPath)   ← 持久化 IVF
  │
  ├─ g_serverReady = true          ← 服务器就绪，开始处理请求
  └─ serverThread.join()
```

### 搜索流程

```
客户端输入查询
       │
       ▼
TcpServer::onMessage()
       │
       ├─ con->recvHttp()           ← 读取完整 HTTP 请求头（直到 \r\n\r\n）
       ├─ parseHttpGet()            ← 解析路径和 q 参数
       │
       ├─ g_serverReady == false?
       │      └── 是 → 返回 503 Service Unavailable
       │
       ├─ query 为空?
       │      └── 是 → 返回 API 端点帮助信息
       │
       ▼
handleHttpRequest(path, query, con)  ← 线程池异步处理
       │
       ├── path == "/search" 或 "/semantic"?
       │      └── 是 → SemanticIndexer::find()
       │                 ├── BertInferEngine::encode_single(query)
       │                 │     ├── WordPieceTokenizer::tokenize()
       │                 │     ├── ONNX Runtime 推理
       │                 │     └── L2 归一化 → 512维向量
       │                 │
       │                 ├── ivfTrained_ == true?       ← 判断 IVF 模式
       │                 │      │
       │                 │      ├── 是 → IVF 加速模式
       │                 │      │   ├── IVFIndex::searchCentroids(query, nprobe)
       │                 │      │   │     └── L2 距离 → partial_sort → 最近 nprobe 个簇
       │                 │      │   ├── SELECT ... WHERE cluster_id IN (c1,c2,...)
       │                 │      │   ├── for each doc:
       │                 │      │   │     score = dot(query_vec, doc_vec)
       │                 │      │   ├── sort by score DESC
       │                 │      │   └── return Top-10 JSON (裸数组)
       │                 │      │
       │                 │      └── 否 → 全表扫描降级
       │                 │          ├── SELECT ... FROM docs   ← 暴力遍历
       │                 │          ├── for each doc:
       │                 │          │     score = dot(query_vec, doc_vec)
       │                 │          ├── sort by score DESC
       │                 │          └── return Top-10 JSON (裸数组)
       │
       ├── path == "/suggest"?
       │      └── 是 → SemanticIndexer::suggest()
       │                 ├── BGE 引擎：
       │                 │   ├── BertInferEngine::encode_single(query) → 512维向量
       │                 │   ├── SELECT title_embedding FROM docs
       │                 │   ├── dot product → top50 → 提取词语 → 归一化
       │                 │   └── score_bge ∈ [0, 1]
       │                 ├── jieba 引擎：
       │                 │   ├── DictProducer::find(query, topK*3)
       │                 │   ├── Levenshtein 编辑距离 → 相似度
       │                 │   └── score_jieba ∈ [0, 1]
       │                 ├── 融合: final = 0.5×score_jieba + 0.5×score_bge
       │                 ├── sort by final DESC
       │                 └── return Top-10 文本数组
       │
       └── 其他路径 → 返回 JSON API 帮助信息
       │
       ▼
  buildHttpResponse(json)          ← 包装 HTTP 响应（含 Content-Length 等头）
       │
       ▼
  con->sendInLoop(httpResponse)    ← 发送 HTTP 响应
```

### 训练数据流（语料 ingestion）

下面的流程图展示了原始语料文件如何一步步变成向量数据并训练 IVF 索引的全过程：

```
XML 语料文件（RSS feed）
       │
       ▼
[ 文件扫描阶段 ]
 DirScanner::traverse(data/doc/)
       │
       ├── 遍历 data/doc/ 目录下的所有 .xml/.html 文件
       └── 生成文件路径列表 files[]
       │
       ▼
[ 解析 & 清洗阶段 ]
 stripHtml() 去除 HTML 标签
       │
       ├── 解析 XML 中的 <doc> 节点
       ├── 提取 <title>、<url>、<content> 字段
       └── tinyxml2 解析 + 正则清洗 HTML 标签
       │
       ▼
[ BERT 编码阶段 ]
 BertInferEngine::encode_single()
       │
       ├── WordPieceTokenizer::tokenize(text)
       │     └── "慢性病" → [CLS] 慢 性 病 [SEP]
       ├── ONNX Runtime 推理
       │     └── 输入 shape [1, 128] token_ids → 输出 shape [1, 768]
       ├── 取最后一层 hidden_state 的均值池化
       │     └── 768维 → 维度压缩 → 512维向量
       ├── L2 归一化 → 单位向量
       │
       ├──[content 分支] → content_embedding (512维)
       └──[title 分支]   → title_embedding  (512维)
       │
       ▼
[ SQLite 写入阶段 ]
 事务批量写入（提升性能）
       │
       ├── BEGIN TRANSACTION
       ├── for each doc:
       │     INSERT INTO docs
       │       (title, url, content, embedding, title_embedding, cluster_id)
       │       VALUES (?, ?, ?, ?, ?, NULL)
       │     └── embedding / title_embedding 以 BLOB 形式存 RAW 二进制 float 数组
       ├── COMMIT
       └── 一次事务处理约 500~1000 条，避免 SQLite 频繁 fsync
       │
       ▼
[ IVF 训练阶段 ]
 数据全部写入后，自动开始 K-Means 分堆训练
       │
       ├── 从 SQLite 读取所有 embedding（SELECT embedding FROM docs）
       ├── 确定 K 值：K = max(10, √N)
       │     └── 例如 N=10000 篇文档 → K = max(10, 100) = 100 堆
       │
       ├── 1. K-Means++ 初始化
       │     ├── 第一个组长：随机选一个数据点（固定种子 seed=42）
       │     └── 后面的组长：离已选组长越远的点越容易被选中
       │
       ├── 2. 分配（E 步）
       │     ├── 每个向量跟所有组长比 L2 距离
       │     └── 归入最近的组
       │
       ├── 3. 更新（M 步）
       │     ├── 每组取所有成员坐标的平均值 → 新组长
       │     └── 如果某组没人（空组），从邻居组复制并加微小噪声
       │
       ├── 4. 迭代 2↔3 直到稳定或达到最大 50 轮
       │
       ├── 5. 构建倒排列表
       │     └── 记录每个组的成员编号
       │
       ├── 6. 回写 SQLite
       │     └── UPDATE docs SET cluster_id = ? WHERE id = ?
       │
       └── 7. 持久化 IVF 文件
             └── IVFIndex::save(ivfIndexPath)
                   ├── 二进制格式：[dim | K | ntotal | centroids | assignments]
                   └── 下次启动直接 load()，无需重新训练
       │
       ▼
[ 服务器就绪 ]
 g_serverReady = true
       └── 开始接受客户端请求
             ├── /search → IVF 加速模式
             └── /suggest → 双引擎推荐
```

**关键说明**：

- **首次启动**：如果 SQLite 是空的（`size() == 0`），会自动执行上面的完整训练流程，这个过程可能需要 1~5 分钟（取决于语料量）。
- **再次启动**：如果已有 IVF 文件，直接 `load()` 跳过训练，秒级启动。
- **增删语料**：当前设计暂不支持增量更新，如果需要添加新语料，删除 `data/` 目录下的数据库文件和 IVF 文件，重启服务即可重建。
- **数据持久化**：embedding 存入 SQLite 后，下次即使没有 IVF 文件也能全表扫描搜索（降级模式），只是速度慢一些。

---

## 技术原理

### 为什么用 BERT 替代 jieba？

| 维度 | jieba + TF-IDF（旧方案） | BERT 语义搜索（新方案） |
|------|------------------------|----------------------|
| **匹配方式** | 关键词字面匹配 | 语义空间向量匹配 |
| **同义词处理** | "电脑"搜不到"计算机" | "电脑"和"计算机"向量距离很近 |
| **上下文理解** | 不支持（每个词独立统计） | 支持（Transformer 双向注意力） |
| **预处理复杂度** | 加载 5+ 个词典文件 → 分词 → 去停用词 → TF-IDF | 直接输入原始文本 |
| **向量维度** | 稀疏（词典大小，数万维） | 稠密（固定 512 维） |
| **检索方式** | 倒排索引 + 集合交集 | 向量点积 + 排序 |

### 为什么用 SQLite 替代文件 I/O？

| 维度 | 旧方案（newripepage.dat） | 新方案（SQLite） |
|------|-------------------------|-----------------|
| **写操作** | `open + lseek + write` 手动管理偏移 | `INSERT INTO docs VALUES(...)` |
| **读操作** | `open + lseek + read` 按偏移量读取 | `SELECT * FROM docs` |
| **并发安全** | 无保障 | ACID 事务隔离 |
| **崩溃恢复** | 文件可能损坏 | WAL 模式 + 原子提交 |
| **元数据** | 无，偏移量需额外记录 | `PRIMARY KEY, COUNT(*), ...` |
| **扩展性** | 新增字段需重写文件格式 | `ALTER TABLE ADD COLUMN` |

### 向量检索的数学原理

1. **BERT 编码器**：将变长文本映射到固定 512 维浮点数向量空间
2. **L2 归一化**：`v = v / ||v||`，使每个向量的欧几里得长度为 1
3. **点积 = 余弦相似度**：当 `||A|| = ||B|| = 1` 时：
   - `cos(θ) = (A·B) / (||A||·||B||) = A·B`
   - 即点积直接等于余弦相似度，无需额外除法
4. **排序**：对所有文档计算 `score = query_emb · doc_emb`，取 TopK 最大值

### IVF + K-Means 分堆搜索（近似最近邻）

搜文章的时候，如果一篇篇比过去，文章越多越慢。

换个思路：**先把文章分成几堆，搜的时候只看最近的几堆**。

**打个比方**：
> 图书馆有一万本书，你想找跟某本书类似的。
> 笨办法：一本本翻，跟所有书都对比一遍。
> IVF 的办法：先把书按内容分到 100 个书架上，搜的时候先看哪个书架最接近你的书，然后只翻这个书架上的书。

**K-Means 分堆的步骤**：

```
输入：一万篇文章的向量，想分成 K 堆，最多调 50 轮
输出：每堆有个"组长"坐标，每篇文章记下属于哪堆

┌────────────────────────────────────────────┐
│ ① 选组长（初始化）                          │
│    - 固定随机种子 42，保证每次分堆结果一样     │
│    - 第一个组长随便挑                        │
│    - 后面的组长尽量挑离已有组长远的            │
│      （这样各堆不会挤在一起）                  │
├────────────────────────────────────────────┤
│ ② 反复调整（最多 50 轮）：                   │
│    a) 分派：每篇文章去找离它最近的组长         │
│       如果某堆没人了 → 从隔壁复制一个加点抖动   │
│    b) 更新：每堆算个新组长（组内文章的平均位置） │
│    c) 检查：如果没人换组了 → 提前结束           │
├────────────────────────────────────────────┤
│ ③ 建目录：记下每篇文章属于哪堆                │
└────────────────────────────────────────────┘
```

**nprobe 搜索过程**：

```
查询词 → 转成向量
      │
      ├── 算向量跟 K 个组长的距离
      ├── 挑最近的 nprobe 个组（默认 5 个）
      │
      ▼
┌────────────────────────────────────────┐
│  只看这 5 个组里的文章                  │
│  看的文章数 ≈ 总数 × 5 / K             │
│  （原来看 10000 篇 → 现在看 ~500 篇）   │
└────────────────────────────────────────┘
      │
      ├── 对这些文章算精确的相似度排序
      └── 返回前 10 条结果

性能对比（一万篇文章，分 100 堆，搜 5 个堆）：
  - 原来：算 10000 次
  - IVF：算 100 次（找堆）+ 500 次（堆里算分）= 600 次
  - 快了大概 16 倍！
```

**重要参数**：

| 参数 | 默认值 | 大白话 |
|------|--------|--------|
| K（堆数） | max(10, √文章数) | 文章越多堆越多，但不能超过总数一半 |
| nprobe | 5 | 搜的时候看几个堆，越多越准但越慢 |
| maxIter | 50 | 最多调几轮，够了就停 |

---

## 许可证

本项目基于 MIT 许可证开源。
