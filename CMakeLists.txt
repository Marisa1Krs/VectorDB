# 指定CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)
# 必须放在 project() 之前或之后，指定默认构建类型为 Debug
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
# 强制添加调试信息（即使不小心切换到 Release 也生效）
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
# 项目名称
project(ServerProject)

# 设置C++标准（ONNX Runtime 1.25.1 要求 C++17）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 指定头文件目录
# 【变更说明】已删除 cppjieba、simhash 头文件目录
# 原代码：
#   ${PROJECT_SOURCE_DIR}/include/cppjieba
#   ${PROJECT_SOURCE_DIR}/include/simhash
# 上述两行已被删除，因为：
#   - jieba 分词已被 BERT 模型替代
#   - simhash 去重已被语义向量检索替代
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/json
    ${PROJECT_SOURCE_DIR}/model/onnxruntime_lib/include  # ONNX Runtime C API
)

# 显式列出所有需要编译的源文件
# 【变更说明】已改为显式列表，替代原来的 aux_source_directory
# 已排除以下文件（它们依赖 jieba，已被删除）：
#   - DictProducer.cpp（词典构建，不再需要）
#   - PageLibPreprocessor.cpp（倒排索引构建，已被向量检索替代）
#   - SplitTool.cpp（jieba 分词包装器）
#   - WebPage.cpp（包含 jieba 分词处理）
set(SRCS
    ${PROJECT_SOURCE_DIR}/src/Acceptor.cpp
    ${PROJECT_SOURCE_DIR}/src/Configer.cpp
    ${PROJECT_SOURCE_DIR}/src/DictProducer.cpp       # jieba 词典加载 + 编辑距离匹配（降准匹配层）
    ${PROJECT_SOURCE_DIR}/src/DirScanner.cpp
    ${PROJECT_SOURCE_DIR}/src/EventLoop.cpp
    ${PROJECT_SOURCE_DIR}/src/EventLoopGroup.cpp
    ${PROJECT_SOURCE_DIR}/src/FileProcessor.cpp
    ${PROJECT_SOURCE_DIR}/src/InetAddress.cpp
    ${PROJECT_SOURCE_DIR}/src/InferEngine.cpp        # BERT 推理引擎
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/mylog.cpp
    ${PROJECT_SOURCE_DIR}/src/PageLib.cpp
    ${PROJECT_SOURCE_DIR}/src/SemanticIndexer.cpp    # 语义搜索引擎（含 jieba+BGE 融合 suggest）
    ${PROJECT_SOURCE_DIR}/src/Socket.cpp
    ${PROJECT_SOURCE_DIR}/src/SockIO.cpp
    ${PROJECT_SOURCE_DIR}/src/Task.cpp
    ${PROJECT_SOURCE_DIR}/src/taskQueue.cpp
    ${PROJECT_SOURCE_DIR}/src/TcpConnetion.cpp
    ${PROJECT_SOURCE_DIR}/src/TcpServer.cpp
    ${PROJECT_SOURCE_DIR}/src/threadpool.cpp
)

# 生成可执行程序，名为server
add_executable(server ${SRCS})

# 链接库
# 【变更说明】
# 已添加 sqlite3（替代 newripepage.dat 文件存储）
# 原代码使用 ::write/lseek/read 操作原始文件，现使用 SQLite 的 C API
target_link_libraries(server PRIVATE
    tinyxml2
    onnxruntime        # ONNX Runtime 推理引擎（位于 model/onnxruntime_lib/lib/libonnxruntime.so）
    icuuc              # ICU Unicode 支持（WordPieceTokenizer 需要）
    sqlite3            # SQLite 嵌入式数据库（替代文件 I/O）
)

# 指定 ONNX Runtime 和 SQLite 库文件搜索路径
target_link_directories(server PRIVATE
    ${PROJECT_SOURCE_DIR}/model/onnxruntime_lib/lib
)

# 设置运行时动态库搜索路径
set_target_properties(server PROPERTIES
    INSTALL_RPATH "${PROJECT_SOURCE_DIR}/model/onnxruntime_lib/lib"
    BUILD_RPATH "${PROJECT_SOURCE_DIR}/model/onnxruntime_lib/lib"
)
# ####################################### 客户端子目录 #######################################
add_subdirectory(${PROJECT_SOURCE_DIR}/cli)
# ###########################################################################################
