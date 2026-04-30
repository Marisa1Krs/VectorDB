#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include "json.hpp"
#include "mylog.h"
#include "Configer.h"
#include "SemanticIndexer.h"
#include "InetAddress.h"
#include "Socket.h"
#include "Acceptor.h"
#include "SockIO.h"
#include "EventLoop.h"
#include "EventLoopGroup.h"
#include "TcpServer.h"
#include "threadpool.h"
#include "Task.h"

TcpServer *server = nullptr;
threadpool *tpool = nullptr;

// 服务器就绪标志：构建语义索引期间为 false，完成后设为 true
// 在构建索引期间，客户端连接将被拒绝并收到提示
volatile bool g_serverReady = false;

// 缓存的静态 HTML 页面内容（启动时从文件加载）
string g_staticHtmlContent;

using std::cout;
using std::endl;
using std::string;
using std::vector;
using json = nlohmann::json;

vector<string> splitBySpace(const std::string &input)
{
    vector<string> result;
    string current;

    for (char c : input)
    {
        if (c == ' ')
        {
            if (!current.empty())
            {
                result.push_back(current);
                current.clear();
            }
        }
        else
        {
            current += c;
        }
    }

    if (!current.empty())
    {
        result.push_back(current);
    }

    return result;
}

// ============================================================
// HTTP 协议辅助函数
// ============================================================

// 简易 URL 解码：将 %XX 替换为字符
string urlDecode(const string& input) {
    string result;
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '%' && i + 2 < input.size()) {
            char hex[3] = {input[i+1], input[i+2], '\0'};
            char* end;
            long code = strtol(hex, &end, 16);
            if (*end == '\0') {
                result += static_cast<char>(code);
                i += 2;
                continue;
            }
        } else if (input[i] == '+') {
            result += ' ';
            continue;
        }
        result += input[i];
    }
    return result;
}

// 解析 HTTP GET 请求，提取路径和查询参数 q
// 输入格式: "GET /search?q=keyword HTTP/1.1\r\n..."
void parseHttpGet(const string& request, string& path, string& query) {
    size_t lineEnd = request.find("\r\n");
    if (lineEnd == string::npos) {
        path = "/"; query = "";
        return;
    }
    string reqLine = request.substr(0, lineEnd);

    size_t firstSpace = reqLine.find(' ');
    if (firstSpace == string::npos) { path = "/"; query = ""; return; }
    size_t secondSpace = reqLine.find(' ', firstSpace + 1);
    if (secondSpace == string::npos) { path = "/"; query = ""; return; }

    string uri = reqLine.substr(firstSpace + 1, secondSpace - firstSpace - 1);

    size_t qmark = uri.find('?');
    if (qmark != string::npos) {
        path = uri.substr(0, qmark);
        string queryStr = uri.substr(qmark + 1);
        size_t pos = 0;
        while (pos < queryStr.size()) {
            size_t amp = queryStr.find('&', pos);
            string pair = queryStr.substr(pos, amp == string::npos ? string::npos : amp - pos);
            size_t eq = pair.find('=');
            if (eq != string::npos) {
                string key = pair.substr(0, eq);
                string val = pair.substr(eq + 1);
                if (key == "q" || key == "query" || key == "keywords") {
                    query = urlDecode(val);
                    break;
                }
            }
            if (amp == string::npos) break;
            pos = amp + 1;
        }
    } else {
        path = uri;
    }
}

// 构建 HTTP 响应字符串
string buildHttpResponse(const json& body, int statusCode = 200, const string& statusText = "OK") {
    string bodyStr = body.dump();
    string response = "HTTP/1.1 " + std::to_string(statusCode) + " " + statusText + "\r\n"
                      "Content-Type: application/json; charset=utf-8\r\n"
                      "Content-Length: " + std::to_string(bodyStr.size()) + "\r\n"
                      "Connection: keep-alive\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "\r\n" +
                      bodyStr;
    return response;
}

// 构建 HTTP 响应（支持任意 Content-Type，用于返回 HTML 等非 JSON 内容）
string buildHttpResponse(const string& body, const string& contentType, int statusCode, const string& statusText) {
    string response = "HTTP/1.1 " + std::to_string(statusCode) + " " + statusText + "\r\n"
                      "Content-Type: " + contentType + "\r\n"
                      "Content-Length: " + std::to_string(body.size()) + "\r\n"
                      "Connection: keep-alive\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "\r\n" +
                      body;
    return response;
}

// ============================================================
// HTTP 请求处理函数：使用语义搜索替代倒排索引
// ============================================================
void handleHttpRequest(const string& path, const string& query, shared_ptr<TcpConnetion> con)
{
    LOG_INFO("HTTP path=%s, query=%s", path.c_str(), query.c_str());
    json msgJson;

    if (path == "/search" || path == "/semantic") {
        // 语义搜索（替代原 PageLibPreprocessor 的倒排索引搜索）
        // 使用 BERT 模型进行语义匹配，替代 jieba 分词 + TF-IDF
        if (!query.empty()) {
            msgJson = SemanticIndexer::getPtr()->find(query);
        } else {
            msgJson["error"] = "请提供搜索关键词，例如 GET /search?q=关键词";
        }
    } else if (path == "/suggest") {
        // 词语推荐 — 使用 BERT 语义相似度匹配文档标题
        // 替代原 DictProducer::find() 基于编辑距离的词典推荐
        if (!query.empty()) {
            msgJson = SemanticIndexer::getPtr()->suggest(query);
        } else {
            msgJson["error"] = "请提供关键词，例如 GET /suggest?q=搜索";
        }
    } else {
        // 根路径或未知路径，返回 API 使用帮助
        json endpoints = json::array();
        json ep1, ep2, ep3;
        ep1["path"] = "/search?q=关键词";
        ep1["description"] = "语义搜索（BERT 模型，替代原倒排索引）";
        ep2["path"] = "/semantic?q=关键词";
        ep2["description"] = "同上，语义搜索";
        ep3["path"] = "/suggest?q=关键词";
        ep3["description"] = "词语推荐 — 基于 BERT 语义相似度的文档标题推荐（替代原 DictProducer）";
        endpoints.push_back(ep1);
        endpoints.push_back(ep2);
        endpoints.push_back(ep3);
        msgJson["usage"] = "搜索引擎 HTTP API — 基于 BERT 语义搜索";
        msgJson["note"] = "已移除 jieba 分词、倒排索引、词典构建，全部改用模型推理搜索";
        msgJson["endpoints"] = endpoints;
    }

    string httpResp = buildHttpResponse(msgJson);
    con->sendInLoop(httpResp);
}

void onNewConnet(const shared_ptr<TcpConnetion> &con)
{
    cout << "新链接到来 " << con->getFd() << endl;
}

void onMessage(const shared_ptr<TcpConnetion> &con)
{
    string httpReq = con->recvHttp();

    string path, query;
    parseHttpGet(httpReq, path, query);

    // 根路径 → 返回静态 HTML 页面（即使索引未就绪也可访问前端界面）
    if (path == "/" || path.empty()) {
        if (!g_staticHtmlContent.empty()) {
            string response = buildHttpResponse(g_staticHtmlContent, "text/html; charset=utf-8", 200, "OK");
            con->sendInLoop(response);
        } else {
            json errJson;
            errJson["error"] = "静态页面未加载，请检查 cli/html/cli.html 文件";
            con->sendInLoop(buildHttpResponse(errJson, 500, "Internal Server Error"));
        }
        return;
    }

    // 服务器正在构建语义索引，拒绝处理请求
    if (!g_serverReady) {
        json errJson;
        errJson["error"] = "服务器正在初始化语义索引（BERT 编码中），请稍后再试";
        errJson["status"] = "initializing";
        con->sendInLoop(buildHttpResponse(errJson, 503, "Service Unavailable"));
        LOG_WARN("拒绝请求：服务器正在初始化语义索引");
        return;
    }

    // 无查询参数时返回 API 帮助信息
    if (query.empty()) {
        json helpJson;
        json endpoints = json::array();
        json ep1, ep2;
        ep1["path"] = "/search?q=关键词";
        ep1["description"] = "语义搜索（BERT 模型）";
        ep2["path"] = "/semantic?q=关键词";
        ep2["description"] = "同上，语义搜索";
        endpoints.push_back(ep1);
        endpoints.push_back(ep2);
        helpJson["usage"] = "搜索引擎 HTTP API — 基于 BERT 语义搜索";
        helpJson["note"] = "已移除 jieba 分词、倒排索引、词典构建";
        helpJson["endpoints"] = endpoints;
        con->sendInLoop(buildHttpResponse(helpJson));
        return;
    }

    // 提交任务到线程池处理
    Task *t = new Task();
    t->setTask(std::bind(&handleHttpRequest, path, query, con));
    tpool->addTask(t);
}

void onClose(const shared_ptr<TcpConnetion> &con)
{
    cout << "链接已经关闭" << endl;
}

int main()
{
    // 配置文件路径（唯一保留的硬编码路径，作为配置入口）
    string confPath = "/home/marisa/code1/VectorDB/config/serch.conf";
    Configer con(confPath);
    auto& cfg = con.getConfigMap();

    // 初始化日志：路径、缓冲区大小、日志级别
    string logPath = cfg["logPath"];
    int logBufSize = std::stoi(cfg["logBufferSize"]);
    int logLevel = std::stoi(cfg["logLevel"]);
    mylog::init(logPath, logBufSize, static_cast<LogLevel>(logLevel));

    // 加载静态 HTML 页面到内存缓存
    string htmlPath = confPath.substr(0, confPath.rfind('/'));
    htmlPath = htmlPath.substr(0, htmlPath.rfind('/'));
    htmlPath += "/cli/html/cli.html";
    std::ifstream ifs(htmlPath);
    if (ifs) {
        g_staticHtmlContent.assign(
            (std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>()
        );
        LOG_INFO("静态页面已加载: %s (%zu bytes)", htmlPath.c_str(), g_staticHtmlContent.size());
    } else {
        LOG_WARN("静态页面文件不存在: %s，根路径将返回 500 错误", htmlPath.c_str());
    }

    // ============================================================
    // 第一阶段：启动 TCP 服务器和线程池
    // ============================================================
    string serverIp = cfg["serverIp"];
    string serverPort = cfg["serverPort"];
    int subLoopCount = std::stoi(cfg["subEventLoopCount"]);

    server = new TcpServer(serverIp, serverPort, subLoopCount);

    int threadCount = std::stoi(cfg["threadPoolThreadCount"]);
    int queueSize = std::stoi(cfg["threadPoolQueueSize"]);
    tpool = new threadpool(threadCount, queueSize);

    server->setCallBack(onNewConnet, onMessage, onClose);
    tpool->start();

    // 在后台线程中启动 TcpServer
    std::thread serverThread([]() {
        LOG_INFO("TCP 服务器已启动，主 EventLoop 监听新连接（语义索引构建中...）");
        server->start();
    });

    // ============================================================
    // 第二阶段：构建语义索引（替代原倒排索引 + 词典构建）
    //
    // 【变更说明】
    // 原代码：
    //   DictProducer::init(con);
    //   DictProducer::getPtr()->buildEnDict();
    //   DictProducer::getPtr()->buildCnDict();
    //   PageLibPreprocessor::init(con);
    //   PageLibPreprocessor::getPtr()->doProcess();
    //
    // 上述代码已被完全删除，原因：
    //   - jieba 分词（DictProducer）被 BERT 模型替代
    //   - 倒排索引（PageLibPreprocessor）被向量检索替代
    //   - 不再需要构建中英文词典
    //   - 不再需要 TF-IDF 计算
    //
    // 新逻辑：
    //   1. 初始化 SemanticIndexer（打开 SQLite 数据库）
    //   2. 如果数据库为空，解析 XML → BERT 编码 → 写入 SQLite
    //   3. 如果已有数据，跳过构建（程序重启后直接使用已有索引）
    // ============================================================
    LOG_INFO("开始初始化语义搜索引擎...");

    string modelPath = cfg["modelPath"];
    string vocabPath = cfg["vocabPath"];
    SemanticIndexer::init(modelPath, vocabPath, con);

    // 检查是否需要构建索引
    if (SemanticIndexer::getPtr()->size() == 0) {
        LOG_INFO("SQLite 数据库为空，开始构建语义索引...");
        SemanticIndexer::getPtr()->buildIndex();
        LOG_INFO("语义索引构建完成");
    } else {
        LOG_INFO("SQLite 数据库已有 %zu 篇文档，跳过索引构建", SemanticIndexer::getPtr()->size());
    }

    // ============================================================
    // 第三阶段：索引构建完成，允许处理客户端请求
    // ============================================================
    g_serverReady = true;
    LOG_INFO("语义索引已就绪，服务器开始处理客户端请求");

    // 主线程等待服务器线程结束
    serverThread.join();
    return 0;
}
