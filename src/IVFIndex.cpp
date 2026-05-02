/**
 * @file IVFIndex.cpp
 * @brief 把向量"分堆"搜索的实现代码
 *
 * 说白了就是三步：
 * 1. 把所有文章向量分成 K 堆（这叫"聚类"）
 * 2. 建个目录：哪篇文章在哪个堆里
 * 3. 搜的时候先看查询词离哪几个堆最近，只看这些堆里的文章
 *    — 不用全部文章都看一遍，省时间
 */

#include "IVFIndex.h"

#include <random>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <limits>

// ============================================================
// 构造函数
// ============================================================

IVFIndex::IVFIndex(int dim)
    : dim_(dim), K_(0) {}

// ============================================================
// 核心：K-Means 训练，把 N 个向量分成 K 个组
// ============================================================

std::vector<int> IVFIndex::train(const float* data, int N, int K, int maxIter) {
    K_ = K;        // 记下要分几组
    ntotal_ = N;   // 记下总共有多少向量

    // ---- 1. 先随便挑 K 个点当"组长"（初始化） ----
    _initCentroidsKmeansPP(data, N);

    // ---- 2. 反复调整分组，直到稳定 ----
    std::vector<int> newAssignments(N, 0);          // 每个向量当前归哪个组
    std::vector<std::vector<float>> newCentroids(   // 新算出来的组长坐标
        K, std::vector<float>(dim_, 0.0f));
    std::vector<int> clusterSizes(K, 0);            // 每个组里有几个向量

    for (int iter = 0; iter < maxIter; ++iter) {
        // ---- 2a. 分派：每个向量归到离它最近的组长 ----
        int changed = 0;  // 记录有多少向量换了组
        for (int i = 0; i < N; ++i) {
            const float* vec = data + i * dim_;      // 第 i 个向量
            int bestCluster = 0;
            float bestDist = _l2sqr(vec, centroids_[0].data());

            // 跟每个组长比距离，找最近的
            for (int c = 1; c < K; ++c) {
                float dist = _l2sqr(vec, centroids_[c].data());
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = c;
                }
            }
            if (newAssignments[i] != bestCluster) {
                ++changed;  // 跟上次分配结果不一样，说明还在调整
            }
            newAssignments[i] = bestCluster;
        }

        // ---- 2b. 更新：重新算每个组的"新组长"（组内平均值） ----
        // 先清零
        std::fill(clusterSizes.begin(), clusterSizes.end(), 0);
        for (auto& centroid : newCentroids) {
            std::fill(centroid.begin(), centroid.end(), 0.0f);
        }

        // 把每个组里的向量坐标加起来
        for (int i = 0; i < N; ++i) {
            int c = newAssignments[i];       // 第 i 个向量属于组 c
            clusterSizes[c]++;               // 组 c 人数 +1
            const float* vec = data + i * dim_;
            for (int j = 0; j < dim_; ++j) {
                newCentroids[c][j] += vec[j];  // 累加坐标
            }
        }

        // 除以人数 = 平均值，这就是新的组长坐标
        for (int c = 0; c < K; ++c) {
            if (clusterSizes[c] > 0) {
                float inv = 1.0f / static_cast<float>(clusterSizes[c]);
                for (int j = 0; j < dim_; ++j) {
                    newCentroids[c][j] *= inv;  // 求平均
                }
            } else {
                // 特殊情况：某个组没人了（空组）
                // 从隔壁组复制一个坐标，稍微加点随机抖动
                int src = (c + 1) % K;
                float noise = 0.01f;
                for (int j = 0; j < dim_; ++j) {
                    newCentroids[c][j] = centroids_[src][j]
                        + noise * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
                }
            }
        }

        // 用新组长替换旧组长
        centroids_.swap(newCentroids);
        assignments_ = newAssignments;

        // 如果没人换组了，说明分组稳定了，提前结束
        if (changed == 0) {
            break;
        }
    }

    // ---- 3. 建目录：每个向量归到最终确定的组 ----
    buildInvertedLists(data, N);

    return assignments_;
}

// ============================================================
// 选初始"组长"（K-Means++ 算法）
// 思路：第一个随便挑，后面的尽量挑离已有组长远的，这样各组更分散
// ============================================================

void IVFIndex::_initCentroidsKmeansPP(const float* data, int N) {
    centroids_.resize(K_);
    for (auto& c : centroids_) {
        c.resize(dim_);
    }

    // 第一个组长：随便挑一个数据点
    std::mt19937 rng(42);  // 随机种子固定为 42，这样每次跑结果一样
    std::uniform_int_distribution<int> dist(0, N - 1);
    int firstIdx = dist(rng);
    std::memcpy(centroids_[0].data(), data + firstIdx * dim_, dim_ * sizeof(float));

    // 后面的组长：离已有组长越远的点，越容易被挑中
    // 这样各组不会挤在一起
    std::vector<float> minDists(N, std::numeric_limits<float>::max());

    for (int c = 1; c < K_; ++c) {
        // 算每个点到最近组长的距离
        float totalDist = 0.0f;
        for (int i = 0; i < N; ++i) {
            const float* vec = data + i * dim_;
            float d = _l2sqr(vec, centroids_[c - 1].data());
            if (d < minDists[i]) {
                minDists[i] = d;  // 更新为更近的距离
            }
            totalDist += minDists[i];
        }

        // 轮盘赌：距离越大的点，被选中的概率越大
        std::uniform_real_distribution<float> realDist(0.0f, totalDist);
        float threshold = realDist(rng);  // 在 0~总距离之间随机一个数
        float cumulative = 0.0f;
        int chosen = 0;
        for (int i = 0; i < N; ++i) {
            cumulative += minDists[i];
            if (cumulative >= threshold) {  // 累加距离直到超过随机阈值
                chosen = i;
                break;
            }
        }
        // 选中的点当新组长
        std::memcpy(centroids_[c].data(), data + chosen * dim_, dim_ * sizeof(float));
    }
}

// ============================================================
// 建库：把每个向量分到离它最近的组
// 建成后 inverted_lists_[组编号] 里存的是这个组里所有向量的编号
// ============================================================

void IVFIndex::buildInvertedLists(const float* data, int N) {
    ntotal_ = N;
    inverted_lists_.clear();
    inverted_lists_.resize(K_);
    if (assignments_.size() != static_cast<size_t>(N)) {
        assignments_.resize(N);
    }

    // 每个向量都去找离它最近的组长，登记入组
    for (int i = 0; i < N; ++i) {
        const float* vec = data + i * dim_;
        int bestCluster = 0;
        float bestDist = _l2sqr(vec, centroids_[0].data());

        for (int c = 1; c < K_; ++c) {
            float dist = _l2sqr(vec, centroids_[c].data());
            if (dist < bestDist) {
                bestDist = dist;
                bestCluster = c;
            }
        }

        inverted_lists_[bestCluster].push_back(i);  // 把 i 号向量加入组 bestCluster
        assignments_[i] = bestCluster;               // 记下 i 号向量属于哪个组
    }
}

// ============================================================
// 搜索：给一个查询向量，找出离它最近的 nprobe 个组
// 后续就只看这几个组里的文章，不扫描全部
// ============================================================

std::vector<std::pair<int, float>> IVFIndex::searchCentroids(
    const float* query, int nprobe) const
{
    // 算一下查询点到每个组长的距离
    std::vector<std::pair<int, float>> centroidDist;
    centroidDist.reserve(K_);

    for (int c = 0; c < K_; ++c) {
        float dist = _l2sqr(query, centroids_[c].data());
        centroidDist.push_back({c, dist});  // (组编号, 距离)
    }

    // 按距离从小到大排序，最近的排在前面
    std::partial_sort(
        centroidDist.begin(),
        centroidDist.begin() + std::min(nprobe, K_),
        centroidDist.end(),
        [](const std::pair<int, float>& a,
           const std::pair<int, float>& b) {
            return a.second < b.second;
        });

    // 只返回前 nprobe 个最近的组
    int n = std::min(nprobe, K_);
    centroidDist.resize(n);
    return centroidDist;
}

// ============================================================
// 获取某个组里所有向量的编号
// ============================================================

const std::vector<int>& IVFIndex::getInvertedList(int clusterId) const {
    return inverted_lists_[clusterId];
}

// ============================================================
// 保存到文件：把训练好的分组信息存起来，下次启动直接读
// 文件格式：[维度(int) | 组数(int) | 总数(int) | 组长坐标(K×dim个float) | 归属记录(N个int)]
// ============================================================

void IVFIndex::save(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return;

    // 先写三个数字：维度、组数、总数
    ofs.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    ofs.write(reinterpret_cast<const char*>(&K_), sizeof(K_));
    ofs.write(reinterpret_cast<const char*>(&ntotal_), sizeof(ntotal_));

    // 再写每个组长的坐标
    for (int c = 0; c < K_; ++c) {
        ofs.write(reinterpret_cast<const char*>(centroids_[c].data()),
                  dim_ * sizeof(float));
    }

    // 最后写每个向量归到了哪个组
    ofs.write(reinterpret_cast<const char*>(assignments_.data()),
              ntotal_ * sizeof(int));
}

// ============================================================
// 从文件加载：把之前存的分组信息读回来
// ============================================================

void IVFIndex::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return;

    // 读三个数字
    ifs.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    ifs.read(reinterpret_cast<char*>(&K_), sizeof(K_));
    ifs.read(reinterpret_cast<char*>(&ntotal_), sizeof(ntotal_));

    // 读每个组长的坐标
    centroids_.resize(K_);
    for (int c = 0; c < K_; ++c) {
        centroids_[c].resize(dim_);
        ifs.read(reinterpret_cast<char*>(centroids_[c].data()),
                 dim_ * sizeof(float));
    }

    // 读每个向量的归属记录
    assignments_.resize(ntotal_);
    ifs.read(reinterpret_cast<char*>(assignments_.data()),
             ntotal_ * sizeof(int));

    // 根据归属记录重建"组→成员"的目录
    inverted_lists_.clear();
    inverted_lists_.resize(K_);
    for (int i = 0; i < ntotal_; ++i) {
        int c = assignments_[i];
        if (c >= 0 && c < K_) {
            inverted_lists_[c].push_back(i);  // 第 i 个向量属于组 c
        }
    }
}

// ============================================================
// 计算两个向量之间的距离（欧几里得距离的平方）
// 就是每个分量差值的平方加起来
// ============================================================

float IVFIndex::_l2sqr(const float* a, const float* b) const {
    float sum = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float diff = a[i] - b[i];   // 同一个位置上的差值
        sum += diff * diff;          // 平方后累加
    }
    return sum;
}
