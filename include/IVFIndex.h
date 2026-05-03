/**
 * @file IVFIndex.h
 * @brief 把向量"分堆"搜索，不用挨个比（声明的头文件）
 *
 * 原理很简单：如果有一万篇文章，逐一比对太慢。
 * 先用 K-Means 算法把文章分成 K 个"堆"，
 * 搜的时候只看离查询词最近的几堆就行，不用看全部。
 *
 * 步骤：
 * 1. 训练：把所有文章向量分成 K 个组（叫"聚类"）
 * 2. 建库：每篇文章归到离它最近的组
 * 3. 查询：先找离查询词最近的 nprobe 个组 → 只看这几个组里的文章
 *
 * 适用于 512 维 BERT 向量的近似搜索（不精确但快很多）。
 */

#pragma once

#include <vector>
#include <string>
#include <utility>

class IVFIndex {
public:
    explicit IVFIndex(int dim = 512);
    ~IVFIndex() = default;

    // 禁止拷贝
    IVFIndex(const IVFIndex&) = default;
    IVFIndex& operator=(const IVFIndex&) = default;

    // ============================================================
    // 训练：用 K-Means 把数据分成 K 堆
    // @param data    所有向量拼成的大数组（N行×dim列）
    // @param N       总共有多少个向量
    // @param K       要分成几个堆
    // @param maxIter 最多迭代多少轮（默认 50 轮）
    // @return 每个向量属于哪个组，assignments[i] 表示第 i 个向量的组编号
    // ============================================================
    std::vector<int> train(const float* data, int N, int K, int maxIter = 50);

    // ============================================================
    // 建库：把 N 个向量分别归到离自己最近的组里
    // 更新 inverted_lists_（每个组的成员列表）和 assignments_（归属记录）
    // ============================================================
    void buildInvertedLists(const float* data, int N);

    // ============================================================
    // 搜索：找离查询词最近的 nprobe 个组
    // @param query   查询词的向量（dim 维）
    // @param nprobe  想看最近的几个组（默认 1 个）
    // @return 按距离从小到大排的 (组编号, 距离值) 列表
    // ============================================================
    std::vector<std::pair<int, float>> searchCentroids(
        const float* query, int nprobe = 1) const;

    // ============================================================
    // 获取某个组里所有文章的编号列表
    // ============================================================
    const std::vector<int>& getInvertedList(int clusterId) const;

    // ============================================================
    // 保存/加载：把训练好的分组信息存到文件 / 从文件读回来
    // ============================================================
    void save(const std::string& path) const;
    void load(const std::string& path);

    // ============================================================
    // Getters
    // ============================================================
    int getK()   const { return K_; }
    int getDim() const { return dim_; }
    int size()   const { return ntotal_; }

private:
    int dim_;                             // 每个向量的维度（512）
    int K_;                               // 分成几个组
    int ntotal_ = 0;                      // 总共有多少个向量

    std::vector<std::vector<float>> centroids_;   // 每组中心点的坐标（K×dim）
    std::vector<int> assignments_;                 // 每个向量归到了哪个组（N 个值）
    std::vector<std::vector<int>> inverted_lists_; // 每个组里有哪些向量（K 个列表）

    // ============================================================
    // 计算两个点之间的距离（平方值）
    // ============================================================
    float _l2sqr(const float* a, const float* b) const;

    // ============================================================
    // 选初始的组中心点（K-Means++ 算法）
    // 思路：第一个随机选，后面的尽量选离已选中心远的
    // 这样分出来的组更均匀
    // ============================================================
    void _initCentroidsKmeansPP(const float* data, int N);
};
