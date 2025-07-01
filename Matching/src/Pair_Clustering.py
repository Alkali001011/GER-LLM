from collections import defaultdict

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def cal_eps_by_test_data_itself(fea, cluster_percentile):
    """根据测试数据自身特征计算DBSCAN的eps参数
    Args:
        fea: 2D数组，形状为 [n_pairs, n_features]，所有候选实体对的特征矩阵
        cluster_percentile: 百分位数参数，用于确定距离阈值
    Returns:
        eps: 适用于DBSCAN算法的邻域半径参数
    """
    # 计算所有测试对之间的欧氏距离矩阵
    # 输入：fea为N×M矩阵，N是数据对数量，M是特征维度
    # 输出：N×N对称矩阵，pair_distances[i,j]表示第i对和第j对数据之间的欧氏距离
    pair_distances = pairwise_distances(fea)

    # 提取上三角部分（不含对角线）的距离值
    # --------------------------------------------------
    # 由于距离矩阵是对称矩阵（d(i,j)=d(j,i)），对角线值为0（数据对与自身的距离）
    # 我们需要排除重复值和无效的对角线值：
    # 1. 获取距离矩阵维度（即测试数据对的总数）
    n_pairs = pair_distances.shape[0]  # 矩阵行数 = 数据对数量

    # 2. 生成上三角矩阵的索引（triangular upper indices）
    # np.triu_indices(n, k=1) 返回二维数组的上三角部分索引
    # 参数说明：
    #   - n: 矩阵维度
    #   - k=1: 从主对角线向上偏移1的位置开始（即排除对角线）
    # 例如：3x3矩阵，k=1时获取索引 (0,1), (0,2), (1,2)
    upper_indices = np.triu_indices(n_pairs, k=1)

    # 3. 通过索引提取非重复的有效距离值
    # 此时得到的是包含所有唯一非零距离值的一维数组
    # 原始矩阵共有N×N元素，提取后数组长度为 N(N-1)/2
    dis = pair_distances[upper_indices]

    # 根据距离分布计算百分位阈值
    # --------------------------------------------------
    # 例如 cluster_percentile=1 时，取距离值从小到大的1%位置作为阈值
    # 这意味着我们期望将最密集区域的1%距离作为邻域半径
    eps = np.percentile(dis, cluster_percentile)

    return eps


def generate_cluster_mappings(preds, candidate_pairs):
    """
    生成簇标签与候选实体对的双向映射，并输出聚类统计信息

    参数：
        preds : np.ndarray - DBSCAN聚类结果标签数组
        candidate_pairs : List[str] - 字符串表示的候选实体对列表

    返回：
        cluster_to_pairs : 字典 - {簇标签: [实体对str, ...]}
        pair_to_cluster : 字典 - {实体对str: 簇标签}
    """
    cluster_to_pairs = defaultdict(list)
    pair_to_cluster = {}

    # 统计原始噪声点数量
    n_noise_original = (preds == -1).sum()

    # 获取原始有效簇ID（排除噪声点）
    valid_clusters = [c for c in preds if c != -1]
    max_cluster = max(valid_clusters) if valid_clusters else -1

    # 创建统一噪声簇ID（仅当存在噪声点时创建）
    has_noise = n_noise_original > 0
    noise_cluster_id = max_cluster + 1 if has_noise and max_cluster != -1 else -1

    # 遍历处理每个候选对
    for pair_str, cluster_id in zip(candidate_pairs, preds):
        if cluster_id == -1:
            # 统一分配到噪声簇
            final_cluster = noise_cluster_id
        else:
            final_cluster = cluster_id

        pair_to_cluster[pair_str] = final_cluster
        cluster_to_pairs[final_cluster].append(pair_str)

    # 转换默认字典为普通字典
    cluster_to_pairs = dict(cluster_to_pairs)

    # 统计聚类结果
    total_clusters = len(cluster_to_pairs)
    n_noise_final = len(cluster_to_pairs.get(noise_cluster_id, [])) if has_noise else 0

    # 打印聚类信息
    print()
    print('[Clustering Report]:')
    print(f"  Total clusters: {total_clusters} ")
    print(f"  Noise points: {n_noise_final} ({'grouped into a single noise cluster' if has_noise else 'no noise points'})")

    # 统计簇大小分布
    cluster_sizes = {}
    for cid, pairs in cluster_to_pairs.items():
        if cid == noise_cluster_id:
            cluster_sizes["Noise Cluster"] = len(pairs)
        else:
            cluster_sizes[f"Cluster {cid}"] = len(pairs)

    # 按簇大小降序排序
    sorted_sizes = sorted(cluster_sizes.items(), key=lambda x: -x[1])
    print("\n[Cluster Size Distribution]：")
    for name, size in sorted_sizes:
        print(f"  {name}: {size} entity pairs")

    return cluster_to_pairs, pair_to_cluster


def hierarchical_noise_clustering(embeddings,
                                  primary_clusterer_params=None,
                                  secondary_clusterer_params=None,
                                  enable_reassignment=True):
    """分层噪声聚类框架

    参数:
        embeddings (np.ndarray): 输入的特征嵌入矩阵，形状为 (n_samples, n_features)
        primary_clusterer_params (dict): 主聚类器参数配置
        secondary_clusterer_params (dict): 二次聚类器参数配置
        enable_reassignment (bool): 是否启用噪声点重新分配到最近簇的功能

    返回:
        np.ndarray: 更新后的聚类标签数组，形状为 (n_samples,)
    """

    # 参数默认配置（使用逻辑或操作实现优雅的默认值设置）
    primary_clusterer_params = primary_clusterer_params or {
        'min_cluster_size': 12,  # 主聚类最小簇大小（过滤极微小簇）
        'min_samples': 5,  # 核心点判定阈值（值越小，核心点越多）
        'cluster_selection_method': 'leaf',  # 选择树结构末端的分割策略（生成更多小簇）
        'metric': 'euclidean',  # 使用欧氏距离
        'core_dist_n_jobs': -1  # 使用所有CPU核心
    }

    secondary_clusterer_params = secondary_clusterer_params or {
        'min_cluster_size': 8,  # 二次聚类更宽松的阈值（捕捉更小的潜在簇）
        'min_samples': 3,  # 更低的密度要求（适应噪声区域稀疏分布）
        'cluster_selection_method': 'leaf',  # 同上
        'metric': 'euclidean',
        'core_dist_n_jobs': -1
    }

    # === 第一阶段：主聚类 ===
    # 初始化主聚类器（生成最小生成树用于后续分析）
    primary_clusterer = HDBSCAN(
        **primary_clusterer_params,  # 解包参数字典
        gen_min_span_tree=True  # 生成最小生成树（可视化可选项）
    )
    # 执行主聚类并获取初始标签
    primary_labels = primary_clusterer.fit_predict(embeddings)

    # 提取噪声点索引（HDBSCAN将噪声标记为-1）
    noise_mask = (primary_labels == -1)
    # 获取噪声点的特征向量
    noise_embeddings = embeddings[noise_mask]

    # === 第二阶段：噪声点二次聚类 ===
    if len(noise_embeddings) > 0:  # 存在噪声点时执行
        # 初始化二次聚类器（参数更宽松以发现潜在小簇）
        secondary_clusterer = HDBSCAN(**secondary_clusterer_params)
        # 在噪声点子集上执行二次聚类
        secondary_labels = secondary_clusterer.fit_predict(noise_embeddings)

        # 标签重新编号逻辑（防止与主聚类标签冲突）
        max_primary_label = np.max(primary_labels)  # 主聚类最大有效标签
        # 过滤二次聚类中的噪声点（-1），并为有效标签生成新编号
        valid_secondary_labels = (
                secondary_labels[secondary_labels != -1]  # 筛选有效标签
                + max_primary_label + 1  # 偏移量确保唯一性
        )

        # 更新主标签数组
        # 1. 先将所有噪声点标签替换为二次聚类结果（包含新的-1）
        primary_labels[noise_mask] = secondary_labels
        # 2. 将二次聚类中找到的有效簇写入正确位置
        primary_labels[noise_mask][secondary_labels != -1] = valid_secondary_labels

    # === 剩余噪声点分配到最近邻簇 ===
    if enable_reassignment and np.any(primary_labels == -1):
        primary_labels = noise_reassignment(embeddings, primary_labels)

    return primary_labels  # 返回最终标签


def adaptable_clustering(embeddings: np.ndarray,
                         method: str,
                         clusterer_params: dict = None,
                         secondary_clusterer_params: dict = None,
                         enable_reassignment: bool = True,
                         reassignment_n_neighbors: int = 5) -> np.ndarray:
    """
    一个可适配的聚类框架，支持多种算法并模仿分层噪声处理逻辑。

    注意:
    - 对于 'dbscan'，此函数将完整执行两阶段聚类和噪声再分配。
    - 对于 'kmeans' 和 'agglomerative'，由于这些算法不产生噪声点，
      因此只会执行第一阶段的主聚类。

    参数:
        embeddings (np.ndarray): 输入的特征嵌入矩阵。
        method (str): 要使用的聚类算法，可选值为 ['kmeans', 'dbscan', 'agglomerative']。
        clusterer_params (dict): 主聚类器的参数配置。
        secondary_clusterer_params (dict): 二次聚类器参数配置 (仅用于 'dbscan')。
        enable_reassignment (bool): 是否启用噪声点重新分配。
        reassignment_n_neighbors (int): 噪声点再分配时使用的邻居数量。

    返回:
        np.ndarray: 最终的聚类标签数组。
    """

    # --- 第一阶段：主聚类 ---
    if method == 'kmeans':
        params = clusterer_params or {'n_clusters': 90, 'n_init': 'auto', 'random_state': 42}
        clusterer = KMeans(**params)
    elif method == 'dbscan':
        params = clusterer_params or {'eps': 0.06, 'min_samples': 5, 'n_jobs': -1}
        clusterer = DBSCAN(**params)
    elif method == 'agglomerative':
        params = clusterer_params or {'n_clusters': 90, 'linkage': 'ward'}
        clusterer = AgglomerativeClustering(**params)
    else:
        raise ValueError(f"Unsupported clustering method: '{method}'. Supported methods are 'kmeans', 'dbscan' and "
                         f"'agglomerative'")

    primary_labels = clusterer.fit_predict(embeddings)

    # --- 第二阶段：噪声点二次聚类 (仅当算法能产生噪声时执行) ---
    noise_mask = (primary_labels == -1)
    noise_embeddings = embeddings[noise_mask]

    if method == 'dbscan' and len(noise_embeddings) > 0:
        sec_params = secondary_clusterer_params or {'eps': 0.16, 'min_samples': 3, 'n_jobs': -1}
        secondary_clusterer = DBSCAN(**sec_params)
        secondary_labels = secondary_clusterer.fit_predict(noise_embeddings)

        max_primary_label = np.max(primary_labels)
        valid_secondary_mask = (secondary_labels != -1)
        valid_secondary_labels = secondary_labels[valid_secondary_mask] + max_primary_label + 1

        temp_secondary_labels = np.copy(secondary_labels)
        temp_secondary_labels[valid_secondary_mask] = valid_secondary_labels
        primary_labels[noise_mask] = temp_secondary_labels

    # --- 第三阶段：剩余噪声点分配到最近邻簇 ---
    if method == 'dbscan' and enable_reassignment and np.any(primary_labels == -1):
        # 调用您提供的 noise_reassignment 函数，并传递邻居数量参数
        primary_labels = noise_reassignment(
            embeddings,
            primary_labels,
            n_neighbors=reassignment_n_neighbors
        )

    return primary_labels


def noise_reassignment(embeddings, labels, n_neighbors=5):
    """将剩余噪声点分配到最近邻簇（软分配）

    参数:
        embeddings (np.ndarray): 特征矩阵
        labels (np.ndarray): 含噪声标签（-1）的聚类结果
        n_neighbors (int): 参与投票的最近邻数量

    返回:
        np.ndarray: 更新后的标签数组
    """
    # 检查是否存在未分配的噪声点
    noise_mask = (labels == -1)
    if not np.any(noise_mask):
        return labels  # 无噪声点时直接返回

    # 提取有效簇数据（排除噪声点）
    valid_data = embeddings[~noise_mask]  # 有效样本特征
    valid_labels = labels[~noise_mask]  # 对应的有效标签

    # 训练KNN模型（使用有效数据构建搜索空间）
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(valid_data)
    # 为每个噪声点查找最近的k个有效样本
    distances, indices = nbrs.kneighbors(embeddings[noise_mask])

    # 投票分配策略
    neighbor_labels = valid_labels[indices]  # 获取邻居标签（形状: (n_noise, k)）
    # 对每个噪声点的邻居标签进行统计，选择众数
    voted_labels = [
        np.bincount(row).argmax()  # 计算每个行的众数
        for row in neighbor_labels  # 逐行处理噪声点的邻居
    ]
    # 更新标签数组中的噪声点
    labels[noise_mask] = voted_labels

    return labels


def print_clustering_report(labels):
    """输出关键聚类指标"""
    n_noise = np.sum(labels == -1)
    cluster_sizes = np.bincount(labels[labels != -1])

    print()
    print(f"""========== Clustering_Report ==========
Total number of samples: {len(labels)}
Number of valid clusters: {len(cluster_sizes)}
Remaining noise points: {n_noise} ({n_noise / len(labels):.1%})
Maximum cluster size: {np.max(cluster_sizes)} ({np.max(cluster_sizes) / len(labels):.1%})
Average cluster size: {np.mean(cluster_sizes):.1f} ± {np.std(cluster_sizes):.1f}
============================
    """)

