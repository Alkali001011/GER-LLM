import copy
import random
from random import sample


def generate_random_batches(cluster_to_pairs, batch_size):
    """
    将聚类后的实体对按随机性策略分批次。
    该策略首先将所有聚类中的所有实体对收集起来，然后随机打乱它们的顺序，
    最后按指定的batch_size进行切分。

    Args:
        cluster_to_pairs (dict): 聚类标签到实体对列表的映射，
                                 格式为 {cluster_label: [pair1, pair2,...]}
        batch_size (int): 每个批次的目标样本数量

    Returns:
        list: 批次列表，每个元素为实体对列表。每个批次中的样本是随机选择的，
              最后一个批次可能小于batch_size。
    """
    # 深拷贝原始聚类数据，以防外部数据结构被修改，尽管此函数主要进行读取和重组。
    # 对于此特定函数，如果 cluster_to_pairs 中的列表不会在别处被修改，
    # 仅收集所有对到一个新列表可能不需要深拷贝整个结构。
    # 但为了与您提供的其他函数保持一致性，这里使用深拷贝。
    all_pairs = []
    # 为了不直接修改原始传入的 cluster_to_pairs 字典中的列表内容
    # （例如，如果其他地方期望这些列表保持原样），我们迭代其值的副本。
    # 如果 cluster_to_pairs 本身就是临时的，或者其内部列表可以被消耗，
    # 那么深拷贝可能不是严格必需的，但作为通用函数是更安全的选择。
    # 实际上，我们只是读取这些列表，所以深拷贝cluster_to_pairs本身可能不是必须的，
    # 而是应该注意不要修改其内部的列表。
    # 一个更轻量级的方法是只深拷贝列表内容，或者如果pair是不可变对象则不需要。
    # 假设pair是字典或自定义对象，我们只复制引用到all_pairs。

    # 步骤 1: 收集所有实体对到一个列表中
    # 我们不修改原始的 cluster_to_pairs，而是从中提取元素
    temp_cluster_data = copy.deepcopy(cluster_to_pairs)  # 使用深拷贝以确保原始数据安全
    for cluster_label in temp_cluster_data:
        all_pairs.extend(temp_cluster_data[cluster_label])

    # 如果没有实体对，则返回空列表
    if not all_pairs:
        return []

    # 步骤 2: 随机打乱所有实体对的顺序
    random.shuffle(all_pairs)

    # 步骤 3: 按 batch_size 切分批次
    batches = []
    for i in range(0, len(all_pairs), batch_size):
        batch = all_pairs[i:i + batch_size]
        batches.append(batch)

    return batches


def generate_similar_batches(cluster_to_pairs, batch_size):
    """
    将聚类后的实体对按相似性策略分批次，确保每个批次尽可能包含同一个聚类簇的样本。

    Args:
        cluster_to_pairs (dict): 聚类标签到实体对列表的映射，格式为 {cluster_label: [pair1, pair2,...]}
        batch_size (int): 每个批次的目标样本数量

    Returns:
        list: 批次列表，每个元素为实体对列表。
              策略是优先用同一个聚类的样本填满一个批次。
              如果一个聚类的样本不足以填满一个批次，则会继续从下一个聚类中取样本来补充当前批次。

    策略说明：
        1. 依次遍历每个聚类。
        2. 从当前聚类中不断取出样本放入当前批次，直到批次满或当前聚类耗尽。
        3. 如果批次已满，则将其存入结果列表，并开启新的空批次。
        4. 如果当前聚类已耗尽但批次未满，则继续从下一个聚类中取样本补充当前批次。
        5. 所有聚类的所有样本都处理完毕后，如果最后一个批次中仍有样本（即使未满），也将其加入结果列表。
    """
    # 深拷贝原始聚类数据，防止修改外部传入的字典结构
    # We will be popping items, so a deep copy is essential if the original needs to be preserved.
    # If the original cluster_to_pairs lists can be modified, a shallow copy of the dict
    # and shallow copies of the lists might be sufficient, but deepcopy is safer.
    remaining_clusters_data = copy.deepcopy(cluster_to_pairs)
    batches = []  # 存储最终生成的批次
    current_batch = []  # 当前正在构建的批次

    # 获取聚类标签列表，可以决定处理顺序（例如，可以按聚类大小排序）
    # 为简单起见，这里按字典默认顺序处理
    cluster_labels = list(remaining_clusters_data.keys())

    for cluster_label in cluster_labels:
        pairs_in_this_cluster = remaining_clusters_data[cluster_label]

        # 从当前聚类中取样本，直到该聚类耗尽
        while pairs_in_this_cluster:
            # 如果当前批次已满，则保存并开始新批次
            if len(current_batch) == batch_size:
                batches.append(current_batch)
                current_batch = []

            # 从当前聚类的样本列表中取出一个样本
            pair = pairs_in_this_cluster.pop(0) # 从列表头部取出
            current_batch.append(pair)

    # 处理完所有聚类后，如果current_batch中仍有剩余样本（最后一个批次可能未满）
    if current_batch:
        batches.append(current_batch)

    return batches


def generate_diverse_batches(cluster_to_pairs, batch_size):
    """
    将聚类后的实体对按多样性策略分批次，确保每个批次尽可能包含不同聚类的样本

    Args:
        cluster_to_pairs (dict): 聚类标签到实体对列表的映射，格式为 {cluster_label: [pair1, pair2,...]}
        batch_size (int): 每个批次的目标样本数量

    Returns:
        list: 批次列表，每个元素为实体对列表，保证每个batch内尽可能多不同cluster的样本

    策略说明：
        1. 当剩余聚类数量 >= batch_size时，随机选择batch_size个不同聚类，每个聚类取一个样本
        2. 当剩余聚类数量 < batch_size时，轮询现有聚类，每个聚类取一个样本，直到填满batch
        3. 自动清理已耗尽样本的聚类，动态维护剩余样本状态
    """
    # 深拷贝原始聚类数据，防止修改外部传入的字典结构
    remain_clusters = copy.deepcopy(cluster_to_pairs)
    batches = []  # 存储最终生成的批次

    while True:
        # 获取当前可用的聚类标签列表（动态更新）
        current_clusters = list(remain_clusters.keys())

        # 终止条件：所有聚类样本已分配完毕
        if not current_clusters:
            break

        # Case 1: 剩余聚类数足够组成完整batch（>= batch_size）
        if len(current_clusters) >= batch_size:
            # 从当前聚类中随机选择batch_size个不同聚类（无放回抽样）
            selected_clusters = sample(current_clusters, batch_size)

            batch = []  # 当前正在构建的批次
            for cluster in selected_clusters:
                # 从该聚类中取出第一个样本（队列式弹出，保证先进先出）
                if remain_clusters[cluster]:  # 防御性检查，防止其他循环已清空该聚类
                    pair = remain_clusters[cluster].pop(0)
                    batch.append(pair)

                    # 若该聚类样本已耗尽，从剩余聚类字典中删除该键
                    if not remain_clusters[cluster]:
                        del remain_clusters[cluster]

            # 将构建好的批次加入结果列表
            batches.append(batch)

        # Case 2: 剩余聚类数不足，需轮询填充
        else:
            batch = []  # 当前正在构建的批次
            # 持续填充直到满足batch_size或样本耗尽
            while len(batch) < batch_size and current_clusters:
                # 遍历当前所有聚类（实时获取最新聚类列表）
                for cluster in list(remain_clusters.keys()):
                    # 提前终止：已填满batch
                    if len(batch) >= batch_size:
                        break

                    # 检查当前聚类是否仍有样本（可能被前序循环处理过）
                    if remain_clusters.get(cluster):
                        # 取出该聚类的第一个样本
                        pair = remain_clusters[cluster].pop(0)
                        batch.append(pair)

                        # 清理空聚类
                        if not remain_clusters[cluster]:
                            del remain_clusters[cluster]

                # 更新当前可用聚类列表（可能已被动态修改）
                current_clusters = list(remain_clusters.keys())

            # 将非空的批次加入结果列表
            if batch:
                batches.append(batch)
            else:
                break  # 无剩余样本时彻底终止循环

    return batches


