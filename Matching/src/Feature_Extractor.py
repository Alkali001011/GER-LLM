import numpy as np
from sklearn.preprocessing import RobustScaler

from Blocking.src.tools.utils import haversine
from Matching.src.tools.utils import serialize_entity, process_alphanumeric, semantic_based_sims, \
    cal_similarities_between_categories, cal_similarities_between_names, cal_similarities_between_addresses


def extract_features_from_candidate_pairs(method, candidate_pairs, encoder=None):
    if method == 'SEMANTIC_BASED':  # 基于语义的特征提取
        return get_feature_based_on_semantic(candidate_pairs, encoder)
    elif method == 'PROP_BASED':  # 基于结构感知的特征提取
        if encoder is None:
            return get_feature_based_on_properties(candidate_pairs)  # 使用特定相似度计算方法计算属性相似度
        else:
            return get_feature_based_on_properties_with_encoder(candidate_pairs, encoder)  # 使用PLM计算属性embeddings的相似度


def get_feature_based_on_semantic(candidate_pairs, encoder):
    # candidate_pairs 候选实体对的列表，元素为元组对
    # 输出：过了一遍bert模型后的embedding     numbers of candidates x 768
    serialized_pairs = []  # 存储所有序列化的实体对

    for pair in candidate_pairs:
        se1, se2 = pair
        # 生成结构化文本对
        s_pair = serialize_entity(se1) + ' [SEP] ' + serialize_entity(se2)
        serialized_pairs.append(s_pair)

    # # 调试：打印前两个样本
    # print("序列化样本0:", serialized_pairs[0])
    # print("序列化样本1:", serialized_pairs[1])

    return encoder.encode(serialized_pairs)  # 整体编码文本对 默认采用平均池化


def get_feature_based_on_properties(candidate_pairs):
    fea = []

    for pair in candidate_pairs:
        se1, se2 = pair

        name_sim = cal_similarities_between_names(se1.name, se2.name)
        category_sim = cal_similarities_between_categories(se1.category, se2.category)
        address_sim = cal_similarities_between_addresses(se1.address, se2.address)

        fea.append([name_sim, category_sim, address_sim])

    return np.array(fea)


def get_feature_based_on_properties_with_encoder(candidate_pairs, encoder):
    item_former_list = []  # 把前一个数据集所有样本的属性都放进去，相当于s
    item_latter_list = []

    for pair in candidate_pairs:
        se1, se2 = pair

        # 假定属性不存在空值
        item_former_list.append(se1.name)
        item_former_list.append(se1.category.replace(' ', '/'))
        item_former_list.append(process_alphanumeric(se1.address))

        item_latter_list.append(se2.name)
        item_latter_list.append(se2.category.replace(' ', '/'))
        item_latter_list.append(process_alphanumeric(se2.address))

    em1 = encoder.encode(item_former_list)
    em2 = encoder.encode(item_latter_list)

    sims = semantic_based_sims(em1, em2)

    fea = []

    for i in range(0, len(sims), 3):  # 每次跳3个元素
        group = sims[i:i + 3]  # 截取当前三个元素
        fea.append(group)  # 添加分组到结果列表

    return np.array(fea)


def get_feature_based_on_multi_fea_fusion(candidate_pairs, city):
    # # 1. 计算每对候选实体对的两个实体之间的mean embeddings的余弦相似度
    # se_former_list = []
    # se_latter_list = []
    #
    # for pair in candidate_pairs:
    #     se1, se2 = pair
    #
    #     se_former_list.append(serialize_entity(se1))
    #     se_latter_list.append(serialize_entity(se2))
    #
    # em1 = encoder.encode(se_former_list, 'mean')
    # em2 = encoder.encode(se_latter_list, 'mean')
    #
    # sims_mean = semantic_based_sims(em1, em2)

    # 2. 处理每对候选实体对的两个实体之间文本属性的相似度
    sims_text = []

    for pair in candidate_pairs:
        se1, se2 = pair

        name_sim = cal_similarities_between_names(se1.name, se2.name)
        category_sim = cal_similarities_between_categories(se1.category, se2.category)
        address_sim = cal_similarities_between_addresses(se1.address, se2.address)

        if city == 'pit':
            sims_text.append([name_sim, address_sim])  # pit数据集没有category字段，所以这里不加上
        else:
            sims_text.append([name_sim, category_sim, address_sim])

    # 3. 处理每对候选实体对的两个实体之间的空间信息
    sim_dist = []
    dist = []
    max_dist = -1

    for pair in candidate_pairs:
        se1, se2 = pair

        d = haversine((se1.longitude, se1.latitude), (se2.longitude, se2.latitude))

        dist.append(d)

        max_dist = max(max_dist, d)

    for d in dist:
        sim_dist.append(1 - (d / max_dist))

    # # 4. 特征拼接
    # fea = []
    # for sm, st, sd in zip(sims_mean, sims_text, sim_dist):
    #     combined = [sm] + st + [sd]
    #     fea.append(combined)
    #
    # return fea

    # 4. 特征拼接
    fea = []
    for st, sd in zip(sims_text, sim_dist):
        combined = st + [sd]
        fea.append(combined)

    # return np.array(fea)  # 返回python 数组类型

    # 将列表转换为numpy数组
    fea_array = np.array(fea)  # 形状为 (3658,4) 的数组

    # # 初始化MinMaxScaler（默认缩放到[0,1]）
    # scaler = MinMaxScaler()
    #
    # # 执行归一化（按列处理，即对每个特征维度独立缩放）
    # fea_normalized = scaler.fit_transform(fea_array)

    # 对欧氏距离更鲁棒的归一化
    scaler = RobustScaler()
    fea_normalized = scaler.fit_transform(fea_array)

    return fea_normalized  # 返回归一化后的特征矩阵
