import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from Blocking.src.tools.utils import break_down_time


def get_max_weight_matching_answers(all_answers, all_confidences, all_batches):
    """
    通过构建二分图并执行最大权匹配来解决实体对匹配中的冲突。

    参数:
        all_answers (list): 整数列表 (0 或 1)，指示一个候选对是否被初步匹配。
        all_confidences (list): 浮点数列表，指示初步匹配的置信度。
        all_batches (list): 元组列表，每个元组是一个实体对 (se_l, se_r)。

    返回:
        list: 一个新的 all_answers 列表 (0 或 1)，反映了最大权匹配的结果，
              长度与输入的 all_answers 相同。
    """
    print('Begin to Conflict Resolution:')
    start_time = time.time()

    # 步骤 1: 筛选初始匹配的候选对，并存储其信息
    candidate_edges_info = []
    for i in range(len(all_batches)):
        if all_answers[i] == 1:
            se_l, se_r = all_batches[i]
            confidence = all_confidences[i]
            candidate_edges_info.append({
                'u': se_l,
                'v': se_r,
                'weight': confidence,
                'original_index': i
            })

    if not candidate_edges_info:
        return [0] * len(all_batches)

    # 步骤 2: 识别唯一的左侧实体 (u_nodes) 和右侧实体 (v_nodes)
    u_nodes_set = set()
    v_nodes_set = set()
    for edge_data in candidate_edges_info:
        u_nodes_set.add(edge_data['u'])
        v_nodes_set.add(edge_data['v'])

    u_nodes_list = list(u_nodes_set)
    v_nodes_list = list(v_nodes_set)

    if not u_nodes_list or not v_nodes_list:
        return [0] * len(all_batches)

    # 步骤 3: 创建从实体到其在列表中的索引的映射
    u_to_idx = {node: i for i, node in enumerate(u_nodes_list)}
    v_to_idx = {node: i for i, node in enumerate(v_nodes_list)}

    num_u = len(u_nodes_list)
    num_v = len(v_nodes_list)

    # 定义一个占位成本，用于表示不存在或不希望选择的边。
    # 这个值必须大于所有可能的真实成本(-置信度)。
    # 假设置信度在 [0,1] 范围内，则真实成本在 [-1,0] 范围内。
    # 因此，1.0 是一个合适的占位成本。
    PLACEHOLDER_COST = 1.0

    # 步骤 4: 构建成本矩阵
    # 初始化成本矩阵：所有成本设为占位成本 PLACEHOLDER_COST
    cost_matrix = np.full((num_u, num_v), PLACEHOLDER_COST)

    edge_original_indices_map = {}

    for edge_data in candidate_edges_info:
        u_idx = u_to_idx[edge_data['u']]
        v_idx = v_to_idx[edge_data['v']]
        current_cost = -edge_data['weight']  # 成本 = -置信度

        if current_cost < cost_matrix[u_idx, v_idx]:
            cost_matrix[u_idx, v_idx] = current_cost
            edge_original_indices_map[(u_idx, v_idx)] = edge_data['original_index']

    # 步骤 5: 应用匈牙利算法求解最小成本指派
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError as e:
        # 捕获潜在的错误，例如如果矩阵仍然因某种原因不可行
        # (虽然我们已经处理了 inf 的问题，但以防万一其他类型的无效值如 NaN 存在于置信度数据中)
        print(f"成本矩阵在调用 linear_sum_assignment 时出错: {e}")
        print("成本矩阵内容:")
        print(cost_matrix)
        # 在这种情况下，可能返回全0的结果或者抛出异常，取决于具体需求
        return [0] * len(all_batches)

    # 步骤 6: 根据最大权匹配的结果生成新的 all_answers
    new_all_answers = [0] * len(all_batches)

    for r_idx, c_idx in zip(row_ind, col_ind):
        # 检查这个指派的成本是否小于占位成本，以确保它是一个真实的匹配
        if cost_matrix[r_idx, c_idx] < PLACEHOLDER_COST:
            original_idx = edge_original_indices_map.get((r_idx, c_idx))
            if original_idx is not None:
                new_all_answers[original_idx] = 1

    end_time = time.time()

    run_time_minutes = end_time - start_time

    minutes, seconds, milliseconds = break_down_time(run_time_minutes)

    print('Successfully finish the Conflict Resolution!')
    print(f"The time of the Conflict Resolution: "
          f"{minutes}m {seconds}s {milliseconds}ms")
    print()

    # for i in range(len(all_batches)):
    #     if all_answers[i] == 1 and new_all_answers[i] == 0:
    #         pair = all_batches[i]
    #         se1, se2 = pair
    #         print(se1.display_properties())
    #         print(se2.display_properties())
    #         print('----------------------------------')
    #         print()

    return new_all_answers
