import argparse
import os
import sys
import time
from queue import Queue

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Blocking.src.tools.Quadtree import Quadtree, reassign_value2static_variable
from Blocking.src.tools.dbscan import build_relationship_betweem_aoi_poi
from Blocking.src.tools.utils import break_down_time, load_pkl, jaccard_similarity, compute_dist, \
    clear_directory, get_data_nj, get_aoi_data_nj, save_poi_id2se_nj, web_mercator_x, web_mercator_y, remove_brackets, \
    save2pkl, get_data_pit, get_aoi_data_pit, save_poi_id2se_pit, levenshtein_similarity_normalized, get_data_hz, \
    save_poi_id2se_hz


def aqsb(list_se, city):
    if city == 'nj':
        # 对于整个南京——最终的矩形边界：31.2, 32.63, 118.33, 119.303321
        # 对于江宁这一小块区域：31.928912246338754, 31.949171622741847, 118.78479874404546, 118.82612463290835
        upper_left_corner_x = web_mercator_x(118.78479874404546)
        upper_left_corner_y = web_mercator_y(31.949171622741847)
        lower_right_corner_x = web_mercator_x(118.82612463290835)
        lower_right_corner_y = web_mercator_y(31.928912246338754)
    elif city == 'pit':
        upper_left_corner_x = web_mercator_x(-80.1089065368723)
        upper_left_corner_y = web_mercator_y(40.50751849049542)
        lower_right_corner_x = web_mercator_x(-79.8481037)
        lower_right_corner_y = web_mercator_y(40.3509222)
    elif city == 'hz':
        upper_left_corner_x = web_mercator_x(120.239)
        upper_left_corner_y = web_mercator_y(30.2366)
        lower_right_corner_x = web_mercator_x(120.2694)
        lower_right_corner_y = web_mercator_y(30.2124)
    else:
        raise ValueError(f"Unsupported city: '{city}'. Supported cities are 'nj', 'pit', 'hz'.")

    width = lower_right_corner_x - upper_left_corner_x
    height = lower_right_corner_y - upper_left_corner_y

    print('Begin to Adaptive Quadtree with Soft Boundaries:')
    start_time = time.time()

    quadtree = Quadtree(0, upper_left_corner_x, upper_left_corner_y, width, height)

    # 将所有的空间实体一并插入四叉树块
    for se in list_se:
        quadtree.insert(se)

    quadtree_leaves = []

    q = Queue()

    q.put(quadtree)

    while not q.empty():
        item = q.get()

        diagonal = item.cal_diagonal()
        # density = item.cal_density()

        if item.level == 0:
            number_of_aoi = len(Quadtree.aoi2poi)
        else:
            number_of_aoi = item.cal_number_of_aoi()

        # --------------------停止条件-----------------------
        if diagonal <= 15 or number_of_aoi <= 2:
            quadtree_leaves.append(item)
            continue
        # --------------------停止条件-----------------------

        item.split()

        q.put(item.nodes[0])
        q.put(item.nodes[1])
        q.put(item.nodes[2])
        q.put(item.nodes[3])

    end_time = time.time()

    run_time_minutes = end_time - start_time

    minutes, seconds, milliseconds = break_down_time(run_time_minutes)

    print('Successfully finish the Adaptive Quadtree with Soft Boundaries!')
    print(f"The time of Adaptive Quadtree with Soft Boundaries: "
          f"{minutes}m {seconds}s {milliseconds}ms")
    print('-------------------------------------------------------------------------------------------')

    return quadtree_leaves


def calculate_quadflex_performance(blocks, ground_truth_address, id2se_address, save_address, city):
    print('Begin to calculate_performance:')
    start_time = time.time()

    set_gt = load_pkl(ground_truth_address)

    set_pairs = set()

    for quadtree in blocks:
        merged_set = quadtree.internal_se | quadtree.external_se
        # 根据belong属性将merged_set拆分为两个集合
        set_0 = {se for se in merged_set if se.belong == 0}
        set_1 = {se for se in merged_set if se.belong == 1}

        for se_0 in set_0:
            for se_1 in set_1:
                # ---------------------------------------------------------------------------------------------------
                if city == 'nj':
                    set_dp = set(se_0.name)

                    # 新增清洗步骤：处理mt_name
                    cleaned_mt = remove_brackets(se_1.name)
                    set_mt = set(cleaned_mt)
                    sim = jaccard_similarity(set_dp, set_mt)

                    if set_mt.issubset(set_dp) or set_dp.issubset(set_mt) or set_dp.issubset(set(se_1.name)) or \
                            set(se_1.name).issubset(set_dp):
                        sim = 1
                    if sim >= 0.225:
                        dist = compute_dist(se_0.latitude, se_0.longitude, se_1.latitude, se_1.longitude)
                        if dist <= 400:  # 351
                            set_pairs.add((se_0.id, se_1.id))
                elif city == 'pit':
                    sim = levenshtein_similarity_normalized(se_0.name, se_1.name)
                    set_dp = set(se_0.name)

                    # 新增清洗步骤：处理mt_name
                    cleaned_mt = remove_brackets(se_1.name)
                    set_mt = set(cleaned_mt)

                    if set_mt.issubset(set_dp) or set_dp.issubset(set_mt) or set_dp.issubset(set(se_1.name)) or \
                            set(se_1.name).issubset(set_dp):
                        sim = 1
                    if sim >= 0.4:
                        dist = compute_dist(se_0.latitude, se_0.longitude, se_1.latitude, se_1.longitude)
                        if dist <= 491:
                            set_pairs.add((se_0.id, se_1.id))
                elif city == 'hz':
                    cleaned_dp = remove_brackets(se_0.name)  # 高德数据集需要去括号
                    set_dp = set(cleaned_dp)
                    set_mt = set(se_1.name)

                    sim = levenshtein_similarity_normalized(cleaned_dp, se_1.name)

                    if set_mt.issubset(set_dp) or set_dp.issubset(set_mt) or set_dp.issubset(set(se_1.name)) or \
                            set(se_1.name).issubset(set_dp):
                        sim = 1

                    if sim >= 0.55:
                        dist = compute_dist(se_0.latitude, se_0.longitude, se_1.latitude, se_1.longitude)
                        if dist <= 350:
                            set_pairs.add((se_0.id, se_1.id))
                else:
                    raise ValueError(f"Unsupported city: '{city}'. Supported cities are 'nj', 'pit', 'hz'.")

    print('The number of pairs to determine whether they match: ' + str(len(set_pairs)))

    n2 = len(set_pairs.intersection(set_gt))
    print('    hit the gt: ' + str(n2))

    n1 = len(set_gt)

    pc = n2 / n1

    print(f'    PC: {pc:.3f}')

    n3 = len(set_pairs)

    pq = n2 / n3

    print(f'    PQ: {pq:.3f}')

    if city == 'nj':
        n4 = 12176 * 828
    elif city == 'pit':
        n4 = 2474 * 2383
    elif city == 'hz':
        n4 = 1982 * 2959
    else:
        raise ValueError(f"Unsupported city: '{city}'. Supported cities are 'nj', 'pit', 'hz'.")

    rr = 1 - (n3 / n4)

    print(f'    RR: {rr:.3f}')

    end_time = time.time()

    run_time_minutes = end_time - start_time

    minutes, seconds, milliseconds = break_down_time(run_time_minutes)

    print('Successfully finish the performance calculate!')
    print(f"The time of Calculation of the performance : "
          f"{minutes}m {seconds}s {milliseconds}ms")
    print('-------------------------------------------------------------------------------------------')
    print()

    poi_id2se = load_pkl(id2se_address)

    set_candidate_pairs = set()
    for pair in set_pairs:
        dp_id, mt_id = pair
        dp_se = poi_id2se[dp_id]
        mt_se = poi_id2se[mt_id]
        set_candidate_pairs.add((dp_se, mt_se))

    save2pkl(set_candidate_pairs, save_address)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city',
                        type=str,
                        required=True,
                        choices=['nj', 'pit', 'hz'],
                        help='The city to process. Supported choices: nj, pit, hz.')
    args = parser.parse_args()
    city = args.city

    current_script_path = os.path.abspath(__file__)  # Get the absolute path of the current script (main_blocking.py)

    # Go up three levels from the script path to get the path of the project root directory
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    dataset_base_subdir = 'data'
    blocking_base_subdir = 'Blocking'
    processed_data_blocking_subdir = os.path.join(blocking_base_subdir, 'processed_data')
    outputs_blocking_subdir = os.path.join(blocking_base_subdir, 'outputs')

    if city == 'nj':
        city_dataset_subdir = os.path.join(dataset_base_subdir, 'nj')
        processed_data_directory = os.path.join(base_path, processed_data_blocking_subdir, city)

        dp_poi_address = os.path.join(base_path, city_dataset_subdir, 'dp_poi_12176.csv')
        mt_poi_address = os.path.join(base_path, city_dataset_subdir, 'mt_poi_828.csv')
        aoi_address = os.path.join(base_path, city_dataset_subdir, 'aoi_180.csv')
        ground_truth_address = os.path.join(base_path, city_dataset_subdir, 'set_ground_truth_411.pkl')

        poi_id2se_address = os.path.join(processed_data_directory, f'{city}_poi_id2se.pkl')
        candidate_pairs_address = os.path.join(base_path, outputs_blocking_subdir, f'{city}_candidate_pairs.pkl')

    elif city == 'pit':
        city_dataset_subdir = os.path.join(dataset_base_subdir, 'pit')
        processed_data_directory = os.path.join(base_path, processed_data_blocking_subdir, city)

        dp_poi_address = os.path.join(base_path, city_dataset_subdir, 'osm_poi_2383.csv')
        mt_poi_address = os.path.join(base_path, city_dataset_subdir, 'fsq_poi_2474.csv')
        aoi_address = os.path.join(base_path, city_dataset_subdir, 'aoi_181.csv')
        ground_truth_address = os.path.join(base_path, city_dataset_subdir, 'set_ground_truth_1237.pkl')

        poi_id2se_address = os.path.join(processed_data_directory, f'{city}_poi_id2se.pkl')
        candidate_pairs_address = os.path.join(base_path, outputs_blocking_subdir, f'{city}_candidate_pairs.pkl')

    elif city == 'hz':
        city_dataset_subdir = os.path.join(dataset_base_subdir, 'hz')
        processed_data_directory = os.path.join(base_path, processed_data_blocking_subdir, city)

        dp_poi_address = os.path.join(base_path, city_dataset_subdir, 'gd_poi_1982.csv')
        mt_poi_address = os.path.join(base_path, city_dataset_subdir, 'dp_poi_2959.csv')
        aoi_address = os.path.join(base_path, city_dataset_subdir, 'aoi_107.csv')
        ground_truth_address = os.path.join(base_path, city_dataset_subdir, 'set_ground_truth_808.pkl')

        poi_id2se_address = os.path.join(processed_data_directory, f'{city}_poi_id2se.pkl')
        candidate_pairs_address = os.path.join(base_path, outputs_blocking_subdir, f'{city}_candidate_pairs.pkl')

    else:
        raise ValueError(f"Unsupported city: '{city}'. Supported cities are 'nj', 'pit', 'hz'.")

    # 0. prepare
    clear_directory(processed_data_directory)

    # 1. get poi data
    if city == 'nj':
        se_list = get_data_nj(dp_poi_address, mt_poi_address)
        aoi_list = get_aoi_data_nj(aoi_address)
    elif city == 'pit':
        se_list = get_data_pit(dp_poi_address, mt_poi_address)
        aoi_list = get_aoi_data_pit(aoi_address)
    elif city == 'hz':
        se_list = get_data_hz(dp_poi_address, mt_poi_address)
        aoi_list = get_aoi_data_nj(aoi_address)  # aois of nj&hz are from dp
    else:
        raise ValueError(f"Unsupported city: '{city}'. Supported cities are 'nj', 'pit', 'hz'.")

    # 3. map pois to ids
    if city == 'nj':
        save_poi_id2se_nj(dp_poi_address, mt_poi_address, poi_id2se_address)
    elif city == 'pit':
        save_poi_id2se_pit(dp_poi_address, mt_poi_address, poi_id2se_address)
    elif city == 'hz':
        save_poi_id2se_hz(dp_poi_address, mt_poi_address, poi_id2se_address)
    else:
        raise ValueError(f"Unsupported city: '{city}'. Supported cities are 'nj', 'pit', 'hz'.")

    # 4. detect the boundaries of the AOIs and get the relationship between the AOIs and POIs
    if city == 'nj':
        poi2aoi, aoi2poi = build_relationship_betweem_aoi_poi(se_list, aoi_list,
                                                              38.54, 5, ground_truth_address, city)
    elif city == 'pit':
        poi2aoi, aoi2poi = build_relationship_betweem_aoi_poi(se_list, aoi_list,
                                                              353.17, 5, ground_truth_address, city)
    elif city == 'hz':
        poi2aoi, aoi2poi = build_relationship_betweem_aoi_poi(se_list, aoi_list,
                                                              55.13, 5, ground_truth_address, city)
    else:
        raise ValueError(f"Unsupported city: '{city}'. Supported cities are 'nj', 'pit', 'hz'.")

    # 5. Reassign the two static variables of the Quadtree
    reassign_value2static_variable(aoi2poi, poi2aoi, poi_id2se_address)

    # 6. Adaptive Quadtree with Soft Boundaries
    leaves = aqsb(se_list, city)

    # 7. Evaluate algorithm and return the candidate pairs
    calculate_quadflex_performance(leaves, ground_truth_address, poi_id2se_address, candidate_pairs_address, city)
