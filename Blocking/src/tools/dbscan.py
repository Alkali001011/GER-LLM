import time

from Blocking.src.tools.utils import measurement_accuracy, haversine, break_down_time, load_pkl


# 寻找aoi的poi邻居
def find_neighbors_aoi(se_aoi, se_list, eps):
    neighbors = []
    for se in se_list:
        if haversine((se_aoi.longitude, se_aoi.latitude), (se.longitude, se.latitude)) < eps:
            neighbors.append(se)
    return neighbors


# 寻找poi的poi邻居
def find_neighbors_poi(se_poi, se_list, eps):
    se_list = [se for se in se_list if se.id != se_poi.id]  # 在se_list中去除当前se

    neighbors = []
    for se in se_list:
        if haversine((se_poi.longitude, se_poi.latitude), (se.longitude, se.latitude)) < eps:
            neighbors.append(se)
    return neighbors


def init_relationship_dict_between_aoi_poi(se_list, aoi_list):
    aoi2poi = {}
    poi2aoi = {}

    for se in se_list:
        poi2aoi[se.id] = -1  # -1记为当前poi没有被访问过

    for aoi in aoi_list:
        aoi2poi[aoi.id] = set()

    return poi2aoi, aoi2poi


def build_relationship_betweem_aoi_poi(se_list, aoi_list, eps, min_pts, gt_address, city):
    print('Start to detect the boundaries of the AOIs:')
    start_time = time.time()

    poi2aoi, aoi2poi = init_relationship_dict_between_aoi_poi(se_list, aoi_list)

    for se_aoi in aoi_list:
        neighbors = find_neighbors_aoi(se_aoi, se_list, eps)  # 计算当前aoi在其eps邻域内有多少个邻居点(poi)
        if len(neighbors) < min_pts:  # 如果邻居点数小于min_pts，说明这个aoi不是核心点，不是功能意义上的aoi，要予以修正
            for neighbor in neighbors:
                if poi2aoi[neighbor.id] == -1:  # 如果当前poi未被访问过
                    poi2aoi[neighbor.id] = [se_aoi.id]  # 记录它属于哪个aoi
                    aoi2poi[se_aoi.id].add(neighbor.id)  # 记录哪个aoi拥有该poi
        else:  # 如果当前aoi邻居点多于min_pts，则认为它是核心点
            i = 0
            while i < len(neighbors):  # 遍历邻居节点列表
                point = neighbors[i]  # 把一个邻居节点拿出来
                if poi2aoi[point.id] == -1:  # 如果它没有被访问过
                    poi2aoi[point.id] = [se_aoi.id]  # 记录它属于哪个aoi
                    aoi2poi[se_aoi.id].add(point.id)  # 记录哪个aoi拥有该poi

                    new_neighbors = find_neighbors_poi(point, se_list, eps)  # 去递归地找它的邻居节点(poi)
                    if len(new_neighbors) >= min_pts:  # 如果邻居节点＞min_pts，则认为它也是属于本aoi
                        neighbors += new_neighbors  # 将其邻居加入到邻居列表中，以进行进一步的扩展
                i = i + 1

    # 删除字典中的空键
    # 创建一个空列表用于存储空集合的键
    empty_keys = [aoi_id for aoi_id, poi_id_set in aoi2poi.items() if not poi_id_set]
    # 遍历empty_keys列表，删除这些键
    for key in empty_keys:
        del aoi2poi[key]
    # 将所有非空集合的值从set转换为list
    for aoi_id in aoi2poi:
        aoi2poi[aoi_id] = list(aoi2poi[aoi_id])

    poi2aoi = {poi_id: value for poi_id, value in poi2aoi.items() if value != -1}

    elapsed_time = time.time() - start_time
    # Convert time to minutes, seconds, and milliseconds
    minutes, seconds, milliseconds = break_down_time(elapsed_time)

    print(f'    The number of AOIs which has covered POIs: {len(aoi2poi)}')
    print(f'    The number of POIs covered by AOIs: {len(poi2aoi)}')
    if city == 'nj':
        print(f'    The number of Outlier POIs: {13004 - len(poi2aoi)}')
    elif city == 'pit':
        print(f'    The number of Outlier POIs: {4857 - len(poi2aoi)}')
    elif city == 'hz':
        print(f'    The number of Outlier POIs: {4941 - len(poi2aoi)}')
    else:
        raise ValueError(f"Unsupported city: '{city}'. Supported cities are 'nj', 'pit', 'hz'.")

    labels = load_pkl(gt_address)
    measurement_accuracy(aoi2poi, labels)

    print('End the process!')
    print(f"The time of algorithm: {minutes}m {seconds}s {milliseconds}ms")
    print('-------------------------------------------------------------------------------------------')

    return poi2aoi, aoi2poi
