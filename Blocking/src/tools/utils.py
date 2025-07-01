import math
import os
import pickle
import re
import shutil

import Levenshtein
import numpy as np
from math import radians, sin, cos, atan2, sqrt

import pandas as pd

from Blocking.src.tools.SE import SE


def load_pkl(filename):
    with open(filename, 'rb') as file:  # 'rb' 表示“二进制读取”模式
        data = pickle.load(file)
    return data


def save2pkl(dat, path):
    with open(path, mode='wb') as f:
        pickle.dump(dat, f)


def haversine(lonlat1, lonlat2):
    """
    Calculate the great circle distance between two points on the earth, specified in decimal degrees.
    """
    lon1, lat1, lon2, lat2 = map(radians, [lonlat1[0], lonlat1[1], lonlat2[0], lonlat2[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Earth's average radius in meters
    return c * r


def invert_aoi_poi_mapping(aoi_to_poi):
    poi_to_aoi = {}
    for aoi_id, poi_list in aoi_to_poi.items():
        for poi_id in poi_list:
            if poi_id in poi_to_aoi:
                poi_to_aoi[poi_id].append(aoi_id)
            else:
                poi_to_aoi[poi_id] = [aoi_id]
    return poi_to_aoi


def web_mercator_x(longitude):
    return longitude * 20037508.342789 / 180


def web_mercator_y(latitude):
    return (math.log(math.tan((90 + latitude) * math.pi / 360)) / (math.pi / 180)) * 20037508.34789 / 180 * (-1)


def clear_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # 删除文件或链接
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 删除子目录及其所有内容
        except Exception as e:
            print(f'删除 {item_path} 时发生错误。错误信息: {e}')


def break_down_time(time):
    # input: 秒
    # output: 分钟，秒，毫秒
    minutes = time // 60
    seconds = time % 60
    milliseconds = round((time - int(time)) * 1000)
    return int(minutes), int(seconds), int(milliseconds)


def jaccard_similarity(set1, set2):
    # 计算交集
    intersection = set1.intersection(set2)
    # 计算并集
    union = set1.union(set2)
    # 计算杰卡德相似度
    jaccard_sim = len(intersection) / len(union)
    return jaccard_sim


def compute_dist(lat1, lon1, lat2, lon2):
    r = 6373.0

    try:
        float(lat1)
        float(lon1)
        float(lat2)
        float(lon2)
    except ValueError:
        return -1

    # 角度制转弧度制
    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))

    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    dist = round(r * c * 1000)  # 经纬度距离（米）

    return dist


def save_poi_id2se_nj(address_1, address_2, save_address):
    poi_id2se = {}

    df_dianping = pd.read_csv(address_1)

    for index, row in df_dianping.iterrows():
        se = SE(row['shop_id'], row['name'], row['address'], row['big_category'] + ' ' + row['small_category'],
                row['longitude'], row['latitude'], 0)
        poi_id2se[row['shop_id']] = se

    df_meituan = pd.read_csv(address_2)

    for index, row in df_meituan.iterrows():
        se = SE(row['id'], row['name'], row['address'], row['category'], row['lng'], row['lat'], 1)
        poi_id2se[row['id']] = se

    save2pkl(poi_id2se, save_address)


def save_poi_id2se_hz(address_1, address_2, save_address):
    poi_id2se = {}

    df_dianping = pd.read_csv(address_1)

    for index, row in df_dianping.iterrows():
        se = SE(row['id'], row['name'], row['address'], f"{row['big_category']} {row['medium_category']} {row['small_category']}",
                row['marlon'], row['marlat'], 0)
        poi_id2se[row['id']] = se
    # Index(['id', 'name', 'dtype', 'typecode', 'address', 'tel', 'adname', 'marlon',
    #        'marlat', 'wgs84lon', 'wgs84lat', 'big_category', 'medium_category',
    #        'small_category'],
    #       dtype='object')

    df_meituan = pd.read_csv(address_2)

    for index, row in df_meituan.iterrows():
        se = SE(row['shop_id'], row['name'], row['address'], f"{row['big_category']} {row['small_category']}",
                row['longitude'], row['latitude'], 1)
        poi_id2se[row['shop_id']] = se
    # Index(['shop_id', 'name', 'branchname', 'alt_name', 'phone', 'regionname',
    #        'address', 'cross_road', 'big_category', 'small_category', 'longitude',
    #        'latitude'],
    #       dtype='object')

    save2pkl(poi_id2se, save_address)


def save_poi_id2se_pit(address_1, address_2, save_address):
    poi_id2se = {}

    df_dianping = pd.read_csv(address_1)

    # Index(['index', 'name', 'latitude', 'longitude', 'address', 'postalcode'], dtype='object')

    for index, row in df_dianping.iterrows():
        se = SE(row['index'], row['name'], row['address'], "", row['longitude'], row['latitude'], 0)
        poi_id2se[row['index']] = se

    df_meituan = pd.read_csv(address_2)

    # Index(['index', 'name', 'latitude', 'longitude', 'address', 'postalcode'], dtype='object')

    for index, row in df_meituan.iterrows():
        se = SE(row['index'], row['name'], row['address'], "", row['longitude'], row['latitude'], 1)
        poi_id2se[row['index']] = se

    save2pkl(poi_id2se, save_address)


def get_data_nj(address_1, address_2):
    list_se = []

    # 读取大众点评的POI数据
    df_dianping = pd.read_csv(address_1)
    list_se.extend(df_dianping.apply(
        lambda row: SE(row['shop_id'], row['name'], row['address'], f"{row['big_category']} {row['small_category']}",
                       row['longitude'], row['latitude'], 0), axis=1))

    # 读取美团的POI数据
    df_meituan = pd.read_csv(address_2)
    list_se.extend(df_meituan.apply(
        lambda row: SE(row['id'], row['name'], row['address'], row['category'], row['lng'], row['lat'], 1), axis=1))

    return list_se


def get_data_hz(address_1, address_2):
    list_se = []

    # 读取高德地图的POI数据
    df_dianping = pd.read_csv(address_1)
    list_se.extend(df_dianping.apply(
        lambda row: SE(row['id'], row['name'], row['address'], f"{row['big_category']} {row['medium_category']} {row['small_category']}",
                       row['marlon'], row['marlat'], 0), axis=1))
    # Index(['id', 'name', 'dtype', 'typecode', 'address', 'tel', 'adname', 'marlon',
    #        'marlat', 'wgs84lon', 'wgs84lat', 'big_category', 'medium_category',
    #        'small_category'],
    #       dtype='object')

    # 读取dp的POI数据
    df_meituan = pd.read_csv(address_2)
    list_se.extend(df_meituan.apply(
        lambda row: SE(row['shop_id'], row['name'], row['address'], f"{row['big_category']} {row['small_category']}",
                       row['longitude'], row['latitude'], 1), axis=1))
    # Index(['shop_id', 'name', 'branchname', 'alt_name', 'phone', 'regionname',
    #        'address', 'cross_road', 'big_category', 'small_category', 'longitude',
    #        'latitude'],
    #       dtype='object')

    return list_se


def get_data_pit(address_1, address_2):
    list_se = []

    # 读取大众点评的POI数据
    df_dianping = pd.read_csv(address_1)
    list_se.extend(df_dianping.apply(
        lambda row: SE(row['index'], row['name'], row['address'], "", row['longitude'], row['latitude'], 0), axis=1))
    # Index(['index', 'name', 'latitude', 'longitude', 'address', 'postalcode'], dtype='object')

    # 读取美团的POI数据
    df_meituan = pd.read_csv(address_2)
    list_se.extend(df_meituan.apply(
        lambda row: SE(row['index'], row['name'], row['address'], "", row['longitude'], row['latitude'], 1), axis=1))
    # Index(['index', 'name', 'latitude', 'longitude', 'address', 'postalcode'], dtype='object')

    return list_se


def get_aoi_data_nj(aoi_address):
    list_aoi = []

    df_aoi = pd.read_csv(aoi_address)
    list_aoi.extend(df_aoi.apply(
        lambda row: SE(row['shop_id'], row['name'], row['address'], f"{row['big_category']} {row['small_category']}",
                       row['longitude'], row['latitude'], 2), axis=1))
    return list_aoi


def get_aoi_data_pit(aoi_address):
    list_aoi = []

    df_aoi = pd.read_csv(aoi_address)
    list_aoi.extend(df_aoi.apply(
        lambda row: SE(row['index'], row['name'], row['address'], "", row['longitude'], row['latitude'], 2), axis=1))
    return list_aoi


def measurement_accuracy(data, labels):
    count = 0

    for pair in labels:
        dp_id, mt_id = pair
        for key, value in data.items():
            if dp_id in value and mt_id in value:
                count = count + 1
                break

    print("    The actual matching entity pairs contained in AOIs: " + str(count))


def remove_brackets(text):
    """精准清洗函数：仅删除括号及其内容，保留其他所有字符"""
    text = str(text)
    # 递归删除所有类型的括号及其内容（支持嵌套）
    while True:
        cleaned = re.sub(
            r'[（(][^()（）]*?[)）]',  # 匹配最内层括号对
            '',
            text,
            flags=re.UNICODE
        )
        if cleaned == text:
            break
        text = cleaned
    return text


def levenshtein_similarity_normalized(s1, s2):
    """
    计算两个字符串之间归一化的列文斯坦相似度。
    结果在0和1之间，1表示字符串完全相同。
    """
    s1_str = str(s1) if pd.notna(s1) else ""
    s2_str = str(s2) if pd.notna(s2) else ""

    # 在计算距离和长度时，统一转为小写，以忽略大小写差异
    s1_lower = s1_str.lower()
    s2_lower = s2_str.lower()

    if not s1_lower and not s2_lower:  # 如果两个字符串处理后都为空
        return 1.0
    if not s1_lower or not s2_lower:  # 如果其中一个字符串处理后为空
        return 0.0

    distance = Levenshtein.distance(s1_lower, s2_lower)
    max_len = max(len(s1_lower), len(s2_lower))

    if max_len == 0:  # 以防万一，但理论上已被前面的空字符串检查覆盖
        return 1.0

    similarity = 1 - (distance / max_len)
    return similarity
