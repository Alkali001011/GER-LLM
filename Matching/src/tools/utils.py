import re

import numpy as np
import torch
import Levenshtein

from Blocking.src.tools.utils import remove_brackets


# 以下是针对PLM bert的分词优化函数：

# 中文地址预处理
def process_alphanumeric(text):
    """处理字母数字组合：去除字母和数字间的零，并转小写
    示例：
    process_alphanumeric("A07")
    'a7'
    process_alphanumeric("B00123C0")
    'b123c0'
    process_alphanumeric("XY0099Z")
    'xy99z'
    """
    if not isinstance(text, str):
        # 可以选择返回空字符串，或者根据业务逻辑处理
        # print(f"Warning: process_alphanumeric received non-string input: {text} of type {type(text)}")
        text = str(text)

    # 步骤1: 移除字母和非零数字之间的零
    pattern = r'([A-Za-z])0+([1-9])'
    while True:
        new_text = re.sub(pattern, r'\1\2', text, flags=re.IGNORECASE)
        if new_text == text:
            break
        text = new_text

    # 步骤2: 所有字母转小写
    text = text.lower()

    return text


# 序列化空间实体
def serialize_entity(entity):
    # 处理地址
    address = process_alphanumeric(entity.address)  # 去除字母和数字间的零，并转小写
    address = remove_brackets(address)  # 去除括号和括号内的内容
    address = normalize_address(address)  # 去除停用词

    # 处理分类
    category = normalize_category(entity.category)  # 统一分隔符为空格、删去stopwords(最后还存在空格的)
    category = category.replace(' ', '/')  # 层级化

    # 处理名称
    name = remove_brackets(entity.name)

    return f"[COL] name [VAL] {name} [COL] category [VAL] {category} [COL] address [VAL] {address}"


def semantic_based_sims(feas1: np.ndarray, feas2: np.ndarray, eps=1e-6) -> list:
    """
    计算两组特征向量之间的余弦相似度（显式归一化 + 数值容错处理）

    参数:
        feas1 (np.ndarray): 输入特征向量组1，形状 (n_samples, hidden_dim)
        feas2 (np.ndarray): 输入特征向量组2，形状需与feas1一致
        eps (float): 允许的浮点误差容限，默认1e-6

    返回:
        list: 余弦相似度列表，数值严格落在[-1, 1]区间内

    异常:
        ValueError: 输入数据非法或计算出现显著越界值时抛出
    """

    # ==================== 输入验证 ====================
    # 检查输入形状一致性（避免维度不匹配导致计算错误）
    assert feas1.shape == feas2.shape, f"形状不匹配: {feas1.shape} vs {feas2.shape}"

    # 检查NaN（非数值数据会导致计算结果不可控）
    assert not np.isnan(feas1).any() and not np.isnan(feas2).any(), "输入包含NaN"

    # 检查无穷大（极端值会导致数值不稳定）
    assert not np.isinf(feas1).any() and not np.isinf(feas2).any(), "输入包含Inf"

    # ==================== 数据转换与归一化 ====================
    # 将numpy数组转换为PyTorch张量（支持GPU加速）
    # 指定float32类型：兼顾精度和计算效率，避免int类型溢出
    embeddings1 = torch.as_tensor(feas1, dtype=torch.float32)
    embeddings2 = torch.as_tensor(feas2, dtype=torch.float32)

    # 显式L2归一化（关键步骤）
    # p=2: 计算L2范数，dim=1: 沿特征维度归一化
    # 公式: v_norm = v / sqrt(sum(v_i^2))，确保向量长度为1
    embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
    embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

    # ==================== 核心计算逻辑 ====================
    # 计算余弦相似度（等价于归一化后的向量点积）
    # 数学原理: cosθ = (A·B) / (||A||·||B||)
    # 因已显式归一化，此处简化为直接计算点积
    similarity = (embeddings1 * embeddings2).sum(dim=1)  # 输出形状 (n_samples,)

    # ==================== 结果合法性检查 ====================
    # 检测显著越界值（超出理论范围±1容限）
    # 设计逻辑：允许微小浮点误差，但拒绝明显异常
    over_max = similarity > (1.0 + eps)  # 上界越界掩码
    under_min = similarity < (-1.0 - eps)  # 下界越界掩码

    # 如果发现显著越界值（超出容限范围）
    if over_max.any() or under_min.any():
        # 获取越界样本的索引（便于后续调试）
        invalid_indices = torch.where(over_max | under_min)[0].tolist()
        raise ValueError(f"发现显著越界值，位置索引: {invalid_indices}")

    # ==================== 结果后处理 ====================
    # 强制约束结果范围（处理微小浮点误差）
    # 示例：计算值1.0000001会被修正为1.0
    return torch.clamp(similarity, min=-1.0, max=1.0).tolist()


def check_data(feas1, feas2):
    # 检查 NaN 和 Inf
    assert not np.isnan(feas1).any(), "feas1 包含 NaN"
    assert not np.isinf(feas1).any(), "feas1 包含 Inf"
    assert not np.isnan(feas2).any(), "feas2 包含 NaN"
    assert not np.isinf(feas2).any(), "feas2 包含 Inf"

    # 检查零向量
    norms1 = np.linalg.norm(feas1, axis=1)
    norms2 = np.linalg.norm(feas2, axis=1)
    assert (norms1 > 1e-6).all(), "feas1 存在零向量"
    assert (norms2 > 1e-6).all(), "feas2 存在零向量"


def normalize_category(category):
    """
    标准化处理 category 属性的文本：统一分隔符为空格、删去stopwords(最后还存在空格的)
    1. 统一分隔符并强制分割停用词
    2. 精准移除停用词
    """
    if not category:
        return ""

    # 1. 符号统一化为空格
    unified = re.sub(r'[/｜,，、＆&+＋]', ' ', category)

    # 2. 合并连续空格
    unified = re.sub(r'\s+', ' ', unified).strip()

    # 3. 定义完整停用词库
    stopwords = {
        '其他', '类', '型', '相关', '特色',
        '更多', '经营', '套餐', '量贩式', '连锁'
    }

    # 4. 在停用词周围插入分隔符
    for word in stopwords:
        unified = re.sub(rf'({word})', r' \1 ', unified)

    # 5. 分割并过滤
    tokens = [token for token in unified.split() if token not in stopwords]

    return ' '.join(tokens)


def lcs(a, b):
    """最长公共子序列动态规划实现（空间优化版）"""
    if not a or not b:
        return 0

    # 使用滚动数组降低空间复杂度
    prev_row = [0] * (len(b) + 1)
    curr_row = [0] * (len(b) + 1)

    for char_a in a:
        for j, char_b in enumerate(b, 1):
            if char_a == char_b:
                curr_row[j] = prev_row[j - 1] + 1
            else:
                curr_row[j] = max(prev_row[j], curr_row[j - 1])
        prev_row, curr_row = curr_row, [0] * (len(b) + 1)

    return prev_row[-1]


def bidirectional_edit_similarity(s1, s2):
    """双向编辑相似度（综合Jaro-Winkler和Levenshtein）"""
    return max(
        Levenshtein.jaro_winkler(s1, s2),
        Levenshtein.jaro_winkler(s2, s1),  # 双向计算解决顺序敏感问题
        Levenshtein.ratio(s1, s2)
    )


def ordered_jaccard_similarity(s1, s2):
    """词序敏感的加权Jaccard相似度"""
    # 将输入字符串按空格分割为词汇列表
    tokens1 = s1.split()  # 示例: "美食 汉堡" → ["美食", "汉堡"]
    tokens2 = s2.split()

    # 处理空值边界情况
    if not tokens1 and not tokens2:  # 两者均为空字符串
        return 1.0  # 定义为完全相似
    if not tokens1 or not tokens2:  # 任意一方为空，另一方非空
        return 0.0  # 定义为完全不相似

    # ===== 词序对齐度计算 =====
    # 统计相同位置上的相同词汇数量（精确位置匹配）
    # 示例: tokens1=["美食","汉堡"], tokens2=["汉堡","美食"] → matched_positions=0
    matched_positions = sum(t1 == t2 for t1, t2 in zip(tokens1, tokens2))

    # 计算词序惩罚因子（指数衰减模型）
    # 公式: 0.9^(总词数-匹配位置数)
    # 特性: 每存在一个不匹配位置，相似度衰减10%
    # 示例: tokens长度=2, matched_positions=0 → 0.9^(2-0)=0.81
    order_penalty = 0.9 ** (len(tokens1) - matched_positions)

    # ===== 词袋重叠度计算 =====
    # 将词汇列表转换为集合，计算Jaccard相似度（忽略顺序）
    set1, set2 = set(tokens1), set(tokens2)
    intersection = len(set1 & set2)  # 交集：共同存在的词汇数
    union = len(set1 | set2)  # 并集：所有唯一词汇总数

    # 计算基础Jaccard相似度（词袋重叠比例）
    # 公式: Jaccard = 交集大小 / 并集大小
    # 注意: 当并集为空时返回0.0（避免除零错误）
    jaccard = intersection / union if union else 0.0

    # ===== 综合相似度 =====
    # 最终相似度 = 词序惩罚因子 × 词袋相似度
    # 设计逻辑: 既奖励词汇重叠，又惩罚词序不一致
    return order_penalty * jaccard


# 计算category属性的相似度
def cal_similarities_between_categories(cat1, cat2):
    """
    1. 空值处理
    2. 统一分隔符为空格、删去停用词
    3. 在去空格前，计算词序敏感的加权Jaccard相似度
    4. 去除空格
    5. 从属关系特判
    6. 多维相似度计算
    """
    # 空值处理
    if not cat1 and not cat2:  # 如果都为空，则认为是匹配的
        return 1.0
    if not cat1 or not cat2:  # 如果只有一个为空，认为是不匹配的
        return 0.0

    # 输入标准化：统一分隔符为空格、删去stopwords(最后还存在空格的)
    norm1 = normalize_category(cat1)
    norm2 = normalize_category(cat2)

    # 在去除空格前，计算一下
    oj_sim = ordered_jaccard_similarity(norm1, norm2)

    # 去除空格
    norm1 = norm1.replace(" ", "")
    norm2 = norm2.replace(" ", "")

    # 从属关系特判
    if is_subset_or_contains(norm1, norm2):
        return 1.0

    # 多维度相似度计算
    similarities = [
        bidirectional_edit_similarity(norm1, norm2),
        oj_sim,
        lcs(norm1, norm2) / max(len(norm1), len(norm2))  # LCS归一化
    ]

    return max(similarities)


def is_subset_or_contains(str1, str2):
    # 判断子串关系（如 "ab" 是 "abc" 的子串）
    if str1 in str2 or str2 in str1:
        return True
    # 判断字符集合的子集关系（如字符种类包含）
    set1, set2 = set(str1), set(str2)
    return set1.issubset(set2) or set2.issubset(set1)


# 计算name属性的相似度
def cal_similarities_between_names(name_1, name_2):
    """
    1. 特判空值情况
    2. 去除括号内容
    3. 从属关系特判
    4. 多维相似度计算
    """
    if len(name_1) == 0 and len(name_2) == 0:  # 都为空，视为匹配
        return 1
    if len(name_1) == 0 or len(name_2) == 0:  # 单方为空，视为不匹配
        return 0

    # 去括号
    name_1 = remove_brackets(name_1)
    name_2 = remove_brackets(name_2)

    if is_subset_or_contains(name_1, name_2):  # 存在从属关系，视为匹配
        return 1

    # 多维度相似度计算
    similarities = [
        bidirectional_edit_similarity(name_1, name_2),
        lcs(name_1, name_2) / max(len(name_1), len(name_2))  # LCS归一化
    ]

    return max(similarities)


def normalize_address(addr):
    # 删除修饰词
    remove_words = ["对面", "旁", "附近", "近", "旁边", "斜对面", "出口",
                    "向北80米", "向南走约13米", "正后方", "斜对面",
                    "交叉口", "交界处", "入口", "出口", "通道口", "巷子内",
                    "直走", "旁边直走", "大门正后方", "向东", "向西", "向南", "向北",
                    "南京市", "南京", "江宁区", "江宁", "江苏省", "江苏", "开发区", "开发", "经济技术开发区"]
    for w in remove_words:
        addr = addr.replace(w, "")

    # 保留核心要素
    return addr.strip()


# 计算address属性的相似度
def cal_similarities_between_addresses(address_1, address_2):
    """
    1. 去括号
    2. 删除停用词
    3. 空值特判
    4. 从属关系特判
    5. 多维相似度计算
    """
    # 去括号
    address_1 = remove_brackets(address_1)
    address_2 = remove_brackets(address_2)

    # 标准化
    address_1 = normalize_address(address_1)
    address_2 = normalize_address(address_2)

    if len(address_1) == 0 and len(address_2) == 0:  # 都为空，视为匹配
        return 1
    if len(address_1) == 0 or len(address_2) == 0:  # 单方为空，视为不匹配
        return 0

    if is_subset_or_contains(address_1, address_2):  # 存在从属关系，视为匹配
        return 1

    # 多维度相似度计算
    similarities = [
        bidirectional_edit_similarity(address_1, address_2),
        lcs(address_1, address_2) / max(len(address_1), len(address_2))  # LCS归一化
    ]

    return max(similarities)


def analyze_features(feature_list):
    """
    输入: feature_list (list of lists) - 包含3658个5维向量的列表
    输出: 各维度的均值和方差的字典
    """
    # 转换为numpy数组
    features = np.array(feature_list)

    # 检查维度是否正确
    assert features.shape == (3658, 4), "输入特征维度不符合要求"

    # 按列计算均值和方差
    means = np.mean(features, axis=0)
    variances = np.var(features, axis=0)

    # 输出结果
    stats = {}
    for i in range(4):
        stats[f"feature_{i + 1}"] = {
            "mean": round(means[i], 4),
            "variance": round(variances[i], 4)
        }
    return stats

