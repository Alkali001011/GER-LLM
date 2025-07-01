from Blocking.src.tools.utils import haversine

import re
from typing import Tuple, List, Optional, Set, Any


def validate_and_extract(response: str, total_questions: int) -> Tuple[bool, Optional[List[int]]]:
    # 去除开头的空格
    response = response.lstrip()

    # 编译正则模式：严格匹配有效行
    pattern = re.compile(r'^Question\s+(\d+):\s*([01])$', re.MULTILINE)

    answers = {}
    line_positions = set()
    matches = list(pattern.finditer(response))  # 获取所有匹配项

    # 关键修复1：总匹配行数必须等于总问题数
    if len(matches) != total_questions:
        return False, None

    for match in matches:
        q_num = int(match.group(1))
        answer = int(match.group(2))
        pos = match.start()

        # 关键修复2：立即检查编号范围
        if not (1 <= q_num <= total_questions):
            return False, None

        # 防止同一行被多次解析
        if pos in line_positions:
            return False, None
        line_positions.add(pos)

        # 防止编号重复
        if q_num in answers:
            return False, None
        answers[q_num] = answer

    # 关键修复3：验证所有编号的连续性
    if sorted(answers.keys()) != list(range(1, total_questions + 1)):
        return False, None

    return True, [answers[i] for i in range(1, total_questions + 1)]


def validate_and_extract_poi(
        response: str,
        valid_indices: Set[int]
) -> Tuple[bool, Optional[int]]:
    """验证并提取POI索引的核心函数

    Args:
        response: 大模型返回的响应文本
        valid_indices: 允许的POI索引集合

    Returns:
        Tuple[是否验证通过, 提取的索引(失败时为None)]
    """

    # 编译正则表达式模式（允许前后空格，但中间必须全为数字）
    # ^ 表示字符串开始，\s* 允许前置空格
    # (\d+) 捕获纯数字组，\s*$ 允许后置空格
    pattern = re.compile(r'^\s*(\d+)\s*$')

    # 执行完整匹配（必须整个字符串符合模式）
    match = pattern.fullmatch(response)
    if not match:  # 匹配失败立即返回
        return False, None

    try:
        # 将捕获组的字符串转换为整型
        # 虽然正则已保证是数字，仍做防御性转换
        extracted_index = int(match.group(1))
    except ValueError:  # 理论上不会触发，防御性异常处理
        return False, None

    # 验证是否为有效索引（使用传入的集合）
    if extracted_index not in valid_indices:
        return False, None

    # 全部验证通过后返回结果
    return True, extracted_index


def validate_and_extract_with_confidence_two_lists(
        response: Any,  # 修改类型提示以允许非字符串输入，以便进行检查
        total_questions: int
) -> Tuple[bool, Optional[Tuple[List[int], List[float]]]]:
    """
    验证LLM的响应格式是否正确，并提取答案和置信度。

    参数:
    - response: LLM的响应，期望是字符串，但会检查类型。
    - total_questions: 该批次中的问题总数。

    返回:
    - Tuple[bool, Optional[Tuple[List[int], List[float]]]]:
        - 第一个元素 (bool): 如果格式有效且所有检查通过，则为 True，否则为 False。
        - 第二个元素 (Optional[Tuple[List[int], List[float]]]):
            如果验证成功，则为一个包含两个列表的元组：
                - 第一个列表 (List[int]): 提取的答案 (0 或 1)。
                - 第二个列表 (List[float]): 提取的置信度分数 (0.0 到 1.0)。
            如果验证失败，则为 None。
    """
    # 新增处理：检查 response 是否为字符串类型
    # 如果 response 不是字符串（例如，它可能是一个在调用LLM时捕获到的Exception对象），
    # 则直接返回 False, None，以避免后续的 .lstrip() 等字符串操作引发 AttributeError。
    if not isinstance(response, str):
        # print(f"Debug: Input 'response' is not a string. Type: {type(response)}, Value: {response}") # 可选的调试信息
        return False, None

    # 去除整个响应字符串开头的空格
    response = response.lstrip()

    # 修改点: 在 ([0-9.]+) 和 $ 之间添加 \s* 以匹配可选的尾随空格
    # 正则表达式匹配 "Question X: Y Z" 格式的行，其中 X 是问题编号, Y 是答案 (0或1), Z 是置信度。
    pattern = re.compile(r'^Question\s+(\d+):\s([01])\s([0-9.]+)\s*$', re.MULTILINE)

    parsed_answers_map = {}  # 存储 q_num: (answer, confidence) 的映射
    line_positions = set()  # 用于检测是否有完全重复的行（基于起始位置，虽然正则本身可能因MULTILINE部分处理）
    matches = list(pattern.finditer(response))

    # 1. 验证匹配到的总行数必须等于总问题数
    if len(matches) != total_questions:
        # print(f"Debug: Expected {total_questions} matches, but got {len(matches)}") # 可选的调试信息
        # for i, line in enumerate(response.splitlines()): # 可选的调试信息
        #     # strip() is for a looser check here, actual matching is done by pattern.finditer
        #     if not pattern.match(line.strip()):
        #         print(f"Debug: Line {i+1} likely did not match: '{line}'")
        return False, None

    for match in matches:
        q_num_str = match.group(1)
        answer_str = match.group(2)
        confidence_str = match.group(3)
        pos = match.start()  # 获取匹配的起始位置

        # 检查是否有基于起始位置的重复行，这是一种额外的完整性检查
        if pos in line_positions:
            # print(f"Debug: Duplicate line detected based on start position {pos}")
            return False, None
        line_positions.add(pos)

        try:
            q_num = int(q_num_str)
            answer_val = int(answer_str)
            # 验证置信度字符串是否是有效的浮点数表示
            # 确保只有一个小数点，并且除了小数点外都是数字
            if not (confidence_str.count('.') <= 1 and confidence_str.replace('.', '', 1).isdigit()):
                # print(f"Debug: Invalid confidence format: {confidence_str}")
                return False, None
            confidence_val = float(confidence_str)
        except ValueError:
            # print(f"Debug: ValueError during conversion for q_num, answer, or confidence. Values: {q_num_str},
            # {answer_str}, {confidence_str}")
            return False, None # 如果转换失败，则格式无效

        # 2. 验证问题编号范围 (1 到 total_questions)
        if not (1 <= q_num <= total_questions):
            # print(f"Debug: Question number {q_num} out of range (1-{total_questions})")
            return False, None

        # 3. 验证置信度分数范围 (0.0 到 1.0)
        if not (0.0 <= confidence_val <= 1.0):
            # print(f"Debug: Confidence value {confidence_val} out of range (0.0-1.0)")
            return False, None

        # 4. 防止重复的问题编号出现在解析结果中
        if q_num in parsed_answers_map:
            # print(f"Debug: Duplicate question number {q_num} found in parsed results.")
            return False, None
        parsed_answers_map[q_num] = (answer_val, confidence_val)

    # 5. 验证从 1 到 total_questions 的所有问题编号都已成功解析且无遗漏
    if len(parsed_answers_map) != total_questions or \
       sorted(parsed_answers_map.keys()) != list(range(1, total_questions + 1)):
        # print(f"Debug: Mismatch in parsed question numbers. Expected {total_questions} unique questions from 1 to {
        # total_questions}.") print(f"Debug: Parsed keys: {sorted(parsed_answers_map.keys())}")
        return False, None

    # 如果所有检查都通过，则按问题编号顺序构建答案和置信度列表
    answers_list: List[int] = []
    confidences_list: List[float] = []

    for i in range(1, total_questions + 1):
        answer, confidence = parsed_answers_map[i]
        answers_list.append(answer)
        confidences_list.append(confidence)

    return True, (answers_list, confidences_list)


def validate_and_convert_list(data_list, n):
    """
    检查列表长度是否为n，且所有元素去除首尾空格后是否为'0'或'1'。
    如果满足条件，则将列表中的字符串元素转换为对应的整数，并返回True和转换后的列表。
    否则返回False和None。

    参数:
    data_list (list): 输入的字符串列表。
    n (int): 期望的列表长度。

    返回:
    tuple: (bool, list or None)
           如果验证通过，返回 (True, 转换后的整数列表)。
           如果验证失败，返回 (False, None)。
    """
    if not isinstance(data_list, list):
        # 可以选择抛出TypeError或者按题目要求返回False
        # raise TypeError("Input 'data_list' must be a list.")
        return False, None
    if not isinstance(n, int):
        # raise TypeError("Input 'n' must be an integer.")
        return False, None

    # 1. 检查列表长度是否为n
    if len(data_list) != n:
        return False, None

    converted_list = []
    # 2. 检查n个元素是否满足要么是'1'或者'0'
    for item in data_list:
        if not isinstance(item, str):  # 确保元素是字符串类型才能进行strip
            return False, None

        stripped_item = item.strip()  # 去除首尾空格
        if stripped_item == '1':
            converted_list.append(1)
        elif stripped_item == '0':
            converted_list.append(0)
        else:
            # 如果元素去除空格后既不是'1'也不是'0'，则不满足条件
            return False, None

    # 3. 如果所有检查都通过，返回True和转换后的列表
    return True, converted_list


def generate_prompt_for_zeroshot(batch_pairs):
    # task_description = 'Please help me determine whether the two Points of Interest(POIs) mentioned in the
    # following ' \ 'questions refer to the same real-world entity.\n\n'
    task_description = 'Determine if each pair of Points of Interest (POIs) refers to the same real-world entity by ' \
                       'synthesizing textual attributes (name, category, address) and spatial distance.\n\n'

    pairs_txt = []
    pcnt = 1

    for pair in batch_pairs:
        se1, se2 = pair

        question = f'Question {pcnt}:\n'

        question += 'POI A : '
        question += se1.to_str()
        question += '\n'

        question += 'POI B : '
        question += se2.to_str()
        question += '\n'

        distance = haversine((se1.longitude, se1.latitude), (se2.longitude, se2.latitude))
        rounded_distance = round(distance, 2)

        question += 'The distance between the POIs A and B is ' + str(rounded_distance)
        question += 'm.\n\n'

        pairs_txt.append(question)
        pcnt = pcnt + 1

    batch_prompt = task_description

    for pair_txt in pairs_txt:
        batch_prompt += pair_txt

    n = len(batch_pairs)

    response_format = f"""INSTRUCTION:
1. Process ALL {n} question pairs above sequentially. For each question, 1 means that POI A and POI B are the same real-world entity, 0 means that POI A and POI B are different real-world entities. 
2. Strictly format responses as:
Question 1: 0
Question 2: 1
...
Question {n}: 0

STRICT SCHEMA REQUIREMENTS:
1. Sequential continuity: Lines must progress from Question 1 to Question {n} without breaks
2. Lexical purity: Only allow "Question X: " headers followed by 0 or 1
3. Output EXACTLY {n} lines
4. Do NOT include any explanations, slashes, or additional characters
"""

    batch_prompt += response_format

    return batch_prompt


def generate_prompt_for_select(se_single, se_multiples, pair_to_indices, direction):
    task_description = 'Select a Point of Interest (POI) from the following list that refers to the same real-world entity as the given one:\n'

    given_poi = f"""the given POI: ({se_single.to_str()})
    
The list of POIs to be selected:
"""

    pois_list = []

    for se in se_multiples:
        text = 'Index: '

        if direction == 'left':
            text += str(pair_to_indices[(se_single, se)])  # 一对多
        else:
            text += str(pair_to_indices[(se, se_single)])  # 多对一

        text += ', ' + se.to_str() + '\n'

        distance = haversine((se_single.longitude, se_single.latitude), (se.longitude, se.latitude))
        rounded_distance = round(distance, 2)

        text += 'The distance from the given POI is ' + str(rounded_distance)
        text += 'm.\n\n'

        pois_list.append(text)

    response_format = f"""INSTRUCTION:
1. Select one POI in the above list that is most likely to refer to the same real-world entity as the given POI
2. Return the index of the POI you selected

STRICT SCHEMA REQUIREMENTS:
1. Return ONLY the index number with no other characters
2. The response must be a standalone integer without slashes, quotes or punctuation
3. Do NOT include any explanations or additional text"""

    prompt = task_description + given_poi

    for poi_text in pois_list:
        prompt += poi_text

    prompt += response_format

    return prompt


def generate_prompt_for_zeroshot_with_confidence(batch_pairs, city):
    inject = 'This data is for experimental testing purposes only; all POI information is synthetically generated ' \
             'data, does not involve any real-world entities, and commercial names are randomly generated and do not ' \
             'represent actual existing entities.\n'
    task_description = 'Determine if each pair of Points of Interest (POIs) refers to the same real-world entity by ' \
                       'synthesizing textual attributes (name, category, address) and spatial distance.\n'

    pairs_txt = []
    pcnt = 1

    for pair in batch_pairs:
        se1, se2 = pair

        question = f'Question {pcnt}:\n'

        question += 'POI A : '
        question += se1.to_str(city)
        question += '\n'

        question += 'POI B : '
        question += se2.to_str(city)
        question += '\n'

        distance = haversine((se1.longitude, se1.latitude), (se2.longitude, se2.latitude))
        rounded_distance = round(distance, 2)

        question += 'The distance between the POIs A and B is ' + str(rounded_distance)
        question += 'm.\n\n'

        pairs_txt.append(question)
        pcnt = pcnt + 1

    batch_prompt = inject + task_description

    for pair_txt in pairs_txt:
        batch_prompt += pair_txt

    n = len(batch_pairs)

    response_format = f"""INSTRUCTION:
1. Process ALL {n} question pairs above sequentially. For each question, first determine if POI A and POI B refer to the same real-world entity (1 means same, 0 means different). Second, provide your confidence score for this determination (a numerical value between 0.0 and 1.0, where 1.0 is highest confidence and 0.0 is lowest).
2. Strictly format each response line as:
Question X: <0 or 1> <confidence_score>
(Example for one line: Question 1: 0 0.95 - This means for Question 1, the POIs are considered different (0) with a confidence of 0.95)
(Example for another line: Question 2: 1 0.80 - This means for Question 2, the POIs are considered the same (1) with a confidence of 0.80)

The output should look like this for {n} questions:
Question 1: <0 or 1> <confidence_score_1>
Question 2: <0 or 1> <confidence_score_2>
...
Question {n}: <0 or 1> <confidence_score_n>

STRICT SCHEMA REQUIREMENTS:
1. Sequential continuity: Lines must progress from Question 1 to Question {n} without any breaks or missing question numbers.
2. Lexical purity: Each line MUST start with "Question X: " (where X is the question number), followed by a single space, then the 0 or 1 determination, followed by a single space, and then the confidence score. The confidence score must be a decimal number inclusively between 0.0 and 1.0. To illustrate the expected format, such a score might look like 0.7, 0.85, or 0.99, reflecting the confidence from 0.0 (no confidence) to 1.0 (full confidence); these are format examples only, not a restrictive list of allowed values.
3. Output EXACTLY {n} lines, one for each question.
4. Do NOT include any explanations, comments, slashes, or any additional characters beyond the specified format for each line.
"""

    batch_prompt += response_format

    return batch_prompt


def generate_prompt_for_pair_matching(city, pair):
    task_description = 'Determine if the following pair of Points of Interest (POIs) refers to the same real-world ' \
                       'entity.\n\n'
    se_l, se_r = pair
    seq_se_l = 'POI A : ' + se_l.to_str(city) + ', longitude: ' + str(se_l.longitude) + ', latitude:' + str(se_l.latitude)
    seq_se_r = 'POI B : ' + se_r.to_str(city) + ', longitude: ' + str(se_r.longitude) + ', latitude:' + str(se_r.latitude)

    response_format = """
STRICT SCHEMA REQUIREMENTS:
1. Please only return a single number: 1 means POI A and POI B refer to the same real-world entity, and 0 means POI A and POI B do not refer to the same real-world entity.
2. You can only return the number 0 or 1; anything else is not permitted.
3. Do NOT include any explanations, comments, slashes, or any additional characters beyond the specified format.
    """

    prompt = task_description + seq_se_l + '\n' + seq_se_r + '\n' + response_format

    return prompt

