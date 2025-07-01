import os
import time
from collections import defaultdict
import asyncio

from openai import OpenAI
from typing import Optional, Dict, Any

from Blocking.src.tools.utils import break_down_time
from Matching.src.Batch_Prompting import generate_prompt_for_select, validate_and_extract_poi, generate_prompt_for_zeroshot_with_confidence, \
    validate_and_extract_with_confidence_two_lists
from ParallelLLM.src.pllm.client import Client


async def process_tasks(tasks):
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


# use the llm by siliconflow without reasoning (deepseek-v3)
def execute_full_llm_chat_by_siliconflow(
        batch_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Optional[Dict[str, Any]]
) -> str:
    """
    执行流式大模型对话（硅流平台DeepSeek-V3版本）

    参数：
    - batch_prompt: 用户输入的提示文本
    - max_retries: API调用最大重试次数（默认3）
    - retry_delay: 重试等待时间（秒，默认1.0）
    - temperature: 生成温度（0~2，默认0.7）
    - max_tokens: 最大生成token数（默认2048）
    - **kwargs: 其他可选API参数（如top_p, stop等）

    返回：
    - 完整生成内容（字符串）
    """
    # 初始化客户端
    client = OpenAI(
        api_key="sk-rtydrhhztctmnvcfuuwtlfmoqwcroxovszakvoktuirciefk",
        base_url="https://api.siliconflow.cn/v1"
    )

    content = ""  # 存储完整生成内容
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # 创建流式请求
            completion = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": batch_prompt}],
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # 处理流式响应
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    chunk_content = delta.content or ""  # 处理None值

                    # 实时输出（可选调试）
                    # print(chunk_content, end='', flush=True)
                    content += chunk_content

            # 成功时跳出重试循环
            break

        except Exception as e:
            print(f"\nAPI调用失败: {str(e)}")
            retry_count += 1

            if retry_count <= max_retries:
                print(f"{retry_delay}秒后重试({retry_count}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"超过最大重试次数({max_retries})") from e

    return content.strip()  # 去除首尾空白


# use the llm by aliyun with reasoning (QwQ-32B)
def execute_qwq32b_chat(
        batch_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 32768,
        **kwargs: Optional[Dict[str, Any]]
) -> str:
    """
    参数：
    - batch_prompt: 用户输入的提示文本
    - max_retries: API调用最大重试次数（默认3）
    - retry_delay: 重试等待时间（秒，默认1.0）
    - temperature: 生成温度（0~2，默认0.7）
    - max_tokens: 最大生成token数（默认32768）
    - **kwargs: 其他可选API参数（如top_p, stop等）
    """
    # 初始化OpenAI客户端（针对阿里云平台配置）
    client = OpenAI(
        api_key="sk-f6e9eedcc35b498daa5160179c669e38",
        # 阿里云QWen模型专用接入点
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 设置阿里云DashScope的兼容端点
    )

    content = ""  # 初始化内容收集容器
    retry_count = 0  # 重试计数器

    # 重试控制循环（最多尝试 max_retries+1 次）
    while retry_count <= max_retries:
        try:
            # 创建流式聊天补全请求
            completion = client.chat.completions.create(
                model="qwq-32b",  # 指定QWen-32B模型
                messages=[{
                    "role": "user",  # 单轮对话模式
                    "content": batch_prompt  # 用户输入的提示
                }],
                stream=True,  # 启用流式传输
                temperature=temperature,  # 控制生成随机性（0-2）
                max_tokens=max_tokens,  # 限制生成最大长度
                **kwargs  # 透传其他API参数（如top_p, stop等）
            )

            # 流式响应处理
            for chunk in completion:
                # 检查是否包含有效数据块
                if chunk.choices:
                    delta = chunk.choices[0].delta  # 获取增量内容
                    # 安全获取content属性（避免None报错）
                    chunk_content = getattr(delta, 'content', '') or ""
                    content += chunk_content  # 累积最终响应

            # 成功时直接返回（跳出循环）
            return content.strip()  # 去除首尾空白字符

        # 异常处理模块
        except Exception as e:
            print(f"API调用失败: {str(e)}")  # 打印错误日志
            retry_count += 1  # 增加重试计数器

            # 判断是否继续重试
            if retry_count <= max_retries:
                print(f"{retry_delay}秒后重试({retry_count}/{max_retries})...")
                time.sleep(retry_delay)  # 等待重试延迟
            else:
                # 超过重试次数后抛出包装异常
                raise RuntimeError(f"超过最大重试次数({max_retries})") from e

    # 此处为安全返回（实际在正常流程中不会执行到这里）
    return content.strip()


# use the llm by siliconflow with reasoning (deepseek-r1)
def execute_deepseek_r1_chat(
        batch_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 16384,
        **kwargs: Optional[Dict[str, Any]]
) -> str:
    """
    参数：
    - batch_prompt: 用户输入的提示文本
    - max_retries: API调用最大重试次数（默认3）
    - retry_delay: 重试等待时间（秒，默认1.0）
    - temperature: 生成温度（0~2，默认0.7）
    - max_tokens: 最大生成token数（默认16384）
    - **kwargs: 其他可选API参数（如top_p, stop等）
    """
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key="sk-rtydrhhztctmnvcfuuwtlfmoqwcroxovszakvoktuirciefk",
        base_url="https://api.siliconflow.cn/v1"
    )

    content = ""  # 初始化内容收集容器
    retry_count = 0  # 重试计数器

    # 重试控制循环（最多尝试 max_retries+1 次）
    while retry_count <= max_retries:
        try:
            # 创建流式聊天补全请求
            completion = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",  # 指定DeepSeek-R1模型
                messages=[{
                    "role": "user",  # 单轮对话模式
                    "content": batch_prompt  # 用户输入的提示
                }],
                stream=True,  # 启用流式传输
                temperature=temperature,  # 控制生成随机性（0-2）
                max_tokens=max_tokens,  # 限制生成最大长度
                **kwargs  # 透传其他API参数（如top_p, stop等）
            )

            # 流式响应处理
            for chunk in completion:
                # 检查是否包含有效数据块
                if chunk.choices:
                    delta = chunk.choices[0].delta  # 获取增量内容
                    # 安全获取content属性（避免None报错）
                    chunk_content = getattr(delta, 'content', '') or ""
                    content += chunk_content  # 累积最终响应

            # 成功时直接返回（跳出循环）
            return content.strip()  # 去除首尾空白字符

        # 异常处理模块
        except Exception as e:
            print(f"API调用失败: {str(e)}")  # 打印错误日志
            retry_count += 1  # 增加重试计数器

            # 判断是否继续重试
            if retry_count <= max_retries:
                print(f"{retry_delay}秒后重试({retry_count}/{max_retries})...")
                time.sleep(retry_delay)  # 等待重试延迟
            else:
                # 超过重试次数后抛出包装异常
                raise RuntimeError(f"超过最大重试次数({max_retries})") from e

    # 此处为安全返回（实际在正常流程中不会执行到这里）
    return content.strip()


# 处理siliconflow处理不了的api请求，且api有速率限制
def generate_prompts_and_interact_with_llm_limit_rate(batches, llm, city):
    # 选择执行器
    if llm == "gemini-2.0-flash":
        executor = execute_full_llm_chat_by_closeai_gemini
        max_calls_per_minute = 4
    elif llm == "gemini-2.5-flash-preview-04-17":
        executor = execute_full_llm_chat_by_closeai_gemini_2_5
        max_calls_per_minute = 4

    sleep_time = 60 / max_calls_per_minute
    print('sleep_time:' + str(sleep_time))

    print('Begin to match with the LLM:')
    start_time = time.time()

    all_answers = []
    all_confidence = []

    count = 1
    for batch in batches:
        print(f'Processing the batch {count}.')
        # prompt = generate_prompt_for_zeroshot(batch)
        prompt = generate_prompt_for_zeroshot_with_confidence(batch, city)

        # 与大模型交互，找到答案
        response = executor(prompt)
        time.sleep(sleep_time)  # 让与大模型的交互休眠

        flag, tup = validate_and_extract_with_confidence_two_lists(response, len(batch))  # 判断大模型的答案是否合法

        while not flag:  # 如果一直不合法
            # 让大模型重新生成回复
            response = executor(prompt)
            time.sleep(sleep_time)  # 让与大模型的交互休眠

            flag, tup = validate_and_extract_with_confidence_two_lists(response, len(batch))  # 继续判断大模型的答案是否合法

        answers, confidences = tup
        all_answers.extend(answers)
        all_confidence.extend(confidences)

        count = count + 1

    end_time = time.time()

    run_time_minutes = end_time - start_time

    minutes, seconds, milliseconds = break_down_time(run_time_minutes)

    print('Successfully finish the match with LLM!')
    print(f"The time of the match with LLM: "
          f"{minutes}m {seconds}s {milliseconds}ms")
    print()

    return all_answers, all_confidence


# 处理chatgpt系列模型
def generate_prompts_and_interact_with_llm_chatgpt(batches, llm, city):
    # 选择执行器
    if llm == "gpt-4.1-mini":
        executor = execute_full_llm_chat_by_api2gpt_chat
    elif llm == "o4-mini":
        executor = execute_full_llm_chat_by_api2gpt_chat_reasoning

    print('Begin to match with the LLM:')
    start_time = time.time()

    all_answers = []
    all_confidence = []

    count = 1
    for batch in batches:
        print(f'Processing the batch {count}.')
        prompt = generate_prompt_for_zeroshot_with_confidence(batch, city)

        # 与大模型交互，找到答案
        response = executor(prompt)

        flag, tup = validate_and_extract_with_confidence_two_lists(response, len(batch))  # 判断大模型的答案是否合法

        while not flag:  # 如果一直不合法
            # 让大模型重新生成回复
            response = executor(prompt)

            flag, tup = validate_and_extract_with_confidence_two_lists(response, len(batch))  # 继续判断大模型的答案是否合法

        answers, confidences = tup
        all_answers.extend(answers)
        all_confidence.extend(confidences)

        count = count + 1

    end_time = time.time()

    run_time_minutes = end_time - start_time

    minutes, seconds, milliseconds = break_down_time(run_time_minutes)

    print('Successfully finish the match with LLM!')
    print(f"The time of the match with LLM: "
          f"{minutes}m {seconds}s {milliseconds}ms")
    print()

    return all_answers, all_confidence


# 冲突消解部分与大模型的交互
def get_response_from_llm(llm_queries, pair_to_indices, llm='llava'):
    # 选择执行器
    if llm == "deepseek-v3":
        executor = execute_full_llm_chat_by_siliconflow
    elif llm == "QwQ-32B":
        executor = execute_qwq32b_chat
    elif llm == "deepseek-r1":
        executor = execute_deepseek_r1_chat

    # 预期格式：[{"conflict_id": "123", "selected_index": 0}, ...]
    llm_responses = []
    count = 1

    for query in llm_queries:
        candidate_pairs = query["candidate_pairs"]
        valid_indices = set()
        for pair in candidate_pairs:
            valid_indices.add(pair_to_indices[pair])

        response = executor(query["prompt"])

        flag, answer = validate_and_extract_poi(response, valid_indices)  # 判断大模型的答案是否合法

        while not flag:  # 如果一直不合法
            # 让大模型重新生成回复
            response = executor(query["prompt"])

            flag, answer = validate_and_extract_poi(response, valid_indices)  # 继续判断大模型的答案是否合法

        llm_responses.append({
            "conflict_id": query["conflict_id"],  # 冲突ID
            "selected_index": answer
        })
        print('Has been processed: ' + str(count))
        count = count + 1

    return llm_responses


# 冲突消解
def conflict_resolution(batches, all_answers, llm='llava'):
    print('Begin to conflict resolution with the LLM:')
    start_time = time.time()

    # 展平所有批次数据：将嵌套的batch结构转换为平面列表 ---------------------------
    all_batches = []
    for batch in batches:
        all_batches.extend(batch)  # 合并所有batch中的候选对
    # 创建候选对到答案索引的映射字典（用于快速定位）
    # 格式：{候选对元组: 在all_answers中的索引}
    pair_to_indices = {pair: i for i, pair in enumerate(all_batches)}

    # 阶段1：处理左侧冲突（大众点评ID一对多美团ID）------------------------------
    # 步骤1.1 收集左侧冲突（基于原始答案）
    left_conflicts = defaultdict(list)  # key: 大众点评ID, value: 匹配的候选对列表
    for pair, answer in zip(all_batches, all_answers):
        if answer == 1:  # 只处理原始标记为匹配的候选对
            left_id = pair[0].id  # 获取左侧实体ID（大众点评）
            left_conflicts[left_id].append(pair)  # 记录冲突

    # 步骤1.2 生成左侧冲突组（筛选出实际存在冲突的）
    conflict_groups_left = [
        {
            "type": "left",  # 冲突类型标识
            "conflict_id": k,  # 引发冲突的大众点评ID
            "candidate_pairs": v  # 该ID对应的所有候选对
        } for k, v in left_conflicts.items() if len(v) > 1  # 仅保留多个匹配的情况
    ]

    # 步骤1.3 准备左侧冲突的LLM查询
    llm_queries_left = []
    for group in conflict_groups_left:
        # 获取冲突的候选实体对列表
        candidate_pairs = group["candidate_pairs"]
        # 获取“一”的实体
        se_single = candidate_pairs[0][0]
        # 获取“多”的实体
        se_multiples = []
        for pair in candidate_pairs:
            se_multiples.append(pair[1])
        # 生成prompt
        prompt = generate_prompt_for_select(se_single, se_multiples, pair_to_indices, 'left')

        llm_queries_left.append({
            "conflict_id": group["conflict_id"],  # 冲突ID（大众点评ID）
            "candidate_pairs": group["candidate_pairs"],  # 候选对列表
            "prompt": prompt  # 交给的大模型的提示
        })

    # 步骤1.4 获取LLM响应
    print('------------------------------------------')
    print('Starting to resolve one-to-many conflicts:')
    print('The number of one-to-many conflicts:' + str(len(llm_queries_left)))
    llm_responses_left = get_response_from_llm(llm_queries_left, pair_to_indices, llm)
    print('------------------------------------------')
    print()

    # 步骤1.5 根据LLM响应修正左侧冲突答案
    for response in llm_responses_left:
        # 解析响应数据
        conflict_id = response["conflict_id"]
        selected_answer_index = response["selected_index"]  # 改名为更清晰的变量名

        # 获取该冲突组对应的所有候选对
        candidate_pairs = next(
            g["candidate_pairs"] for g in conflict_groups_left
            if g["conflict_id"] == conflict_id
        )

        # 获取这些候选对在all_answers中的全部索引
        all_indices = {pair_to_indices[p] for p in candidate_pairs}  # 改用集合提升查询效率

        # 重置所有相关答案为0
        for i in all_indices:
            all_answers[i] = 0
        # 设置选定答案
        all_answers[selected_answer_index] = 1

    # 阶段2：处理右侧冲突（美团ID一对多大众点评ID）------------------------------
    # 注意：此时all_answers已包含左侧冲突的修正结果

    # 步骤2.1 重新收集右侧冲突（基于更新后的答案）
    right_conflicts = defaultdict(list)  # key: 美团ID, value: 匹配的候选对列表
    for pair, answer in zip(all_batches, all_answers):
        if answer == 1:  # 使用修正后的答案
            right_id = pair[1].id  # 获取右侧实体ID（美团）
            right_conflicts[right_id].append(pair)

    # 步骤2.2 生成右侧冲突组
    conflict_groups_right = [
        {
            "type": "right",  # 冲突类型标识
            "conflict_id": k,  # 引发冲突的美团ID
            "candidate_pairs": v  # 该ID对应的所有候选对
        } for k, v in right_conflicts.items() if len(v) > 1  # 仅保留多个匹配的情况
    ]

    # 步骤2.3 准备右侧冲突的LLM查询
    llm_queries_right = []
    for group in conflict_groups_right:
        # 获取冲突的候选实体对列表
        candidate_pairs = group["candidate_pairs"]
        # 获取“一”的实体
        se_single = candidate_pairs[0][1]  # 因为是右侧冲突，所以得拿右边
        # 获取“多”的实体
        se_multiples = []
        for pair in candidate_pairs:
            se_multiples.append(pair[0])  # 这时候就要拿左边的实体
        # 生成prompt
        prompt = generate_prompt_for_select(se_single, se_multiples, pair_to_indices, 'right')

        llm_queries_right.append({
            "conflict_id": group["conflict_id"],  # 冲突ID（美团ID）
            "candidate_pairs": group["candidate_pairs"],  # 候选对列表
            "prompt": prompt
        })

    # 步骤2.4 获取LLM响应
    print('------------------------------------------')
    print('Starting to resolve many-to-one conflicts:')
    print('The number of many-to-one conflicts:' + str(len(llm_queries_right)))
    llm_responses_right = get_response_from_llm(llm_queries_right, pair_to_indices, llm)
    print('------------------------------------------')
    print()

    # 步骤2.5 根据LLM响应修正右侧冲突答案
    for response in llm_responses_right:
        conflict_id = response["conflict_id"]
        selected_answer_index = response["selected_index"]

        # 查找对应的候选对列表
        candidate_pairs = next(
            g["candidate_pairs"] for g in conflict_groups_right
            if g["conflict_id"] == conflict_id
        )

        all_indices = {pair_to_indices[p] for p in candidate_pairs}

        # 重置所有相关答案为0
        for i in all_indices:
            all_answers[i] = 0
        # 设置选定答案
        all_answers[selected_answer_index] = 1

    # 计算时间
    end_time = time.time()

    run_time_minutes = end_time - start_time

    minutes, seconds, milliseconds = break_down_time(run_time_minutes)

    print('Successfully finish the conflict resolution with LLM!')
    print(f"The time of the conflict resolution with LLM: "
          f"{minutes}m {seconds}s {milliseconds}ms")

    return all_answers  # 返回最终修正后的答案列表


def generate_prompts_and_parallel_interact_with_llm(batches, parallel_size, llm, city):
    print('Begin to match with the LLM:')
    start_time = time.time()

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    base_config_path = os.path.join(project_root, 'ParallelLLM', 'examples')

    if llm == 'DeepSeek-V2.5':
        config_file = 'example_config_ds_v2.5.yaml'
    elif llm == 'DeepSeek-V3':
        config_file = 'example_config_ds_v3.yaml'
    elif llm == 'DeepSeek-R1':
        config_file = 'example_config_ds_r1.yaml'
    elif llm == 'Qwen3-32B':
        config_file = 'example_config_qwen3_32b.yaml'
    elif llm == 'QwQ-32B':
        config_file = 'example_config_qwq_32b.yaml'
    elif llm == 'Qwen3-14B':
        config_file = 'example_config_qwen3_14b.yaml'
    elif llm == 'DeepSeek-R1-Distill-Qwen-14B':
        config_file = 'example_config_ds_r1_distill_qwen_14b.yaml'
    elif llm == 'Qwen3-8B':
        config_file = 'example_config_qwen3_8b.yaml'
    elif llm == 'DeepSeek-R1-0528-Qwen3-8B':
        config_file = 'example_config_ds_r1_0528_qwen3_8b.yaml'
    else:
        raise ValueError(f"Unsupported LLM model for config path: {llm}")

    client = Client(os.path.join(base_config_path, config_file))

    all_answers = []
    all_confidence = []

    num_batches = len(batches)
    for i in range(0, num_batches, parallel_size):
        if i + parallel_size > num_batches:
            print('Processing the batch: ' + str(i + 1) + ' to ' + str(num_batches))
        else:
            print('Processing the batch: ' + str(i + 1) + ' to ' + str(i + parallel_size))

        current_batches = batches[i:i + parallel_size]  # current_batches 是一个列表，其中每个元素是一个 "batch_item"
        current_prompts = []

        for batch_item in current_batches:  # batch_item 是原始 batches 列表中的一个元素
            prompt = generate_prompt_for_zeroshot_with_confidence(batch_item, city)
            current_prompts.append(prompt)

        tasks = [client.generate(q) for q in current_prompts]
        batch_results = asyncio.run(process_tasks(tasks))

        current_answers = []
        current_confidence = []
        count_true = 0

        # 使用 enumerate 获取索引 j，用于访问 current_batches 中对应的 batch_item
        for j, result in enumerate(batch_results):
            original_batch_item_for_this_result = current_batches[j]
            # 修改点 2: 传递 len(original_batch_item_for_this_result) 而不是 parallel_size
            flag, tup = validate_and_extract_with_confidence_two_lists(result, len(original_batch_item_for_this_result))

            if flag:
                count_true = count_true + 1
                answers, confidences = tup
                current_answers.extend(answers)
                current_confidence.extend(confidences)

        while count_true != len(current_prompts):  # 使用 len(current_prompts)
            tasks = [client.generate(q) for q in current_prompts]
            batch_results = asyncio.run(process_tasks(tasks))

            current_answers = []
            current_confidence = []
            count_true = 0

            for j, result in enumerate(batch_results):
                original_batch_item_for_this_result = current_batches[j]
                flag, tup = validate_and_extract_with_confidence_two_lists(result, len(original_batch_item_for_this_result))

                if flag:
                    count_true = count_true + 1
                    answers, confidences = tup
                    current_answers.extend(answers)
                    current_confidence.extend(confidences)

        all_answers.extend(current_answers)
        all_confidence.extend(current_confidence)

    end_time = time.time()

    run_time_minutes = end_time - start_time

    minutes, seconds, milliseconds = break_down_time(run_time_minutes)

    print()
    print('Successfully finish the match with LLM!')
    print(f"The time of the match with LLM: "
          f"{minutes}m {seconds}s {milliseconds}ms")

    return all_answers, all_confidence


# use the llm by closeai without reasoning (gemini-2.0-flash)
def execute_full_llm_chat_by_closeai_gemini(
        batch_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Optional[Dict[str, Any]]
) -> str:
    """
    Executes a streaming large language model chat (CloseAI Gemini Pro version).

    Args:
    - batch_prompt: User input prompt text.
    - max_retries: Maximum number of API call retries (default 3).
    - retry_delay: Wait time between retries in seconds (default 1.0).
    - temperature: Generation temperature (0~2, default 0.7).
    - max_tokens: Maximum number of tokens to generate (default 2048).
    - **kwargs: Other optional API parameters (e.g., top_p, stop).

    Returns:
    - Complete generated content (string).
    """
    # Initialize client
    client = OpenAI(
        api_key="sk-Y7ZuexcXDUWLwJpumBIEMnsyaSkx6AliPoB4m8v0dvsPIG1f",
        base_url="https://api.closeai.im/v1/"
    )

    content = ""  # To store the complete generated content
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # Create a streaming request
            completion = client.chat.completions.create(
                model="gemini-2.0-flash",  # Model name for Gemini Pro
                messages=[{"role": "user", "content": batch_prompt}],
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Process the streaming response
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    chunk_content = delta.content or ""  # Handle None values

                    # Real-time output (optional for debugging)
                    # print(chunk_content, end='', flush=True)
                    content += chunk_content

            # Break out of the retry loop on success
            break

        except Exception as e:
            print(f"\nAPI call failed: {str(e)}")
            retry_count += 1

            if retry_count <= max_retries:
                print(f"Retrying in {retry_delay} seconds ({retry_count}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Exceeded maximum retry attempts ({max_retries})") from e

    return content.strip()  # Remove leading/trailing whitespace


# use the llm by closeai with reasoning (gemini-2.5-flash-preview-04-17)
def execute_full_llm_chat_by_closeai_gemini_2_5(
        batch_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 65536,  # You might want to adjust this based on the model's capabilities
        **kwargs: Optional[Dict[str, Any]]
) -> str:
    """
    Executes a streaming large language model chat (CloseAI Gemini 2.5 Pro Exp 03-25 version).

    Args:
    - batch_prompt: User input prompt text.
    - max_retries: Maximum number of API call retries (default 3).
    - retry_delay: Wait time between retries in seconds (default 1.0).
    - temperature: Generation temperature (0~2, default 0.7).
    - max_tokens: Maximum number of tokens to generate (default 2048).
    - **kwargs: Other optional API parameters (e.g., top_p, stop).

    Returns:
    - Complete generated content (string).
    """
    # Initialize client
    client = OpenAI(
        api_key="sk-Y7ZuexcXDUWLwJpumBIEMnsyaSkx6AliPoB4m8v0dvsPIG1f",
        base_url="https://api.closeai.im/v1/"
    )

    content = ""  # To store the complete generated content
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # Create a streaming request
            completion = client.chat.completions.create(
                model="gemini-2.5-flash-preview-04-17",  # Specific Gemini model name
                messages=[{"role": "user", "content": batch_prompt}],
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Process the streaming response
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    chunk_content = delta.content or ""  # Handle None values

                    # Real-time output (optional for debugging)
                    # print(chunk_content, end='', flush=True)
                    content += chunk_content

            # Break out of the retry loop on success and return
            return content.strip()  # Remove leading/trailing whitespace

        except Exception as e:
            print(f"\nAPI call failed: {str(e)}")
            retry_count += 1

            if retry_count <= max_retries:
                print(f"Retrying in {retry_delay} seconds ({retry_count}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                # After all retries, raise the last encountered error
                raise RuntimeError(
                    f"Exceeded maximum retry attempts ({max_retries}) for model gemini-2.5-pro-exp-03-25") from e

    # This line should ideally not be reached if logic is correct (either returns in try or raises in except)
    # However, to satisfy linters or as a fallback, you might return the accumulated content or an empty string.
    # Given the loop structure, if it exits without returning or raising, content would be what was accumulated
    # before the last failure.
    return content.strip()


# use the llm by api_2_gpt without reasoning (gpt-4.1-mini)
def execute_full_llm_chat_by_api2gpt_chat(
        batch_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Optional[Dict[str, Any]]
) -> str:
    client = OpenAI(
        api_key="AK-888d1659-4507-4ee2-971c-074352d40424",
        base_url="https://api.api2gpt.com/v1/"
    )

    content = ""  # To store the complete generated content
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # Create a streaming request
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",  # Model name for Gemini Pro
                messages=[{"role": "user", "content": batch_prompt}],
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Process the streaming response
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    chunk_content = delta.content or ""  # Handle None values

                    # Real-time output (optional for debugging)
                    # print(chunk_content, end='', flush=True)
                    content += chunk_content

            # Break out of the retry loop on success
            break

        except Exception as e:
            print(f"\nAPI call failed: {str(e)}")
            retry_count += 1

            if retry_count <= max_retries:
                print(f"Retrying in {retry_delay} seconds ({retry_count}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Exceeded maximum retry attempts ({max_retries})") from e

    return content.strip()  # Remove leading/trailing whitespace


# use the llm by api_2_gpt with reasoning (o4-mini)
def execute_full_llm_chat_by_api2gpt_chat_reasoning(
        batch_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        **kwargs: Optional[Dict[str, Any]]
) -> str:
    # Initialize client
    client = OpenAI(
        api_key="AK-888d1659-4507-4ee2-971c-074352d40424",
        base_url="https://api.api2gpt.com/v1/"
    )

    content = ""  # To store the complete generated content
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # Create a streaming request
            completion = client.chat.completions.create(
                model="o4-mini",  # Specific Gemini model name
                messages=[{"role": "user", "content": batch_prompt}],
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Process the streaming response
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    chunk_content = delta.content or ""  # Handle None values

                    # Real-time output (optional for debugging)
                    # print(chunk_content, end='', flush=True)
                    content += chunk_content

            # Break out of the retry loop on success and return
            return content.strip()  # Remove leading/trailing whitespace

        except Exception as e:
            print(f"\nAPI call failed: {str(e)}")
            retry_count += 1

            if retry_count <= max_retries:
                print(f"Retrying in {retry_delay} seconds ({retry_count}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                # After all retries, raise the last encountered error
                raise RuntimeError(
                    f"Exceeded maximum retry attempts ({max_retries}) for model gemini-2.5-pro-exp-03-25") from e

    return content.strip()