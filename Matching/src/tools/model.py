from transformers import BertModel, BertTokenizer
from typing import Literal
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch


class InputFeatures(object):
    """
    BERT类模型输入特征的标准化容器，用于封装单一样本的所有输入特征及相关元数据。

    功能：
        - 存储模型所需的原始输入特征（token IDs, 注意力掩码, 段落标记）
        - 关联样本的标签及辅助信息（样本ID、任务ID）
        - 便于数据管道与模型接口之间的数据传递

    典型使用场景：
        1. 数据预处理阶段将原始文本转换为特征对象
        2. 训练/验证时批量加载特征对象
        3. 多任务学习中区分不同任务的数据
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, exm_id, task_id=-1):
        # Token ID序列（含填充），形状: [max_seq_length]
        self.input_ids = input_ids
        # 注意力掩码（1=有效token，0=填充），形状: [max_seq_length]
        self.input_mask = input_mask
        # 段落标记（0=第一句，1=第二句），形状: [max_seq_length]
        # 单句场景全为0，句子对场景如[0,0,0,0,1,1]
        self.segment_ids = segment_ids
        # 样本标签（分类任务为类别ID，回归任务为浮点值）
        # 例如：情感分类中的0（负面）/1（正面）
        self.label_id = label_id
        # 样本唯一标识符，用于：
        # - 调试时追踪特定样本
        # - 后期结果分析（如错误样本复查）
        # - 数据集划分后的样本追踪
        self.exm_id = exm_id
        # 多任务ID（默认-1表示单任务）
        # 例如：0-文本分类，1-序列标注，2-相似度计算
        self.task_id = task_id


def convert_one_example_to_features(tuple, max_seq_length, cls_token, sep_token, pad_token, tokenizer):
    """
    将单个文本样本转换为BERT类模型输入特征

    参数:
        tuple (str): 输入文本（单句）
        max_seq_length (int): 模型最大输入长度（包括特殊标记）
        cls_token (str): 分类标记（如[CLS]）
        sep_token (str): 分隔标记（如[SEP]）
        pad_token (int): 填充token的ID
        tokenizer: 分词器对象

    返回:
        tuple: (
            input_ids (list[int]): token ID序列（含填充）
            input_mask (list[int]): 注意力掩码（1=有效token，0=填充）
            segment_ids (list[int]): 段落标记（单句场景全为0）
        )
    """
    # 步骤1：对输入文本进行分词（子词切分）
    tokens = tokenizer.tokenize(tuple)
    # 步骤2：处理长度超限情况（保留max_seq_length-2个位置给CLS和SEP）
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]  # 直接截断尾部（例如保留前510个token）
    # 步骤3：添加特殊标记构建完整序列（格式：[CLS] + tokens + [SEP]）
    tokens = [cls_token] + tokens + [sep_token]
    # 步骤4：生成段落标记（单句场景全为0，包含CLS和SEP）
    segment_ids = [0] * (len(tokens))

    # 步骤5：将子词转换为数字ID
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # 步骤6：初始化注意力掩码（有效token位置为1）
    input_mask = [1] * len(input_ids)
    # 步骤7：计算填充长度并执行填充
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)  # 尾部填充pad_token
    input_mask = input_mask + ([0] * padding_length)  # 掩码填充
    segment_ids = segment_ids + ([0] * padding_length)  # 段落标记填充0

    # 步骤8：校验输出维度
    assert len(input_ids) == max_seq_length  # 确保输入ID长度正确
    assert len(input_mask) == max_seq_length  # 确保掩码长度正确
    assert len(segment_ids) == max_seq_length  # 确保段落标记长度正确

    return input_ids, input_mask, segment_ids


def convert_one_example_to_features_sep(tuple, max_seq_length, cls_token, sep_token, pad_token, tokenizer):
    """
        处理包含分隔符（如[SEP]）的句子对，生成BERT输入特征

        参数:
            tuple (str): 包含分隔符的文本，格式为"句子A[SEP]句子B"
            max_seq_length (int): 模型最大输入长度（含特殊标记）
            cls_token (str): 分类标记（如[CLS]）
            sep_token (str): 分隔标记（如[SEP]）
            pad_token (int): 填充符的ID
            tokenizer: 分词器对象

        返回:
            input_ids (list[int]): token ID序列（含填充）
            input_mask (list[int]): 注意力掩码（1=有效token，0=填充）
            segment_ids (list[int]): 段落标记（0=左句，1=右句）
    """
    # 步骤1：分割文本为左句和右句
    left = tuple.split(sep_token)[0]  # 提取分隔符左侧文本（如句子A）
    right = tuple.split(sep_token)[1]  # 提取分隔符右侧文本（如句子B）
    # 步骤2：分词处理
    ltokens = tokenizer.tokenize(left)  # 左句分词结果（子词列表）
    rtokens = tokenizer.tokenize(right)  # 右句分词结果（子词列表）

    # 步骤3：计算需要截断的token数量
    # 总token数 = 左句长度 + 右句长度 + CLS(1) + SEP(2)
    # 若超过max_seq_length，计算超出量（例如总长度258，max_seq=256，则more=2）
    more = len(ltokens) + len(rtokens) - max_seq_length + 3  # -3为CLS+SEP*2保留空间
    # 步骤4：动态截断策略（一般情况下用不到）
    if more > 0:  # 需要截断
        if more < len(rtokens):  # 右句足够长，优先截断右句
            rtokens = rtokens[:(len(rtokens) - more)]
        elif more < len(ltokens):  # 左句足够长，截断左句
            ltokens = ltokens[:(len(ltokens) - more)]
        else:  # 两边都不够截断，强制各保留50个token（极端情况）
            rtokens = rtokens[:50]
            ltokens = ltokens[:50]

    # 步骤5：构建最终token序列
    # 格式: [CLS] + 左句 + [SEP] + 右句 + [SEP]
    tokens = [cls_token] + ltokens + [sep_token] + rtokens + [sep_token]
    # 步骤6：生成段落标记
    # 左句部分（含CLS和第一个SEP）标记为0，右句部分（含第二个SEP）标记为1
    segment_ids = [0] * (len(ltokens) + 2) + [1] * (len(rtokens) + 1)  # +2=CLS+SEP，+1=SEP
    # 步骤7：转换为ID序列
    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将子词转换为数字ID
    # 步骤8：生成注意力掩码（有效token为1）
    input_mask = [1] * len(input_ids)  # 初始化有效标记
    # 步骤9：填充处理
    padding_length = max_seq_length - len(input_ids)  # 计算需要填充的长度
    input_ids = input_ids + ([pad_token] * padding_length)  # 尾部填充
    input_mask = input_mask + ([0] * padding_length)  # 填充部分掩码为0
    segment_ids = segment_ids + ([0] * padding_length)  # 填充部分段落标记为0

    return input_ids, input_mask, segment_ids


def convert_fea_to_tensor00(features_list, batch_size, do_train):
    """
    将预处理后的特征对象转换为PyTorch DataLoader

    参数:
        features_list: list[InputFeatures] - 特征对象列表，每个元素对应一个样本的输入特征
        batch_size: int - 每个批次的样本数量
        do_train: int - 训练模式标志（1=训练模式，0=评估/预测模式）

    返回:
        DataLoader - 可迭代的批数据加载器，包含以下张量：
            input_ids, input_mask, segment_ids, label_ids, exm_ids, task_ids

    修改说明:
        原代码存在特征提取错误，已修复：直接使用features_list中的InputFeatures对象
        原错误代码：features = [x[0] for x in features_list]
        修正后代码：features = features_list
    """

    # features = [x[0] for x in features_list]

    # 关键修复点：直接使用特征对象列表（原错误假设features_list是二维列表）
    features = features_list  # 每个元素是InputFeatures对象，而非子列表

    # 将特征对象属性转换为PyTorch张量 -------------------------------------------------
    # input_ids张量 形状: [num_samples, max_seq_length]
    all_input_ids = torch.tensor(
        [f.input_ids for f in features],  # 遍历所有样本的input_ids
        dtype=torch.long  # 数据类型为长整型（符合BERT的token ID要求）
    )

    # 注意力掩码张量 形状: [num_samples, max_seq_length]
    all_input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long
    )

    # 段落标记张量 形状: [num_samples, max_seq_length]
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long
    )

    # 标签ID张量 形状: [num_samples]
    all_label_ids = torch.tensor(
        [f.label_id for f in features],  # 分类标签或回归值（需确保类型匹配）
        dtype=torch.long  # 注意：回归任务应改为torch.float
    )

    # 样本ID张量 形状: [num_samples]（用于追踪样本来源）
    all_exm_ids = torch.tensor(
        [f.exm_id for f in features],
        dtype=torch.long
    )

    # 任务ID张量 形状: [num_samples]（多任务场景使用）
    all_task_ids = torch.tensor(
        [f.task_id for f in features],
        dtype=torch.long
    )

    # 创建TensorDataset对象 --------------------------------------------------------
    # 将六个张量打包为一个数据集，保证各维度第一维（样本数）一致
    dataset = TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
        all_exm_ids,
        all_task_ids
    )

    # 配置DataLoader --------------------------------------------------------------
    if do_train == 0:
        # 评估/预测模式：顺序采样，保持数据顺序
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,  # 按指定批次大小划分
            shuffle=False  # 显式禁止打乱（即使sampler=SequentialSampler）
        )
    else:
        # 训练模式：随机采样，提升模型泛化能力
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True  # 丢弃最后不足batch_size的批次（避免小批次影响BN层）
        )

    return dataloader


def convert_examples_to_features(text=None, labels=None, max_seq_length=128, tokenizer=None,
                                 cls_token="[CLS]", sep_token='[SEP]',
                                 pad_token=0, task_ids=None):
    """
    将原始文本数据转换为BERT类模型所需的输入特征

    参数:
        text: list[str/list] - 文本列表，元素可以是单句或句子对（如相似度任务中的句子对）
        labels: list[int] - 对应文本的标签列表（若为None则自动填充0）
        max_seq_length: int - 模型最大输入长度（包括特殊标记如[CLS]/[SEP]）
        tokenizer: 分词器对象 - 用于将文本转换为token ID序列
        cls_token: str - 分类标记（如BERT的[CLS]）
        sep_token: str - 分隔标记（如BERT的[SEP]）
        pad_token: int - 填充token的ID（通常为0）
        task_ids: list[int] - 多任务学习的任务ID列表（每个样本对应一个任务ID）

    返回:
        list[InputFeatures] - 包含所有样本特征对象的列表
    """

    features = []  # 初始化特征存储列表

    # 处理无标签情况（例如预测时）
    if labels is None:
        labels = [0] * len(text)  # 创建全零伪标签列表

    # print(text)

    # 遍历每个样本，也就是遍历每个句子
    for ex_index, pair in enumerate(text):  # ex_index 序号， pair每个句子其实就是一个实体对，所以这里写成了pair
        # print(ex_index)
        # print(pair)

        # 进度打印（每200个样本打印一次）
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(text)))

        # 判断当前样本是否包含分隔符（可能指示句子对）
        if sep_token in pair:
            # 调用带分隔符的处理函数（例如处理句子对）
            input_ids, input_mask, segment_ids = convert_one_example_to_features_sep(
                pair, max_seq_length, cls_token, sep_token, pad_token, tokenizer)
        else:
            # 调用常规处理函数（单句处理）
            input_ids, input_mask, segment_ids = convert_one_example_to_features(
                pair, max_seq_length, cls_token, sep_token, pad_token, tokenizer)

        # 校验特征长度是否合法
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # 构建特征对象（处理多任务场景）
        if task_ids:  # 如果提供任务ID列表
            features.append(
                InputFeatures(input_ids=input_ids,  # token ID序列
                              input_mask=input_mask,  # 注意力掩码（1=有效token，0=填充）
                              segment_ids=segment_ids,  # 段落ID（0=第一句，1=第二句）
                              label_id=labels[ex_index],  # 当前样本的标签
                              exm_id=ex_index,  # 样本索引ID
                              task_id=task_ids[ex_index]))  # 多任务ID（每个样本对应一个任务）
        else:  # 单任务场景
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=labels[ex_index],
                              exm_id=ex_index,
                              task_id=-1))  # 无任务ID时填充-1

    return features


def convert_text_to_tensor(texts, encoder, tokenizer, pooling_strategy,
                           batch_size=32, do_train=0):
    """
    将文本列表转换为神经网络可处理的张量表示

    参数:
        texts: list[str] - 需要编码的原始文本列表
        encoder: nn.Module - 预训练的语言模型编码器（如BERT）
        tokenizer - 分词器对象，用于文本到token的转换
        batch_size: int - 每次处理的样本数量
        do_train: int - 是否训练模式标志（0表示推理模式）
        pooling_strategy: 池化策略选择 ["cls", "mean"]
            - "cls" : 使用[CLS]标记的嵌入（默认）
            - "mean" : 使用平均池化嵌入
    返回:
        torch.Tensor - 所有文本的编码结果张量，形状为 (样本数, 隐藏层维度)
    """
    # 获取模型所在的设备（CPU/GPU）
    device = next(encoder.parameters()).device  # 自动适配模型当前设备

    # 步骤1：将原始文本转换为特征对象列表
    fea = convert_examples_to_features(
        text=texts,
        labels=None,  # 无标签模式
        max_seq_length=128,  # 模型最大输入长度
        tokenizer=tokenizer,
        cls_token="[CLS]",  # 分类标记
        sep_token='[SEP]',  # 分隔标记
        pad_token=0,  # 填充符ID
        task_ids=None  # 单任务场景
    )
    # print(fea)

    # 步骤2：将特征对象列表转换为PyTorch DataLoader
    data_loader = convert_fea_to_tensor00(
        features_list=fea,
        batch_size=batch_size,
        do_train=do_train  # 控制是否打乱数据（训练模式需打乱）
    )

    # 步骤3：设置模型为评估模式（关闭Dropout等训练层）
    encoder.eval()

    # 初始化结果张量容器
    sample_tensor = None

    for batch in data_loader:
        # 解包批次数据（移除实体掩码相关部分）
        reviews, mask, segment, _, _, _ = batch

        # 移动数据至模型设备
        reviews = reviews.to(device)  # input_ids
        mask = mask.to(device)  # attention_mask
        segment = segment.to(device)  # token_type_ids

        with torch.no_grad():
            outputs = encoder(reviews, mask, segment)

            # 根据策略选择池化方法
            if pooling_strategy == "cls":
                feat = get_cls_embedding(outputs)
            elif pooling_strategy == "mean":
                feat = get_mean_pooling_embedding(outputs, mask)
            else:
                raise ValueError(f"无效的池化策略: {pooling_strategy}")

            # 累积结果
            if sample_tensor is None:
                sample_tensor = feat
            else:
                sample_tensor = torch.cat((sample_tensor, feat), dim=0)

    return sample_tensor


def get_cls_embedding(outputs: torch.Tensor) -> torch.Tensor:
    """CLS标记池化方案：取[CLS]位置的嵌入"""
    return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]


def get_mean_pooling_embedding(outputs: torch.Tensor,
                               attention_mask: torch.Tensor) -> torch.Tensor:
    """平均池化方案：根据有效token计算均值"""
    last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
    input_mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]

    # 计算有效token的加权和
    sum_emb = torch.sum(last_hidden * input_mask, dim=1)  # [batch, hidden]
    valid_len = torch.sum(input_mask, dim=1)  # [batch, 1]

    # 避免除以零（当样本全为填充时）
    valid_len = torch.clamp(valid_len, min=1e-9)
    return sum_emb / valid_len


class EncoderModel:
    def __init__(self, encoder_name='BERT', instruction=None):
        self.encoder_name = encoder_name
        if encoder_name == 'BERT':
            # 1. 初始化tokenizer（含never_split）
            # self.tokenizer = BertTokenizer.from_pretrained(
            #     '/home/ssd_0//lm_model/bert-base-chinese',
            #     never_split=['#NUM#']  # 保护#NUM#不被拆分
            # )
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased',
                never_split=['#NUM#']  # 保护#NUM#不被拆分
            )
            # 2. 添加任务相关的新特殊token
            special_tokens = ['[COL]', '[VAL]']  # 仅添加新标记
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

            # 3. 加载模型并立即扩展词表
            self.encoder = BertModel.from_pretrained('bert-base-chinese')
            self.encoder.resize_token_embeddings(len(self.tokenizer))  # 关键：先扩词表再加载权重

    def encode(self, texts, pooling_strategy='mean'):
        if self.encoder_name == 'BERT':
            ts = convert_text_to_tensor(
                texts, self.encoder, self.tokenizer, pooling_strategy, batch_size=32, do_train=0)
            return ts.numpy()
