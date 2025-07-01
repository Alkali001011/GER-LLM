# Parallel LLM Client Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A parallel Large Language Model invocation framework providing:
- Load balancing across multiple LLM providers
- Hybrid synchronous/asynchronous interfaces
- Automatic failover and intelligent retry mechanisms

## Quick Start

### Installation

```bash
git clone
cd ParallelLLM
pip install -e .
```

### Usage

1. Modify the config file: `examples/example_config.yaml`

```yaml
llm:
  use: siliconflow
  siliconflow:
    - api_key: "sk-xxx"
      api_base: "https://api.siliconflow.cn/v1/chat/completions"
      model: "model name"
      rate_limit: 20
    
    - api_key: "sk-xxx"
      api_base: "https://api.siliconflow.cn/v1/chat/completions"
      model: "model name"
      rate_limit: 20
```

2. Ask simple questions:

```python
from pllm import Client

client = Client("examples/example_config_ds_v2.5.yaml")

response = await client.generate("解释一下量子计算的基本原理")
chat_response = await client.chat([
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "写一个Python函数计算斐波那契数列。"}
])
```

3. Ask multiple questions in parallel:

```python
from pllm import Client
import asyncio

client = Client("examples/example_config_ds_v2.5.yaml")

questions = [
    "解释量子隧穿效应",
    "写一个快速排序算法",
    "什么是Transformer架构？",
    "如何用Python计算圆周率？",
    "解释深度神经网络的工作原理",
    "写一个正则表达式匹配邮箱",
    "什么是元学习？",
    "解释梯度消失问题",
    "写一个递归阶乘函数",
    "什么是注意力机制？"
]
tasks = [client.generate(q) for q in questions]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

## TODO

- [ ] Currently only supports siliconflow, need to add more LLM providers
- [ ] Change into English comments
