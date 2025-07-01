import asyncio
from pllm import Client

async def main():
    # 初始化客户端
    client = Client("input/config/llm.yaml")
    
    # 新增并行测试示例
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
    
    # 并行执行10个请求
    print("\n=== 并行测试 ===")
    tasks = [client.generate(q) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 输出结果统计
    success = sum(1 for r in results if not isinstance(r, Exception))
    print(f"\n成功: {success}/10, 失败: {10-success}")
    for i, (q, r) in enumerate(zip(questions, results)):
        status = "✓" if not isinstance(r, Exception) else f"✗ ({str(r)})"
        print(f"{i+1}. {q[:20]}... {status}")
    
    # 查看详细统计
    print("\n使用统计:")
    print(client.get_stats())
    
    # 异步生成文本
    response = await client.generate("解释一下量子计算的基本原理")
    print(response)
    
    # 使用聊天接口
    chat_response = await client.chat([
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "写一个Python函数计算斐波那契数列。"}
    ])
    print(chat_response["choices"][0]["message"]["content"])

# 对于不支持异步的环境，可以使用同步接口
def sync_example():
    client = Client("input/config/llm.yaml")
    response = client.generate_sync("什么是机器学习？")
    print(response)

if __name__ == "__main__":
    # 运行异步示例
    asyncio.run(main())
    
    # 或者运行同步示例
    # sync_example()
    