import asyncio
from pllm import Client

async def main_async():
    """异步embedding示例"""
    # 初始化客户端（使用embedding专用配置）
    client = Client("input/config/embedding.yaml")
    
    texts = [
        "量子计算的基本原理",
        "深度学习的数学基础",
        "Transformer架构的核心思想",
        "Python的异步编程模型",
        "机器学习的常见算法"
        "量子计算的基本原理",
        "深度学习的数学基础",
        "Transformer架构的核心思想",
        "Python的异步编程模型",
        "机器学习的常见算法"
    ]
    
    # 并行执行embedding请求
    print("=== 异步Embedding测试 ===")
    tasks = [client.embedding(text) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 输出结果统计
    success = sum(1 for r in results if not isinstance(r, Exception))
    print(f"\n成功: {success}/{len(texts)}, 失败: {len(texts)-success}")
    
    # 显示部分embedding结果
    for i, (text, emb) in enumerate(zip(texts, results)):
        if not isinstance(emb, Exception):
            print(f"{i+1}. {text[:15]}... embedding长度: {len(emb)}")
            print(f"    前5个值: {emb[:5]}...")  # 只显示前5个维度
        else:
            print(f"{i+1}. {text[:15]}... 失败: {str(emb)}")
    
    # 查看详细统计
    print("\n使用统计:")
    print(client.get_stats())

def main_sync():
    """同步embedding示例"""
    client = Client("input/config/embedding.yaml")
    
    # 单个请求示例
    text = "大语言模型的工作原理"
    print("\n=== 同步单个请求测试 ===")
    embedding = client.embedding_sync(text)
    print(f"文本: {text}")
    print(f"Embedding长度: {len(embedding)}")
    print(f"前5个值: {embedding[:5]}...")
    
    # 批量处理示例
    print("\n=== 同步批量处理测试 ===")
    texts = ["机器学习", "深度学习", "强化学习"]
    embeddings = [client.embedding_sync(t) for t in texts]
    for t, emb in zip(texts, embeddings):
        print(f"{t}: {len(emb)}维")
    
    # 查看统计
    print("\n使用统计:")
    print(client.get_stats())

if __name__ == "__main__":
    # 运行异步示例
    print("运行异步示例...")
    asyncio.run(main_async())
    
    # 运行同步示例
    print("\n\n运行同步示例...")
    main_sync() 