import unittest
import asyncio
import logging
from pllm import Client

class TestMultiKeyBalancing(unittest.IsolatedAsyncioTestCase):  # 继承测试基类
    def setUp(self):
        # 配置详细日志
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("test")

    async def test_multi_key_balancing(self):  # 改为类方法
        """测试多密钥负载均衡及响应验证"""
        client = Client("input/config/base.yaml", log_level=logging.DEBUG)
        responses = []
        
        # 创建任务并收集响应
        tasks = [client.generate(f"测试请求 {i}") for i in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 记录部分响应内容
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                self.logger.error(f"请求 {i} 失败: {str(resp)}")
            else:
                self.logger.debug(f"请求 {i} 响应片段: {resp[:60]}...")  # 显示前60字符防止日志过大
        
        # 验证基础统计
        stats = client.get_stats()
        self.logger.info(f"使用统计: {stats}")
        
        # 验证密钥使用情况（保持原有断言）
        key_usage = [c["total_requests"] for c in stats["siliconflow"]]
        self.assertGreater(len(key_usage), 1, "未检测到多密钥负载均衡")

if __name__ == "__main__":
    unittest.main()  # 添加标准测试运行器 