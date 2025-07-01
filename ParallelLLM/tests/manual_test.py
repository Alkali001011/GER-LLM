import unittest
import asyncio
import logging
from pllm import Client

class TestClientFunctionality(unittest.IsolatedAsyncioTestCase):
    """客户端基础功能测试套件"""
    
    @classmethod
    def setUpClass(cls):
        """类级别初始化"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger("TestClient")  # 创建类级别日志器
    
    async def asyncSetUp(self):
        """每个测试方法前的初始化"""
        self.client = Client("input/config/base.yaml", log_level=logging.DEBUG)
    
    async def test_basic_functionality(self):
        """测试生成和聊天基础功能"""
        try:
            # 测试生成功能
            response = await self.client.generate("解释什么是机器学习")
            self.assertIsInstance(response, str)
            
            # 测试聊天功能
            chat_response = await self.client.chat([
                {"role": "system", "content": "你是一个有用的AI助手"},
                {"role": "user", "content": "写一个Python函数计算两个数的最大公约数"}
            ])
            self.logger.debug("聊天响应示例：%s", chat_response)  # 使用正确日志器
            
            # 验证统计信息
            stats = self.client.get_stats()
            self.logger.debug("完整统计结构：%s", stats)

            # 计算总请求数和总tokens
            total_requests = 0
            total_tokens = 0
            
            # 遍历所有提供商
            for provider in stats:
                # 遍历该提供商的所有客户端实例
                for client_stats in stats[provider]:
                    total_requests += client_stats['total_requests']
                    total_tokens += client_stats['total_tokens']

            self.assertGreaterEqual(total_requests, 2, f"总请求数不足，当前：{total_requests}")
            self.assertGreaterEqual(total_tokens, 50, f"总token数不足，当前：{total_tokens}")
            
        except Exception as e:
            self.logger.error("测试失败，当前统计信息：%s", stats, exc_info=True)
            raise

if __name__ == "__main__":
    unittest.main() 