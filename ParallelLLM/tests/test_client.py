import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
import os
import sys
import yaml
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pllm import Client

class TestPLLMClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # 创建临时目录
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        
        # 创建测试配置文件
        self.config = {
            'llm': {
                'use': 'test_provider',
                'test_provider': {
                    'api_key': 'test-key',
                    'api_base': 'https://test.api/v1',
                    'model': 'test-model'
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        self.client = Client(self.config_path)
        
        # 如果有健康检查协程，手动启动它
        if hasattr(self.client.balancer, '_health_check_coro'):
            self.health_check_task = asyncio.create_task(
                self.client.balancer._health_check_coro()
            )
    
    async def asyncTearDown(self):
        # 取消健康检查任务
        if hasattr(self, 'health_check_task'):
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # 清理临时目录
        self.temp_dir.cleanup()
    
    @patch('aiohttp.ClientSession.post')
    async def test_generate(self, mock_post):
        # 模拟API响应
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 10}
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # 测试生成方法
        response = await self.client.generate("Test prompt")
        self.assertEqual(response, "Test response")
        
        # 验证API调用参数
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['messages'][0]['content'], "Test prompt")
    
    @patch('aiohttp.ClientSession.post')
    async def test_chat(self, mock_post):
        # 模拟API响应
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Chat response"}}],
            "usage": {"total_tokens": 15}
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # 测试聊天方法
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        response = await self.client.chat(messages)
        
        # 验证响应
        self.assertEqual(response["choices"][0]["message"]["content"], "Chat response")
        
        # 验证API调用参数
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['messages'], messages)
    
    def test_sync_methods(self):
        # 模拟异步方法
        self.client.generate = AsyncMock(return_value="Sync test")
        self.client.chat = AsyncMock(return_value={"choices": [{"message": {"content": "Sync chat"}}]})
        
        # 测试同步方法
        self.assertEqual(self.client.generate_sync("Test"), "Sync test")
        self.assertEqual(
            self.client.chat_sync([{"role": "user", "content": "Hi"}])["choices"][0]["message"]["content"], 
            "Sync chat"
        )

if __name__ == '__main__':
    unittest.main() 