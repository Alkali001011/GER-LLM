import asyncio
import logging
from typing import Dict, List, Optional, Union, Any

import yaml

from .balancer import LoadBalancer

class Client:
    """
    PLLM Client - 并行LLM调用客户端
    
    简单易用的LLM负载均衡客户端，自动选择最佳API密钥进行调用
    
    Args:
        config_path: 配置文件路径，YAML格式
        log_level: 日志级别，默认为INFO
    """
    
    def __init__(self, config_path: str, log_level: int = logging.INFO):
        # 配置日志
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("pllm")
        
        # 初始化负载均衡器
        self.balancer = LoadBalancer(config_path)
        self.logger.info(f"PLLM Client initialized with {len(list(self.balancer._all_clients()))} API clients")
    
    async def chat(self, 
                  messages: List[Dict[str, str]], 
                  temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None,
                  retry_policy: str = 'fixed',
                  **kwargs) -> Dict[str, Any]:
        """
        发送聊天请求到LLM服务
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数
            retry_policy: 重试策略（infinite, fixed, retry_once）
            **kwargs: 其他参数传递给底层API
            
        Returns:
            API响应的JSON对象
        """
        return await self.balancer.execute_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_policy=retry_policy,
            **kwargs
        )
    
    async def generate(self, 
                     prompt: str, 
                     retry_policy: str = 'fixed',
                     **kwargs) -> str:
        """
        简化的文本生成接口
        
        Args:
            prompt: 用户输入的提示词
            retry_policy: 重试策略
            **kwargs: 其他参数传递给chat方法
            
        Returns:
            生成的文本内容
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self.balancer.execute_request(
            messages, 
            retry_policy=retry_policy,
            **kwargs
        )
        return response["choices"][0]["message"]["content"]
    
    async def embedding(self, 
                      text: str,
                      encoding_format: str = 'float',
                      retry_policy: str = 'fixed',
                      **kwargs) -> List[float]:
        """
        获取文本的embedding向量
        
        Args:
            text: 需要编码的文本
            encoding_format: 编码格式（默认float）
            retry_policy: 重试策略
            **kwargs: 其他API参数
            
        Returns:
            embedding向量列表
        """
        response = await self.balancer.execute_embedding_request(
            input_text=text,
            encoding_format=encoding_format,
            retry_policy=retry_policy,
            **kwargs
        )
        return response['data'][0]['embedding']
    
    def chat_sync(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        同步执行聊天请求
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}]
            **kwargs: 透传至chat()方法参数（temperature/max_tokens等）
            
        Returns:
            包含完整响应数据的字典（同chat()方法）
        """
        return asyncio.run(self.chat(messages, **kwargs))
    
    def generate_sync(self,
                    prompt: str,
                    retry_policy: str = 'fixed',
                    **kwargs) -> str:
        """
        同步执行文本生成
        
        Args:
            prompt: 用户输入的提示文本
            retry_policy: 重试策略配置（infinite, fixed, retry_once）
            **kwargs: 透传至generate()方法参数
            
        Returns:
            生成的文本内容（同generate()方法）
        """
        return asyncio.run(self.generate(prompt, retry_policy=retry_policy, **kwargs))
    
    def embedding_sync(self, 
                     text: str,
                     encoding_format: str = 'float',
                     **kwargs) -> List[float]:
        """
        同步执行embedding请求
        
        Args:
            text: 需要编码的文本
            encoding_format: 编码格式（默认float）
            **kwargs: 透传至embedding()方法参数
            
        Returns:
            embedding向量列表
        """
        return asyncio.run(
            self.embedding(text, encoding_format=encoding_format, **kwargs)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有客户端的使用统计信息"""
        stats = {}
        # 添加调试日志
        print(f"Available providers: {self.balancer.clients.keys()}") 
        for provider, clients in self.balancer.clients.items():
            print(f"Processing provider: {provider} with {len(clients)} clients")
            provider_stats = []
            for i, client in enumerate(clients):
                print(f"Client {i}: requests={client.total_requests}, tokens={client.total_tokens}")
                provider_stats.append({
                    "id": i,
                    "active": client.is_active,
                    "error_count": client.error_count,
                    "total_requests": client.total_requests,
                    "total_tokens": client.total_tokens
                })
            stats[provider] = provider_stats
        return stats 