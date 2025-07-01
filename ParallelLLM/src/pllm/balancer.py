from typing import Dict, List, Any, Optional
import time
import logging
import asyncio
import yaml
from collections import deque
import os
from dotenv import load_dotenv

import aiohttp
import random

class LLMClient:
    """管理单个LLM API客户端，包括速率限制和错误跟踪"""
    
    def __init__(self, provider: str, config: Dict):
        self.provider = provider
        self.config = config
        self.last_used = 0
        self.error_count = 0
        self.is_active = True
        self.rate_limit = int(config.get('rate_limit', 5))  # 默认每分钟5次请求
        self.request_queue = deque(maxlen=self.rate_limit)
        self.total_tokens = 0
        self.total_requests = 0
        self.active_requests = 0  # 新增活跃请求计数器
        self.logger = logging.getLogger(f"pllm.{provider}")
    
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        if not self.is_active:
            return False
        
        # 检查速率限制
        now = time.time()
        if len(self.request_queue) >= self.rate_limit:
            oldest = self.request_queue[0]
            if now - oldest < 60:  # 1分钟窗口
                return False
        return True
    
    def record_usage(self, response: Dict[str, Any]) -> None:
        """记录API使用情况（仅处理标准LLM响应格式）"""
        # 仅记录明确包含usage字段的响应
        usage = response.get('usage', {})
        
        self.total_tokens += usage.get('total_tokens', 0)
        self.total_requests += 1
        
        # 添加额度预警逻辑
        quota = self.config.get('quota')
        if quota and self.total_tokens > quota * 0.8:
            self.logger.warning(f"API quota nearing limit for {self.provider}: {self.total_tokens}/{quota}")
        
        self.request_queue.append(time.time())
        self.last_used = time.time()
    
    def mark_error(self, error: Exception) -> None:
        """标记错误并更新客户端状态"""
        self.error_count += 1
        self.logger.error(f"API error with {self.provider}: {str(error)}")
        
        if self.error_count > 3:  # 连续3次错误后标记为不可用
            self.is_active = False
            self.logger.warning(f"LLM client {self.provider} marked as inactive due to errors")

class LoadBalancer:
    """智能负载均衡器，管理多个LLM API端点"""
    
    def __init__(self, config_path: str):
        self.clients: Dict[str, List[LLMClient]] = {}
        self.logger = logging.getLogger("pllm.balancer")
        self.load_config(config_path)
        self.start_health_check()
    
    def load_config(self, config_path: str) -> None:
        """加载并解析YAML配置文件，支持每个提供商有多个API密钥"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 初始化每个提供商的客户端
            llm_config = config.get('llm', {})
            if not llm_config:
                raise ValueError("Missing 'llm' section in config")
                
            providers_str = llm_config.get('use', '')
            if not providers_str:
                raise ValueError("No providers specified in config")
                
            self.active_providers = [p.strip() for p in providers_str.split(',')]
            self.clients = {}  # 确保初始化clients字典

            # 修复：遍历所有支持的provider配置
            for provider in self.active_providers:
                provider_configs = llm_config.get(provider, [])
                if not provider_configs:
                    raise ValueError(f"No configuration found for provider: {provider}")
                
                # 支持单配置和多配置格式
                if isinstance(provider_configs, dict):
                    provider_configs = [provider_configs]
                
                self.clients[provider] = [
                    LLMClient(provider, cfg) for cfg in provider_configs
                ]
                self.logger.info(f"Initialized {len(provider_configs)} clients for {provider}")
                
        except Exception as e:
            self.logger.error(f"Config load failed: {str(e)}")
            raise
    
    def get_best_client(self) -> LLMClient:
        """改进后的负载均衡算法，支持同分随机选择"""
        candidates = []
        for provider in self.active_providers:
            for client in self.clients.get(provider, []):
                if client.is_available():
                    # 新的评分标准（数值越大优先级越高）：
                    # 1. 活跃请求数最少（负值使更少请求的客户端得分更高）
                    # 2. 错误计数最少
                    # 3. 速率限制余量最多
                    # 4. 最近使用时间最久远
                    score = (
                        -client.active_requests * 1000,  # 主要因素
                        -client.error_count * 100,
                        (client.rate_limit - len(client.request_queue)) * 10,
                        -client.last_used  # 次要因素
                    )
                    candidates.append((score, client))
        
        if candidates:
            # 找到最高分
            max_score = max(candidates, key=lambda x: x[0])[0]
            # 收集所有达到最高分的客户端
            best_candidates = [client for score, client in candidates if score == max_score]
            # 随机选择一个
            best_client = random.choice(best_candidates)
            best_client.active_requests += 1  # 预占请求槽位
            return best_client
        raise Exception("No available LLM clients")
    
    async def execute_request(self, messages: List[Dict[str, str]], retry_policy: str = 'fixed', **kwargs) -> Dict[str, Any]:
        """执行LLM请求，支持不同重试策略"""
        max_retries = 3
        retries = 0
        last_error = None
        
        while True:
            try:
                client = self.get_best_client()
                self.logger.debug(f"Selected client: {client.provider}")
                response = await self._call_api(client, messages, **kwargs)
                client.record_usage(response)
                return response
                
            except Exception as e:
                retries += 1
                last_error = e
                client.mark_error(e)
                
                if retry_policy == 'fixed':
                    if retries >= max_retries:
                        self.logger.error(f"All retries failed (policy={retry_policy})")
                        raise Exception(f"Failed after {retries} retries: {str(e)}")
                    self.logger.warning(f"Retry {retries}/{max_retries}")
                    
                elif retry_policy == 'infinite':
                    self.logger.warning(f"Retry {retries} (infinite mode), last error: {str(e)}")
                    await asyncio.sleep(1)  # 防止密集重试
                    
                else:
                    raise ValueError(f"Invalid retry policy: {retry_policy}")
    
    async def _call_api(self, client: LLMClient, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """调用特定客户端的API（适配Siliconflow参数规范）"""
        request_params = {
            "model": client.config['model'],
            "messages": messages,
            "stream": kwargs.get('stream', False),
            "max_tokens": kwargs.get('max_tokens', 4096),
            "stop": kwargs.get('stop'),
            "response_format": kwargs.get('response_format'),
            "tools": kwargs.get('tools')
        }

        # 清理空值参数
        request_params = {k: v for k, v in request_params.items() if v is not None}

        # 添加其他自定义参数（排除已明确处理的参数）
        reserved_params = {'temperature', 'top_p', 'top_k', 'frequency_penalty', 
                          'stream', 'max_tokens', 'stop', 'response_format', 'tools'}
        for key, value in kwargs.items():
            if key not in reserved_params and value is not None:
                request_params[key] = value

        # 执行API调用
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {client.config['api_key']}",
                "Content-Type": "application/json"
            }
            
            if 'headers' in client.config:
                headers.update(client.config['headers'])
                
            async with session.post(
                client.config['api_base'],
                headers=headers,
                json=request_params,
                timeout=None  # 移除固定超时限制
            ) as response:
                try:
                    if response.status != 200:
                        error_text = (await response.text()).strip() or "No error message"
                        raise Exception(f"API request failed: {response.status}, {error_text}")
                    return await response.json()
                finally:
                    client.active_requests -= 1  # 确保请求计数正确释放
    
    def start_health_check(self) -> None:
        """启动定期健康检查任务"""
        async def check():
            while True:
                await asyncio.sleep(300)  # 每5分钟检查一次
                self.logger.debug("Running health check")
                for client in self._all_clients():
                    if not client.is_active:
                        client.error_count = 0
                        client.is_active = True
                        self.logger.info(f"Reactivated client: {client.provider}")
        
        # 检查是否有运行中的事件循环，如果没有则不启动健康检查
        try:
            asyncio.get_running_loop()
            asyncio.create_task(check())
            self.logger.debug("Health check task started")
        except RuntimeError:
            self.logger.debug("No running event loop, health check will not start automatically")
            # 存储协程以便稍后手动启动
            self._health_check_coro = check
    
    def _all_clients(self) -> List[LLMClient]:
        """
        获取所有客户端的生成器
        
        Returns:
            所有已初始化的LLM客户端的生成器
        """
        for provider_clients in self.clients.values():
            for client in provider_clients:
                yield client 

    async def execute_embedding_request(self, input_text: str, retry_policy: str = 'fixed', **kwargs) -> Dict[str, Any]:
        """执行Embedding请求"""
        max_retries = 3
        retries = 0
        last_error = None
        
        while True:
            try:
                client = self.get_best_client()
                self.logger.debug(f"Selected client for embedding: {client.provider}")
                response = await self._call_embedding_api(client, input_text, **kwargs)
                client.record_usage(response)
                return response
                
            except Exception as e:
                retries += 1
                last_error = e
                client.mark_error(e)
                
                if retry_policy == 'fixed':
                    if retries >= max_retries:
                        self.logger.error(f"All embedding retries failed (policy={retry_policy})")
                        raise Exception(f"Embedding failed after {retries} retries: {str(e)}")
                    self.logger.warning(f"Embedding retry {retries}/{max_retries}")
                    
                elif retry_policy == 'infinite':
                    self.logger.warning(f"Embedding retry {retries} (infinite mode), last error: {str(e)}")
                    await asyncio.sleep(1)
                    
                else:
                    raise ValueError(f"Invalid retry policy: {retry_policy}")

    async def _call_embedding_api(self, client: LLMClient, input_text: str, **kwargs) -> Dict[str, Any]:
        """调用特定客户端的Embedding API"""
        request_params = {
            "model": client.config['model'],
            "input": input_text,
            "encoding_format": kwargs.get('encoding_format', 'float')
        }

        # 清理空值参数
        request_params = {k: v for k, v in request_params.items() if v is not None}

        # 添加其他自定义参数
        reserved_params = {'encoding_format'}
        for key, value in kwargs.items():
            if key not in reserved_params and value is not None:
                request_params[key] = value

        # 执行API调用
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {client.config['api_key']}",
                "Content-Type": "application/json"
            }
            
            if 'headers' in client.config:
                headers.update(client.config['headers'])
                
            async with session.post(
                client.config['api_base'],
                headers=headers,
                json=request_params,
                timeout=None
            ) as response:
                try:
                    if response.status != 200:
                        error_text = (await response.text()).strip() or "No error message"
                        raise Exception(f"Embedding API failed: {response.status}, {error_text}")
                    return await response.json()
                finally:
                    client.active_requests -= 1 