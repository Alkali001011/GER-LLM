"""
LLM性能基准测试工具
"""
import os
import time
import asyncio
import json
import logging
from datetime import datetime
from typing import List
from pllm import Client

class Benchmark:
    """LLM性能基准测试工具"""
    
    def __init__(self, config_path: str, output_dir: str = "output/benchmark"):
        """
        初始化基准测试工具
        
        Args:
            config_path: 客户端配置文件路径
            output_dir: 测试结果输出目录
        """
        self.client = Client(config_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 需要自定义日志格式和存储位置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("Benchmark")

    def generate_questions(self, n: int) -> List[str]:
        """
        生成测试问题集
        
        Args:
            n: 需要生成的问题数量
            
        Returns:
            包含指定数量问题的列表
        """
        base_question = "请用300字左右解释什么是{}"
        topics = [
            "机器学习", "深度学习", "神经网络",
            "自然语言处理", "计算机视觉", "强化学习"
        ]
        return [base_question.format(topics[i % len(topics)]) for i in range(n)]

    async def parallel_test(self, questions: List[str], workers: int = 10) -> dict:
        """
        执行并行压力测试
        
        Args:
            questions: 要测试的问题列表
            workers: 最大并发请求数
            
        Returns:
            包含测试指标和详细结果的字典
        """
        start_time = time.perf_counter()
        
        # 分批处理避免内存溢出
        batch_size = workers
        results = []
        failed = 0
        details = []  # 新增结果记录
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            tasks = [self.client.generate(q, retry_policy='infinite') for q in batch]
            
            try:
                batch_start = time.perf_counter()
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 记录每个问题的详细结果
                for q, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        details.append({
                            "question": q,
                            "success": False,
                            "error": str(result),
                            "latency": time.perf_counter() - batch_start
                        })
                    else:
                        details.append({
                            "question": q,
                            "response": result,
                            "success": True,
                            "latency": time.perf_counter() - batch_start
                        })
                results.extend(batch_results)
            except Exception as e:
                self.logger.error(f"批量请求失败: {str(e)}")
                failed += len(batch)
        
        total_time = time.perf_counter() - start_time
        success = len(results) - failed
        
        return {
            "mode": "parallel",
            "total_questions": len(questions),
            "success": success,
            "failed": failed,
            "total_time": round(total_time, 2),
            "qps": round(success / total_time, 2) if total_time > 0 else 0,
            "details": details  # 添加详细记录
        }

    async def sequential_test(self, questions: List[str]) -> dict:
        """
        执行顺序基准测试
        
        Args:
            questions: 要测试的问题列表
            
        Returns:
            包含测试指标和详细结果的字典
        """
        start_time = time.perf_counter()
        success = 0
        failed = 0
        details = []  # 新增结果记录
        
        for q in questions:
            try:
                start = time.perf_counter()
                response = await self.client.generate(q, retry_policy='infinite')
                latency = time.perf_counter() - start
                details.append({
                    "question": q,
                    "response": response,
                    "success": True,
                    "latency": latency
                })
                success += 1
            except Exception as e:
                self.logger.error(f"请求失败: {str(e)}")
                details.append({
                    "question": q,
                    "success": False,
                    "error": str(e),
                    "latency": time.perf_counter() - start
                })
                failed += 1
        
        total_time = time.perf_counter() - start_time
        
        return {
            "mode": "sequential",
            "total_questions": len(questions),
            "success": success,
            "failed": failed,
            "total_time": round(total_time, 2),
            "qps": round(success / total_time, 2) if total_time > 0 else 0,
            "details": details  # 添加详细记录
        }

    def generate_report(self, results: dict) -> str:
        """
        生成测试报告
        
        Args:
            results: 测试结果数据
            
        Returns:
            报告文件存储路径
        """
        # 需要记录完整问题集用于结果分析
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_questions": self.generate_questions(0),
            "results": results,
            "stats": self.client.get_stats()
        }
        
        # 保存文件
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d%H%M')}.json"
        path = os.path.join(self.output_dir, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        return path

async def main():
    """测试执行入口"""
    # 配置测试参数
    TEST_CONFIG = "input/config/base.yaml"
    QUESTIONS_NUM = 100  # 总测试问题数
    WORKERS = 10         # 最大并发数
    
    benchmark = Benchmark(TEST_CONFIG)
    questions = benchmark.generate_questions(QUESTIONS_NUM)
    
    # 执行并行测试
    parallel_result = await benchmark.parallel_test(questions, WORKERS)
    
    # 执行顺序测试（需要重置客户端统计）
    benchmark.client = Client(TEST_CONFIG)
    sequential_result = await benchmark.sequential_test(questions)
    
    # 生成报告
    report_path = benchmark.generate_report(
        results={"parallel": parallel_result, "sequential": sequential_result},
    )
    
    print(f"测试报告已保存至：{report_path}")

if __name__ == "__main__":
    asyncio.run(main())
