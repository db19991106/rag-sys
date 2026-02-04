"""
评估服务 - 用于评估检索系统的性能、准确性和相关性
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import time
import json
from pathlib import Path
from utils.logger import logger
from config import settings


class RetrievalEvaluator:
    """检索系统评估器"""
    
    def __init__(self):
        self.eval_results: List[Dict] = []  # 评估结果历史
        self.metrics_history: Dict[str, List[float]] = {}  # 指标历史
        self.eval_dir = Path(settings.vector_db_dir) / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_retrieval(self, queries: List[str], expected_results: List[List[str]], 
                          retrieve_func, top_k: int = 10) -> Dict[str, float]:
        """
        评估检索系统的准确性
        
        Args:
            queries: 查询列表
            expected_results: 每个查询的期望结果列表
            retrieve_func: 检索函数
            top_k: 评估的top_k值
            
        Returns:
            评估指标字典
        """
        if len(queries) != len(expected_results):
            raise ValueError("查询列表和期望结果列表长度不匹配")
        
        start_time = time.time()
        
        # 计算评估指标
        mrr_scores = []
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        response_times = []
        
        for i, (query, expected) in enumerate(zip(queries, expected_results)):
            # 执行检索
            query_start = time.time()
            result = retrieve_func(query)
            query_time = time.time() - query_start
            response_times.append(query_time)
            
            # 提取检索结果的内容
            retrieved_contents = []
            if hasattr(result, 'results'):
                retrieved_contents = [r.content for r in result.results[:top_k]]
            elif isinstance(result, list):
                retrieved_contents = [r['content'] for r in result[:top_k]]
            
            # 计算MRR
            mrr = self._calculate_mrr(retrieved_contents, expected)
            mrr_scores.append(mrr)
            
            # 计算NDCG
            ndcg = self._calculate_ndcg(retrieved_contents, expected, top_k)
            ndcg_scores.append(ndcg)
            
            # 计算精准率、召回率和F1
            precision, recall, f1 = self._calculate_precision_recall_f1(retrieved_contents, expected)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # 计算平均指标
        avg_mrr = np.mean(mrr_scores)
        avg_ndcg = np.mean(ndcg_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        avg_response_time = np.mean(response_times)
        total_time = time.time() - start_time
        
        # 构建评估结果
        metrics = {
            'mrr@k': avg_mrr,
            'ndcg@k': avg_ndcg,
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1@k': avg_f1,
            'avg_response_time': avg_response_time,
            'total_evaluation_time': total_time,
            'num_queries': len(queries),
            'top_k': top_k
        }
        
        # 保存评估结果
        self.eval_results.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'num_queries': len(queries),
            'top_k': top_k
        })
        
        # 更新指标历史
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        # 保存评估结果到文件
        self._save_evaluation_results(metrics)
        
        logger.info(f"检索评估完成: MRR@{top_k}={avg_mrr:.4f}, NDCG@{top_k}={avg_ndcg:.4f}, "
                   f"Precision@{top_k}={avg_precision:.4f}, Recall@{top_k}={avg_recall:.4f}, "
                   f"F1@{top_k}={avg_f1:.4f}, Avg Response Time={avg_response_time:.4f}s")
        
        return metrics
    
    def _calculate_mrr(self, retrieved: List[str], expected: List[str]) -> float:
        """
        计算平均倒数排名 (MRR)
        """
        for i, item in enumerate(retrieved):
            if item in expected:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg(self, retrieved: List[str], expected: List[str], top_k: int) -> float:
        """
        计算归一化折扣累积增益 (NDCG)
        """
        # 计算DCG
        dcg = 0.0
        for i, item in enumerate(retrieved[:top_k]):
            if item in expected:
                dcg += 1.0 / np.log2(i + 2)  # i从0开始，所以+2
        
        # 计算理想DCG
        ideal_dcg = 0.0
        for i in range(min(len(expected), top_k)):
            ideal_dcg += 1.0 / np.log2(i + 2)
        
        if ideal_dcg == 0:
            return 0.0
        
        return dcg / ideal_dcg
    
    def _calculate_precision_recall_f1(self, retrieved: List[str], expected: List[str]) -> Tuple[float, float, float]:
        """
        计算精准率、召回率和F1分数
        """
        if not retrieved and not expected:
            return 1.0, 1.0, 1.0
        if not retrieved:
            return 0.0, 0.0, 0.0
        if not expected:
            return 0.0, 0.0, 0.0
        
        # 计算交集
        intersection = set(retrieved) & set(expected)
        
        # 计算精准率
        precision = len(intersection) / len(retrieved)
        
        # 计算召回率
        recall = len(intersection) / len(expected)
        
        # 计算F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return precision, recall, f1
    
    def evaluate_intent_recognition(self, queries: List[str], expected_intents: List[str], 
                                   recognize_func) -> Dict[str, float]:
        """
        评估意图识别的准确性
        
        Args:
            queries: 查询列表
            expected_intents: 每个查询的期望意图
            recognize_func: 意图识别函数
            
        Returns:
            评估指标字典
        """
        if len(queries) != len(expected_intents):
            raise ValueError("查询列表和期望意图列表长度不匹配")
        
        start_time = time.time()
        
        correct_count = 0
        confidence_scores = []
        response_times = []
        
        for query, expected_intent in zip(queries, expected_intents):
            # 执行意图识别
            query_start = time.time()
            result = recognize_func(query)
            query_time = time.time() - query_start
            response_times.append(query_time)
            
            # 提取识别结果
            recognized_intent = None
            confidence = 0.0
            
            if isinstance(result, tuple) and len(result) >= 2:
                recognized_intent = result[0]
                if len(result) >= 2:
                    confidence = result[1]
            elif isinstance(result, dict):
                recognized_intent = result.get('intent')
                confidence = result.get('confidence', 0.0)
            
            # 检查是否正确
            if recognized_intent == expected_intent:
                correct_count += 1
            
            confidence_scores.append(confidence)
        
        # 计算准确率
        accuracy = correct_count / len(queries)
        avg_confidence = np.mean(confidence_scores)
        avg_response_time = np.mean(response_times)
        total_time = time.time() - start_time
        
        # 构建评估结果
        metrics = {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_response_time': avg_response_time,
            'total_evaluation_time': total_time,
            'num_queries': len(queries)
        }
        
        # 保存评估结果
        self.eval_results.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'evaluation_type': 'intent_recognition',
            'num_queries': len(queries)
        })
        
        logger.info(f"意图识别评估完成: 准确率={accuracy:.4f}, 平均置信度={avg_confidence:.4f}, "
                   f"平均响应时间={avg_response_time:.4f}s")
        
        return metrics
    
    def evaluate_system_performance(self, test_queries: List[str], retrieve_func, 
                                   concurrent_users: int = 10, iterations: int = 100) -> Dict[str, float]:
        """
        评估系统的性能
        
        Args:
            test_queries: 测试查询列表
            retrieve_func: 检索函数
            concurrent_users: 并发用户数
            iterations: 每个用户的迭代次数
            
        Returns:
            性能指标字典
        """
        import concurrent.futures
        
        start_time = time.time()
        
        # 执行性能测试
        response_times = []
        errors = []
        
        def user_task(user_id):
            user_errors = []
            user_times = []
            
            for i in range(iterations):
                # 随机选择一个查询
                query = np.random.choice(test_queries)
                
                try:
                    task_start = time.time()
                    retrieve_func(query)
                    task_time = time.time() - task_start
                    user_times.append(task_time)
                except Exception as e:
                    user_errors.append(str(e))
            
            return user_times, user_errors
        
        # 使用线程池模拟并发用户
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_task, i) for i in range(concurrent_users)]
            
            for future in concurrent.futures.as_completed(futures):
                user_times, user_errors = future.result()
                response_times.extend(user_times)
                errors.extend(user_errors)
        
        # 计算性能指标
        avg_response_time = np.mean(response_times) if response_times else 0
        p95_response_time = np.percentile(response_times, 95) if response_times else 0
        p99_response_time = np.percentile(response_times, 99) if response_times else 0
        throughput = (concurrent_users * iterations) / (time.time() - start_time)
        error_rate = len(errors) / (concurrent_users * iterations)
        
        # 构建性能评估结果
        metrics = {
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'p99_response_time': p99_response_time,
            'throughput': throughput,
            'error_rate': error_rate,
            'concurrent_users': concurrent_users,
            'iterations_per_user': iterations,
            'total_requests': concurrent_users * iterations,
            'total_time': time.time() - start_time
        }
        
        # 保存评估结果
        self.eval_results.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'evaluation_type': 'performance',
            'concurrent_users': concurrent_users,
            'iterations': iterations
        })
        
        logger.info(f"性能评估完成: 平均响应时间={avg_response_time:.4f}s, "
                   f"P95响应时间={p95_response_time:.4f}s, "
                   f"P99响应时间={p99_response_time:.4f}s, "
                   f"吞吐量={throughput:.2f} QPS, "
                   f"错误率={error_rate:.4f}")
        
        return metrics
    
    def _save_evaluation_results(self, metrics: Dict[str, float]):
        """
        保存评估结果到文件
        
        Args:
            metrics: 评估指标字典
        """
        timestamp = int(time.time())
        filename = self.eval_dir / f"evaluation_{timestamp}.json"
        
        result = {
            'timestamp': timestamp,
            'metrics': metrics,
            'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存到: {filename}")
    
    def get_evaluation_history(self) -> List[Dict]:
        """
        获取评估历史
        
        Returns:
            评估历史列表
        """
        return self.eval_results
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """
        获取指标历史
        
        Returns:
            指标历史字典
        """
        return self.metrics_history
    
    def generate_evaluation_report(self, output_file: str = None) -> str:
        """
        生成评估报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            评估报告内容
        """
        if not self.eval_results:
            return "暂无评估结果"
        
        # 生成报告
        report = f"# 检索系统评估报告\n"
        report += f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 汇总评估结果
        report += "## 评估摘要\n"
        report += f"总评估次数: {len(self.eval_results)}\n\n"
        
        # 分析指标历史
        if self.metrics_history:
            report += "## 指标分析\n"
            for metric_name, values in self.metrics_history.items():
                if values:
                    avg_value = np.mean(values)
                    max_value = np.max(values)
                    min_value = np.min(values)
                    report += f"- {metric_name}: 平均值={avg_value:.4f}, 最大值={max_value:.4f}, 最小值={min_value:.4f}\n"
        
        # 最近的评估结果
        if self.eval_results:
            report += "\n## 最近评估结果\n"
            latest_result = self.eval_results[-1]
            report += f"时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_result['timestamp']))}\n"
            report += "指标:\n"
            for key, value in latest_result['metrics'].items():
                report += f"- {key}: {value:.4f}\n"
        
        # 保存报告
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"评估报告已保存到: {output_path}")
        
        return report


# 全局评估器实例
evaluator = RetrievalEvaluator()
