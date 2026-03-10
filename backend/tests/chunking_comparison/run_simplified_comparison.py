#!/usr/bin/env python3
"""
简化版文档切分方法对比测试

使用基础评估方法替代RAGAS：
- 关键词召回率
- 关键词精确率
- 上下文相似度
- LLM答案评估

测试流程：
1. 文档切分阶段：两种方法切分同一批文档
2. 向量存储阶段：BGE-M3 嵌入 + FAISS 存储
3. 评估测试阶段：基础指标评估
4. 测试报告：生成 Markdown 报告
"""

import json
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class SimplifiedEvaluator:
    """简化版评估器 - 不依赖RAGAS"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def calculate_keyword_metrics(
        self,
        retrieved_contexts: List[str],
        expected_keywords: List[str]
    ) -> Dict[str, float]:
        """计算关键词相关指标"""
        if not expected_keywords:
            return {"keyword_recall": 0.0, "keyword_precision": 0.0, "keyword_f1": 0.0}
        
        # 合并所有检索上下文
        all_context = " ".join(retrieved_contexts).lower()
        
        # 计算关键词命中
        hits = 0
        for keyword in expected_keywords:
            if keyword.lower() in all_context:
                hits += 1
        
        recall = hits / len(expected_keywords) if expected_keywords else 0.0
        
        # 计算精确率（基于检索内容中关键词密度）
        total_keywords_in_context = sum(
            all_context.count(kw.lower()) for kw in expected_keywords
        )
        context_length = len(all_context)
        precision = min(1.0, total_keywords_in_context / (len(expected_keywords) * 3)) if context_length > 0 else 0.0
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "keyword_recall": recall,
            "keyword_precision": precision,
            "keyword_f1": f1
        }
    
    def calculate_context_similarity(
        self,
        query: str,
        retrieved_contexts: List[str],
        embedding_service
    ) -> float:
        """计算查询与检索上下文的相似度"""
        try:
            # 编码查询
            query_embedding = embedding_service.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            # 编码上下文
            context_text = " ".join(retrieved_contexts[:3])  # 只取前3个上下文
            context_embedding = embedding_service.encode([context_text])
            context_embedding = np.array(context_embedding).astype('float32')
            
            # 计算余弦相似度
            similarity = np.dot(query_embedding[0], context_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(context_embedding[0])
            )
            
            return float(similarity)
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0
    
    def evaluate_answer_with_llm(
        self,
        query: str,
        answer: str,
        ground_truth: str
    ) -> Dict[str, float]:
        """使用LLM评估答案质量"""
        if not self.llm_client:
            return {"faithfulness": 0.5, "answer_relevance": 0.5}
        
        prompt = f"""请评估以下问答对的质量，给出0-1的分数。

【问题】{query}

【生成的答案】{answer}

【标准答案】{ground_truth}

请评估：
1. 忠实性: 答案是否准确反映了问题，没有虚假信息（0-1）
2. 相关性: 答案是否完整回答了问题（0-1）

请严格按以下JSON格式返回，不要返回其他内容：
{{"faithfulness": 0.8, "relevance": 0.9}}"""

        try:
            result = self.llm_client.generate(prompt)
            if isinstance(result, dict):
                response_text = result.get('text', str(result))
            else:
                response_text = str(result)
            
            # 解析JSON
            json_match = re.search(r'\{[^{}]*"faithfulness"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "faithfulness": float(data.get('faithfulness', 0.5)),
                    "answer_relevance": float(data.get('relevance', 0.5))
                }
        except Exception as e:
            logger.warning(f"LLM评估失败: {e}")
        
        return {"faithfulness": 0.5, "answer_relevance": 0.5}


class SimplifiedChunkingTest:
    """简化版切分对比测试"""
    
    def __init__(self, sample_limit: int = 30):
        self.docs_dir = PROJECT_ROOT / "data" / "docs"
        self.test_data_path = PROJECT_ROOT / "tests" / "chunking_comparison" / "ragas_eval_dataset.json"
        self.output_dir = PROJECT_ROOT / "tests" / "chunking_comparison" / "comparison_output"
        self.report_path = PROJECT_ROOT.parent / "test_reports" / "report.md"
        
        # 向量数据库路径
        self.vector_db_layered = self.output_dir / "vector_db_layered"
        self.vector_db_naive = self.output_dir / "vector_db_naive"
        
        # 切分参数
        self.chunk_size = 512
        self.overlap_percent = 0.1
        self.top_k = 5
        self.sample_limit = sample_limit
        
        # 组件
        self.embedding_service = None
        self.llm_client = None
        self.evaluator = None
        
        # 结果
        self.layered_chunks = []
        self.naive_chunks = []
        self.layered_index = None
        self.naive_index = None
        self.layered_metadata = {}
        self.naive_metadata = {}
        
    def load_documents(self) -> Dict[str, str]:
        """加载所有文档"""
        logger.info("="*60)
        logger.info("加载文档...")
        logger.info("="*60)
        
        documents = {}
        for file_path in self.docs_dir.glob("*.md"):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents[file_path.stem] = f.read()
        
        logger.info(f"加载了 {len(documents)} 个文档")
        return documents
    
    def run_layered_chunking(self, documents: Dict[str, str]) -> List[Dict]:
        """执行分层智能切分"""
        logger.info("")
        logger.info("="*60)
        logger.info("执行分层智能切分...")
        logger.info("="*60)
        
        from services.layered_chunker import LayeredChunker, LayeredChunkConfig
        
        config = LayeredChunkConfig(
            max_chunk_size=int(self.chunk_size * 1.5),
            overlap=int(self.chunk_size * self.overlap_percent),
            min_chunk_size=50,
            preserve_table=True,
            preserve_flow=True,
            preserve_code=True
        )
        
        chunker = LayeredChunker(config)
        all_chunks = []
        
        for doc_name, content in documents.items():
            chunks = chunker.chunk(content, doc_name)
            for chunk in chunks:
                chunk['doc_name'] = doc_name
            all_chunks.extend(chunks)
            logger.info(f"  {doc_name}: {len(chunks)} 个片段")
        
        logger.info(f"分层切分总计: {len(all_chunks)} 个 chunks")
        self.layered_chunks = all_chunks
        return all_chunks
    
    def run_naive_chunking(self, documents: Dict[str, str]) -> List[Dict]:
        """执行分隔符切分"""
        logger.info("")
        logger.info("="*60)
        logger.info("执行分隔符切分...")
        logger.info("="*60)
        
        from services.chunker import RAGFlowChunker
        
        chunker = RAGFlowChunker()
        all_chunks = []
        
        for doc_name, content in documents.items():
            try:
                chunks = chunker._naive_merge(
                    content,
                    chunk_token_num=self.chunk_size,
                    delimiter="\n\n",
                    overlapped_percent=self.overlap_percent
                )
                for i, chunk_content in enumerate(chunks):
                    all_chunks.append({
                        'content': chunk_content,
                        'doc_name': doc_name,
                        'chunk_id': f"{doc_name}_naive_{i}",
                        'type': 'text'
                    })
                logger.info(f"  {doc_name}: {len(chunks)} 个片段")
            except Exception as e:
                logger.error(f"切分 {doc_name} 失败: {e}")
        
        logger.info(f"分隔符切分总计: {len(all_chunks)} 个 chunks")
        self.naive_chunks = all_chunks
        return all_chunks
    
    def _load_embedding_model(self):
        """加载嵌入模型"""
        if self.embedding_service is not None:
            return
        
        from services.embedding import EmbeddingService
        from models import EmbeddingConfig, EmbeddingModelType
        from config import settings
        
        logger.info("加载嵌入模型...")
        
        self.embedding_service = EmbeddingService()
        config = EmbeddingConfig(
            model_name=settings.embedding_model_name,
            model_path=settings.embedding_model_name,
            model_type=EmbeddingModelType.BGE,
            device="cuda"
        )
        self.embedding_service.load_model(config)
        
        logger.info(f"嵌入模型加载完成，维度: {self.embedding_service.get_dimension()}")
    
    def _load_llm(self):
        """加载LLM模型"""
        if self.llm_client is not None:
            return
        
        try:
            from services.rag_generator import LocalLLMClient
            from models import GenerationConfig
            from config import settings
            
            logger.info("加载 LLM 模型...")
            
            config = GenerationConfig(
                llm_provider="local",
                local_model_path=settings.local_llm_model_path,
                device=settings.local_llm_device,
                temperature=0.7,
                max_tokens=512
            )
            self.llm_client = LocalLLMClient(config)
            logger.info("LLM 模型加载完成")
        except Exception as e:
            logger.warning(f"LLM 模型加载失败: {e}，将跳过LLM评估")
            self.llm_client = None
    
    def build_vector_index(self, chunks: List[Dict], index_path: Path) -> Tuple[Any, Dict]:
        """构建向量索引"""
        self._load_embedding_model()
        
        import faiss
        
        logger.info(f"构建向量索引: {index_path}")
        
        # 生成嵌入向量
        texts = [c['content'] for c in chunks]
        embeddings = self.embedding_service.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        # 构建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.add(embeddings)
        
        # 保存索引
        index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path / "index.faiss"))
        
        # 保存元数据
        metadata = {}
        for i, chunk in enumerate(chunks):
            metadata[str(i)] = {
                'content': chunk['content'],
                'doc_name': chunk.get('doc_name', ''),
                'chunk_id': chunk.get('chunk_id', str(i)),
                'type': chunk.get('type', 'text')
            }
        
        with open(index_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"索引构建完成: {len(chunks)} 个向量")
        
        return index, metadata
    
    def build_all_indexes(self):
        """构建所有向量索引"""
        logger.info("")
        logger.info("="*60)
        logger.info("构建向量索引...")
        logger.info("="*60)
        
        # 分层切分索引
        self.layered_index, self.layered_metadata = self.build_vector_index(
            self.layered_chunks, self.vector_db_layered
        )
        
        # 分隔符切分索引
        self.naive_index, self.naive_metadata = self.build_vector_index(
            self.naive_chunks, self.vector_db_naive
        )
    
    def search(self, query: str, index: Any, metadata: Dict, top_k: int = 5) -> List[Dict]:
        """检索相关上下文"""
        query_embedding = self.embedding_service.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            idx_str = str(idx)
            if idx_str in metadata:
                result = metadata[idx_str].copy()
                result['score'] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """使用LLM生成答案"""
        if not self.llm_client:
            return "LLM未加载"
        
        context_text = "\n\n---\n\n".join([f"[参考文档{i+1}]\n{ctx}" for i, ctx in enumerate(contexts[:3])])
        
        prompt = f"""请根据以下参考文档回答用户问题。如果参考文档中没有相关信息，请说明"文档中未找到相关信息"。

用户问题：{query}

参考文档：
{context_text}

请给出简洁准确的回答："""

        try:
            result = self.llm_client.generate(prompt)
            if isinstance(result, dict):
                return result.get('text', str(result)).strip()
            return str(result).strip()
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return "生成答案时发生错误"
    
    def run_evaluation(self) -> Dict[str, Any]:
        """运行简化版评估"""
        logger.info("")
        logger.info("="*60)
        logger.info("运行简化版评估...")
        logger.info("="*60)
        
        self._load_llm()
        self.evaluator = SimplifiedEvaluator(self.llm_client)
        
        # 加载测试数据
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        samples = test_data['samples']
        
        # 限制样本数量
        if self.sample_limit and self.sample_limit < len(samples):
            samples = samples[:self.sample_limit]
            logger.info(f"样本数量限制为: {self.sample_limit}")
        
        logger.info(f"加载了 {len(samples)} 个测试样本")
        
        # 结果存储
        results = {
            "layered": {"samples": [], "statistics": defaultdict(list)},
            "naive": {"samples": [], "statistics": defaultdict(list)}
        }
        
        # 对每个样本进行评估
        for i, sample in enumerate(samples):
            query = sample['question']
            ground_truth = sample['ground_truth']
            expected_keywords = sample.get('metadata', {}).get('expected_keywords', [])
            
            if (i + 1) % 5 == 0:
                logger.info(f"处理进度: {i+1}/{len(samples)}")
            
            # 分层切分评估
            layered_contexts = self.search(query, self.layered_index, self.layered_metadata, self.top_k)
            layered_context_texts = [c['content'] for c in layered_contexts]
            
            # 关键词指标
            layered_keyword_metrics = self.evaluator.calculate_keyword_metrics(
                layered_context_texts, expected_keywords
            )
            
            # 相似度
            layered_similarity = self.evaluator.calculate_context_similarity(
                query, layered_context_texts, self.embedding_service
            )
            
            # 生成答案
            layered_answer = self.generate_answer(query, layered_context_texts)
            
            # LLM评估
            layered_llm_metrics = self.evaluator.evaluate_answer_with_llm(
                query, layered_answer, ground_truth
            )
            
            # 合并指标
            layered_scores = {
                **layered_keyword_metrics,
                "context_similarity": layered_similarity,
                **layered_llm_metrics
            }
            
            results["layered"]["samples"].append({
                "query": query,
                "answer": layered_answer[:200],  # 截断答案
                "scores": layered_scores
            })
            for metric, score in layered_scores.items():
                results["layered"]["statistics"][metric].append(score)
            
            # 分隔符切分评估
            naive_contexts = self.search(query, self.naive_index, self.naive_metadata, self.top_k)
            naive_context_texts = [c['content'] for c in naive_contexts]
            
            naive_keyword_metrics = self.evaluator.calculate_keyword_metrics(
                naive_context_texts, expected_keywords
            )
            
            naive_similarity = self.evaluator.calculate_context_similarity(
                query, naive_context_texts, self.embedding_service
            )
            
            naive_answer = self.generate_answer(query, naive_context_texts)
            
            naive_llm_metrics = self.evaluator.evaluate_answer_with_llm(
                query, naive_answer, ground_truth
            )
            
            naive_scores = {
                **naive_keyword_metrics,
                "context_similarity": naive_similarity,
                **naive_llm_metrics
            }
            
            results["naive"]["samples"].append({
                "query": query,
                "answer": naive_answer[:200],
                "scores": naive_scores
            })
            for metric, score in naive_scores.items():
                results["naive"]["statistics"][metric].append(score)
        
        # 计算统计信息
        for method in ["layered", "naive"]:
            for metric in results[method]["statistics"]:
                scores = results[method]["statistics"][metric]
                if scores:
                    results[method]["statistics"][metric] = {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "count": len(scores)
                    }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成 Markdown 测试报告"""
        logger.info("")
        logger.info("="*60)
        logger.info("生成测试报告...")
        logger.info("="*60)
        
        layered_stats = results["layered"]["statistics"]
        naive_stats = results["naive"]["statistics"]
        
        report = f"""# 文档切分方法对比测试报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、测试配置

| 参数 | 值 |
|------|------|
| 测试文档数 | 18 个 |
| 测试样本数 | {len(results['layered']['samples'])} |
| 分层切分 chunks | {len(self.layered_chunks)} |
| 分隔符切分 chunks | {len(self.naive_chunks)} |
| chunk_size | {self.chunk_size} tokens |
| overlap | {self.overlap_percent*100:.0f}% |
| 检索 top_k | {self.top_k} |

## 二、总体指标对比

| 指标 | 分层切分 | 分隔符切分 | 差异 | 说明 |
|------|----------|------------|------|------|
"""
        
        metrics_display = [
            ("keyword_recall", "关键词召回率", "检索内容覆盖关键词的比例"),
            ("keyword_precision", "关键词精确率", "检索内容中关键词的密度"),
            ("keyword_f1", "关键词F1", "召回率和精确率的调和平均"),
            ("context_similarity", "上下文相似度", "查询与检索内容的语义相似度"),
            ("faithfulness", "忠实性", "答案是否基于上下文"),
            ("answer_relevance", "答案相关性", "答案是否回答问题"),
        ]
        
        for metric_key, metric_name, metric_desc in metrics_display:
            layered_val = layered_stats.get(metric_key, {}).get('mean', 0)
            naive_val = naive_stats.get(metric_key, {}).get('mean', 0)
            diff = layered_val - naive_val
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            report += f"| {metric_name} | {layered_val:.4f} | {naive_val:.4f} | {diff_str} | {metric_desc} |\n"
        
        report += f"""
## 三、详细统计数据

### 3.1 分层智能切分

| 指标 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
"""
        
        for metric_key, metric_name, _ in metrics_display:
            stats = layered_stats.get(metric_key, {})
            if isinstance(stats, dict) and 'mean' in stats:
                report += f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n"
        
        report += f"""
### 3.2 分隔符切分

| 指标 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
"""
        
        for metric_key, metric_name, _ in metrics_display:
            stats = naive_stats.get(metric_key, {})
            if isinstance(stats, dict) and 'mean' in stats:
                report += f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n"
        
        # 性能差异分析
        report += f"""
## 四、性能差异分析

### 4.1 各指标差异对比

| 指标 | 更优方法 | 差异幅度 |
|------|----------|----------|
"""
        
        for metric_key, metric_name, _ in metrics_display:
            layered_val = layered_stats.get(metric_key, {}).get('mean', 0)
            naive_val = naive_stats.get(metric_key, {}).get('mean', 0)
            diff = layered_val - naive_val
            winner = "分层切分" if diff > 0 else ("分隔符切分" if diff < 0 else "持平")
            report += f"| {metric_name} | {winner} | {diff:+.4f} |\n"
        
        report += f"""
### 4.2 可能原因分析

1. **关键词召回率**
   - 分层切分：按语义边界切分，保持信息完整性
   - 分隔符切分：chunk数量较少，但单个chunk信息量大

2. **上下文相似度**
   - 取决于检索内容与查询的语义匹配程度
   - 分层切分chunk更精细，可能提高检索精度

3. **忠实性与相关性**
   - 取决于LLM生成能力
   - 上下文质量影响答案质量

## 五、方法选择建议

### 5.1 综合评估

"""
        
        # 计算综合得分
        layered_total = sum(layered_stats.get(m, {}).get('mean', 0) for m, _, _ in metrics_display)
        naive_total = sum(naive_stats.get(m, {}).get('mean', 0) for m, _, _ in metrics_display)
        
        if layered_total > naive_total:
            recommendation = "分层智能切分在本次测试中整体表现更优，建议优先使用。"
        elif naive_total > layered_total:
            recommendation = "分隔符切分在本次测试中整体表现更优，建议优先使用。"
        else:
            recommendation = "两种方法各有优势，建议根据具体场景选择。"
        
        report += f"""{recommendation}

### 5.2 场景选择建议

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 结构化文档 | 分层切分 | 保留章节结构，上下文清晰 |
| 短文本/碎片化文档 | 分隔符切分 | 简单高效，chunk数量少 |
| 需要表格完整性 | 分层切分 | 保护表格不拆分 |
| 跨章节关联查询 | 分隔符切分 | 无章节边界限制 |

---

*报告由 RAG 系统切分对比测试工具自动生成*
"""
        
        return report
    
    def run(self):
        """运行完整测试流程"""
        start_time = time.time()
        
        try:
            # 1. 加载文档
            documents = self.load_documents()
            
            # 2. 执行切分
            self.run_layered_chunking(documents)
            self.run_naive_chunking(documents)
            
            # 3. 构建向量索引
            self.build_all_indexes()
            
            # 4. 运行评估
            results = self.run_evaluation()
            
            # 5. 生成报告
            report = self.generate_report(results)
            
            # 保存详细结果
            self.output_dir.mkdir(parents=True, exist_ok=True)
            results_path = self.output_dir / "detailed_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            # 保存报告
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            total_time = time.time() - start_time
            logger.info("")
            logger.info("="*60)
            logger.info("测试完成!")
            logger.info("="*60)
            logger.info(f"总耗时: {total_time:.2f} 秒")
            logger.info(f"测试报告: {self.report_path}")
            logger.info(f"详细结果: {results_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"测试执行失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    test = SimplifiedChunkingTest(sample_limit=30)
    test.run()


if __name__ == "__main__":
    main()
