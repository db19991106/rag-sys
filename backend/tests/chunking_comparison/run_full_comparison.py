#!/usr/bin/env python3
"""
文档切分方法全面对比测试

对比分层智能切分 vs 分隔符切分在 RAGAS 四个核心指标上的表现：
- Context Precision (上下文精确度)
- Context Recall (上下文召回率)
- Faithfulness (忠实性)
- Answer Relevance (答案相关性)

测试流程：
1. 文档切分阶段：两种方法切分同一批文档
2. 向量存储阶段：BGE-M3 嵌入 + FAISS 存储
3. 评估测试阶段：RAGAS 评估
4. 测试报告：生成 Markdown 报告
"""

import json
import os
import sys
import time
import shutil
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


class ChunkingComparisonTest:
    """文档切分方法对比测试"""
    
    def __init__(self, sample_limit: int = None):
        self.docs_dir = PROJECT_ROOT / "data" / "docs"
        self.test_data_path = PROJECT_ROOT / "tests" / "chunking_comparison" / "ragas_eval_dataset.json"
        self.output_dir = PROJECT_ROOT / "tests" / "chunking_comparison" / "comparison_output"
        self.report_path = PROJECT_ROOT.parent / "test_reports" / "report.md"
        
        # 向量数据库路径
        self.vector_db_layered = self.output_dir / "vector_db_layered"
        self.vector_db_naive = self.output_dir / "vector_db_naive"
        
        # 切分参数
        self.chunk_size = 512  # token 数
        self.overlap_percent = 0.1
        self.top_k = 5
        
        # 样本限制
        self.sample_limit = sample_limit
        
        # 组件
        self.embedding_service = None
        self.llm_client = None
        self.ragas_evaluator = None
        
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
        for name, content in documents.items():
            logger.info(f"  - {name}: {len(content)} 字符")
        
        return documents
    
    def run_layered_chunking(self, documents: Dict[str, str]) -> List[Dict]:
        """执行分层智能切分"""
        logger.info("")
        logger.info("="*60)
        logger.info("执行分层智能切分...")
        logger.info("="*60)
        
        from services.layered_chunker import LayeredChunker, LayeredChunkConfig
        
        config = LayeredChunkConfig(
            max_chunk_size=int(self.chunk_size * 1.5),  # ~768 字符
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
        
        config = EmbeddingConfig(
            model_name=settings.embedding_model_name,
            model_path=settings.embedding_model_name,  # 使用相同路径
            model_type=EmbeddingModelType.BGE,
            device="cuda"  # 默认使用 GPU
        )
        self.embedding_service = EmbeddingService()
        self.embedding_service.load_model(config)
        
        logger.info(f"嵌入模型加载完成，维度: {self.embedding_service.get_dimension()}")
    
    def build_vector_index(self, chunks: List[Dict], index_path: Path) -> Tuple[Any, Dict]:
        """构建向量索引"""
        self._load_embedding_model()
        
        import faiss
        
        logger.info(f"构建向量索引: {index_path}")
        
        # 生成嵌入向量
        texts = [c['content'] for c in chunks]
        embeddings = self.embedding_service.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        # 构建 FAISS 索引
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
        # 生成查询向量
        query_embedding = self.embedding_service.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # 搜索
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            idx_str = str(idx)
            if idx_str in metadata:
                result = metadata[idx_str].copy()
                result['score'] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def _load_llm(self):
        """加载 LLM 模型"""
        if self.llm_client is not None:
            return
        
        from services.rag_generator import LocalLLMClient
        from models import GenerationConfig
        from config import settings
        
        logger.info("加载 LLM 模型...")
        
        config = GenerationConfig(
            llm_provider="local",
            llm_model=settings.llm_model,  # 必须设置，否则 _check_memory_requirement 无法识别模型大小
            local_model_path=settings.local_llm_model_path,
            device=settings.local_llm_device,
            temperature=0.7,
            max_tokens=512
        )
        self.llm_client = LocalLLMClient(config)
        
        logger.info("LLM 模型加载完成")
    
    def _load_ragas_evaluator(self):
        """加载 RAGAS 评估器"""
        if self.ragas_evaluator is not None:
            return
        
        from services.ragas_integration import create_ragas_evaluator
        from config import settings
        
        self._load_llm()
        
        logger.info("初始化 RAGAS 评估器...")
        # 传递模型路径和设备信息，用于创建 LangChain HuggingFacePipeline
        self.ragas_evaluator = create_ragas_evaluator(
            llm_client=self.llm_client,
            model_path=settings.local_llm_model_path,
            device=settings.local_llm_device
        )
        
        logger.info("RAGAS 评估器初始化完成")
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """使用 LLM 生成答案"""
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
        """运行 RAGAS 评估"""
        logger.info("")
        logger.info("="*60)
        logger.info("运行 RAGAS 评估...")
        logger.info("="*60)
        
        self._load_ragas_evaluator()
        
        # 加载测试数据
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        samples = test_data['samples']
        
        # 限制样本数量
        if self.sample_limit and self.sample_limit < len(samples):
            samples = samples[:self.sample_limit]
            logger.info(f"样本数量限制为: {self.sample_limit}")
        
        logger.info(f"加载了 {len(samples)} 个测试样本")
        
        # 评估指标
        metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        
        # 结果存储
        results = {
            "layered": {"samples": [], "statistics": defaultdict(list)},
            "naive": {"samples": [], "statistics": defaultdict(list)}
        }
        
        # 对每个样本进行评估
        for i, sample in enumerate(samples):
            query = sample['question']
            ground_truth = sample['ground_truth']
            
            if (i + 1) % 20 == 0:
                logger.info(f"处理进度: {i+1}/{len(samples)}")
            
            # 分层切分评估
            layered_contexts = self.search(query, self.layered_index, self.layered_metadata, self.top_k)
            layered_context_texts = [c['content'] for c in layered_contexts]
            layered_answer = self.generate_answer(query, layered_context_texts)
            
            try:
                layered_result = self.ragas_evaluator.evaluate_single(
                    query=query,
                    answer=layered_answer,
                    contexts=layered_context_texts,
                    ground_truth=ground_truth,
                    metrics=metrics
                )
                layered_scores = layered_result.get('scores', {})
            except Exception as e:
                logger.warning(f"分层切分评估失败 (样本 {i}): {e}")
                layered_scores = {}
            
            results["layered"]["samples"].append({
                "query": query,
                "answer": layered_answer,
                "contexts": layered_context_texts,
                "scores": layered_scores
            })
            for metric, score in layered_scores.items():
                if score is not None:
                    results["layered"]["statistics"][metric].append(score)
            
            # 分隔符切分评估
            naive_contexts = self.search(query, self.naive_index, self.naive_metadata, self.top_k)
            naive_context_texts = [c['content'] for c in naive_contexts]
            naive_answer = self.generate_answer(query, naive_context_texts)
            
            try:
                naive_result = self.ragas_evaluator.evaluate_single(
                    query=query,
                    answer=naive_answer,
                    contexts=naive_context_texts,
                    ground_truth=ground_truth,
                    metrics=metrics
                )
                naive_scores = naive_result.get('scores', {})
            except Exception as e:
                logger.warning(f"分隔符切分评估失败 (样本 {i}): {e}")
                naive_scores = {}
            
            results["naive"]["samples"].append({
                "query": query,
                "answer": naive_answer,
                "contexts": naive_context_texts,
                "scores": naive_scores
            })
            for metric, score in naive_scores.items():
                if score is not None:
                    results["naive"]["statistics"][metric].append(score)
        
        # 计算统计信息
        for method in ["layered", "naive"]:
            for metric in metrics:
                scores = results[method]["statistics"].get(metric, [])
                if scores:
                    results[method]["statistics"][metric] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "min": np.min(scores),
                        "max": np.max(scores),
                        "count": len(scores)
                    }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]):
        """生成 Markdown 测试报告"""
        logger.info("")
        logger.info("="*60)
        logger.info("生成测试报告...")
        logger.info("="*60)
        
        # 确保目录存在
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        layered_stats = results["layered"]["statistics"]
        naive_stats = results["naive"]["statistics"]
        
        report = f"""# 文档切分方法对比测试报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、测试配置

| 参数 | 值 |
|------|------|
| 测试文档数 | {len(self.layered_chunks) + len(self.naive_chunks)} chunks |
| 测试样本数 | {len(results['layered']['samples'])} |
| 分层切分 chunks | {len(self.layered_chunks)} |
| 分隔符切分 chunks | {len(self.naive_chunks)} |
| chunk_size | {self.chunk_size} tokens |
| overlap | {self.overlap_percent*100:.0f}% |
| 检索 top_k | {self.top_k} |

## 二、总体指标对比

| 指标 | 分层切分 | 分隔符切分 | 差异 | 说明 |
|------|----------|------------|------|------|
| Context Precision | {layered_stats.get('context_precision', {}).get('mean', 'N/A'):.4f} | {naive_stats.get('context_precision', {}).get('mean', 'N/A'):.4f} | {(layered_stats.get('context_precision', {}).get('mean', 0) - naive_stats.get('context_precision', {}).get('mean', 0)):.4f} | 检索结果的相关性 |
| Context Recall | {layered_stats.get('context_recall', {}).get('mean', 'N/A'):.4f} | {naive_stats.get('context_recall', {}).get('mean', 'N/A'):.4f} | {(layered_stats.get('context_recall', {}).get('mean', 0) - naive_stats.get('context_recall', {}).get('mean', 0)):.4f} | 检索覆盖度 |
| Faithfulness | {layered_stats.get('faithfulness', {}).get('mean', 'N/A'):.4f} | {naive_stats.get('faithfulness', {}).get('mean', 'N/A'):.4f} | {(layered_stats.get('faithfulness', {}).get('mean', 0) - naive_stats.get('faithfulness', {}).get('mean', 0)):.4f} | 答案是否基于上下文 |
| Answer Relevance | {layered_stats.get('answer_relevancy', {}).get('mean', 'N/A'):.4f} | {naive_stats.get('answer_relevancy', {}).get('mean', 'N/A'):.4f} | {(layered_stats.get('answer_relevancy', {}).get('mean', 0) - naive_stats.get('answer_relevancy', {}).get('mean', 0)):.4f} | 答案是否回答问题 |

## 三、详细统计数据

### 3.1 分层智能切分

| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 样本数 |
|------|------|--------|--------|--------|--------|
"""
        
        for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            stats = layered_stats.get(metric, {})
            if isinstance(stats, dict) and 'mean' in stats:
                report += f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |\n"
        
        report += f"""
### 3.2 分隔符切分

| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 样本数 |
|------|------|--------|--------|--------|--------|
"""
        
        for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            stats = naive_stats.get(metric, {})
            if isinstance(stats, dict) and 'mean' in stats:
                report += f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |\n"
        
        # 性能差异分析
        report += f"""
## 四、性能差异分析

### 4.1 各指标差异对比

"""
        
        # 计算各指标的胜者
        winners = {}
        for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            layered_mean = layered_stats.get(metric, {}).get('mean', 0)
            naive_mean = naive_stats.get(metric, {}).get('mean', 0)
            if layered_mean > naive_mean:
                winners[metric] = "分层切分"
            elif naive_mean > layered_mean:
                winners[metric] = "分隔符切分"
            else:
                winners[metric] = "持平"
        
        report += "| 指标 | 更优方法 | 差异幅度 |\n"
        report += "|------|----------|----------|\n"
        for metric, winner in winners.items():
            diff = layered_stats.get(metric, {}).get('mean', 0) - naive_stats.get(metric, {}).get('mean', 0)
            report += f"| {metric} | {winner} | {diff:+.4f} |\n"
        
        # 原因分析
        report += f"""
### 4.2 可能原因分析

1. **上下文精确度**
   - 分层切分优势：按语义边界切分，每个 chunk 主题明确
   - 分隔符切分优势：段落完整，信息密度高

2. **上下文召回率**
   - 分层切分：可能因章节边界导致跨章节信息分散
   - 分隔符切分：按段落切分，相关信息更集中

3. **忠实性**
   - 取决于检索上下文是否包含生成答案所需信息
   - 与 chunk 大小和内容完整性相关

4. **答案相关性**
   - 取决于 LLM 生成能力
   - 上下文质量影响答案质量

## 五、方法选择建议

"""
        
        # 综合评估
        layered_wins = sum(1 for w in winners.values() if w == "分层切分")
        naive_wins = sum(1 for w in winners.values() if w == "分隔符切分")
        
        if layered_wins > naive_wins:
            recommendation = "分层智能切分在本次测试中整体表现更优，建议优先使用。"
        elif naive_wins > layered_wins:
            recommendation = "分隔符切分在本次测试中整体表现更优，建议优先使用。"
        else:
            recommendation = "两种方法各有优势，建议根据具体场景选择。"
        
        report += f"""### 5.1 综合评估

{recommendation}

### 5.2 场景选择建议

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 结构化文档 | 分层切分 | 保留章节结构，上下文清晰 |
| 短文本/碎片化文档 | 分隔符切分 | 简单高效，召回率高 |
| 需要表格完整性 | 分层切分 | 保护表格不拆分 |
| 跨章节关联查询 | 分隔符切分 | 无章节边界限制 |

---

*报告由 RAG 系统切分对比测试工具自动生成*
"""
        
        # 保存报告
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"测试报告已保存: {self.report_path}")
        
        return report
    
    def run(self):
        """运行完整测试流程"""
        start_time = time.time()
        
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
        results_path = self.output_dir / "detailed_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        total_time = time.time() - start_time
        logger.info("")
        logger.info("="*60)
        logger.info("测试完成!")
        logger.info("="*60)
        logger.info(f"总耗时: {total_time:.2f} 秒")
        logger.info(f"测试报告: {self.report_path}")
        logger.info(f"详细结果: {results_path}")
        
        return results


def main():
    """主函数"""
    test = ChunkingComparisonTest(sample_limit=30)
    test.run()


if __name__ == "__main__":
    main()
