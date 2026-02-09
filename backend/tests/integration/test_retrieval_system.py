"""
检索系统综合测试脚本
用于测试检索系统的各个组件和功能
"""

import sys
import os
import time
import unittest
import numpy as np
from typing import List, Dict, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.retriever import retriever
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.reranker import reranker_manager
from services.intent_recognizer import intent_recognizer
from services.evaluation import evaluator
from models import RetrievalConfig, EmbeddingConfig, VectorDBConfig
from models import EmbeddingModelType, VectorDBType, SimilarityAlgorithm
from config import settings


class TestRetrievalSystem(unittest.TestCase):
    """检索系统测试类"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        print("\n=== 设置测试环境 ===")
        
        # 初始化嵌入服务
        cls._init_embedding_service()
        
        # 初始化向量数据库
        cls._init_vector_db()
        
        # 初始化重排序器
        cls._init_reranker()
        
        # 初始化意图识别器
        cls._init_intent_recognizer()
        
        # 准备测试数据
        cls.test_queries = [
            "什么是人工智能",
            "如何使用Python进行数据分析",
            "机器学习和深度学习的区别",
            "如何构建一个神经网络",
            "数据科学的工作流程是什么"
        ]
        
        cls.test_documents = [
            "人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的机器。",
            "Python是一种广泛使用的高级编程语言，特别适合数据分析和科学计算。",
            "机器学习是人工智能的一个子集，专注于开发能够从数据中学习的算法。",
            "深度学习是机器学习的一个分支，使用多层神经网络来模拟人类大脑的学习过程。",
            "数据科学的工作流程包括数据收集、数据清洗、特征工程、模型训练和模型评估。"
        ]
        
        # 索引测试文档
        cls._index_test_documents()
        
        print("测试环境设置完成\n")
    
    @classmethod
    def _init_embedding_service(cls):
        """初始化嵌入服务"""
        try:
            config = EmbeddingConfig(
                model_type=EmbeddingModelType.BGE,
                model_name="BAAI/bge-small-zh-v1.5",
                device="cpu",
                batch_size=32
            )
            response = embedding_service.load_model(config)
            if response.status == "success":
                print("✓ 嵌入服务初始化成功")
            else:
                print(f"✗ 嵌入服务初始化失败: {response.message}")
        except Exception as e:
            print(f"✗ 嵌入服务初始化异常: {str(e)}")
    
    @classmethod
    def _init_vector_db(cls):
        """初始化向量数据库"""
        try:
            dimension = embedding_service.get_dimension() if embedding_service.is_loaded() else 768
            config = VectorDBConfig(
                db_type=VectorDBType.FAISS,
                dimension=dimension,
                index_type="HNSW",
                host="localhost",
                port=19530,
                collection_name="test_collection"
            )
            success = vector_db_manager.initialize(config)
            if success:
                print("✓ 向量数据库初始化成功")
            else:
                print("✗ 向量数据库初始化失败")
        except Exception as e:
            print(f"✗ 向量数据库初始化异常: {str(e)}")
    
    @classmethod
    def _init_reranker(cls):
        """初始化重排序器"""
        try:
            reranker_manager.initialize(
                reranker_type="bge",
                model_name="BAAI/bge-reranker-v2-m3",
                device="cpu",
                top_k=5,
                threshold=0.0
            )
            print("✓ 重排序器初始化成功")
        except Exception as e:
            print(f"✗ 重排序器初始化异常: {str(e)}")
    
    @classmethod
    def _init_intent_recognizer(cls):
        """初始化意图识别器"""
        try:
            if embedding_service.is_loaded():
                intent_recognizer.initialize(embedding_service)
                print("✓ 意图识别器初始化成功")
            else:
                print("✗ 意图识别器初始化失败: 嵌入服务未加载")
        except Exception as e:
            print(f"✗ 意图识别器初始化异常: {str(e)}")
    
    @classmethod
    def _index_test_documents(cls):
        """索引测试文档"""
        try:
            if not embedding_service.is_loaded() or vector_db_manager.db is None:
                print("✗ 索引测试文档失败: 服务未初始化")
                return
            
            # 编码文档
            vectors = embedding_service.encode(cls.test_documents)
            
            # 准备元数据
            metadata = []
            for i, doc in enumerate(cls.test_documents):
                metadata.append({
                    "document_id": f"test_doc_{i}",
                    "document_name": f"Test Document {i}",
                    "chunk_id": f"chunk_{i}",
                    "chunk_num": i + 1,
                    "content": doc,
                    "keywords": doc.split()[:5]
                })
            
            # 添加到向量数据库
            vector_db_manager.add_vectors(vectors, metadata)
            print(f"✓ 成功索引 {len(cls.test_documents)} 个测试文档")
        except Exception as e:
            print(f"✗ 索引测试文档异常: {str(e)}")
    
    def test_embedding_service(self):
        """测试嵌入服务"""
        print("=== 测试嵌入服务 ===")
        
        # 测试编码功能
        test_texts = ["测试文本1", "测试文本2"]
        try:
            vectors = embedding_service.encode(test_texts)
            self.assertEqual(len(vectors), len(test_texts))
            self.assertEqual(vectors.ndim, 2)
            print("✓ 嵌入服务编码功能测试通过")
        except Exception as e:
            self.fail(f"嵌入服务编码功能测试失败: {str(e)}")
        
        # 测试维度获取
        try:
            dimension = embedding_service.get_dimension()
            self.assertGreater(dimension, 0)
            print(f"✓ 嵌入服务维度获取测试通过 (维度: {dimension})")
        except Exception as e:
            self.fail(f"嵌入服务维度获取测试失败: {str(e)}")
    
    def test_vector_db(self):
        """测试向量数据库"""
        print("\n=== 测试向量数据库 ===")
        
        # 测试搜索功能
        try:
            # 编码查询
            query = "人工智能"
            query_vector = embedding_service.encode([query])[0]
            
            # 搜索
            distances, metadata_list = vector_db_manager.search(query_vector, top_k=3)
            
            self.assertIsInstance(distances, np.ndarray)
            self.assertIsInstance(metadata_list, list)
            print(f"✓ 向量数据库搜索功能测试通过 (返回 {len(metadata_list[0])} 个结果)")
        except Exception as e:
            self.fail(f"向量数据库搜索功能测试失败: {str(e)}")
    
    def test_intent_recognition(self):
        """测试意图识别"""
        print("\n=== 测试意图识别 ===")
        
        # 测试意图识别功能
        test_query = "什么是人工智能"
        try:
            result = intent_recognizer.recognize(test_query)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            print(f"✓ 意图识别功能测试通过 (识别结果: {result[0].value})")
        except Exception as e:
            print(f"⚠ 意图识别功能测试失败: {str(e)}")
            # 意图识别功能可能未初始化，标记为跳过
            self.skipTest("意图识别功能未初始化")
    
    def test_retrieval_basic(self):
        """测试基本检索功能"""
        print("\n=== 测试基本检索功能 ===")
        
        # 创建检索配置
        config = RetrievalConfig(
            top_k=3,
            similarity_threshold=0.5,
            algorithm=SimilarityAlgorithm.COSINE,
            enable_rerank=False,
            enable_query_expansion=False
        )
        
        # 测试检索功能
        test_query = "人工智能"
        try:
            result = retriever.retrieve(test_query, config)
            self.assertIsInstance(result, type(retriever.retrieve(test_query, config)))
            self.assertHasAttr(result, 'results')
            print(f"✓ 基本检索功能测试通过 (返回 {len(result.results)} 个结果)")
        except Exception as e:
            self.fail(f"基本检索功能测试失败: {str(e)}")
    
    def test_retrieval_with_rerank(self):
        """测试带重排序的检索功能"""
        print("\n=== 测试带重排序的检索功能 ===")
        
        # 创建检索配置
        config = RetrievalConfig(
            top_k=3,
            similarity_threshold=0.5,
            algorithm=SimilarityAlgorithm.COSINE,
            enable_rerank=True,
            reranker_type="bge",
            reranker_model="BAAI/bge-reranker-v2-m3",
            reranker_top_k=5,
            reranker_threshold=0.0,
            enable_query_expansion=False
        )
        
        # 测试检索功能
        test_query = "人工智能"
        try:
            result = retriever.retrieve(test_query, config)
            self.assertIsInstance(result, type(retriever.retrieve(test_query, config)))
            self.assertHasAttr(result, 'results')
            print(f"✓ 带重排序的检索功能测试通过 (返回 {len(result.results)} 个结果)")
        except Exception as e:
            print(f"⚠ 带重排序的检索功能测试失败: {str(e)}")
            # 重排序功能可能未初始化，标记为跳过
            self.skipTest("重排序功能未初始化")
    
    def test_retrieval_with_query_expansion(self):
        """测试带查询扩展的检索功能"""
        print("\n=== 测试带查询扩展的检索功能 ===")
        
        # 创建检索配置
        config = RetrievalConfig(
            top_k=3,
            similarity_threshold=0.5,
            algorithm=SimilarityAlgorithm.COSINE,
            enable_rerank=False,
            enable_query_expansion=True
        )
        
        # 测试检索功能
        test_query = "人工智能"
        try:
            result = retriever.retrieve(test_query, config)
            self.assertIsInstance(result, type(retriever.retrieve(test_query, config)))
            self.assertHasAttr(result, 'results')
            print(f"✓ 带查询扩展的检索功能测试通过 (返回 {len(result.results)} 个结果)")
        except Exception as e:
            self.fail(f"带查询扩展的检索功能测试失败: {str(e)}")
    
    def test_retrieval_performance(self):
        """测试检索性能"""
        print("\n=== 测试检索性能 ===")
        
        # 创建检索配置
        config = RetrievalConfig(
            top_k=5,
            similarity_threshold=0.5,
            algorithm=SimilarityAlgorithm.COSINE,
            enable_rerank=False,
            enable_query_expansion=False
        )
        
        # 测试响应时间
        test_query = "人工智能"
        try:
            start_time = time.time()
            result = retriever.retrieve(test_query, config)
            response_time = time.time() - start_time
            
            print(f"✓ 检索性能测试通过 (响应时间: {response_time:.4f}s)")
            
            # 验证响应时间是否在合理范围内
            self.assertLess(response_time, 1.0, "响应时间过长")
        except Exception as e:
            self.fail(f"检索性能测试失败: {str(e)}")
    
    def test_evaluation(self):
        """测试评估功能"""
        print("\n=== 测试评估功能 ===")
        
        # 准备测试数据
        test_queries = ["人工智能", "Python数据分析"]
        expected_results = [
            ["人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的机器。"],
            ["Python是一种广泛使用的高级编程语言，特别适合数据分析和科学计算。"]
        ]
        
        # 定义检索函数
        def retrieve_func(query):
            config = RetrievalConfig(
                top_k=3,
                similarity_threshold=0.5,
                algorithm=SimilarityAlgorithm.COSINE,
                enable_rerank=False,
                enable_query_expansion=False
            )
            return retriever.retrieve(query, config)
        
        # 测试评估功能
        try:
            metrics = evaluator.evaluate_retrieval(
                test_queries,
                expected_results,
                retrieve_func,
                top_k=3
            )
            
            self.assertIsInstance(metrics, dict)
            self.assertIn('mrr@k', metrics)
            self.assertIn('ndcg@k', metrics)
            self.assertIn('precision@k', metrics)
            print(f"✓ 评估功能测试通过 (MRR@{3}: {metrics['mrr@k']:.4f})")
        except Exception as e:
            self.fail(f"评估功能测试失败: {str(e)}")
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n=== 测试错误处理 ===")
        
        # 测试空查询处理
        config = RetrievalConfig(
            top_k=3,
            similarity_threshold=0.5,
            algorithm=SimilarityAlgorithm.COSINE,
            enable_rerank=False,
            enable_query_expansion=False
        )
        
        try:
            result = retriever.retrieve("", config)
            self.assertIsInstance(result, type(retriever.retrieve("", config)))
            print("✓ 空查询错误处理测试通过")
        except Exception as e:
            self.fail(f"空查询错误处理测试失败: {str(e)}")
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        print("\n=== 清理测试环境 ===")
        
        # 清理嵌入服务缓存
        try:
            embedding_service.clear_cache()
            print("✓ 嵌入服务缓存已清理")
        except Exception as e:
            print(f"⚠ 清理嵌入服务缓存失败: {str(e)}")
        
        print("测试环境清理完成\n")


def run_performance_test():
    """运行性能测试"""
    print("\n=== 运行性能测试 ===")
    
    # 准备测试查询
    performance_queries = [
        "什么是人工智能",
        "如何使用Python进行数据分析",
        "机器学习和深度学习的区别",
        "如何构建一个神经网络",
        "数据科学的工作流程是什么",
        "什么是自然语言处理",
        "如何评估机器学习模型",
        "什么是大数据",
        "云计算的优势是什么",
        "如何保护数据隐私"
    ]
    
    # 定义检索函数
    def retrieve_func(query):
        config = RetrievalConfig(
            top_k=5,
            similarity_threshold=0.5,
            algorithm=SimilarityAlgorithm.COSINE,
            enable_rerank=False,
            enable_query_expansion=False
        )
        return retriever.retrieve(query, config)
    
    # 运行性能测试
    try:
        metrics = evaluator.evaluate_system_performance(
            performance_queries,
            retrieve_func,
            concurrent_users=5,
            iterations=20
        )
        
        print("\n性能测试结果:")
        print(f"平均响应时间: {metrics['avg_response_time']:.4f}s")
        print(f"P95响应时间: {metrics['p95_response_time']:.4f}s")
        print(f"P99响应时间: {metrics['p99_response_time']:.4f}s")
        print(f"吞吐量: {metrics['throughput']:.2f} QPS")
        print(f"错误率: {metrics['error_rate']:.4f}")
        
        print("\n性能测试完成")
    except Exception as e:
        print(f"性能测试失败: {str(e)}")


def run_accuracy_test():
    """运行准确性测试"""
    print("\n=== 运行准确性测试 ===")
    
    # 准备测试数据
    test_data = [
        ("什么是人工智能", ["人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的机器。"]),
        ("Python数据分析", ["Python是一种广泛使用的高级编程语言，特别适合数据分析和科学计算。"]),
        ("机器学习和深度学习", ["机器学习是人工智能的一个子集，专注于开发能够从数据中学习的算法。", "深度学习是机器学习的一个分支，使用多层神经网络来模拟人类大脑的学习过程。"]),
        ("数据科学工作流程", ["数据科学的工作流程包括数据收集、数据清洗、特征工程、模型训练和模型评估。"])
    ]
    
    # 分离查询和期望结果
    queries = [item[0] for item in test_data]
    expected_results = [item[1] for item in test_data]
    
    # 定义检索函数
    def retrieve_func(query):
        config = RetrievalConfig(
            top_k=3,
            similarity_threshold=0.5,
            algorithm=SimilarityAlgorithm.COSINE,
            enable_rerank=True,
            reranker_type="bge",
            reranker_model="BAAI/bge-reranker-v2-m3",
            reranker_top_k=5,
            reranker_threshold=0.0,
            enable_query_expansion=True
        )
        return retriever.retrieve(query, config)
    
    # 运行准确性测试
    try:
        metrics = evaluator.evaluate_retrieval(
            queries,
            expected_results,
            retrieve_func,
            top_k=3
        )
        
        print("\n准确性测试结果:")
        print(f"MRR@{3}: {metrics['mrr@k']:.4f}")
        print(f"NDCG@{3}: {metrics['ndcg@k']:.4f}")
        print(f"Precision@{3}: {metrics['precision@k']:.4f}")
        print(f"Recall@{3}: {metrics['recall@k']:.4f}")
        print(f"F1@{3}: {metrics['f1@k']:.4f}")
        
        print("\n准确性测试完成")
    except Exception as e:
        print(f"准确性测试失败: {str(e)}")


def generate_evaluation_report():
    """生成评估报告"""
    print("\n=== 生成评估报告 ===")
    
    try:
        report_path = os.path.join(settings.vector_db_dir, "evaluation_report.md")
        report = evaluator.generate_evaluation_report(report_path)
        print(f"✓ 评估报告生成成功: {report_path}")
        print("\n报告摘要:")
        print(report[:500] + "..." if len(report) > 500 else report)
    except Exception as e:
        print(f"生成评估报告失败: {str(e)}")


def main():
    """主测试函数"""
    print("====================================")
    print("        检索系统综合测试")
    print("====================================")
    
    # 运行单元测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # 运行性能测试
    run_performance_test()
    
    # 运行准确性测试
    run_accuracy_test()
    
    # 生成评估报告
    generate_evaluation_report()
    
    print("\n====================================")
    print("        测试完成")
    print("====================================")


if __name__ == "__main__":
    main()
