"""
HyDE (Hypothetical Document Embedding) 功能测试
测试 LLM 生成的假设文档效果
"""

import sys
import os

# 添加 backend 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger
from services.enhanced.query_enhancer import QueryEnhancer


def test_hyde():
    """测试 HyDE 假设文档生成"""
    logger.info("=" * 60)
    logger.info("HyDE 假设文档生成测试")
    logger.info("模型: Qwen2.5-0.5B-Instruct")
    logger.info("=" * 60)
    
    # 创建 QueryEnhancer 实例
    enhancer = QueryEnhancer()
    
    # 测试用例
    test_cases = [
        {
            "query": "公司的报销流程是什么？",
            "scene_tags": ["Ambiguous"],  # 触发 HyDE
            "description": "普通查询 - 单假设模式"
        },
        {
            "query": "苹果的产品有哪些？",
            "scene_tags": ["Ambiguous"],  # 触发 HyDE，且有歧义实体
            "description": "歧义查询 - 多假设模式（苹果=公司/水果）"
        },
        {
            "query": "Python的优势是什么？",
            "scene_tags": ["Ambiguous"],
            "description": "歧义查询 - 多假设模式（Python=语言/蛇）"
        },
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        query = case["query"]
        scene_tags = case["scene_tags"]
        description = case["description"]
        
        logger.info("=" * 60)
        logger.info(f"测试用例 {i+1}: {description}")
        logger.info(f"查询: {query}")
        logger.info(f"场景标签: {scene_tags}")
        logger.info("-" * 60)
        
        try:
            # 调用 HyDE 生成
            enhanced = enhancer.enhance(
                query=query,
                session_context={},
                scene_tags=scene_tags
            )
            
            # 输出结果
            logger.info(f"增强类型: {enhanced.enhancements_applied}")
            logger.info(f"假设文档数量: {len(enhanced.hypotheses)}")
            
            for j, hyp in enumerate(enhanced.hypotheses):
                logger.info(f"  假设 {j+1}:")
                logger.info(f"    实体: {hyp.entity or 'N/A'}")
                logger.info(f"    置信度: {hyp.confidence}")
                logger.info(f"    内容: {hyp.text[:200]}{'...' if len(hyp.text) > 200 else ''}")
            
            results.append({
                "query": query,
                "success": True,
                "hypotheses_count": len(enhanced.hypotheses)
            })
            
        except Exception as e:
            logger.error(f"测试失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })
    
    # 汇总结果
    logger.info("=" * 60)
    logger.info("测试汇总")
    logger.info("=" * 60)
    
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"成功: {success_count}/{len(results)}")
    
    for r in results:
        status = "✅" if r["success"] else "❌"
        logger.info(f"{status} {r['query']}")
    
    return results


if __name__ == "__main__":
    test_hyde()
