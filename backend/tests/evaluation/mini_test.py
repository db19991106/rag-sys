#!/usr/bin/env python3
"""
最小化测试构建脚本
避免系统日志干扰
"""

import os
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

def main():
    print("🚀 最小化构建测试")
    print("=" * 60)
    
    # 禁用系统日志
    os.environ['NO_SYSTEM_LOG'] = '1'
    
    # 直接配置日志
    log_file = Path("/root/autodl-tmp/rag/backend/tests/evaluation/logs/mini_test.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 简单日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding='utf-8')
        ],
        force=True,  # 强制重新配置
    )
    
    logger = logging.getLogger("MiniBuilder")
    logger.info("=" * 60)
    logger.info("🚀 最小化构建测试")
    logger.info("=" * 60)
    
    try:
        # 1. 检查配置
        from eval_config import get_config
        config = get_config('financial')
        logger.info(f"✅ 配置加载成功")
        logger.info(f"文档目录: {config['docs_dir']}")
        logger.info(f"向量库目录: {config['vector_db_dir']}")
        
        # 2. 扫描文档
        if not config["docs_dir"].exists():
            logger.error(f"❌ 文档目录不存在: {config['docs_dir']}")
            return False
        
        doc_files = list(config["docs_dir"].glob("*.md"))
        logger.info(f"📁 找到 {len(doc_files)} 个文档")
        
        # 3. 测试切分
        logger.info("✂️ 测试财务报销制度切分...")
        from services.financial_chunker_v2 import FinancialDocumentChunker
        chunker = FinancialDocumentChunker(max_chunk_size=1000)
        
        # 使用第一个文档进行测试
        if doc_files:
            test_file = doc_files[0]
            logger.info(f"\n📄 测试文档: {test_file.name}")
            
            # 读取文档内容
            try:
                content = Path(test_file).read_text(encoding='utf-8')
                logger.info(f"   ✅ 文档读取成功，内容长度: {len(content)} 字符")
                
                # 切分文档
                chunks = chunker.chunk_document(content, test_file.stem)
                logger.info(f"   ✅ 切分成功: {len(chunks)} 个片段")
                
                # 显示前3个片段
                for i, chunk in enumerate(chunks[:3], 1):
                    logger.info(f"   片段 {i}:")
                    logger.info(f"     类型: {chunk.chunk_type}")
                    logger.info(f"     预览: {chunk.content[:100]}...")
                    
                logger.info(f"✅ 切分测试完成")
                return True
                
        else:
            logger.warning("⚠️ 没有文档可供测试")
            return False
            
            except Exception as e:
                print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ 测试完成！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)