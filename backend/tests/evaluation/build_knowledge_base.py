#!/usr/bin/env python3
"""
离线知识库构建脚本
用于批量处理文档、切分、向量化并存入向量数据库

使用方法:
    python build_knowledge_base.py
    python build_knowledge_base.py --config financial
    python build_knowledge_base.py --docs-dir /path/to/docs --chunking-method financial_v2
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# 添加backend目录到Python路径
# 当前文件路径: backend/tests/evaluation/build_knowledge_base.py
# 需要从 evaluation -> tests -> backend
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

# 禁用stdout重定向，避免日志递归问题
import os
os.environ['RAG_DISABLE_STDOUT_REDIRECT'] = 'true'

# 导入配置
from eval_config import (
    CURRENT_CONFIG,
    get_config,
    DOCS_DIR,
    VECTOR_DB_DIR,
    MODELS_DIR,
    TEST_DATASET_PATH,
    SUPPORTED_EXTENSIONS,
)

# 导入服务
from services.document_parser import DocumentParser
from services.chunker import Chunker, ChunkType
from services.financial_chunker_v2 import FinancialDocumentChunker
from services.embedding import embedding_service
from services.vector_db import vector_db_manager, VectorDBConfig
from models import EmbeddingConfig, EmbeddingModelType, VectorDBType
from config import settings
# from utils.logger import logger  # 注释掉，避免冲突

# 创建 chunker 实例
chunker = Chunker()


class KnowledgeBaseBuilder:
    """知识库构建器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化构建器

        Args:
            config: 配置字典
        """
        self.config = config
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_docs": 0,
            "processed_docs": 0,
            "failed_docs": 0,
            "total_chunks": 0,
            "total_vectors": 0,
        }
        self.setup_logging()

    def setup_logging(self):
        """设置日志"""
        log_config = self.config.get("log_config", {})
        log_level = getattr(logging, log_config.get("log_level", "INFO"))
        log_file = log_config.get("log_file")

        # 确保日志目录存在
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

        # 配置根日志记录器
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(str(log_file), encoding="utf-8")
                if log_file
                else logging.NullHandler(),
            ],
            force=True,  # 强制重新配置
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("🚀 离线知识库构建脚本启动")
        self.logger.info(f"📝 日志文件: {log_file}")
        self.logger.info("=" * 80)

    def scan_documents(self, docs_dir: Path) -> List[Path]:
        """
        扫描文档目录

        Args:
            docs_dir: 文档目录

        Returns:
            文档路径列表
        """
        self.logger.info(f"\n📁 扫描文档目录: {docs_dir}")

        if not docs_dir.exists():
            self.logger.error(f"❌ 文档目录不存在: {docs_dir}")
            return []

        supported_exts = self.config.get("supported_extensions", SUPPORTED_EXTENSIONS)
        doc_files = []

        for ext in supported_exts:
            doc_files.extend(docs_dir.glob(f"**/*{ext}"))

        # 去重并排序
        doc_files = sorted(list(set(doc_files)))

        self.logger.info(f"✅ 找到 {len(doc_files)} 个文档")
        for doc_file in doc_files:
            self.logger.info(f"   - {doc_file.name}")

        self.stats["total_docs"] = len(doc_files)
        return doc_files

    def parse_document(self, doc_path: Path) -> Optional[str]:
        """
        解析文档

        Args:
            doc_path: 文档路径

        Returns:
            文档内容
        """
        try:
            self.logger.info(f"\n📄 解析文档: {doc_path.name}")

            # 使用文档解析器
            content = DocumentParser.parse(str(doc_path))

            if not content:
                self.logger.warning(f"⚠️ 文档为空: {doc_path.name}")
                return None

            self.logger.info(f"   ✅ 解析成功，内容长度: {len(content)} 字符")
            return content

        except Exception as e:
            self.logger.error(f"   ❌ 解析失败: {str(e)}")
            return None

    def chunk_document(self, content: str, doc_path: Path) -> List[Dict[str, Any]]:
        """
        切分文档

        Args:
            content: 文档内容
            doc_path: 文档路径

        Returns:
            文档片段列表
        """
        chunking_method = self.config.get("chunking_method", "financial_v2")
        self.logger.info(f"\n✂️  切分文档 (方法: {chunking_method})")

        try:
            chunks = []

            if chunking_method == "financial_v2":
                # 使用财务报销制度切分器V2
                chunker_v2 = FinancialDocumentChunker(
                    max_chunk_size=self.config.get("chunking_config", {}).get(
                        "max_chunk_size", 1000
                    )
                )
                chunk_objects = chunker_v2.chunk_document(content, doc_id=doc_path.stem)

                # 转换为标准格式
                for i, chunk_obj in enumerate(chunk_objects):
                    chunks.append(
                        {
                            "id": f"{doc_path.stem}_chunk_{i + 1}",
                            "content": chunk_obj.content,
                            "metadata": chunk_obj.metadata,
                            "chunk_type": chunk_obj.chunk_type,
                        }
                    )

            elif chunking_method == "intelligent":
                # 使用智能切分
                chunk_result = chunker.chunk(
                    content=content,
                    chunk_type=ChunkType.INTELLIGENT,
                    doc_id=doc_path.stem,
                )

                for i, chunk in enumerate(chunk_result.chunks):
                    chunks.append(
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "chunk_type": chunk.chunk_type.value,
                        }
                    )

            else:
                # 使用默认切分
                chunk_result = chunker.chunk(
                    content=content,
                    chunk_type=ChunkType.NAIVE,
                    doc_id=doc_path.stem,
                )

                for i, chunk in enumerate(chunk_result.chunks):
                    chunks.append(
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "chunk_type": chunk.chunk_type.value,
                        }
                    )

            self.logger.info(f"   ✅ 生成 {len(chunks)} 个片段")
            return chunks

        except Exception as e:
            self.logger.error(f"   ❌ 切分失败: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return []

    def init_embedding_service(self) -> bool:
        """
        初始化嵌入服务

        Returns:
            是否成功
        """
        self.logger.info("\n🔧 初始化嵌入服务...")

        try:
            embedding_type = self.config.get("embedding_model_type", "bge")
            embedding_config = self.config.get("embedding_config", {})

            if embedding_type == "bge":
                model_path = embedding_config.get("model_path")
                if not model_path or not Path(model_path).exists():
                    self.logger.error(f"❌ BGE模型路径不存在: {model_path}")
                    return False

                config = EmbeddingConfig(
                    model_type=EmbeddingModelType.BGE,
                    model_name=model_path,
                    device=embedding_config.get("device", "cuda"),
                )
            else:
                self.logger.error(f"❌ 不支持的嵌入模型类型: {embedding_type}")
                return False

            embedding_service.load_model(config)

            if embedding_service.is_loaded():
                self.logger.info(f"   ✅ 嵌入模型加载成功")
                self.logger.info(f"   📊 模型维度: {embedding_service.get_dimension()}")
                return True
            else:
                self.logger.error("❌ 嵌入模型加载失败")
                return False

        except Exception as e:
            self.logger.error(f"❌ 初始化嵌入服务失败: {str(e)}")
            return False

    def init_vector_db(self) -> bool:
        """
        初始化向量数据库

        Returns:
            是否成功
        """
        self.logger.info("\n💾 初始化向量数据库...")

        try:
            db_type = self.config.get("vector_db_type", "faiss")
            vector_db_config = self.config.get("vector_db_config", {})

            if db_type == "faiss":
                # 检查向量库目录
                vector_db_dir = self.config.get("vector_db_dir", VECTOR_DB_DIR)
                vector_db_path = Path(vector_db_dir)
                vector_db_path.mkdir(parents=True, exist_ok=True)

                config = VectorDBConfig(
                    db_type=VectorDBType.FAISS,
                    dimension=embedding_service.get_dimension(),
                    index_type=vector_db_config.get("index_type", "HNSW"),
                    index_path=str(vector_db_path),
                )
            else:
                self.logger.error(f"❌ 不支持的向量数据库类型: {db_type}")
                return False

            success = vector_db_manager.initialize(config)

            if success:
                status = vector_db_manager.get_status()
                self.logger.info(f"   ✅ 向量数据库初始化成功")
                self.logger.info(f"   📊 当前向量数: {status.total_vectors}")
                return True
            else:
                self.logger.error("❌ 向量数据库初始化失败")
                return False

        except Exception as e:
            self.logger.error(f"❌ 初始化向量数据库失败: {str(e)}")
            return False

    def embed_and_store(self, chunks: List[Dict[str, Any]], doc_path: Path) -> bool:
        """
        向量化并存储

        Args:
            chunks: 文档片段列表
            doc_path: 文档路径

        Returns:
            是否成功
        """
        if not chunks:
            return True

        self.logger.info(f"\n🔢 向量化并存储...")

        try:
            # 准备文本
            texts = [chunk["content"] for chunk in chunks]

            # 批量向量化
            batch_size = self.config.get("batch_config", {}).get(
                "embedding_batch_size", 32
            )
            all_vectors = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                self.logger.info(
                    f"   处理批次 {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch_texts)} 条)"
                )

                vectors = embedding_service.encode(batch_texts)
                all_vectors.extend(vectors)

            # 准备元数据
            metadata = []
            for i, chunk in enumerate(chunks):
                meta = {
                    "chunk_id": chunk["id"],
                    "document_id": doc_path.stem,
                    "document_name": doc_path.name,
                    "chunk_num": i + 1,
                    "content": chunk["content"],
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "metadata": chunk.get("metadata", {}),
                }
                metadata.append(meta)

            # 存入向量数据库
            import numpy as np

            vectors_array = np.array(all_vectors, dtype=np.float32)
            vector_db_manager.add_vectors(vectors_array, metadata)

            self.logger.info(f"   ✅ 存储成功: {len(chunks)} 个向量")
            self.stats["total_vectors"] += len(chunks)
            return True

        except Exception as e:
            self.logger.error(f"   ❌ 向量化或存储失败: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    def process_document(self, doc_path: Path) -> bool:
        """
        处理单个文档

        Args:
            doc_path: 文档路径

        Returns:
            是否成功
        """
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"📄 处理文档: {doc_path.name}")
        self.logger.info(f"{'=' * 80}")

        try:
            # 1. 解析文档
            content = self.parse_document(doc_path)
            if not content:
                self.stats["failed_docs"] += 1
                return False

            # 2. 切分文档
            chunks = self.chunk_document(content, doc_path)
            if not chunks:
                self.logger.warning(f"⚠️ 没有生成片段: {doc_path.name}")
                self.stats["failed_docs"] += 1
                return False

            self.stats["total_chunks"] += len(chunks)

            # 3. 向量化并存储
            success = self.embed_and_store(chunks, doc_path)
            if not success:
                self.stats["failed_docs"] += 1
                return False

            self.stats["processed_docs"] += 1
            self.logger.info(f"✅ 文档处理完成: {doc_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"❌ 处理文档失败: {doc_path.name} - {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            self.stats["failed_docs"] += 1
            return False

    def build(self) -> bool:
        """
        构建知识库

        Returns:
            是否成功
        """
        self.stats["start_time"] = datetime.now()
        self.logger.info(f"\n🚀 开始构建知识库 - {self.stats['start_time']}")

        # 1. 初始化服务
        if not self.init_embedding_service():
            self.logger.error("❌ 嵌入服务初始化失败，停止构建")
            return False

        if not self.init_vector_db():
            self.logger.error("❌ 向量数据库初始化失败，停止构建")
            return False

        # 2. 扫描文档
        docs_dir = self.config.get("docs_dir", DOCS_DIR)
        doc_files = self.scan_documents(Path(docs_dir))

        if not doc_files:
            self.logger.warning("⚠️ 没有找到可处理的文档")
            return False

        # 3. 处理文档
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"🔄 开始批量处理 {len(doc_files)} 个文档")
        self.logger.info(f"{'=' * 80}")

        for i, doc_file in enumerate(doc_files, 1):
            self.logger.info(
                f"\n📊 进度: {i}/{len(doc_files)} ({i / len(doc_files) * 100:.1f}%)"
            )
            self.process_document(doc_file)

        # 4. 保存向量数据库
        self.logger.info("\n💾 保存向量数据库...")
        try:
            if hasattr(vector_db_manager.db, "save"):
                vector_db_manager.db.save()
                self.logger.info("✅ 向量数据库保存成功")
        except Exception as e:
            self.logger.error(f"❌ 保存向量数据库失败: {str(e)}")

        # 5. 输出统计
        self.stats["end_time"] = datetime.now()
        self.print_stats()

        return self.stats["failed_docs"] == 0

    def print_stats(self):
        """打印统计信息"""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 构建统计")
        self.logger.info("=" * 80)
        self.logger.info(f"⏱️  总耗时: {duration:.2f} 秒")
        self.logger.info(f"📄 总文档数: {self.stats['total_docs']}")
        self.logger.info(f"✅ 成功处理: {self.stats['processed_docs']}")
        self.logger.info(f"❌ 失败文档: {self.stats['failed_docs']}")
        self.logger.info(f"✂️  总片段数: {self.stats['total_chunks']}")
        self.logger.info(f"🔢 总向量数: {self.stats['total_vectors']}")

        if self.stats["total_docs"] > 0:
            success_rate = (
                self.stats["processed_docs"] / self.stats["total_docs"]
            ) * 100
            self.logger.info(f"📈 成功率: {success_rate:.1f}%")

        self.logger.info("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="离线知识库构建脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用默认配置构建
    python build_knowledge_base.py
    
    # 使用财务报销制度配置
    python build_knowledge_base.py --config financial
    
    # 指定文档目录和切分方法
    python build_knowledge_base.py --docs-dir /path/to/docs --chunking-method financial_v2
    
    # 指定向量库目录
    python build_knowledge_base.py --vector-db-dir /path/to/vector_db
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "financial", "general"],
        help="使用预定义配置方案 (default: default)",
    )

    parser.add_argument(
        "--docs-dir",
        type=str,
        help=f"文档目录 (默认: {DOCS_DIR})",
    )

    parser.add_argument(
        "--vector-db-dir",
        type=str,
        help=f"向量库目录 (默认: {VECTOR_DB_DIR})",
    )

    parser.add_argument(
        "--chunking-method",
        type=str,
        choices=["financial_v2", "financial", "intelligent", "naive", "enhanced"],
        help="切分方法",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        help="嵌入模型路径",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="运行设备",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)",
    )

    args = parser.parse_args()

    # 获取配置
    config = get_config(args.config)

    # 使用命令行参数覆盖配置
    if args.docs_dir:
        config["docs_dir"] = Path(args.docs_dir)
    if args.vector_db_dir:
        config["vector_db_dir"] = Path(args.vector_db_dir)
    if args.chunking_method:
        config["chunking_method"] = args.chunking_method
    if args.embedding_model:
        config["embedding_config"]["model_path"] = args.embedding_model
    if args.device:
        config["embedding_config"]["device"] = args.device
    if args.log_level:
        # 确保log_config存在
        if "log_config" not in config:
            config["log_config"] = {}
        config["log_config"]["log_level"] = args.log_level

    # 创建构建器并运行
    builder = KnowledgeBaseBuilder(config)
    success = builder.build()

    if success:
        print("\n✅ 知识库构建完成！")
        sys.exit(0)
    else:
        print("\n⚠️ 知识库构建完成，但部分文档处理失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
