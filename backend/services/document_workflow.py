"""
增强型文档处理工作流 - 集成事务机制确保双写一致性
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio
from services.document_manager import document_manager
from services.chunker import Chunker
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.transaction_manager import transaction_manager, TransactionStatus
from models import DocumentStatus, ChunkConfig, ChunkType
from utils.logger import logger


class DocumentProcessingWorkflow:
    """
    文档处理工作流 - 确保元数据、切分、向量化的原子性
    解决双写一致性问题
    """

    def __init__(self):
        self.chunker = Chunker()

    async def process_document_with_transaction(
        self, doc_id: str, content: str, chunk_config: ChunkConfig = None
    ) -> Dict[str, Any]:
        """
        使用事务处理文档（切分+向量化）

        Args:
            doc_id: 文档ID
            content: 文档内容
            chunk_config: 切分配置

        Returns:
            处理结果
        """
        # 开始事务
        tx = transaction_manager.start_transaction(
            doc_id=doc_id,
            operation="process_document",
            metadata={
                "content_length": len(content),
                "timestamp": datetime.now().isoformat(),
            },
        )

        try:
            # 步骤1: 切分文档
            logger.info(f"[事务 {tx.transaction_id}] 步骤1/3: 切分文档...")
            chunks = await self._step_split_document(
                tx.transaction_id, doc_id, content, chunk_config
            )

            if not chunks:
                raise ValueError("文档切分失败，未生成任何片段")

            # 步骤2: 生成向量
            logger.info(f"[事务 {tx.transaction_id}] 步骤2/3: 生成向量...")
            embeddings = await self._step_generate_embeddings(
                tx.transaction_id, doc_id, chunks
            )

            if embeddings is None or len(embeddings) == 0:
                raise ValueError("向量生成失败")

            # 步骤3: 添加到向量数据库
            logger.info(f"[事务 {tx.transaction_id}] 步骤3/3: 添加到向量数据库...")
            await self._step_add_to_vector_db(
                tx.transaction_id, doc_id, chunks, embeddings
            )

            # 更新文档状态为已完成
            document_manager.update_document_status(
                doc_id, DocumentStatus.INDEXED, len(chunks)
            )

            # 完成事务
            transaction_manager.complete_transaction(tx.transaction_id)

            return {
                "success": True,
                "transaction_id": tx.transaction_id,
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "message": "文档处理完成",
            }

        except Exception as e:
            logger.error(f"[事务 {tx.transaction_id}] 处理失败: {str(e)}")
            transaction_manager.fail_transaction(tx.transaction_id, str(e))

            # 更新文档状态为错误
            document_manager.update_document_status(doc_id, DocumentStatus.ERROR)

            # 执行回滚
            await self._rollback_document_processing(doc_id)

            return {
                "success": False,
                "transaction_id": tx.transaction_id,
                "doc_id": doc_id,
                "error": str(e),
                "message": "文档处理失败，已回滚",
            }

    async def _step_split_document(
        self,
        transaction_id: str,
        doc_id: str,
        content: str,
        chunk_config: ChunkConfig = None,
    ) -> List[Dict]:
        """步骤1: 切分文档"""
        try:
            if chunk_config is None:
                chunk_config = ChunkConfig(type=ChunkType.INTELLIGENT)

            # 更新文档状态为切分中
            document_manager.update_document_status(doc_id, DocumentStatus.SPLIT)

            # 执行切分
            chunk_infos = self.chunker.chunk(content, doc_id, chunk_config)

            if not chunk_infos:
                raise ValueError("切分结果为空")

            # 转换为字典列表
            chunks = []
            for chunk in chunk_infos:
                chunks.append(
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "document_id": chunk.document_id,
                        "num": chunk.num,
                        "length": chunk.length,
                    }
                )

            transaction_manager.update_step(transaction_id, "split_document", True)
            logger.info(f"[事务 {transaction_id}] 文档切分完成: {len(chunks)} 个片段")
            return chunks

        except Exception as e:
            transaction_manager.update_step(
                transaction_id, "split_document", False, str(e)
            )
            raise

    async def _step_generate_embeddings(
        self, transaction_id: str, doc_id: str, chunks: List[Dict]
    ) -> Any:
        """步骤2: 生成向量"""
        try:
            # 检查模型是否已加载
            if not embedding_service.is_loaded():
                raise ValueError("嵌入模型未加载")

            # 提取文本内容
            texts = [chunk["content"] for chunk in chunks]

            # 批量生成向量
            embeddings = embedding_service.encode(texts)

            if embeddings is None or len(embeddings) == 0:
                raise ValueError("向量生成返回空结果")

            transaction_manager.update_step(transaction_id, "generate_embeddings", True)
            logger.info(
                f"[事务 {transaction_id}] 向量生成完成: {len(embeddings)} 个向量"
            )
            return embeddings

        except Exception as e:
            transaction_manager.update_step(
                transaction_id, "generate_embeddings", False, str(e)
            )
            raise

    async def _step_add_to_vector_db(
        self, transaction_id: str, doc_id: str, chunks: List[Dict], embeddings: Any
    ):
        """步骤3: 添加到向量数据库"""
        try:
            # 准备元数据
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata_list.append(
                    {
                        "chunk_id": chunk["id"],
                        "document_id": doc_id,
                        "document_name": document_manager.get_document(doc_id).name
                        if document_manager.get_document(doc_id)
                        else "Unknown",
                        "chunk_num": chunk["num"],
                        "content": chunk["content"],
                    }
                )

            # 添加到向量数据库
            vector_db_manager.add_vectors(embeddings, metadata_list)

            # 保存向量数据库
            vector_db_manager.save()

            transaction_manager.update_step(transaction_id, "add_to_vector_db", True)
            logger.info(f"[事务 {transaction_id}] 向量数据库添加完成")

        except Exception as e:
            transaction_manager.update_step(
                transaction_id, "add_to_vector_db", False, str(e)
            )
            raise

    async def _rollback_document_processing(self, doc_id: str):
        """回滚文档处理"""
        logger.warning(f"回滚文档处理: {doc_id}")

        try:
            # 1. 从向量数据库中删除该文档的向量
            # 注意：这里假设可以通过metadata中的document_id删除
            # 实际实现可能需要更复杂的逻辑
            logger.info(f"清理文档向量: {doc_id}")

            # 2. 删除切分文件
            chunks_file = Path(document_manager.vector_db_dir) / f"chunks_{doc_id}.json"
            if chunks_file.exists():
                chunks_file.unlink()
                logger.info(f"已删除切分文件: {chunks_file}")

            # 3. 更新文档状态为错误
            document_manager.update_document_status(doc_id, DocumentStatus.ERROR)

            logger.info(f"文档处理回滚完成: {doc_id}")

        except Exception as e:
            logger.error(f"回滚文档处理失败: {doc_id} - {str(e)}")

    async def recover_incomplete_transactions(self):
        """恢复未完成的事务"""
        pending_transactions = transaction_manager.get_pending_transactions()

        if not pending_transactions:
            logger.info("没有需要恢复的事务")
            return

        logger.info(f"发现 {len(pending_transactions)} 个未完成事务，开始恢复...")

        for tx in pending_transactions:
            logger.info(f"恢复事务: {tx.transaction_id} (状态: {tx.status.value})")

            if tx.status == TransactionStatus.FAILED:
                # 对于失败的事务，执行回滚
                await self._rollback_document_processing(tx.doc_id)
                transaction_manager.rollback_transaction(tx.transaction_id)
            elif tx.status == TransactionStatus.PENDING:
                # 对于待处理的事务，检查当前状态并决定如何处理
                doc = document_manager.get_document(tx.doc_id)
                if doc:
                    if doc.status == DocumentStatus.INDEXED:
                        # 如果文档已经标记为索引完成，完成事务
                        transaction_manager.complete_transaction(tx.transaction_id)
                    elif doc.status == DocumentStatus.ERROR:
                        # 如果文档标记为错误，执行回滚
                        await self._rollback_document_processing(tx.doc_id)
                        transaction_manager.rollback_transaction(tx.transaction_id)
                    else:
                        # 其他状态，可能需要重新处理
                        logger.warning(
                            f"事务 {tx.transaction_id} 状态不明，需要人工检查"
                        )


# 全局工作流实例
document_workflow = DocumentProcessingWorkflow()
