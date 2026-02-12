from fastapi import APIRouter, HTTPException
from datetime import datetime
from models import ChunkConfig, ChunkResponse, ApiResponse
from services.chunker import Chunker
from services.document_manager import document_manager
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from utils.logger import logger
from config import settings


router = APIRouter(prefix="/chunking", tags=["文档切分"])


@router.post("/split", response_model=ChunkResponse)
async def split_document(doc_id: str, config: ChunkConfig, auto_embed: bool = False):
    """
    切分文档

    Args:
        doc_id: 文档ID
        config: 切分配置
        auto_embed: 是否自动向量化（默认False，向后兼容）
    """
    try:
        # 获取文档内容
        content = document_manager.get_document_content(doc_id)
        if not content:
            raise HTTPException(status_code=404, detail="文档不存在或无法读取")

        # 切分文档（传入文件名以启用多类型智能切分）
        chunker = Chunker()
        doc = document_manager.get_document(doc_id)
        filename = doc.name if doc else ""
        chunks = chunker.chunk(content, doc_id, config, filename=filename)

        if not chunks:
            raise HTTPException(status_code=400, detail="文档切分失败，未生成片段")

        # 保存切分结果到文件
        from pathlib import Path
        import json

        chunks_file = Path(settings.vector_db_dir) / f"chunks_{doc_id}.json"
        chunks_file.parent.mkdir(parents=True, exist_ok=True)

        chunks_data = {
            "document_id": doc_id,
            "config": config.dict(),  # 保存完整的切分配置
            "chunks": [chunk.dict() for chunk in chunks],
            "chunk_count": len(chunks),
            "created_at": datetime.now().isoformat(),
        }

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # 更新文档状态
        document_manager.update_document_status(
            doc_id, status="split", chunk_count=len(chunks)
        )

        logger.info(f"文档切分成功: {doc_id}, 生成 {len(chunks)} 个片段")

        # 如果启用自动向量化，立即执行向量化
        auto_embedded = False
        if auto_embed:
            logger.info(f"自动向量化已启用，开始向量化和存储到数据库: {doc_id}")
            try:
                # 检查并加载嵌入模型
                if not embedding_service.is_loaded():
                    logger.info(
                        "嵌入模型未加载，自动加载默认模型: BAAI/bge-base-zh-v1.5"
                    )
                    from models import EmbeddingConfig, EmbeddingModelType
                    import torch

                    # 动态检测设备
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"自动检测到设备: {device}")

                    embedding_config = EmbeddingConfig(
                        model_type=EmbeddingModelType.BGE,
                        model_name="BAAI/bge-base-zh-v1.5",
                        batch_size=32,
                        device=device,
                    )
                    embedding_response = embedding_service.load_model(embedding_config)
                    if embedding_response.status != "success":
                        logger.warning(
                            f"自动加载嵌入模型失败: {embedding_response.message}"
                        )
                    else:
                        logger.info("默认嵌入模型加载成功")

                # 检查向量数据库是否已初始化
                if not vector_db_manager.db:
                    logger.warning("向量数据库未初始化，尝试初始化...")
                    from models import VectorDBConfig, VectorDBType

                    vector_db_config = VectorDBConfig(
                        db_type=VectorDBType.FAISS, dimension=768, index_type="HNSW"
                    )
                    success = vector_db_manager.initialize(vector_db_config)
                    if success:
                        logger.info("向量数据库初始化成功")
                    else:
                        logger.warning("向量数据库初始化失败")

                # 如果模型和数据库都准备好了，执行向量化
                if embedding_service.is_loaded() and vector_db_manager.db:
                    # 生成向量
                    texts = [chunk.content for chunk in chunks]
                    vectors = embedding_service.encode(texts)

                    # 准备元数据
                    metadata = []
                    doc = document_manager.get_document(doc_id)
                    for chunk in chunks:
                        meta = {
                            "chunk_id": chunk.id,
                            "document_id": chunk.document_id,
                            "document_name": doc.name if doc else "Unknown",
                            "chunk_num": chunk.num,
                            "content": chunk.content,
                            "keywords": [],
                        }
                        metadata.append(meta)

                    # 添加到向量数据库
                    vector_db_manager.add_vectors(vectors, metadata)

                    # 更新文档状态为已索引
                    document_manager.update_document_status(doc_id, status="indexed")

                    logger.info(
                        f"✅ 文档自动向量化成功: {doc_id}, {len(chunks)} 个片段已存入向量数据库"
                    )
                    auto_embedded = True
                else:
                    logger.warning(f"⚠️ 自动向量化跳过: 模型或向量数据库未就绪")

            except Exception as e:
                logger.error(f"❌ 自动向量化失败: {str(e)}")
                # 自动向量化失败不影响切分成功返回

        return ChunkResponse(
            chunks=chunks, total=len(chunks), auto_embedded=auto_embedded
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档切分失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档切分失败: {str(e)}")


@router.get("/chunks/{doc_id}", response_model=ChunkResponse)
async def get_document_chunks(doc_id: str):
    """
    获取文档的切分片段列表

    Args:
        doc_id: 文档ID
    """
    try:
        from pathlib import Path
        import json

        chunks_file = Path(settings.vector_db_dir) / f"chunks_{doc_id}.json"

        if not chunks_file.exists():
            raise HTTPException(
                status_code=404, detail="未找到该文档的切分片段，请先切分文档"
            )

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            chunks = [
                ChunkInfo(**chunk_data) for chunk_data in chunks_data.get("chunks", [])
            ]

        return ChunkResponse(chunks=chunks, total=len(chunks))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档片段失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档片段失败: {str(e)}")


@router.post("/embed", response_model=ApiResponse)
async def embed_chunks(doc_id: str):
    """
    向量化片段并存储到向量数据库

    Args:
        doc_id: 文档ID
    """
    try:
        # 检查模型是否已加载，如果未加载则自动加载默认模型
        if not embedding_service.is_loaded():
            logger.info("嵌入模型未加载，自动加载默认模型: BAAI/bge-base-zh-v1.5")
            from models import EmbeddingConfig, EmbeddingModelType
            import torch

            # 动态检测设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"自动检测到设备: {device}")

            embedding_config = EmbeddingConfig(
                model_type=EmbeddingModelType.BGE,
                model_name="BAAI/bge-base-zh-v1.5",
                batch_size=32,
                device=device,
            )
            embedding_response = embedding_service.load_model(embedding_config)
            if embedding_response.status != "success":
                raise HTTPException(
                    status_code=400,
                    detail=f"自动加载嵌入模型失败: {embedding_response.message}",
                )
            logger.info("默认嵌入模型加载成功")

        # 检查向量数据库是否已初始化
        if not vector_db_manager.db:
            raise HTTPException(status_code=400, detail="向量数据库未初始化")

        # 检查向量维度是否匹配
        current_dimension = embedding_service.get_dimension()
        db_dimension = vector_db_manager.db.dimension
        if current_dimension != db_dimension:
            logger.warning(
                f"向量维度不匹配: 当前模型维度={current_dimension}, 数据库维度={db_dimension}"
            )
            logger.warning("需要重新初始化向量数据库以匹配新的维度")
            # 重新初始化向量数据库
            from models import VectorDBConfig, VectorDBType

            vector_db_config = VectorDBConfig(
                db_type=VectorDBType.FAISS,
                dimension=current_dimension,
                index_type="HNSW",
            )
            success = vector_db_manager.initialize(vector_db_config)
            if not success:
                raise HTTPException(
                    status_code=400, detail="无法重新初始化向量数据库以匹配新的维度"
                )
            logger.info(f"向量数据库已重新初始化，维度: {current_dimension}")

        # 获取文档信息
        doc = document_manager.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        # 检查文档是否已切分
        if doc.status != "split" and doc.chunk_count == 0:
            raise HTTPException(
                status_code=400, detail="文档未切分，请先调用 /chunking/split 切分文档"
            )

        # 获取文档内容
        content = document_manager.get_document_content(doc_id)
        if not content:
            raise HTTPException(status_code=404, detail="文档内容无法读取")

        # 切分文档（使用与前端相同的配置）
        # 注意：这里需要与前端切分时使用的配置保持一致
        # 由于配置信息未保存，我们使用默认配置，但应该从某处读取用户之前使用的配置
        chunker = Chunker()

        # 使用默认配置切分（传入文件名以启用多类型智能切分）
        # TODO: 应该保存切分时的配置，向量化时使用相同的配置
        config = ChunkConfig()
        doc = document_manager.get_document(doc_id)
        filename = doc.name if doc else ""
        chunks = chunker.chunk(content, doc_id, config, filename=filename)

        if not chunks:
            raise HTTPException(status_code=400, detail="文档切分失败，未生成片段")

        # 尝试从文件加载已切分的片段
        from pathlib import Path
        import json
        from models import ChunkInfo

        chunks_file = Path(settings.vector_db_dir) / f"chunks_{doc_id}.json"

        chunks = []
        if chunks_file.exists():
            # 从文件加载已切分的片段
            logger.info(f"从文件加载已切分的片段: {chunks_file}")
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                loaded_config = chunks_data.get("config", {})
                chunks = [
                    ChunkInfo(**chunk_data)
                    for chunk_data in chunks_data.get("chunks", [])
                ]
            logger.info(f"成功加载 {len(chunks)} 个片段，使用保存的配置")
        else:
            # 如果文件不存在，需要重新切分
            logger.warning(f"切分文件不存在: {chunks_file}，需要重新切分文档")

            # 获取文档内容
            content = document_manager.get_document_content(doc_id)
            if not content:
                raise HTTPException(status_code=404, detail="文档内容无法读取")

            # 切分文档（传入文件名以启用多类型智能切分）
            chunker = Chunker()
            config = ChunkConfig()
            filename = doc.name if doc else ""
            chunks = chunker.chunk(content, doc_id, config, filename=filename)

            if not chunks:
                raise HTTPException(status_code=400, detail="文档切分失败，未生成片段")

            # 保存切分结果
            from pathlib import Path

            chunks_file = Path(settings.vector_db_dir) / f"chunks_{doc_id}.json"
            chunks_data = {
                "document_id": doc_id,
                "config": config.dict(),
                "chunks": [chunk.dict() for chunk in chunks],
                "chunk_count": len(chunks),
                "created_at": datetime.now().isoformat(),
            }
            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)

            # 检查切分数量是否与记录一致
            if doc.chunk_count > 0 and len(chunks) != doc.chunk_count:
                logger.warning(
                    f"切分数量不一致: 记录中为 {doc.chunk_count} 个片段，"
                    f"实际切分得到 {len(chunks)} 个片段。"
                )

        if not chunks:
            raise HTTPException(status_code=400, detail="未找到可向量化的片段")

        # 生成向量
        texts = [chunk.content for chunk in chunks]
        vectors = embedding_service.encode(texts)

        # 准备元数据
        metadata = []
        for chunk in chunks:
            meta = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "document_name": doc.name,
                "chunk_num": chunk.num,
                "content": chunk.content,
                "keywords": [],  # 可以添加关键词提取
            }
            metadata.append(meta)

        # 添加到向量数据库
        vector_db_manager.add_vectors(vectors, metadata)

        # 更新文档状态
        document_manager.update_document_status(doc_id, status="indexed")

        logger.info(f"文档向量化成功: {doc_id}, {len(chunks)} 个片段")

        return ApiResponse(
            success=True, message=f"成功向量化 {len(chunks)} 个片段并存储到向量数据库"
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"文档向量化失败: {str(e)}")
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"文档向量化失败: {str(e)}")
