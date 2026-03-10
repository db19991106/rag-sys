from fastapi import APIRouter, HTTPException
from datetime import datetime
from models import ChunkConfig, ChunkResponse, ApiResponse, DocumentStatus
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
        import os

        # 使用文档名称作为文件名（去掉扩展名）
        doc_name = os.path.splitext(filename)[0] if filename else doc_id
        if not doc_name:
            doc_name = doc_id
        
        # 保存到 backend/data/chunks 目录（使用当前文件位置定位backend目录）
        backend_dir = Path(__file__).parent.parent  # api/chunking.py -> backend
        chunks_dir = backend_dir / "data" / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = chunks_dir / f"{doc_name}.json"

        chunks_data = {
            "document_id": doc_id,
            "document_name": filename,
            "config": config.dict(),  # 保存完整的切分配置
            "chunks": [chunk.dict() for chunk in chunks],
            "chunk_count": len(chunks),
            "created_at": datetime.now().isoformat(),
        }

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"切分结果已保存到: {chunks_file}")

        # 更新文档状态
        document_manager.update_document_status(
            doc_id, status=DocumentStatus.SPLIT, chunk_count=len(chunks)
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
                        f"嵌入模型未加载，自动加载配置模型: {settings.embedding_model_name}"
                    )
                    from models import EmbeddingConfig, EmbeddingModelType
                    import torch

                    # 动态检测设备
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"自动检测到设备: {device}")

                    # 转换模型类型
                    model_type = EmbeddingModelType.BGE
                    if settings.embedding_model_type == "sentence-transformers":
                        model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS

                    embedding_config = EmbeddingConfig(
                        model_type=model_type,
                        model_name=settings.embedding_model_name,
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

                    # 使用配置中的数据库类型
                    db_type_map = {
                        "faiss": VectorDBType.FAISS,
                        "milvus": VectorDBType.MILVUS,
                        "milvus_lite": VectorDBType.MILVUS_LITE,
                    }
                    db_type = db_type_map.get(settings.vector_db_type, VectorDBType.MILVUS_LITE)

                    vector_db_config = VectorDBConfig(
                        db_type=db_type, dimension=settings.faiss_dimension, index_type="HNSW"
                    )
                    success = vector_db_manager.initialize(vector_db_config)
                    if success:
                        logger.info(f"向量数据库初始化成功: {settings.vector_db_type}")
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
                    document_manager.update_document_status(doc_id, status=DocumentStatus.INDEXED)

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
            logger.info(f"嵌入模型未加载，自动加载配置模型: {settings.embedding_model_name}")
            from models import EmbeddingConfig, EmbeddingModelType
            import torch

            # 动态检测设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"自动检测到设备: {device}")

            # 转换模型类型
            model_type = EmbeddingModelType.BGE
            if settings.embedding_model_type == "sentence-transformers":
                model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS

            embedding_config = EmbeddingConfig(
                model_type=model_type,
                model_name=settings.embedding_model_name,
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

        # 检查向量数据库是否已初始化，如果未初始化则自动初始化
        if not vector_db_manager.db:
            logger.warning("向量数据库未初始化，尝试自动初始化...")
            from models import VectorDBConfig, VectorDBType

            # 获取当前嵌入模型的维度
            current_dimension = embedding_service.get_dimension()
            logger.info(f"使用当前嵌入模型维度: {current_dimension}")

            # 使用配置中的数据库类型
            db_type_map = {
                "faiss": VectorDBType.FAISS,
                "milvus": VectorDBType.MILVUS,
                "milvus_lite": VectorDBType.MILVUS_LITE,
            }
            db_type = db_type_map.get(settings.vector_db_type, VectorDBType.MILVUS_LITE)

            vector_db_config = VectorDBConfig(
                db_type=db_type,
                dimension=current_dimension,
                index_type="HNSW"
            )
            success = vector_db_manager.initialize(vector_db_config)
            if not success:
                raise HTTPException(status_code=400, detail="向量数据库自动初始化失败")
            logger.info(f"向量数据库自动初始化成功: {settings.vector_db_type}")

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

            # 使用配置中的数据库类型
            db_type_map = {
                "faiss": VectorDBType.FAISS,
                "milvus": VectorDBType.MILVUS,
                "milvus_lite": VectorDBType.MILVUS_LITE,
            }
            db_type = db_type_map.get(settings.vector_db_type, VectorDBType.MILVUS_LITE)

            vector_db_config = VectorDBConfig(
                db_type=db_type,
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
        import os
        from models import ChunkInfo

        # 确定切分文件的路径 - 优先使用文档名称作为文件名
        backend_dir = Path(__file__).parent.parent
        chunks_dir = backend_dir / "data" / "chunks"
        
        # 尝试多种文件名格式
        possible_files = []
        
        # 1. 使用文档名称（与 split_document 保持一致）
        if doc and doc.name:
            doc_name = os.path.splitext(doc.name)[0]
            possible_files.append(chunks_dir / f"{doc_name}.json")
        
        # 2. 使用 doc_id（旧格式）
        possible_files.append(Path(settings.vector_db_dir) / f"chunks_{doc_id}.json")
        
        # 3. 使用 doc_id 在 chunks_dir 中
        possible_files.append(chunks_dir / f"{doc_id}.json")

        chunks = []
        chunks_file = None
        for possible_file in possible_files:
            if possible_file.exists():
                chunks_file = possible_file
                break

        if chunks_file and chunks_file.exists():
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
            logger.warning(f"切分文件不存在，尝试的路径: {[str(f) for f in possible_files]}")
            logger.warning("需要重新切分文档")

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

            # 保存切分结果（使用与 split_document 相同的路径）
            if doc and doc.name:
                doc_name = os.path.splitext(doc.name)[0]
            else:
                doc_name = doc_id
            chunks_file = chunks_dir / f"{doc_name}.json"
            
            chunks_data = {
                "document_id": doc_id,
                "document_name": doc.name if doc else "",
                "config": config.dict(),
                "chunks": [chunk.dict() for chunk in chunks],
                "chunk_count": len(chunks),
                "created_at": datetime.now().isoformat(),
            }
            chunks_dir.mkdir(parents=True, exist_ok=True)
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
        document_manager.update_document_status(doc_id, status=DocumentStatus.INDEXED)

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


@router.post("/batch-split", response_model=ApiResponse)
async def batch_split_documents(doc_ids: list[str], config: ChunkConfig, auto_embed: bool = True):
    """
    批量切分文档

    支持两种文档来源：
    1. 上传的文档（通过document_manager管理）
    2. 本地文档（从data/docs目录读取）

    Args:
        doc_ids: 文档ID列表
        config: 切分配置
        auto_embed: 是否自动向量化（默认True）

    Returns:
        批量切分结果统计
    """
    from pathlib import Path
    import json
    import os

    results = {
        "total": len(doc_ids),
        "success": 0,
        "failed": 0,
        "total_chunks": 0,
        "details": []
    }

    # 本地文档目录
    backend_dir = Path(__file__).parent.parent
    local_docs_dir = backend_dir / "data" / "docs"
    chunks_dir = backend_dir / "data" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for doc_id in doc_ids:
        try:
            content = None
            filename = None
            
            # 首先尝试从document_manager获取（上传的文档）
            content = document_manager.get_document_content(doc_id)
            if content:
                doc = document_manager.get_document(doc_id)
                filename = doc.name if doc else f"{doc_id}.md"
                logger.info(f"[批量切分] 从document_manager获取文档: {filename}")
            else:
                # 尝试从本地docs目录读取
                # 支持完整路径格式（如 "人力资源/员工入职管理制度.md"）
                local_doc_path = local_docs_dir / doc_id
                
                if not local_doc_path.exists():
                    # 尝试添加扩展名（如果doc_id没有扩展名）
                    for ext in ['.md', '.txt', '.markdown', '.html']:
                        test_path = local_docs_dir / f"{doc_id}{ext}"
                        if test_path.exists():
                            local_doc_path = test_path
                            break
                
                # 安全检查：确保路径在允许的目录内
                try:
                    local_doc_path.resolve().relative_to(local_docs_dir.resolve())
                except (ValueError, RuntimeError):
                    local_doc_path = None
                
                if local_doc_path and local_doc_path.exists() and local_doc_path.is_file():
                    with open(local_doc_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    filename = local_doc_path.name
                    logger.info(f"[批量切分] 从本地目录读取文档: {filename}")
            
            if not content:
                results["failed"] += 1
                results["details"].append({
                    "doc_id": doc_id,
                    "status": "failed",
                    "error": "文档不存在或无法读取"
                })
                continue

            # 切分文档
            chunker = Chunker()
            chunks = chunker.chunk(content, doc_id, config, filename=filename)

            if not chunks:
                results["failed"] += 1
                results["details"].append({
                    "doc_id": doc_id,
                    "doc_name": filename,
                    "status": "failed",
                    "error": "文档切分失败，未生成片段"
                })
                continue

            # 保存切分结果到文件
            doc_name = os.path.splitext(filename)[0] if filename else doc_id
            chunks_file = chunks_dir / f"{doc_name}.json"

            chunks_data = {
                "document_id": doc_id,
                "document_name": filename,
                "config": config.dict(),
                "chunks": [chunk.dict() for chunk in chunks],
                "chunk_count": len(chunks),
                "created_at": datetime.now().isoformat(),
            }

            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)

            # 自动向量化
            embedded = False
            if auto_embed:
                try:
                    logger.info(f"[批量切分] 开始向量化文档: {filename}")
                    
                    # 确保嵌入模型已加载
                    if not embedding_service.is_loaded():
                        logger.info(f"[批量切分] 嵌入模型未加载，自动加载...")
                        from models import EmbeddingConfig, EmbeddingModelType
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model_type = EmbeddingModelType.BGE
                        if settings.embedding_model_type == "sentence-transformers":
                            model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
                        embedding_config = EmbeddingConfig(
                            model_type=model_type,
                            model_name=settings.embedding_model_name,
                            batch_size=32,
                            device=device,
                        )
                        embedding_response = embedding_service.load_model(embedding_config)
                        if embedding_response.status != "success":
                            logger.warning(f"[批量切分] 自动加载嵌入模型失败: {embedding_response.message}")
                        else:
                            logger.info("[批量切分] 嵌入模型加载成功")

                    # 确保向量数据库已初始化
                    if not vector_db_manager.db:
                        logger.info("[批量切分] 向量数据库未初始化，自动初始化...")
                        from models import VectorDBConfig, VectorDBType
                        current_dimension = embedding_service.get_dimension()
                        
                        # 使用配置中的数据库类型
                        db_type_map = {
                            "faiss": VectorDBType.FAISS,
                            "milvus": VectorDBType.MILVUS,
                            "milvus_lite": VectorDBType.MILVUS_LITE,
                        }
                        db_type = db_type_map.get(settings.vector_db_type, VectorDBType.MILVUS_LITE)
                        
                        vector_db_config = VectorDBConfig(
                            db_type=db_type,
                            dimension=current_dimension,
                            index_type="HNSW"
                        )
                        success = vector_db_manager.initialize(vector_db_config)
                        if success:
                            logger.info(f"[批量切分] 向量数据库初始化成功: {settings.vector_db_type}")
                        else:
                            logger.warning("[批量切分] 向量数据库初始化失败")

                    # 执行向量化
                    if embedding_service.is_loaded() and vector_db_manager.db:
                        texts = [chunk.content for chunk in chunks]
                        vectors = embedding_service.encode(texts)

                        # 准备元数据
                        metadata = []
                        for chunk in chunks:
                            meta = {
                                "chunk_id": chunk.id,
                                "document_id": doc_id,
                                "document_name": filename,
                                "chunk_num": chunk.num,
                                "content": chunk.content,
                                "keywords": [],
                            }
                            metadata.append(meta)

                        # 添加到向量数据库
                        vector_db_manager.add_vectors(vectors, metadata)
                        embedded = True
                        logger.info(f"[批量切分] 文档 {filename} 向量化成功，{len(chunks)} 个片段已存入向量数据库")
                    else:
                        logger.warning(f"[批量切分] 文档 {filename} 向量化跳过: 模型或数据库未就绪")

                except Exception as embed_error:
                    logger.error(f"[批量切分] 文档 {filename} 向量化失败: {str(embed_error)}")

            # 更新或添加文档元数据到 document_manager
            doc = document_manager.get_document(doc_id)
            if doc:
                # 已存在的文档，更新状态
                document_manager.update_document_status(
                    doc_id, 
                    status=DocumentStatus.INDEXED if embedded else DocumentStatus.SPLIT, 
                    chunk_count=len(chunks)
                )
            else:
                # 本地文档（不存在于 document_manager），添加元数据记录
                from models import DocumentInfo
                document_manager.documents[doc_id] = DocumentInfo(
                    id=doc_id,
                    name=filename,
                    size=0,  # 本地文档不记录大小
                    status=DocumentStatus.INDEXED if embedded else DocumentStatus.SPLIT,
                    chunk_count=len(chunks),
                    upload_time=datetime.now(),
                    file_path=str(local_doc_path) if 'local_doc_path' in dir() else None,
                )
                logger.info(f"[批量切分] 已添加文档元数据: {doc_id} ({len(chunks)} chunks)")

            results["success"] += 1
            results["total_chunks"] += len(chunks)
            results["details"].append({
                "doc_id": doc_id,
                "doc_name": filename,
                "status": "success",
                "chunk_count": len(chunks),
                "embedded": embedded
            })

            logger.info(f"批量切分 - 文档 {filename} 成功，生成 {len(chunks)} 个片段，向量化: {embedded}")

        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "doc_id": doc_id,
                "doc_name": filename if filename else "Unknown",
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"批量切分 - 文档 {doc_id} 失败: {str(e)}")

    logger.info(f"批量切分完成: 成功 {results['success']}/{results['total']}，共生成 {results['total_chunks']} 个片段")

    return ApiResponse(
        success=results["failed"] == 0,
        message=f"批量切分完成：成功 {results['success']}/{results['total']}，共生成 {results['total_chunks']} 个片段",
        data=results
    )
