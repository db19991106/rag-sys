from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pathlib import Path
import os
from datetime import datetime
from models import DocumentInfo, DocumentStatus, ApiResponse
from services.document_manager import document_manager
from services.document_parser import DocumentParser
from utils.logger import logger
from utils.file_utils import get_file_size, format_file_size
from config import settings


router = APIRouter(prefix="/sync", tags=["数据同步"])


@router.post("/documents")
async def sync_documents():
    """
    同步文档数据库与文件系统

    1. 扫描文件系统中的文件
    2. 添加数据库中不存在的新文件
    3. 删除数据库中不存在文件的记录
    4. 更新文件大小和时间信息

    Returns:
        同步结果
    """
    try:
        upload_dir = Path(settings.upload_dir)
        
        if not upload_dir.exists():
            return {
                "success": False,
                "message": "上传目录不存在",
                "stats": {
                    "added": 0,
                    "removed": 0,
                    "updated": 0,
                    "total": 0
                }
            }

        # 获取文件系统中的所有文件
        file_system_files = set()
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                file_system_files.add(file_path.name)

        # 获取数据库中的所有文档
        db_documents = document_manager.list_documents()
        db_file_map = {doc.file_path: doc for doc in db_documents if doc.file_path}
        db_file_names = set(db_file_map.keys())

        stats = {
            "added": 0,
            "removed": 0,
            "updated": 0,
            "total": 0
        }

        # 1. 添加新文件
        new_files = file_system_files - db_file_names
        for filename in new_files:
            file_path = upload_dir / filename
            
            # 跳过空文件
            if file_path.stat().st_size == 0:
                logger.warning(f"跳过空文件: {filename}")
                continue

            try:
                # 解析文档
                parser = DocumentParser()
                content = parser.parse(str(file_path))
                
                if content is None:
                    logger.warning(f"无法解析文件: {filename}")
                    continue

                # 生成文档ID（使用文件名的hash）
                import hashlib
                file_id = hashlib.md5(filename.encode()).hexdigest()

                # 创建文档信息
                doc_info = DocumentInfo(
                    id=file_id,
                    name=filename,
                    size=get_file_size(str(file_path)),
                    status=DocumentStatus.PENDING,
                    upload_time=datetime.fromtimestamp(file_path.stat().st_mtime),
                    chunk_count=0,
                    category="未分类",
                    tags=[],
                    file_path=filename
                )

                # 添加到数据库
                document_manager.documents[file_id] = doc_info
                document_manager._save_documents()
                stats["added"] += 1
                logger.info(f"添加新文档: {filename}")

            except Exception as e:
                logger.error(f"添加文件失败 {filename}: {str(e)}")

        # 2. 删除数据库中不存在文件的记录
        deleted_files = db_file_names - file_system_files
        for filename in list(deleted_files):
            try:
                # 找到对应的文档并删除
                doc_to_delete = None
                for doc_id, doc in document_manager.documents.items():
                    if doc.file_path == filename:
                        doc_to_delete = doc_id
                        break
                
                if doc_to_delete:
                    del document_manager.documents[doc_to_delete]
                    document_manager._save_documents()
                    stats["removed"] += 1
                    logger.info(f"删除文档记录: {filename}")

            except Exception as e:
                logger.error(f"删除文档记录失败 {filename}: {str(e)}")

        # 3. 更新文件信息（大小、时间）
        existing_files = file_system_files & db_file_names
        for filename in existing_files:
            file_path = upload_dir / filename
            doc = db_file_map.get(filename)
            
            if doc:
                file_size = get_file_size(str(file_path))
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # 检查是否需要更新
                needs_update = (
                    doc.size != file_size or
                    doc.upload_time != file_mtime
                )
                
                if needs_update:
                    doc.size = file_size
                    doc.upload_time = file_mtime
                    document_manager._save_documents()
                    stats["updated"] += 1
                    logger.info(f"更新文档信息: {filename}")

        stats["total"] = len(file_system_files)

        logger.info(f"文档同步完成: 新增 {stats['added']}, 删除 {stats['removed']}, 更新 {stats['updated']}, 总计 {stats['total']}")

        return {
            "success": True,
            "message": "文档同步成功",
            "stats": stats
        }

    except Exception as e:
        logger.error(f"文档同步失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档同步失败: {str(e)}")


@router.get("/documents/status")
async def get_sync_status():
    """
    获取同步状态

    Returns:
        文件系统和数据库的文件列表对比
    """
    try:
        upload_dir = Path(settings.upload_dir)
        
        if not upload_dir.exists():
            return {
                "file_system": [],
                "database": [],
                "new_files": [],
                "deleted_files": [],
                "synced": True
            }

        # 获取文件系统中的文件
        file_system_files = []
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                file_system_files.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })

        # 获取数据库中的文档
        db_documents = document_manager.list_documents()
        database_files = [
            {
                "id": doc.id,
                "name": doc.file_path or doc.name,
                "size": doc.size,
                "modified": doc.upload_time.isoformat() if doc.upload_time else None
            }
            for doc in db_documents
        ]

        # 对比
        fs_names = set(f["name"] for f in file_system_files)
        db_names = set(d["name"] for d in database_files)

        new_files = list(fs_names - db_names)
        deleted_files = list(db_names - fs_names)
        synced = len(new_files) == 0 and len(deleted_files) == 0

        return {
            "file_system": file_system_files,
            "database": database_files,
            "new_files": new_files,
            "deleted_files": deleted_files,
            "synced": synced
        }

    except Exception as e:
        logger.error(f"获取同步状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取同步状态失败: {str(e)}")


@router.post("/documents/cleanup")
async def cleanup_orphaned_records():
    """
    清理孤立的数据库记录（file_path为null或文件不存在的记录）
    """
    try:
        upload_dir = Path(settings.upload_dir)
        db_documents = document_manager.list_documents()
        
        orphaned_ids = []
        for doc_id, doc in document_manager.documents.items():
            # 检查file_path为null或文件不存在
            if not doc.file_path:
                orphaned_ids.append(doc_id)
            else:
                file_path = upload_dir / doc.file_path
                if not file_path.exists():
                    orphaned_ids.append(doc_id)

        # 删除孤立记录
        for doc_id in orphaned_ids:
            del document_manager.documents[doc_id]
        
        if orphaned_ids:
            document_manager._save_documents()
            logger.info(f"清理了 {len(orphaned_ids)} 个孤立记录")

        return {
            "success": True,
            "message": f"清理了 {len(orphaned_ids)} 个孤立记录",
            "cleaned_count": len(orphaned_ids)
        }

    except Exception as e:
        logger.error(f"清理孤立记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理孤立记录失败: {str(e)}")