from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from models import (
    DocumentUploadResponse, DocumentInfo,
    ApiResponse, ErrorResponse
)
from services.document_manager import document_manager
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from utils.logger import logger
from utils.confirmation import confirmation_manager


router = APIRouter(prefix="/documents", tags=["文档管理"])


class DeleteConfirmationRequest(BaseModel):
    """删除确认请求"""
    confirmation_token: str


class BatchDeleteConfirmationRequest(BaseModel):
    """批量删除确认请求"""
    doc_ids: List[str]
    confirmation_token: str

# 允许的文件类型
ALLOWED_FILE_TYPES = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "text/html": ".html",
}

# 最大文件大小 (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024


def validate_file_upload(file: UploadFile) -> None:
    """
    验证文件上传的安全性和有效性
    
    Args:
        file: 上传的文件对象
        
    Raises:
        HTTPException: 如果文件验证失败
    """
    # 验证文件名
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件名不能为空"
        )
    
    # 验证文件名长度
    if len(file.filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件名过长（最大255字符）"
        )
    
    # 验证文件名安全性（防止路径遍历）
    import os
    if ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件名包含非法字符"
        )
    
    # 验证文件扩展名
    file_ext = os.path.splitext(file.filename)[1].lower()
    valid_extensions = set(ALLOWED_FILE_TYPES.values())
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型。支持的类型: {', '.join(valid_extensions)}"
        )
    
    # 验证MIME类型（如果提供了）
    if file.content_type:
        expected_ext = ALLOWED_FILE_TYPES.get(file.content_type)
        if expected_ext and file_ext != expected_ext:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文件类型与扩展名不匹配。MIME类型: {file.content_type}, 扩展名: {file_ext}"
            )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档
    """
    try:
        # 验证文件
        validate_file_upload(file)
        
        # 读取文件内容并验证大小
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"文件过大（最大 {MAX_FILE_SIZE // (1024*1024)}MB）"
            )
        
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件内容为空"
            )
        
        response = await document_manager.upload_document(file.filename, content)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.get("/list", response_model=List[DocumentInfo])
async def list_documents():
    """
    获取文档列表
    """
    try:
        documents = document_manager.list_documents()
        return documents
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    """
    获取文档详情
    """
    document = document_manager.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    return document


@router.get("/{doc_id}/content")
async def get_document_content(doc_id: str):
    """
    获取文档内容
    """
    content = document_manager.get_document_content(doc_id)
    if content is None:
        raise HTTPException(status_code=404, detail="文档不存在或内容无法读取")
    return {"content": content}


@router.get("/{doc_id}/delete-confirmation-token")
async def get_delete_confirmation_token(doc_id: str):
    """
    获取删除文档的确认令牌
    
    使用此令牌确认删除操作，防止误删
    """
    try:
        # 检查文档是否存在
        document = document_manager.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        # 生成确认令牌
        token = confirmation_manager.generate_token(
            operation="delete",
            resource_id=doc_id
        )
        
        return {
            "confirmation_token": token,
            "expires_in_minutes": 10
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成删除确认令牌失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成确认令牌失败")


@router.delete("/{doc_id}", response_model=ApiResponse)
async def delete_document(doc_id: str, request: DeleteConfirmationRequest):
    """
    删除文档（需要确认令牌）
    
    为了防止误删，需要先调用 /{doc_id}/delete-confirmation-token 获取确认令牌，
    然后在删除请求中提供该令牌。
    """
    try:
        # 验证确认令牌
        if not confirmation_manager.verify_token(
            token=request.confirmation_token,
            operation="delete",
            resource_id=doc_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无效或过期的确认令牌，请重新获取"
            )
        
        # 记录敏感操作
        from utils.error_handler import log_sensitive_operation
        log_sensitive_operation(
            operation="delete_document",
            resource_id=doc_id,
            success=True
        )
        
        success = document_manager.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="文档不存在")
        return ApiResponse(success=True, message="文档删除成功")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档删除失败: {str(e)}")
        from utils.error_handler import log_sensitive_operation
        log_sensitive_operation(
            operation="delete_document",
            resource_id=doc_id,
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(status_code=500, detail=f"文档删除失败")


@router.post("/batch-delete-confirmation-token")
async def get_batch_delete_confirmation_token(doc_ids: List[str]):
    """
    获取批量删除文档的确认令牌
    
    使用此令牌确认批量删除操作，防止误删
    """
    try:
        # 验证文档ID列表
        if not doc_ids:
            raise HTTPException(status_code=400, detail="文档ID列表不能为空")
        
        if len(doc_ids) > 100:
            raise HTTPException(status_code=400, detail="一次最多删除100个文档")
        
        # 检查所有文档是否存在
        for doc_id in doc_ids:
            document = document_manager.get_document(doc_id)
            if not document:
                raise HTTPException(
                    status_code=404, 
                    detail=f"文档不存在: {doc_id}"
                )
        
        # 生成确认令牌（使用所有ID的hash作为resource_id）
        import hashlib
        resource_id = hashlib.sha256(",".join(sorted(doc_ids)).encode()).hexdigest()
        
        token = confirmation_manager.generate_token(
            operation="batch_delete",
            resource_id=resource_id
        )
        
        return {
            "confirmation_token": token,
            "expires_in_minutes": 10,
            "doc_count": len(doc_ids)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成批量删除确认令牌失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成确认令牌失败")


@router.post("/batch-delete", response_model=ApiResponse)
async def batch_delete_documents(request: BatchDeleteConfirmationRequest):
    """
    批量删除文档（需要确认令牌）
    
    为了防止误删，需要先调用 /batch-delete-confirmation-token 获取确认令牌，
    然后在删除请求中提供该令牌。
    """
    try:
        # 验证文档ID列表
        if not request.doc_ids:
            raise HTTPException(status_code=400, detail="文档ID列表不能为空")
        
        # 计算resource_id用于验证令牌
        import hashlib
        resource_id = hashlib.sha256(",".join(sorted(request.doc_ids)).encode()).hexdigest()
        
        # 验证确认令牌
        if not confirmation_manager.verify_token(
            token=request.confirmation_token,
            operation="batch_delete",
            resource_id=resource_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无效或过期的确认令牌，请重新获取"
            )
        
        # 记录敏感操作
        from utils.error_handler import log_sensitive_operation
        log_sensitive_operation(
            operation="batch_delete_documents",
            resource_id=resource_id,
            success=True,
            details={"doc_count": len(request.doc_ids)}
        )
        
        deleted_count = document_manager.batch_delete_documents(request.doc_ids)
        return ApiResponse(
            success=True,
            message=f"成功删除 {deleted_count} 个文档"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量删除文档失败: {str(e)}")
        from utils.error_handler import log_sensitive_operation
        log_sensitive_operation(
            operation="batch_delete_documents",
            resource_id=resource_id,
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(status_code=500, detail=f"批量删除文档失败")


@router.get("/{doc_id}/integrity")
async def check_document_integrity(doc_id: str):
    """
    检查文档数据完整性
    
    检查以下内容：
    1. 文档元数据是否存在
    2. 原始文件是否存在
    3. 切分文件是否存在
    4. 切分数量是否匹配
    5. 向量是否存在
    6. 向量数量是否匹配
    """
    try:
        import json
        from pathlib import Path
        
        checks = {
            "document_exists": False,
            "file_exists": False,
            "chunks_file_exists": False,
            "chunk_count_match": False,
            "vectors_exist": False,
            "vector_count_match": False,
            "overall_status": "error",
            "details": []
        }
        
        # 1. 检查文档元数据
        doc = document_manager.get_document(doc_id)
        if doc:
            checks["document_exists"] = True
            checks["details"].append(f"✓ 文档元数据存在: {doc.name}")
            checks["expected_chunk_count"] = doc.chunk_count
        else:
            checks["details"].append("✗ 文档元数据不存在")
            checks["overall_status"] = "error"
            return checks
        
        # 2. 检查原始文件
        file_path = Path(document_manager.upload_dir) / doc_id
        if file_path.exists():
            checks["file_exists"] = True
            file_size = file_path.stat().st_size
            checks["details"].append(f"✓ 原始文件存在: {file_size} bytes")
        else:
            checks["details"].append("✗ 原始文件不存在")
            checks["overall_status"] = "warning"
        
        # 3. 检查切分文件
        chunks_file = Path(document_manager.vector_db_dir) / f"chunks_{doc_id}.json"
        actual_chunks = []
        
        if chunks_file.exists():
            checks["chunks_file_exists"] = True
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                actual_chunks = chunks_data.get("chunks", [])
            checks["details"].append(f"✓ 切分文件存在: {len(actual_chunks)} 个片段")
        else:
            checks["details"].append("✗ 切分文件不存在")
            checks["overall_status"] = "warning"
        
        # 4. 检查切分数量
        checks["actual_chunk_count"] = len(actual_chunks)
        if doc.chunk_count > 0 and len(actual_chunks) == doc.chunk_count:
            checks["chunk_count_match"] = True
            checks["details"].append(f"✓ 切分数量匹配: {doc.chunk_count}")
        else:
            if doc.chunk_count == 0 and len(actual_chunks) > 0:
                checks["details"].append(f"! 切分数量记录为0，但实际有 {len(actual_chunks)} 个片段")
            elif doc.chunk_count > 0:
                checks["details"].append(f"✗ 切分数量不匹配: 记录={doc.chunk_count}, 实际={len(actual_chunks)}")
            checks["overall_status"] = "warning"
        
        # 5. 检查向量
        if vector_db_manager.db:
            # 获取所有向量元数据
            vector_count = 0
            doc_vector_count = 0
            
            for vector_id, metadata in vector_db_manager.metadata.items():
                if metadata.get("document_id") == doc_id:
                    doc_vector_count += 1
                vector_count += 1
            
            if doc_vector_count > 0:
                checks["vectors_exist"] = True
                checks["details"].append(f"✓ 向量存在: {doc_vector_count} 个")
                
                # 6. 检查向量数量
                if doc.chunk_count > 0:
                    if doc_vector_count == doc.chunk_count:
                        checks["vector_count_match"] = True
                        checks["details"].append(f"✓ 向量数量匹配: {doc_vector_count}")
                    else:
                        checks["details"].append(f"✗ 向量数量不匹配: 记录={doc.chunk_count}, 实际={doc_vector_count}")
                        checks["overall_status"] = "warning"
                elif len(actual_chunks) > 0:
                    if doc_vector_count == len(actual_chunks):
                        checks["vector_count_match"] = True
                        checks["details"].append(f"✓ 向量数量匹配: {doc_vector_count}")
                    else:
                        checks["details"].append(f"✗ 向量数量不匹配: 片段={len(actual_chunks)}, 向量={doc_vector_count}")
                        checks["overall_status"] = "warning"
            else:
                checks["details"].append("✗ 向量不存在")
                checks["overall_status"] = "warning" if doc.status == "indexed" else "pending"
        else:
            checks["details"].append("✗ 向量数据库未初始化")
            checks["overall_status"] = "warning"
        
        # 7. 综合评估
        if all(checks[k] for k in ["document_exists", "file_exists", "chunks_file_exists", "chunk_count_match", "vectors_exist", "vector_count_match"]):
            checks["overall_status"] = "healthy"
            checks["details"].append("✓ 所有检查通过，文档数据完整")
        elif checks["overall_status"] == "error":
            checks["details"].append("✗ 文档存在严重问题，无法使用")
        else:
            checks["details"].append("! 文档存在部分问题，可能需要重新处理")
        
        return checks
        
    except Exception as e:
        logger.error(f"检查文档完整性失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检查文档完整性失败: {str(e)}")


@router.get("/integrity")
async def check_system_integrity():
    """
    检查系统整体数据完整性
    """
    try:
        documents = document_manager.list_documents()
        
        summary = {
            "total_documents": len(documents),
            "healthy_documents": 0,
            "warning_documents": 0,
            "error_documents": 0,
            "details": []
        }
        
        for doc in documents:
            integrity = await check_document_integrity(doc.id)
            status = integrity.get("overall_status", "error")
            
            if status == "healthy":
                summary["healthy_documents"] += 1
            elif status == "warning":
                summary["warning_documents"] += 1
            else:
                summary["error_documents"] += 1
            
            summary["details"].append({
                "doc_id": doc.id,
                "doc_name": doc.name,
                "status": status,
                "issues": [d for d in integrity["details"] if d.startswith("✗") or d.startswith("!")]
            })
        
        return summary
        
    except Exception as e:
        logger.error(f"检查系统完整性失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检查系统完整性失败: {str(e)}")