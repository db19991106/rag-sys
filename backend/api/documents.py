from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from models import (
    DocumentUploadResponse, DocumentInfo,
    ApiResponse, ErrorResponse
)
from services.document_manager import document_manager
from utils.logger import logger


router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档
    """
    try:
        content = await file.read()
        response = await document_manager.upload_document(file.filename, content)
        return response
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


@router.delete("/{doc_id}", response_model=ApiResponse)
async def delete_document(doc_id: str):
    """
    删除文档
    """
    try:
        success = document_manager.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="文档不存在")
        return ApiResponse(success=True, message="文档删除成功")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档删除失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档删除失败: {str(e)}")


@router.post("/batch-delete", response_model=ApiResponse)
async def batch_delete_documents(doc_ids: List[str]):
    """
    批量删除文档
    """
    try:
        deleted_count = document_manager.batch_delete_documents(doc_ids)
        return ApiResponse(
            success=True,
            message=f"成功删除 {deleted_count} 个文档"
        )
    except Exception as e:
        logger.error(f"批量删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量删除文档失败: {str(e)}")