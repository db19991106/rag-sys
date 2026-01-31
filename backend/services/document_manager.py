from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import json
from services.document_parser import DocumentParser
from models import DocumentInfo, DocumentStatus, DocumentUploadResponse
from utils.logger import logger
from utils.file_utils import (
    save_upload_file, delete_file, get_file_size,
    format_file_size, is_allowed_file
)
from config import settings


class DocumentManager:
    """文档管理器 - 处理文档上传、存储和管理"""

    def __init__(self):
        self.documents_db_path = Path(settings.vector_db_dir) / "documents.json"
        self.documents: Dict[str, DocumentInfo] = {}
        self._load_documents()

    def _load_documents(self):
        """从文件加载文档信息"""
        if self.documents_db_path.exists():
            try:
                with open(self.documents_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc_id, doc_data in data.items():
                        doc_data['upload_time'] = datetime.fromisoformat(doc_data['upload_time'])
                        self.documents[doc_id] = DocumentInfo(**doc_data)
                logger.info(f"加载了 {len(self.documents)} 个文档")
            except Exception as e:
                logger.error(f"加载文档数据库失败: {str(e)}")

    def _save_documents(self):
        """保存文档信息到文件"""
        try:
            data = {}
            for doc_id, doc_info in self.documents.items():
                data[doc_id] = doc_info.dict()
                data[doc_id]['upload_time'] = doc_info.upload_time.isoformat()

            with open(self.documents_db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存文档数据库失败: {str(e)}")

    async def upload_document(self, filename: str, content: bytes) -> DocumentUploadResponse:
        """
        上传文档

        Args:
            filename: 文件名
            content: 文件内容

        Returns:
            上传响应
        """
        # 检查文件格式
        if not is_allowed_file(filename):
            return DocumentUploadResponse(
                id="",
                name=filename,
                size=0,
                status=DocumentStatus.ERROR,
                upload_time=datetime.now(),
                message=f"不支持的文件格式，仅支持: {', '.join(settings.allowed_extensions)}"
            )

        # 检查文件大小
        if len(content) > settings.max_file_size:
            return DocumentUploadResponse(
                id="",
                name=filename,
                size=len(content),
                status=DocumentStatus.ERROR,
                upload_time=datetime.now(),
                message=f"文件大小超过限制 ({format_file_size(settings.max_file_size)})"
            )

        # 保存文件
        try:
            file_id, file_path = save_upload_file(content, filename)
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            return DocumentUploadResponse(
                id="",
                name=filename,
                size=len(content),
                status=DocumentStatus.ERROR,
                upload_time=datetime.now(),
                message=f"保存文件失败: {str(e)}"
            )

        # 解析文档
        parser = DocumentParser()
        parsed_content = parser.parse(file_path)

        if parsed_content is None:
            delete_file(file_id)
            return DocumentUploadResponse(
                id="",
                name=filename,
                size=len(content),
                status=DocumentStatus.ERROR,
                upload_time=datetime.now(),
                message="文档解析失败"
            )

        # 保存文档信息
        document_info = DocumentInfo(
            id=file_id,
            name=filename,
            size=get_file_size(file_path),
            status=DocumentStatus.PENDING,
            upload_time=datetime.now(),
            chunk_count=0,
            category="未分类",
            tags=[],
            file_path=Path(file_path).name  # 保存文件名
        )

        self.documents[file_id] = document_info
        self._save_documents()

        logger.info(f"文档上传成功: {filename} (ID: {file_id})")

        return DocumentUploadResponse(
            id=file_id,
            name=filename,
            size=len(content),
            status=DocumentStatus.PENDING,
            upload_time=datetime.now(),
            message="文档上传成功"
        )

    def get_document(self, doc_id: str) -> Optional[DocumentInfo]:
        """获取文档信息"""
        return self.documents.get(doc_id)

    def get_document_content(self, doc_id: str) -> Optional[str]:
        """获取文档内容"""
        doc = self.documents.get(doc_id)
        if not doc or not doc.file_path:
            return None

        upload_dir = Path(settings.upload_dir)
        file_path = upload_dir / doc.file_path

        if not file_path.exists():
            return None

        parser = DocumentParser()
        return parser.parse(str(file_path))

    def list_documents(self, status: Optional[DocumentStatus] = None) -> List[DocumentInfo]:
        """列出所有文档"""
        docs = list(self.documents.values())
        if status:
            docs = [doc for doc in docs if doc.status == status]
        return docs

    def update_document_status(self, doc_id: str, status: DocumentStatus, chunk_count: Optional[int] = None):
        """更新文档状态"""
        if doc_id in self.documents:
            self.documents[doc_id].status = status
            if chunk_count is not None:
                self.documents[doc_id].chunk_count = chunk_count
            self._save_documents()

    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        if doc_id not in self.documents:
            return False

        doc = self.documents[doc_id]

        # 删除文件
        if doc.file_path:
            delete_file(doc.file_path)

        # 删除切分文件
        chunks_file = Path(settings.vector_db_dir) / f"chunks_{doc_id}.json"
        if chunks_file.exists():
            chunks_file.unlink()
            logger.info(f"已删除切分文件: {chunks_file}")

        # 删除记录
        del self.documents[doc_id]
        self._save_documents()

        logger.info(f"文档已删除: {doc_id}")
        return True

    def batch_delete_documents(self, doc_ids: List[str]) -> int:
        """批量删除文档"""
        deleted_count = 0
        for doc_id in doc_ids:
            if self.delete_document(doc_id):
                deleted_count += 1
        return deleted_count


# 全局文档管理器实例
document_manager = DocumentManager()