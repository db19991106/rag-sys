from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
from pathlib import Path
import json
import uuid
import asyncio
from services.document_parser import DocumentParser
from models import DocumentInfo, DocumentStatus, DocumentUploadResponse, DocumentVersion, UpdateHistory
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
        self.versions_db_path = Path(settings.vector_db_dir) / "versions.json"
        self.history_db_path = Path(settings.vector_db_dir) / "update_history.json"
        self.documents: Dict[str, DocumentInfo] = {}
        self.versions: Dict[str, List[DocumentVersion]] = {}
        self.update_history: List[UpdateHistory] = []
        self._load_documents()
        self._load_versions()
        self._load_history()
        
        # 定期更新任务将在应用启动时手动启动
        # asyncio.create_task(self._periodic_update_task())

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

    def _load_versions(self):
        """从文件加载版本信息"""
        if self.versions_db_path.exists():
            try:
                with open(self.versions_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc_id, versions_data in data.items():
                        self.versions[doc_id] = []
                        for version_data in versions_data:
                            version_data['created_at'] = datetime.fromisoformat(version_data['created_at'])
                            self.versions[doc_id].append(DocumentVersion(**version_data))
                logger.info(f"加载了版本信息: {len(self.versions)} 个文档有版本记录")
            except Exception as e:
                logger.error(f"加载版本数据库失败: {str(e)}")

    def _save_versions(self):
        """保存版本信息到文件"""
        try:
            data = {}
            for doc_id, versions in self.versions.items():
                data[doc_id] = []
                for version in versions:
                    version_data = version.dict()
                    version_data['created_at'] = version.created_at.isoformat()
                    data[doc_id].append(version_data)

            with open(self.versions_db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存版本数据库失败: {str(e)}")

    def _load_history(self):
        """从文件加载更新历史"""
        if self.history_db_path.exists():
            try:
                with open(self.history_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for history_data in data:
                        history_data['timestamp'] = datetime.fromisoformat(history_data['timestamp'])
                        self.update_history.append(UpdateHistory(**history_data))
                logger.info(f"加载了 {len(self.update_history)} 条更新历史记录")
            except Exception as e:
                logger.error(f"加载更新历史失败: {str(e)}")

    def _save_history(self):
        """保存更新历史到文件"""
        try:
            data = []
            for history in self.update_history[-1000:]:  # 只保留最近1000条记录
                history_data = history.dict()
                history_data['timestamp'] = history.timestamp.isoformat()
                data.append(history_data)

            with open(self.history_db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存更新历史失败: {str(e)}")

    async def _periodic_update_task(self):
        """定期更新任务"""
        while True:
            try:
                # 每小时检查一次需要更新的文档
                await asyncio.sleep(3600)
                logger.info("执行定期知识库更新检查")
                
                # 检查需要更新的文档
                updated_count = await self._check_and_update_documents()
                if updated_count > 0:
                    logger.info(f"定期更新完成，更新了 {updated_count} 个文档")
            except Exception as e:
                logger.error(f"定期更新任务失败: {str(e)}")

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
            file_path=Path(file_path).name,  # 保存文件名
            version="1.0.0"
        )

        self.documents[file_id] = document_info
        self._save_documents()

        # 创建初始版本
        initial_version = DocumentVersion(
            version_id=str(uuid.uuid4()),
            document_id=file_id,
            version="1.0.0",
            created_at=datetime.now(),
            changes="初始版本",
            file_path=Path(file_path).name
        )
        
        if file_id not in self.versions:
            self.versions[file_id] = []
        self.versions[file_id].append(initial_version)
        self._save_versions()

        # 记录更新历史
        history = UpdateHistory(
            history_id=str(uuid.uuid4()),
            action="upload",
            document_id=file_id,
            document_name=filename,
            user_id="system",
            timestamp=datetime.now(),
            details="初始版本上传"
        )
        self.update_history.append(history)
        self._save_history()

        logger.info(f"文档上传成功: {filename} (ID: {file_id}, 版本: 1.0.0)")

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

    async def update_document_version(self, doc_id: str, content: bytes, changes: str, user_id: str = "system") -> Optional[str]:
        """
        更新文档版本

        Args:
            doc_id: 文档ID
            content: 新的文件内容
            changes: 变更描述
            user_id: 用户ID

        Returns:
            新版本号，更新失败返回 None
        """
        doc = self.documents.get(doc_id)
        if not doc:
            logger.error(f"文档不存在: {doc_id}")
            return None

        try:
            # 保存新文件
            file_id, file_path = save_upload_file(content, doc.name)

            # 解析文档
            parser = DocumentParser()
            parsed_content = parser.parse(file_path)
            if parsed_content is None:
                delete_file(file_id)
                logger.error(f"文档解析失败: {doc_id}")
                return None

            # 生成新版本号
            current_version = doc.version
            version_parts = list(map(int, current_version.split('.')))
            version_parts[2] += 1  # 增加补丁版本
            new_version = '.'.join(map(str, version_parts))

            # 更新文档信息
            doc.size = get_file_size(file_path)
            doc.file_path = Path(file_path).name
            doc.version = new_version
            doc.status = DocumentStatus.PENDING
            self._save_documents()

            # 创建新版本记录
            new_version_record = DocumentVersion(
                version_id=str(uuid.uuid4()),
                document_id=doc_id,
                version=new_version,
                created_at=datetime.now(),
                changes=changes,
                file_path=Path(file_path).name
            )
            
            self.versions[doc_id].append(new_version_record)
            self._save_versions()

            # 记录更新历史
            history = UpdateHistory(
                history_id=str(uuid.uuid4()),
                action="update",
                document_id=doc_id,
                document_name=doc.name,
                user_id=user_id,
                timestamp=datetime.now(),
                details=changes
            )
            self.update_history.append(history)
            self._save_history()

            logger.info(f"文档版本更新成功: {doc.name} (ID: {doc_id}, 版本: {new_version})")
            return new_version
        except Exception as e:
            logger.error(f"更新文档版本失败: {str(e)}")
            return None

    def get_document_versions(self, doc_id: str) -> List[DocumentVersion]:
        """
        获取文档的版本历史

        Args:
            doc_id: 文档ID

        Returns:
            版本历史列表
        """
        return self.versions.get(doc_id, [])

    def rollback_to_version(self, doc_id: str, version_id: str, user_id: str = "system") -> bool:
        """
        回滚到特定版本

        Args:
            doc_id: 文档ID
            version_id: 版本ID
            user_id: 用户ID

        Returns:
            回滚是否成功
        """
        doc = self.documents.get(doc_id)
        if not doc:
            logger.error(f"文档不存在: {doc_id}")
            return False

        # 查找目标版本
        target_version = None
        for version in self.versions.get(doc_id, []):
            if version.version_id == version_id:
                target_version = version
                break

        if not target_version:
            logger.error(f"版本不存在: {version_id}")
            return False

        try:
            # 更新文档信息
            doc.file_path = target_version.file_path
            doc.version = target_version.version
            doc.status = DocumentStatus.PENDING
            self._save_documents()

            # 记录更新历史
            history = UpdateHistory(
                history_id=str(uuid.uuid4()),
                action="rollback",
                document_id=doc_id,
                document_name=doc.name,
                user_id=user_id,
                timestamp=datetime.now(),
                details=f"回滚到版本 {target_version.version}"
            )
            self.update_history.append(history)
            self._save_history()

            logger.info(f"文档回滚成功: {doc.name} (ID: {doc_id}, 版本: {target_version.version})")
            return True
        except Exception as e:
            logger.error(f"回滚文档版本失败: {str(e)}")
            return False

    async def _check_and_update_documents(self) -> int:
        """
        检查并更新需要更新的文档

        Returns:
            更新的文档数量
        """
        updated_count = 0
        
        # 这里可以实现具体的更新逻辑，例如：
        # 1. 检查外部数据源的变更
        # 2. 检查文档的最后修改时间
        # 3. 检查文档的过期时间
        # 4. 执行增量更新
        
        # 示例：检查所有状态为UPDATED的文档
        for doc_id, doc in self.documents.items():
            if doc.status == DocumentStatus.UPDATED:
                # 这里可以添加具体的更新逻辑
                doc.status = DocumentStatus.PENDING
                updated_count += 1
                
                # 记录更新历史
                history = UpdateHistory(
                    history_id=str(uuid.uuid4()),
                    action="auto_update",
                    document_id=doc_id,
                    document_name=doc.name,
                    user_id="system",
                    timestamp=datetime.now(),
                    details="自动更新"
                )
                self.update_history.append(history)

        if updated_count > 0:
            self._save_documents()
            self._save_history()

        return updated_count

    def get_update_history(self, limit: int = 100) -> List[UpdateHistory]:
        """
        获取更新历史

        Args:
            limit: 返回记录数量限制

        Returns:
            更新历史列表
        """
        return self.update_history[-limit:]

    def batch_update_documents(self, doc_updates: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        批量更新文档

        Args:
            doc_updates: 文档更新信息列表

        Returns:
            更新结果字典
        """
        results = {}
        
        for update_info in doc_updates:
            doc_id = update_info.get('doc_id')
            content = update_info.get('content')
            changes = update_info.get('changes', '批量更新')
            user_id = update_info.get('user_id', 'system')
            
            if doc_id and content:
                # 这里需要异步处理，暂时返回结果
                results[doc_id] = "pending"
            else:
                results[doc_id] = "error: missing required fields"
        
        return results


# 全局文档管理器实例
document_manager = DocumentManager()