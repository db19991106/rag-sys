#!/usr/bin/env python3
"""
功能测试模块
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.document_parser import DocumentParser
from services.document_manager import document_manager
from services.retriever import retriever
from services.rag_generator import rag_generator
from services.auth_service import auth_service
from services.audit_logger import audit_logger
from models import RetrievalConfig, GenerationConfig


class TestDocumentParser:
    """测试文档解析功能"""

    def test_parse_text(self):
        """测试文本文件解析"""
        parser = DocumentParser()
        test_content = "这是一个测试文本文件。\n包含多行内容。"
        
        # 创建临时测试文件
        test_file = Path(__file__).parent / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        
        try:
            result = parser.parse(str(test_file))
            assert result is not None
            assert "测试文本文件" in result
            assert "多行内容" in result
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_parse_with_metadata(self):
        """测试带元数据的解析"""
        parser = DocumentParser()
        test_content = "这是一个测试文本文件。"
        
        # 创建临时测试文件
        test_file = Path(__file__).parent / "test_with_metadata.txt"
        test_file.write_text(test_content, encoding='utf-8')
        
        try:
            content, metadata = parser.parse_with_metadata(str(test_file), {"source": "test"})
            assert content is not None
            assert metadata is not None
            assert "测试文本文件" in content
            assert metadata.get("source") == "test"
        finally:
            if test_file.exists():
                test_file.unlink()


class TestDocumentManager:
    """测试文档管理功能"""

    def test_upload_document(self):
        """测试文档上传"""
        import asyncio
        
        async def upload_test():
            test_content = b"This is a test document."
            test_filename = "test_upload.txt"
            
            response = await document_manager.upload_document(test_filename, test_content)
            assert response.id != ""
            assert response.name == test_filename
            assert response.status.value == "pending"
            
            # 清理
            document_manager.delete_document(response.id)
        
        asyncio.run(upload_test())

    def test_list_documents(self):
        """测试文档列表"""
        documents = document_manager.list_documents()
        assert isinstance(documents, list)

    def test_get_document(self):
        """测试获取文档"""
        import asyncio
        
        async def get_test():
            # 先上传一个文档
            test_content = b"Test document content"
            test_filename = "test_get.txt"
            response = await document_manager.upload_document(test_filename, test_content)
            
            try:
                doc = document_manager.get_document(response.id)
                assert doc is not None
                assert doc.id == response.id
            finally:
                document_manager.delete_document(response.id)
        
        asyncio.run(get_test())


class TestRetriever:
    """测试检索功能"""

    def test_retrieve(self):
        """测试基本检索"""
        config = RetrievalConfig(
            top_k=5,
            similarity_threshold=0.6
        )
        
        response = retriever.retrieve("测试检索", config)
        assert response is not None
        assert hasattr(response, 'results')
        assert isinstance(response.results, list)


class TestRAGGenerator:
    """测试RAG生成功能"""

    def test_generate(self):
        """测试基本生成"""
        retrieval_config = RetrievalConfig(
            top_k=3,
            similarity_threshold=0.6
        )
        
        generation_config = GenerationConfig(
            llm_provider="local",
            llm_model="Qwen2.5-7B-Instruct",
            temperature=0.7,
            max_tokens=500
        )
        
        response = rag_generator.generate("测试RAG生成", retrieval_config, generation_config)
        assert response is not None
        assert response.query == "测试RAG生成"
        assert response.answer != ""


class TestAuthService:
    """测试认证服务"""

    def test_authenticate_user(self):
        """测试用户认证"""
        # 测试有效用户
        user = auth_service.authenticate_user("admin", "123456")
        assert user is not None
        assert user.get("username") == "admin"
        
        # 测试无效用户
        user = auth_service.authenticate_user("invalid", "password")
        assert user is None

    def test_create_access_token(self):
        """测试创建访问令牌"""
        data = {"sub": "1", "username": "admin", "role": "admin"}
        token = auth_service.create_access_token(data)
        assert token is not None
        assert isinstance(token, str)

    def test_verify_token(self):
        """测试验证令牌"""
        data = {"sub": "1", "username": "admin", "role": "admin"}
        token = auth_service.create_access_token(data)
        
        payload = auth_service.verify_token(token)
        assert payload is not None
        assert payload.get("sub") == "1"


class TestAuditLogger:
    """测试审计日志功能"""

    def test_log(self):
        """测试记录审计日志"""
        audit_logger.log(
            user_id="1",
            username="admin",
            action="test",
            module="test_module",
            details={"test": "value"}
        )
        
        # 验证日志记录
        logs = audit_logger.get_logs(limit=5)
        assert len(logs) > 0

    def test_log_system_event(self):
        """测试记录系统事件"""
        audit_logger.log_system_event(
            event_type="test_event",
            message="Test system event",
            severity="info"
        )
        
        # 验证日志记录
        logs = audit_logger.get_logs(limit=5)
        assert len(logs) > 0


if __name__ == "__main__":
    # 运行测试
    print("=" * 70)
    print("运行功能测试")
    print("=" * 70)
    print()
    
    # 测试文档解析
    print("1. 测试文档解析...")
    parser_test = TestDocumentParser()
    parser_test.test_parse_text()
    parser_test.test_parse_with_metadata()
    print("✅ 文档解析测试通过")
    print()
    
    # 测试文档管理
    print("2. 测试文档管理...")
    manager_test = TestDocumentManager()
    manager_test.test_upload_document()
    manager_test.test_list_documents()
    manager_test.test_get_document()
    print("✅ 文档管理测试通过")
    print()
    
    # 测试检索
    print("3. 测试检索功能...")
    retriever_test = TestRetriever()
    retriever_test.test_retrieve()
    print("✅ 检索功能测试通过")
    print()
    
    # 测试认证服务
    print("4. 测试认证服务...")
    auth_test = TestAuthService()
    auth_test.test_authenticate_user()
    auth_test.test_create_access_token()
    auth_test.test_verify_token()
    print("✅ 认证服务测试通过")
    print()
    
    # 测试审计日志
    print("5. 测试审计日志...")
    audit_test = TestAuditLogger()
    audit_test.test_log()
    audit_test.test_log_system_event()
    print("✅ 审计日志测试通过")
    print()
    
    print("=" * 70)
    print("🎉 所有功能测试通过!")
    print("=" * 70)
