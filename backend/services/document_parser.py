from typing import Optional
from pathlib import Path
import pypdf
from docx import Document as DocxDocument
import markdown
from utils.logger import logger
from utils.file_utils import get_file_extension


class DocumentParser:
    """文档解析器 - 支持多种文档格式"""

    @staticmethod
    def parse(file_path: str) -> Optional[str]:
        """
        解析文档内容

        Args:
            file_path: 文件路径

        Returns:
            文档文本内容，解析失败返回 None
        """
        ext = get_file_extension(file_path)

        try:
            if ext == '.txt':
                return DocumentParser._parse_txt(file_path)
            elif ext == '.pdf':
                return DocumentParser._parse_pdf(file_path)
            elif ext == '.docx':
                return DocumentParser._parse_docx(file_path)
            elif ext == '.md':
                return DocumentParser._parse_markdown(file_path)
            else:
                logger.error(f"不支持的文件格式: {ext}")
                return None
        except Exception as e:
            logger.error(f"解析文件失败 {file_path}: {str(e)}")
            return None

    @staticmethod
    def _parse_txt(file_path: str) -> str:
        """解析 TXT 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        """解析 PDF 文件"""
        content = []
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    content.append(text)
        return '\n'.join(content)

    @staticmethod
    def _parse_docx(file_path: str) -> str:
        """解析 DOCX 文件"""
        doc = DocxDocument(file_path)
        content = []
        for paragraph in doc.paragraphs:
            if paragraph.text:
                content.append(paragraph.text)
        return '\n'.join(content)

    @staticmethod
    def _parse_markdown(file_path: str) -> str:
        """解析 Markdown 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        # 返回原始 Markdown 内容
        return md_content