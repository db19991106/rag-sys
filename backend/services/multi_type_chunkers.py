"""
多类型文档切分器 - 为不同类型文档提供专门的切分策略
"""

import re
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from utils.logger import logger


class BaseDocumentChunker(ABC):
    """文档切分器基类"""

    # 默认切分限制
    DEFAULT_MAX_CHARS = 2000
    DEFAULT_OVERLAP = 100

    def __init__(self, max_chars: int = None, overlap: int = None):
        self.max_chars = max_chars or self.DEFAULT_MAX_CHARS
        self.overlap = overlap or self.DEFAULT_OVERLAP

    @abstractmethod
    def chunk(self, content: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        切分文档

        Args:
            content: 文档内容
            doc_id: 文档ID

        Returns:
            切分后的片段列表
        """
        pass

    def _split_by_headings(
        self, content: str, heading_pattern: re.Pattern
    ) -> List[Tuple[str, str]]:
        """
        按标题切分文档

        Args:
            content: 文档内容
            heading_pattern: 标题正则表达式

        Returns:
            [(标题, 内容), ...]
        """
        sections = []
        current_title = ""
        current_content = []

        lines = content.split("\n")

        for line in lines:
            if heading_pattern.match(line.strip()):
                # 保存上一个章节
                if current_title or current_content:
                    sections.append((current_title, "\n".join(current_content).strip()))
                # 开始新章节
                current_title = line.strip()
                current_content = [line]
            else:
                current_content.append(line)

        # 保存最后一个章节
        if current_title or current_content:
            sections.append((current_title, "\n".join(current_content).strip()))

        return sections

    def _merge_small_chunks(
        self, chunks: List[Dict], min_chars: int = 500
    ) -> List[Dict]:
        """
        合并过小的片段

        Args:
            chunks: 原始片段
            min_chars: 最小字符数

        Returns:
            合并后的片段
        """
        if not chunks:
            return []

        merged = []
        current_chunk = None

        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk.copy()
            elif len(current_chunk["content"]) < min_chars:
                # 合并到当前片段
                current_chunk["content"] += "\n\n" + chunk["content"]
                # 更新元数据
                if (
                    "titles" in current_chunk["metadata"]
                    and "titles" in chunk["metadata"]
                ):
                    current_chunk["metadata"]["titles"].extend(
                        chunk["metadata"].get("titles", [])
                    )
            else:
                # 保存当前片段，开始新片段
                merged.append(current_chunk)
                current_chunk = chunk.copy()

        # 保存最后一个片段
        if current_chunk:
            merged.append(current_chunk)

        return merged

    def _create_chunk(
        self, content: str, chunk_type: str, metadata: Dict, index: int
    ) -> Dict:
        """创建切分片段"""
        return {
            "content": content.strip(),
            "type": chunk_type,
            "metadata": {
                **metadata,
                "chunk_index": index,
                "char_count": len(content),
            },
        }


class ProductDocumentChunker(BaseDocumentChunker):
    """
    产品文档切分器
    适合：产品介绍、用户手册、使用指南

    特点：
    - 按章节（##）切分
    - 保留功能模块完整性
    - 支持表格和代码块
    """

    def chunk(self, content: str, doc_id: str) -> List[Dict[str, Any]]:
        """切分产品文档"""
        logger.info(f"使用产品文档切分器处理文档: {doc_id}")

        chunks = []
        chunk_index = 0

        # 1. 提取文档标题
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else "未知文档"

        # 2. 按二级标题（##）切分
        # 模式：## 标题内容
        sections = re.split(r"\n(?=##\s+)", content)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # 提取章节标题
            section_title = ""
            title_match = re.search(r"^##\s+(.+)$", section, re.MULTILINE)
            if title_match:
                section_title = title_match.group(1).strip()

            # 如果章节太大，再按三级标题切分
            if len(section) > self.max_chars:
                subsections = self._split_by_subsections(section)
                for sub_section, sub_title in subsections:
                    if len(sub_section) > self.max_chars:
                        # 按段落进一步切分
                        para_chunks = self._split_large_section(
                            sub_section, doc_title, section_title, sub_title
                        )
                        chunks.extend(para_chunks)
                    else:
                        chunk = self._create_chunk(
                            sub_section,
                            "product_section",
                            {
                                "doc_title": doc_title,
                                "section_title": section_title,
                                "subsection_title": sub_title,
                            },
                            chunk_index,
                        )
                        chunks.append(chunk)
                        chunk_index += 1
            else:
                chunk = self._create_chunk(
                    section,
                    "product_section",
                    {
                        "doc_title": doc_title,
                        "section_title": section_title,
                    },
                    chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1

        # 3. 合并过小的片段
        chunks = self._merge_small_chunks(chunks, min_chars=300)

        logger.info(f"产品文档切分完成: 共 {len(chunks)} 个片段")
        return chunks

    def _split_by_subsections(self, section: str) -> List[Tuple[str, str]]:
        """按三级标题（###）切分"""
        subsections = []
        parts = re.split(r"\n(?=###\s+)", section)

        for part in parts:
            if not part.strip():
                continue

            title_match = re.search(r"^###\s+(.+)$", part, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else ""
            subsections.append((part, title))

        return subsections

    def _split_large_section(
        self, content: str, doc_title: str, section_title: str, sub_title: str
    ) -> List[Dict]:
        """对过大的章节按段落切分"""
        chunks = []
        paragraphs = content.split("\n\n")
        current_content = []
        current_size = 0
        chunk_index = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > self.max_chars and current_content:
                # 保存当前片段
                chunk = self._create_chunk(
                    "\n\n".join(current_content),
                    "product_paragraph",
                    {
                        "doc_title": doc_title,
                        "section_title": section_title,
                        "subsection_title": sub_title,
                    },
                    chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1

                # 开始新片段（带重叠）
                current_content = (
                    current_content[-2:] if len(current_content) >= 2 else []
                )
                current_size = sum(len(p) for p in current_content)

            current_content.append(para)
            current_size += para_size

        # 保存最后一个片段
        if current_content:
            chunk = self._create_chunk(
                "\n\n".join(current_content),
                "product_paragraph",
                {
                    "doc_title": doc_title,
                    "section_title": section_title,
                    "subsection_title": sub_title,
                },
                chunk_index,
            )
            chunks.append(chunk)

        return chunks


class TechnicalSpecChunker(BaseDocumentChunker):
    """
    技术规范切分器
    适合：API规范、设计规范、开发规范、架构文档

    特点：
    - 保留代码块完整性
    - 保留接口定义完整性
    - 支持表格（参数说明、状态码等）
    """

    def chunk(self, content: str, doc_id: str) -> List[Dict[str, Any]]:
        """切分技术规范文档"""
        logger.info(f"使用技术规范切分器处理文档: {doc_id}")

        chunks = []
        chunk_index = 0

        # 1. 提取文档标题
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else "未知文档"

        # 2. 保护代码块（```）和表格
        protected_blocks = []

        # 提取代码块
        code_pattern = r"```[\s\S]*?```"
        for match in re.finditer(code_pattern, content):
            placeholder = f"__CODE_BLOCK_{len(protected_blocks)}__"
            protected_blocks.append((placeholder, match.group()))
            content = content[: match.start()] + placeholder + content[match.end() :]

        # 提取表格
        table_pattern = r"\|[^\n]+\|\n\|[-:\|\s]+\|\n(?:\|[^\n]+\|\n?)+"
        for match in re.finditer(table_pattern, content):
            placeholder = f"__TABLE_{len(protected_blocks)}__"
            protected_blocks.append((placeholder, match.group()))
            content = content[: match.start()] + placeholder + content[match.end() :]

        # 3. 按章节切分
        sections = re.split(r"\n(?=##\s+)", content)

        for section in sections:
            if not section.strip():
                continue

            section_title = ""
            title_match = re.search(r"^##\s+(.+)$", section, re.MULTILINE)
            if title_match:
                section_title = title_match.group(1).strip()

            # 恢复被保护的内容
            restored_section = self._restore_protected_content(
                section, protected_blocks
            )

            # 如果章节太大，按接口/功能点切分
            if len(restored_section) > self.max_chars:
                sub_chunks = self._split_by_api_endpoints(
                    restored_section, doc_title, section_title
                )
                for sub_chunk in sub_chunks:
                    chunks.append(
                        self._create_chunk(
                            sub_chunk["content"],
                            sub_chunk["type"],
                            sub_chunk["metadata"],
                            chunk_index,
                        )
                    )
                    chunk_index += 1
            else:
                chunks.append(
                    self._create_chunk(
                        restored_section,
                        "tech_section",
                        {
                            "doc_title": doc_title,
                            "section_title": section_title,
                        },
                        chunk_index,
                    )
                )
                chunk_index += 1

        logger.info(f"技术规范切分完成: 共 {len(chunks)} 个片段")
        return chunks

    def _restore_protected_content(
        self, content: str, protected_blocks: List[Tuple[str, str]]
    ) -> str:
        """恢复被保护的内容块"""
        for placeholder, original in protected_blocks:
            content = content.replace(placeholder, original)
        return content

    def _split_by_api_endpoints(
        self, content: str, doc_title: str, section_title: str
    ) -> List[Dict]:
        """按API端点或功能点切分"""
        chunks = []

        # 尝试按接口路径或功能点识别
        # 模式1：HTTP方法 + 路径
        # 模式2：### 功能/接口名称

        parts = re.split(r"\n(?=###\s+|(?:GET|POST|PUT|DELETE|PATCH)\s+/)", content)

        current_chunk = []
        current_size = 0

        for part in parts:
            if not part.strip():
                continue

            part_size = len(part)

            # 如果是新的功能点且当前片段足够大，保存当前片段
            if (
                re.match(r"^(###\s+|(?:GET|POST|PUT|DELETE|PATCH)\s+/)", part.strip())
                and current_chunk
                and current_size > 500
            ):
                chunk_content = "\n".join(current_chunk)
                # 检测是否为API定义
                chunk_type = (
                    "api_endpoint"
                    if re.search(r"(?:GET|POST|PUT|DELETE|PATCH)\s+/", chunk_content)
                    else "tech_subsection"
                )

                chunks.append(
                    {
                        "content": chunk_content,
                        "type": chunk_type,
                        "metadata": {
                            "doc_title": doc_title,
                            "section_title": section_title,
                        },
                    }
                )

                current_chunk = [part]
                current_size = part_size
            else:
                current_chunk.append(part)
                current_size += part_size

                # 如果超出限制，强制切分
                if current_size > self.max_chars:
                    chunk_content = "\n".join(current_chunk)
                    chunk_type = (
                        "api_endpoint"
                        if re.search(
                            r"(?:GET|POST|PUT|DELETE|PATCH)\s+/", chunk_content
                        )
                        else "tech_subsection"
                    )

                    chunks.append(
                        {
                            "content": chunk_content,
                            "type": chunk_type,
                            "metadata": {
                                "doc_title": doc_title,
                                "section_title": section_title,
                            },
                        }
                    )
                    current_chunk = []
                    current_size = 0

        # 保存最后一部分
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            chunk_type = (
                "api_endpoint"
                if re.search(r"(?:GET|POST|PUT|DELETE|PATCH)\s+/", chunk_content)
                else "tech_subsection"
            )

            chunks.append(
                {
                    "content": chunk_content,
                    "type": chunk_type,
                    "metadata": {
                        "doc_title": doc_title,
                        "section_title": section_title,
                    },
                }
            )

        return chunks


class ComplianceDocumentChunker(BaseDocumentChunker):
    """
    合规文件切分器
    适合：隐私政策、安全制度、合规协议、法律条款

    特点：
    - 保留条款完整性（第X条、第X章）
    - 维护条款之间的逻辑关系
    - 支持条款引用追踪
    
    切分策略：
    1. 结构化切分（优先）：
       - 第一层：按「第 X 章」切分章节
       - 第二层：在章节内按「第 X 条」切分条款
       - 每个条款作为一个基础单元（包含条款标题 + 正文 + 关联表格）
    2. 语义切分（补充）：
       - 超长条款（>2000字）按句子拆分后再语义合并
       - 短条款（<100字）按语义相似度合并
    """

    CHAPTER_PATTERN = re.compile(r"(?:^|\n)(?:#{1,2}\s*)?第[零一二三四五六七八九十百千万\d]+章\s*.+", re.MULTILINE)
    ARTICLE_PATTERN = re.compile(r"(?:^|\n)(?:#{1,6}\s*)?第[零一二三四五六七八九十百千万\d]+条", re.MULTILINE)
    SENTENCE_ENDINGS = re.compile(r"[。；！？\n]")
    TABLE_PATTERN = re.compile(r"(\|(?:[^\n]+\|)(?:\n\|[-:\|\s]+\|)(?:\n\|[^\n]+\|)+)")
    TABLE_PLACEHOLDER = re.compile(r"__TABLE_\d+__")

    def chunk(self, content: str, doc_id: str) -> List[Dict[str, Any]]:
        """切分合规文件"""
        logger.info(f"使用合规文件切分器处理文档: {doc_id}")

        protected_tables = []
        content = self._protect_tables(content, protected_tables)
        
        chunks = []
        
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else "未知文档"

        chapters = self._split_by_chapters(content)
        
        for chapter in chapters:
            chapter_title = chapter.get("title", "")
            chapter_content = chapter.get("content", "")
            
            articles = self._split_articles_in_chapter(chapter_content, chapter_title)
            
            for article in articles:
                article_title = article.get("title", "")
                article_content = self._restore_tables(article.get("content", ""), protected_tables)
                article_numbers = article.get("article_numbers", [])
                
                if len(article_content) > 2000:
                    sub_chunks = self._semantic_split_article(
                        article_content, doc_title, chapter_title, article_numbers
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(
                        self._create_chunk(
                            article_content,
                            "compliance_article",
                            {
                                "doc_title": doc_title,
                                "chapter_title": chapter_title,
                                "article_title": article_title,
                                "articles": article_numbers,
                                "has_table": bool(article.get("has_table", False)),
                            },
                            len(chunks),
                        )
                    )

        chunks = self._merge_short_articles(chunks)

        logger.info(f"合规文件切分完成: 共 {len(chunks)} 个片段")
        return chunks

    def _protect_tables(self, content: str, table_list: List) -> str:
        """保护表格内容"""
        for match in self.TABLE_PATTERN.finditer(content):
            placeholder = f"__TABLE_{len(table_list)}__"
            table_list.append(match.group(1))
            content = content[:match.start()] + "\n" + placeholder + "\n" + content[match.end():]
        return content

    def _restore_tables(self, content: str, table_list: List) -> str:
        """恢复表格内容"""
        for i, table in enumerate(table_list):
            content = content.replace(f"__TABLE_{i}__", table)
        return content

    def _split_by_chapters(self, content: str) -> List[Dict]:
        """按章节切分文档"""
        chapters = []
        
        matches = list(self.CHAPTER_PATTERN.finditer(content))
        
        if not matches:
            return [{"title": "", "content": content}]
        
        for i, match in enumerate(matches):
            chapter_start = match.start()
            chapter_title = match.group().strip()
            
            chapter_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            
            chapter_content = content[chapter_start:chapter_end].strip()
            
            chapters.append({
                "title": chapter_title,
                "content": chapter_content,
            })
        
        return chapters

    def _split_articles_in_chapter(
        self, chapter_content: str, chapter_title: str
    ) -> List[Dict]:
        """在章节内按条款切分"""
        articles = []
        
        matches = list(self.ARTICLE_PATTERN.finditer(chapter_content))
        
        if not matches:
            return [{"title": "", "content": chapter_content, "article_numbers": []}]
        
        for i, match in enumerate(matches):
            article_start = match.start()
            article_title = match.group().strip()
            
            article_match = re.match(r"第([零一二三四五六七八九十百千万\d]+)条", article_title)
            article_number = article_match.group(1) if article_match else str(i + 1)
            
            article_end = matches[i + 1].start() if i + 1 < len(matches) else len(chapter_content)
            
            article_content = chapter_content[article_start:article_end].strip()
            
            has_table = bool(self.TABLE_PATTERN.search(article_content) or self.TABLE_PLACEHOLDER.search(article_content))
            
            articles.append({
                "title": article_title,
                "content": article_content,
                "article_numbers": [article_number],
                "has_table": has_table,
            })
        
        return articles

    def _semantic_split_article(
        self,
        content: str,
        doc_title: str,
        chapter_title: str,
        article_numbers: List[str],
    ) -> List[Dict]:
        """对超长条款按语义切分"""
        chunks = []
        
        sentences = self._split_into_sentences(content)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > 1500 and current_chunk:
                chunk_content = "".join(current_chunk)
                chunks.append(
                    self._create_chunk(
                        chunk_content,
                        "compliance_article_split",
                        {
                            "doc_title": doc_title,
                            "chapter_title": chapter_title,
                            "article_title": "",
                            "articles": article_numbers,
                            "is_split": True,
                        },
                        len(chunks),
                    )
                )
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunk_content = "".join(current_chunk)
            chunks.append(
                self._create_chunk(
                    chunk_content,
                    "compliance_article_split",
                    {
                        "doc_title": doc_title,
                        "chapter_title": chapter_title,
                        "article_title": "",
                        "articles": article_numbers,
                        "is_split": True,
                    },
                    len(chunks),
                )
            )
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本按句子拆分"""
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in "。；！？\n":
                sentences.append(current)
                current = ""
        
        if current:
            sentences.append(current)
        
        return sentences if sentences else [text]

    def _merge_short_articles(self, chunks: List[Dict]) -> List[Dict]:
        """合并短条款（仅在同一章节内合并）"""
        if not chunks:
            return []
        
        merged = []
        current = chunks[0].copy()
        
        for i in range(1, len(chunks)):
            chunk = chunks[i]
            
            current_chapter = current.get("metadata", {}).get("chapter_title", "")
            chunk_chapter = chunk.get("metadata", {}).get("chapter_title", "")
            
            current_articles = current.get("metadata", {}).get("articles", [])
            chunk_articles = chunk.get("metadata", {}).get("articles", [])
            
            same_chapter = current_chapter == chunk_chapter
            both_short = len(current["content"]) < 300 and len(chunk["content"]) < 300
            same_type = current.get("type") == chunk.get("type")
            
            if same_chapter and both_short and same_type:
                current["content"] += "\n\n" + chunk["content"]
                current["metadata"]["articles"] = current_articles + chunk_articles
            else:
                merged.append(current)
                current = chunk.copy()
        
        if current:
            merged.append(current)
        
        return merged


class HRDocumentChunker(BaseDocumentChunker):
    """
    HR文档切分器
    适合：员工手册、绩效考核办法、薪酬制度

    特点：
    - 按章节和条款切分
    - 保留表格（工资标准、考核表等）
    - 支持流程说明
    """

    def chunk(self, content: str, doc_id: str) -> List[Dict[str, Any]]:
        """切分HR文档"""
        logger.info(f"使用HR文档切分器处理文档: {doc_id}")

        chunks = []
        chunk_index = 0

        # 1. 提取文档标题
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else "未知文档"

        # 2. 保护表格
        protected_blocks = []
        table_pattern = r"\|[^\n]+\|\n\|[-:\|\s]+\|\n(?:\|[^\n]+\|\n?)+"
        for match in re.finditer(table_pattern, content):
            placeholder = f"__TABLE_{len(protected_blocks)}__"
            protected_blocks.append((placeholder, match.group()))
            content = content[: match.start()] + placeholder + content[match.end() :]

        # 3. 按章（##）切分
        sections = re.split(r"\n(?=##\s+)", content)

        for section in sections:
            if not section.strip():
                continue

            section_title = ""
            title_match = re.search(r"^##\s+(.+)$", section, re.MULTILINE)
            if title_match:
                section_title = title_match.group(1).strip()

            # 恢复表格
            restored_section = section
            for placeholder, original in protected_blocks:
                restored_section = restored_section.replace(placeholder, original)

            # 如果章节太大，按条切分
            if len(restored_section) > self.max_chars:
                # 尝试匹配"第X条"或"X.X"格式
                article_pattern = r"(?:^|\n)(#{3,4}\s*(?:第[一二三四五六七八九十百千万]+条|第\d+条|\d+\.\d+)[^\n]*)"
                articles = re.split(article_pattern, restored_section)

                current_article_title = ""
                current_content = []

                for article_part in articles:
                    if not article_part.strip():
                        continue

                    # 检查是否是条款标题
                    if re.match(
                        r"^#{3,4}\s*(?:第[一二三四五六七八九十百千万]+条|第\d+条|\d+\.)",
                        article_part.strip(),
                    ):
                        # 保存之前的条款
                        if current_content:
                            chunk_content = "\n".join(current_content)
                            chunks.append(
                                self._create_chunk(
                                    chunk_content,
                                    "hr_article",
                                    {
                                        "doc_title": doc_title,
                                        "section_title": section_title,
                                        "article_title": current_article_title,
                                    },
                                    chunk_index,
                                )
                            )
                            chunk_index += 1

                        current_article_title = article_part.strip()
                        current_content = [article_part]
                    else:
                        current_content.append(article_part)

                        # 检查大小
                        if sum(len(c) for c in current_content) > self.max_chars:
                            chunk_content = "\n".join(current_content)
                            chunks.append(
                                self._create_chunk(
                                    chunk_content,
                                    "hr_article",
                                    {
                                        "doc_title": doc_title,
                                        "section_title": section_title,
                                        "article_title": current_article_title,
                                    },
                                    chunk_index,
                                )
                            )
                            chunk_index += 1
                            current_content = []

                # 保存最后的条款
                if current_content:
                    chunk_content = "\n".join(current_content)
                    chunks.append(
                        self._create_chunk(
                            chunk_content,
                            "hr_article",
                            {
                                "doc_title": doc_title,
                                "section_title": section_title,
                                "article_title": current_article_title,
                            },
                            chunk_index,
                        )
                    )
                    chunk_index += 1
            else:
                chunks.append(
                    self._create_chunk(
                        restored_section,
                        "hr_section",
                        {
                            "doc_title": doc_title,
                            "section_title": section_title,
                        },
                        chunk_index,
                    )
                )
                chunk_index += 1

        # 合并小片段
        chunks = self._merge_small_chunks(chunks, min_chars=400)

        logger.info(f"HR文档切分完成: 共 {len(chunks)} 个片段")
        return chunks


class ProjectManagementChunker(BaseDocumentChunker):
    """
    项目管理文档切分器
    适合：开发流程规范、发布管理规范、项目管理文档

    特点：
    - 按流程阶段切分
    - 保留流程图和时序
    - 支持检查清单
    """

    def chunk(self, content: str, doc_id: str) -> List[Dict[str, Any]]:
        """切分项目管理文档"""
        logger.info(f"使用项目管理文档切分器处理文档: {doc_id}")

        chunks = []
        chunk_index = 0

        # 1. 提取文档标题
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else "未知文档"

        # 2. 保护流程图（``` 代码块）
        protected_blocks = []
        code_pattern = r"```[\s\S]*?```"
        for match in re.finditer(code_pattern, content):
            placeholder = f"__CODE_BLOCK_{len(protected_blocks)}__"
            protected_blocks.append((placeholder, match.group()))
            content = content[: match.start()] + placeholder + content[match.end() :]

        # 3. 保护检查清单表格
        checklist_pattern = r"\[([ x])\][^\n]+"
        checklist_matches = list(re.finditer(checklist_pattern, content))
        if checklist_matches:
            # 合并连续的检查清单
            checklist_blocks = []
            current_block = []
            last_end = 0

            for match in checklist_matches:
                if (
                    last_end == 0 or match.start() - last_end < 100
                ):  # 距离较近视为同一块
                    current_block.append(match.group())
                else:
                    if current_block:
                        placeholder = f"__CHECKLIST_{len(protected_blocks)}__"
                        block_content = "\n".join(current_block)
                        protected_blocks.append((placeholder, block_content))
                    current_block = [match.group()]
                last_end = match.end()

            if current_block:
                placeholder = f"__CHECKLIST_{len(protected_blocks)}__"
                block_content = "\n".join(current_block)
                protected_blocks.append((placeholder, block_content))

        # 替换检查清单块
        for placeholder, block_content in protected_blocks:
            if placeholder.startswith("__CHECKLIST_"):
                content = content.replace(block_content, placeholder)

        # 4. 按阶段/章节切分
        # 尝试按"第X章"或"##"切分
        chapter_pattern = r"(?:^|\n)(#{1,2}\s*(?:第[一二三四五六七八九十百千万]+章|第\d+章|流程|阶段)[^\n]*)"
        sections = re.split(chapter_pattern, content)

        current_chapter_title = ""

        for part in sections:
            if not part.strip():
                continue

            # 检查是否是章节标题
            if re.match(
                r"^#{1,2}\s*(?:第[一二三四五六七八九十百千万]+章|第\d+章|流程|阶段)",
                part.strip(),
            ):
                current_chapter_title = part.strip()
                continue

            # 恢复保护的内容
            restored_part = part
            for placeholder, original in protected_blocks:
                restored_part = restored_part.replace(placeholder, original)

            # 在章节内部按子流程切分（###）
            subsections = re.split(r"\n(?=###\s+)", restored_part)

            for subsection in subsections:
                if not subsection.strip():
                    continue

                subsection_title = ""
                title_match = re.search(r"^###\s+(.+)$", subsection, re.MULTILINE)
                if title_match:
                    subsection_title = title_match.group(1).strip()

                # 如果还是太大，按步骤切分
                if len(subsection) > self.max_chars:
                    # 尝试按步骤编号切分（Step X、1. 2. 3.等）
                    step_pattern = r"(?:^|\n)(?:Step\s+\d+|步骤\s*\d+|\d+\.\s+[^\n]+)"
                    steps = re.split(step_pattern, subsection)

                    current_content = []
                    current_size = 0

                    for step in steps:
                        if not step.strip():
                            continue

                        step_size = len(step)

                        if (
                            current_size + step_size > self.max_chars
                            and current_content
                        ):
                            # 保存当前片段
                            chunks.append(
                                self._create_chunk(
                                    "\n".join(current_content),
                                    "pm_step",
                                    {
                                        "doc_title": doc_title,
                                        "chapter_title": current_chapter_title,
                                        "section_title": subsection_title,
                                    },
                                    chunk_index,
                                )
                            )
                            chunk_index += 1
                            current_content = []
                            current_size = 0

                        current_content.append(step)
                        current_size += step_size

                    # 保存最后的步骤
                    if current_content:
                        chunks.append(
                            self._create_chunk(
                                "\n".join(current_content),
                                "pm_step",
                                {
                                    "doc_title": doc_title,
                                    "chapter_title": current_chapter_title,
                                    "section_title": subsection_title,
                                },
                                chunk_index,
                            )
                        )
                        chunk_index += 1
                else:
                    chunks.append(
                        self._create_chunk(
                            subsection,
                            "pm_section",
                            {
                                "doc_title": doc_title,
                                "chapter_title": current_chapter_title,
                                "section_title": subsection_title,
                            },
                            chunk_index,
                        )
                    )
                    chunk_index += 1

        # 合并小片段
        chunks = self._merge_small_chunks(chunks, min_chars=400)

        logger.info(f"项目管理文档切分完成: 共 {len(chunks)} 个片段")
        return chunks
