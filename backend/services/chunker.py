"""
文档切分器 - RAGFlow 风格的智能切分系统
参考 RAGFlow 的实现，支持多种切分策略和高级特性
"""

from typing import List, Tuple, Union, Optional
import re
from copy import deepcopy

from models import ChunkInfo, ChunkType, ChunkConfig
from utils.logger import logger
from utils.token_counter import num_tokens_from_string


class RAGFlowChunker:
    """
    RAGFlow 风格的文档切分器
    
    核心特性：
    1. 基于 Token 数量的精确切分
    2. 支持自定义分隔符
    3. 支持重叠切分
    4. 支持子分隔符（细粒度切分）
    5. 智能标题识别
    """

    def __init__(self):
        pass

    def chunk(self, content: str, doc_id: str, config: ChunkConfig) -> List[ChunkInfo]:
        """
        根据配置切分文档
        
        Args:
            content: 文档内容
            doc_id: 文档ID
            config: 切分配置
            
        Returns:
            切分后的片段列表
        """
        if not content or not content.strip():
            logger.warning("文档内容为空，无法切分")
            return []

        chunks = []

        try:
            # 根据切分类型选择策略
            if config.type == ChunkType.NAIVE:
                chunk_texts = self._naive_merge(
                    content,
                    chunk_token_num=config.chunk_token_size,
                    delimiter=config.delimiters[0] if config.delimiters else "\n。；！？",
                    overlapped_percent=config.overlapped_percent
                )
            elif config.type == ChunkType.QA:
                chunk_texts = self._qa_merge(
                    content,
                    chunk_token_num=config.chunk_token_size,
                    overlapped_percent=config.overlapped_percent
                )
            elif config.type == ChunkType.PAPER:
                chunk_texts = self._paper_merge(
                    content,
                    chunk_token_num=config.chunk_token_size,
                    overlapped_percent=config.overlapped_percent
                )
            elif config.type == ChunkType.LAWS:
                chunk_texts = self._laws_merge(
                    content,
                    chunk_token_num=config.chunk_token_size,
                    overlapped_percent=config.overlapped_percent
                )
            elif config.type == ChunkType.BOOK:
                chunk_texts = self._book_merge(
                    content,
                    chunk_token_num=config.chunk_token_size,
                    overlapped_percent=config.overlapped_percent
                )
            elif config.type == ChunkType.TABLE:
                chunk_texts = self._table_merge(content)
            elif config.type == ChunkType.CHAR:
                chunk_texts = self._chunk_by_char(
                    content,
                    config.chunk_token_size,
                    config.overlapped_percent
                )
            elif config.type == ChunkType.SENTENCE:
                chunk_texts = self._chunk_by_sentence(
                    content,
                    config.chunk_token_size,
                    config.overlapped_percent
                )
            elif config.type == ChunkType.PARAGRAPH:
                chunk_texts = self._chunk_by_paragraph(content)
            elif config.type == ChunkType.CUSTOM:
                chunk_texts = self._custom_merge(
                    content,
                    config.chunk_token_size,
                    config.delimiters[0] if config.delimiters else "\n",
                    config.overlapped_percent,
                    config.enable_children,
                    config.children_delimiters
                )
            else:
                logger.error(f"不支持的切分类型: {config.type}")
                return []

            # 转换为 ChunkInfo 对象
            chunk_infos = []
            for i, chunk_text in enumerate(chunk_texts):
                if not chunk_text or not chunk_text.strip():
                    continue
                    
                chunk_info = ChunkInfo(
                    id=f"{doc_id}_chunk_{i + 1}",
                    document_id=doc_id,
                    num=i + 1,
                    content=chunk_text.strip(),
                    length=len(chunk_text.strip()),
                    embedding_status="pending"
                )
                chunk_infos.append(chunk_info)

            logger.info(f"文档切分完成: {len(chunk_infos)} 个片段 (方式: {config.type})")
            return chunk_infos

        except Exception as e:
            logger.error(f"文档切分失败: {str(e)}")
            import traceback
            logger.error(f"堆栈跟踪: {traceback.format_exc()}")
            return []

    def _naive_merge(
        self,
        content: str,
        chunk_token_num: int = 128,
        delimiter: str = "\n。；！？",
        overlapped_percent: float = 0
    ) -> List[str]:
        """
        朴素切分 - RAGFlow 的核心切分算法
        
        参考 RAGFlow 的 naive_merge 函数实现
        
        Args:
            content: 文档内容
            chunk_token_num: 每个 chunk 的最大 token 数
            delimiter: 分隔符字符串
            overlapped_percent: 重叠百分比 (0-1)
            
        Returns:
            切分后的文本列表
        """
        if not content:
            return []
        
        cks = [""]
        tk_nums = [0]
        
        def add_chunk(t: str):
            nonlocal cks, tk_nums
            tnum = num_tokens_from_string(t)
            
            # 如果 token 数少于 8，不作为独立 chunk
            if tnum < 8:
                return
            
            # 确保 chunk 的长度不超过 chunk_token_num
            if cks[-1] == "" or tk_nums[-1] > chunk_token_num * (100 - overlapped_percent) / 100.:
                # 创建新的 chunk，并添加重叠内容
                if cks:
                    overlapped = cks[-1]
                    # 添加重叠部分
                    t = overlapped[int(len(overlapped) * (100 - overlapped_percent) / 100.):] + t
                cks.append(t)
                tk_nums.append(tnum)
            else:
                # 添加到当前 chunk
                cks[-1] += t
                tk_nums[-1] += tnum
        
        # 处理自定义分隔符（使用反引号包裹）
        custom_delimiters = [m.group(1) for m in re.finditer(r"`([^`]+)`", delimiter)]
        has_custom = bool(custom_delimiters)
        
        if has_custom:
            # 使用自定义分隔符进行切分
            custom_pattern = "|".join(re.escape(t) for t in sorted(set(custom_delimiters), key=len, reverse=True))
            cks, tk_nums = [], []
            
            split_secs = re.split(r"(%s)" % custom_pattern, content, flags=re.DOTALL)
            for sec in split_secs:
                if re.fullmatch(custom_pattern, sec or ""):
                    continue
                text = sec
                local_tnum = num_tokens_from_string(text)
                
                if local_tnum < 8:
                    continue
                
                cks.append(text)
                tk_nums.append(local_tnum)
        else:
            # 使用普通分隔符
            # 解析分隔符
            dels = []
            s = 0
            for m in re.finditer(r"`([^`]+)`", delimiter, re.I):
                f, t = m.span()
                dels.append(m.group(1))
                dels.extend(list(delimiter[s: f]))
                s = t
            if s < len(delimiter):
                dels.extend(list(delimiter[s:]))
            
            dels = [re.escape(d) for d in dels if d]
            dels = [d for d in dels if d]
            
            if dels:
                pattern = "|".join(dels)
                secs = re.split(r"(%s)" % pattern, content)
                for sec in secs:
                    if not sec or re.match(f"^{pattern}$", sec):
                        continue
                    add_chunk(sec)
            else:
                # 没有分隔符，整体作为一个 chunk
                add_chunk(content)
        
        # 过滤空 chunk
        return [c for c in cks if c and c.strip()]

    def _qa_merge(
        self,
        content: str,
        chunk_token_num: int = 256,
        overlapped_percent: float = 0
    ) -> List[str]:
        """
        问答对切分
        
        识别问答对格式，如：
        - "第X问"
        - "QUESTION X"
        - "问题："
        
        Args:
            content: 文档内容
            chunk_token_num: 每个 chunk 的最大 token 数
            overlapped_percent: 重叠百分比
            
        Returns:
            切分后的文本列表
        """
        # 问答对模式
        question_patterns = [
            r"第([零一二三四五六七八九十百0-9]+)问",
            r"第([0-9]+)问",
            r"QUESTION (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
            r"QUESTION (I+V?|VI*|XI|IX|X)",
            r"问题[：:]\s*",
            r"Q[0-9]+[：:]\s*",
        ]
        
        # 尝试识别问答对
        for pattern in question_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
            if len(matches) >= 2:  # 至少找到2个问题
                # 按问题切分
                chunks = []
                for i in range(len(matches)):
                    start = matches[i].start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                    chunk_text = content[start:end].strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                
                # 如果切分后的 chunk 太大，进一步切分
                result = []
                for chunk in chunks:
                    if num_tokens_from_string(chunk) > chunk_token_num:
                        sub_chunks = self._naive_merge(
                            chunk,
                            chunk_token_num,
                            "\n",
                            overlapped_percent
                        )
                        result.extend(sub_chunks)
                    else:
                        result.append(chunk)
                
                return result
        
        # 如果没有识别到问答对，使用朴素切分
        return self._naive_merge(
            content,
            chunk_token_num,
            "\n",
            overlapped_percent
        )

    def _paper_merge(
        self,
        content: str,
        chunk_token_num: int = 1024,
        overlapped_percent: float = 0.15
    ) -> List[str]:
        """
        论文切分
        
        识别论文的标题层级结构
        
        Args:
            content: 文档内容
            chunk_token_num: 每个 chunk 的最大 token 数
            overlapped_percent: 重叠百分比
            
        Returns:
            切分后的文本列表
        """
        # 论文标题模式
        title_patterns = [
            r"^#{1,6}\s+.+$",  # Markdown 标题
            r"^第[零一二三四五六七八九十百0-9]+章",
            r"^第[零一二三四五六七八九十百0-9]+节",
            r"^[0-9]+\.[0-9.]+\s+.+$",  # 数字编号
            r"^(ABSTRACT|INTRODUCTION|METHODOLOGY|RESULTS|DISCUSSION|CONCLUSION|REFERENCES)\b",
        ]
        
        # 查找所有标题
        titles = []
        for pattern in title_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                titles.append((match.start(), match.end(), match.group()))
        
        # 按位置排序
        titles.sort(key=lambda x: x[0])
        
        if len(titles) >= 2:
            # 按标题切分
            chunks = []
            for i in range(len(titles)):
                start = titles[i][0]
                end = titles[i + 1][0] if i + 1 < len(titles) else len(content)
                chunk_text = content[start:end].strip()
                if chunk_text:
                    chunks.append(chunk_text)
            
            # 如果切分后的 chunk 太大，进一步切分
            result = []
            for chunk in chunks:
                if num_tokens_from_string(chunk) > chunk_token_num:
                    sub_chunks = self._naive_merge(
                        chunk,
                        chunk_token_num,
                        "\n\n",
                        overlapped_percent
                    )
                    result.extend(sub_chunks)
                else:
                    result.append(chunk)
            
            return result
        
        # 如果没有识别到标题，使用朴素切分
        return self._naive_merge(
            content,
            chunk_token_num,
            "\n\n",
            overlapped_percent
        )

    def _laws_merge(
        self,
        content: str,
        chunk_token_num: int = 256,
        overlapped_percent: float = 0.1
    ) -> List[str]:
        """
        法律文档切分
        
        识别法律条文结构
        
        Args:
            content: 文档内容
            chunk_token_num: 每个 chunk 的最大 token 数
            overlapped_percent: 重叠百分比
            
        Returns:
            切分后的文本列表
        """
        # 法律条文模式
        law_patterns = [
            r"第[零一二三四五六七八九十百0-9]+(条|节|章|编)",
            r"[零一二三四五六七八九十百]+[、是 　]",
            r"[\(（][零一二三四五六七八九十百]+[\)）]",
            r"[\(（][0-9]+[\)）]",
            r"[0-9]+(、|\.[　 ]|）|\.[^0-9./a-zA-Z_%><-]{4,})",
        ]
        
        # 查找所有条文
        sections = []
        for pattern in law_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                sections.append((match.start(), match.group()))
        
        if len(sections) >= 2:
            # 按条文切分
            chunks = []
            sections.sort(key=lambda x: x[0])
            for i in range(len(sections)):
                start = sections[i][0]
                end = sections[i + 1][0] if i + 1 < len(sections) else len(content)
                chunk_text = content[start:end].strip()
                if chunk_text:
                    chunks.append(chunk_text)
            
            # 如果切分后的 chunk 太大，进一步切分
            result = []
            for chunk in chunks:
                if num_tokens_from_string(chunk) > chunk_token_num:
                    sub_chunks = self._naive_merge(
                        chunk,
                        chunk_token_num,
                        "\n",
                        overlapped_percent
                    )
                    result.extend(sub_chunks)
                else:
                    result.append(chunk)
            
            return result
        
        # 如果没有识别到条文，使用朴素切分
        return self._naive_merge(
            content,
            chunk_token_num,
            "\n。；！？",
            overlapped_percent
        )

    def _book_merge(
        self,
        content: str,
        chunk_token_num: int = 512,
        overlapped_percent: float = 0.1
    ) -> List[str]:
        """
        书籍切分
        
        识别书籍的章节结构
        
        Args:
            content: 文档内容
            chunk_token_num: 每个 chunk 的最大 token 数
            overlapped_percent: 重叠百分比
            
        Returns:
            切分后的文本列表
        """
        # 书籍章节模式
        chapter_patterns = [
            r"^第[零一二三四五六七八九十百0-9]+章",
            r"^第[零一二三四五六七八九十百0-9]+节",
            r"^第[零一二三四五六七八九十百0-9]+部分",
            r"^PART (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
            r"^Chapter (I+V?|VI*|XI|IX|X)",
            r"^Section [0-9]+",
            r"^Article [0-9]+",
        ]
        
        # 查找所有章节
        chapters = []
        for pattern in chapter_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                chapters.append((match.start(), match.group()))
        
        if len(chapters) >= 2:
            # 按章节切分
            chunks = []
            chapters.sort(key=lambda x: x[0])
            for i in range(len(chapters)):
                start = chapters[i][0]
                end = chapters[i + 1][0] if i + 1 < len(chapters) else len(content)
                chunk_text = content[start:end].strip()
                if chunk_text:
                    chunks.append(chunk_text)
            
            # 如果切分后的 chunk 太大，进一步切分
            result = []
            for chunk in chunks:
                if num_tokens_from_string(chunk) > chunk_token_num:
                    sub_chunks = self._naive_merge(
                        chunk,
                        chunk_token_num,
                        "\n\n",
                        overlapped_percent
                    )
                    result.extend(sub_chunks)
                else:
                    result.append(chunk)
            
            return result
        
        # 如果没有识别到章节，使用朴素切分
        return self._naive_merge(
            content,
            chunk_token_num,
            "\n\n",
            overlapped_percent
        )

    def _table_merge(self, content: str) -> List[str]:
        """
        表格切分
        
        识别 Markdown 和 HTML 表格
        
        Args:
            content: 文档内容
            
        Returns:
            切分后的文本列表
        """
        # Markdown 表格模式
        markdown_table_pattern = re.compile(
            r"""
            (?:\n|^)
            (?:\|.*?\|.*?\n)
            (?:\|(?:\s*[:-]+[-| :]*\s*)\|.*?\n)
            (?:\|.*?\|.*?\n)+
            """,
            re.VERBOSE,
        )
        
        # HTML 表格模式
        html_table_pattern = re.compile(
            r"<table[^>]*>.*?</table>",
            re.IGNORECASE | re.DOTALL
        )
        
        tables = []
        remaining_text = content
        
        # 提取 Markdown 表格
        for match in markdown_table_pattern.finditer(remaining_text):
            tables.append(match.group())
        
        # 提取 HTML 表格
        for match in html_table_pattern.finditer(remaining_text):
            tables.append(match.group())
        
        # 移除表格后的剩余文本
        for table in tables:
            remaining_text = remaining_text.replace(table, "\n\n")
        
        # 切分剩余文本
        text_chunks = self._naive_merge(remaining_text, 256, "\n", 0)
        
        # 添加表格
        result = []
        for chunk in text_chunks:
            if chunk.strip():
                result.append(chunk)
        
        for table in tables:
            if table.strip():
                result.append(table)
        
        return result

    def _custom_merge(
        self,
        content: str,
        chunk_token_num: int,
        delimiter: str,
        overlapped_percent: float,
        enable_children: bool,
        children_delimiters: List[str]
    ) -> List[str]:
        """
        自定义切分
        
        支持主分隔符和子分隔符
        
        Args:
            content: 文档内容
            chunk_token_num: 每个 chunk 的最大 token 数
            delimiter: 主分隔符
            overlapped_percent: 重叠百分比
            enable_children: 是否启用子分隔符
            children_delimiters: 子分隔符列表
            
        Returns:
            切分后的文本列表
        """
        if not delimiter:
            return [content]
        
        # 使用主分隔符切分
        primary_chunks = self._naive_merge(
            content,
            chunk_token_num,
            delimiter,
            overlapped_percent
        )
        
        if not enable_children or not children_delimiters:
            return primary_chunks
        
        # 如果启用子分隔符，进一步切分
        result = []
        children_pattern = "|".join(re.escape(d) for d in children_delimiters if d)
        
        for chunk in primary_chunks:
            if num_tokens_from_string(chunk) > chunk_token_num:
                # 使用子分隔符进一步切分
                sub_chunks = re.split(r"(%s)" % children_pattern, chunk)
                for sub_chunk in sub_chunks:
                    if sub_chunk and not re.match(f"^{children_pattern}$", sub_chunk):
                        result.append(sub_chunk)
            else:
                result.append(chunk)
        
        return result

    def _chunk_by_char(
        self,
        content: str,
        chunk_size: int,
        overlapped_percent: float
    ) -> List[str]:
        """按字符切分"""
        overlap_size = int(chunk_size * overlapped_percent)
        chunks = []
        index = 0
        
        while index < len(content):
            end = min(index + chunk_size, len(content))
            chunk = content[index:end]
            chunks.append(chunk)
            index += chunk_size - overlap_size
        
        return chunks

    def _chunk_by_sentence(
        self,
        content: str,
        chunk_size: int,
        overlapped_percent: float
    ) -> List[str]:
        """按句子切分"""
        # 使用正则表达式分割句子
        sentence_pattern = r'([。！？.!?])'
        sentences = re.split(sentence_pattern, content)
        
        # 重新组合句子和标点
        combined_sentences = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            else:
                if sentences[i]:
                    combined_sentences.append(sentences[i])
        
        chunks = []
        index = 0
        overlap_size = int(len(combined_sentences) * overlapped_percent)
        
        while index < len(combined_sentences):
            # 计算包含重叠的句子范围
            start = max(0, index - overlap_size)
            end = min(index + chunk_size, len(combined_sentences))
            
            chunk_content = ''.join(combined_sentences[start:end])
            if chunk_content.strip():
                chunks.append(chunk_content.strip())
            
            index += chunk_size
        
        return chunks

    def _chunk_by_paragraph(self, content: str) -> List[str]:
        """按段落切分"""
        # 使用换行符分割段落
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                chunks.append(paragraph)
        
        return chunks


# 保持向后兼容
class Chunker(RAGFlowChunker):
    """Chunker 的别名，保持向后兼容"""
    pass