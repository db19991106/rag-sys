"""
分层LLM切分器 - 正则结构切分 + Embedding语义合并

核心流程:
1. 第一层：正则匹配标题，按章节结构切分
2. 第二层：识别表格/代码/流程，直接保护
3. 第三层：句子切分 → Embedding → 相似度合并
4. 第四层：后处理优化

特点:
- 句子级语义切分，粒度更精细
- 基于embedding相似度的智能合并
- 保护特殊内容（表格/代码/流程）完整性
- 可配置相似度阈值和token限制
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from models import LayeredLLMConfig
from utils.logger import logger


class ContentType(Enum):
    """内容类型枚举"""
    TABLE = "table"      # 表格
    CODE = "code"        # 代码块
    FLOW = "flow"        # 流程（Mermaid等）
    TEXT = "text"        # 普通文本


@dataclass
class Sentence:
    """句子单元"""
    text: str
    start: int = 0
    end: int = 0
    embedding: Optional[np.ndarray] = None


@dataclass
class Section:
    """章节单元"""
    level: int
    title: str
    content: str
    content_type: ContentType = ContentType.TEXT
    protected: bool = False  # 是否受保护（表格/代码/流程）


class LayeredLLMChunker:
    """
    分层LLM切分器
    
    使用方法:
        config = LayeredLLMConfig(
            max_chunk_tokens=512,
            similarity_threshold=0.7
        )
        chunker = LayeredLLMChunker(config)
        chunks = chunker.chunk(document_content, doc_id)
    """
    
    # ==================== 第一层：标题识别模式 ====================
    # 注意：\s* 允许零个或多个空格，(?!#) 排除更高级别标题的误匹配
    HEADING_PATTERNS = {
        1: re.compile(r'^#\s*(?!#)(.+)$', re.MULTILINE),      # # 标题（排除 ##）
        2: re.compile(r'^##\s*(?!#)(.+)$', re.MULTILINE),     # ## 标题（排除 ###）
        3: re.compile(r'^###\s*(?!#)(.+)$', re.MULTILINE),    # ### 标题（排除 ####）
        4: re.compile(r'^####\s*(?!#)(.+)$', re.MULTILINE),   # #### 标题（排除 #####）
    }
    
    # ==================== 第二层：特殊内容识别模式 ====================
    # Markdown表格模式
    TABLE_PATTERN = re.compile(
        r'(?:^|\n)(\|[^\n]+\|\n)(\|[-:|\s]+\|\n)(?:\|[^\n]+\|\n?)+',
        re.MULTILINE
    )
    
    # Mermaid流程图模式
    MERMAID_PATTERN = re.compile(
        r'```mermaid\s*([\s\S]*?)```',
        re.IGNORECASE
    )
    
    # 代码块模式
    CODE_BLOCK_PATTERN = re.compile(
        r'```(\w*)\s*([\s\S]*?)```',
        re.MULTILINE
    )
    
    # ==================== 第三层：句子切分模式 ====================
    # 中文句子分隔符
    SENTENCE_DELIMITERS = r'([。；！？])'
    
    def __init__(self, config: LayeredLLMConfig = None):
        self.config = config or LayeredLLMConfig()
        # Embedding服务延迟导入，避免循环依赖
        self._embedding_service = None
        # Token计数器延迟导入
        self._token_counter = None
        # 句子embedding缓存
        self._sentence_cache: Dict[str, np.ndarray] = {}
    
    @property
    def embedding_service(self):
        """延迟加载embedding服务"""
        if self._embedding_service is None:
            from services.embedding import embedding_service
            self._embedding_service = embedding_service
        return self._embedding_service
    
    @property
    def token_counter(self):
        """延迟加载token计数器"""
        if self._token_counter is None:
            from utils.token_counter import num_tokens_from_string
            self._token_counter = num_tokens_from_string
        return self._token_counter
    
    def chunk(self, content: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        执行分层LLM切分
        
        Args:
            content: 文档内容
            doc_id: 文档ID
            
        Returns:
            切分后的chunk列表，每个chunk包含:
            - content: 内容文本
            - type: 内容类型(table/code/flow/text)
            - metadata: 元数据字典
        """
        logger.info(f"[LayeredLLMChunker] 开始处理文档: {doc_id}, 内容长度: {len(content)}")
        
        # 第一层：结构化分层
        sections = self._layer1_structure_parse(content)
        logger.info(f"[Layer1] 识别到 {len(sections)} 个章节")
        
        # 第二层：内容类型识别与保护
        protected_sections = self._layer2_content_protection(sections)
        protected_count = sum(1 for s in protected_sections if s.protected)
        logger.info(f"[Layer2] 保护 {protected_count} 个特殊内容区域")
        
        # 第三层：语义切分
        chunks = self._layer3_semantic_chunking(protected_sections)
        logger.info(f"[Layer3] 语义切分完成，生成 {len(chunks)} 个chunks")
        
        # 第四层：后处理
        chunks = self._layer4_post_process(chunks)
        logger.info(f"[Layer4] 最终生成 {len(chunks)} 个chunks")
        
        return chunks
    
    # ============================================================
    # 第一层：正则结构化分层
    # ============================================================
    
    def _layer1_structure_parse(self, content: str) -> List[Section]:
        """
        第一层：文档结构解析
        
        对Markdown格式文档进行结构解析，实现基于章节的第一轮切分
        通过正则表达式精确匹配#标记，准确识别并提取标题层级结构
        """
        # 识别所有标题位置
        heading_matches = []
        for level, pattern in self.HEADING_PATTERNS.items():
            for match in pattern.finditer(content):
                heading_matches.append({
                    'level': level,
                    'title': match.group(1).strip(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        # 按位置排序
        heading_matches.sort(key=lambda x: x['start'])
        
        # 构建章节列表
        sections = []
        for i, match in enumerate(heading_matches):
            # 计算章节内容范围
            content_start = match['end']
            content_end = heading_matches[i + 1]['start'] if i + 1 < len(heading_matches) else len(content)
            
            section_content = content[content_start:content_end].strip()
            
            sections.append(Section(
                level=match['level'],
                title=match['title'],
                content=section_content
            ))
        
        # 如果没有识别到标题，整个文档作为一个章节
        if not sections:
            sections.append(Section(
                level=0,
                title='文档内容',
                content=content.strip()
            ))
        
        return sections
    
    # ============================================================
    # 第二层：内容类型识别与保护
    # ============================================================
    
    def _layer2_content_protection(self, sections: List[Section]) -> List[Section]:
        """
        第二层：内容类型识别与保护
        
        识别并标记需要保护的特殊内容（表格/代码/流程）
        保护的内容不参与语义切分，直接作为一个完整chunk
        """
        result = []
        
        for section in sections:
            content = section.content
            
            if not content.strip():
                # 如果有标题，将标题作为内容保留（无论是否以句号结尾）
                if section.title and section.title.strip():
                    section.content = section.title
                    logger.debug(f"章节 '{section.title}' 内容为空，将标题作为内容保留")
                else:
                    continue
            
            # 检测是否为特殊内容
            is_table = bool(self.TABLE_PATTERN.search(content))
            is_mermaid = bool(self.MERMAID_PATTERN.search(content))
            is_code = bool(self.CODE_BLOCK_PATTERN.search(content)) and not is_mermaid
            
            # 标记保护类型（优先级：表格 > 流程 > 代码）
            if is_table and self.config.preserve_table:
                section.content_type = ContentType.TABLE
                section.protected = True
                logger.debug(f"章节 '{section.title}' 识别为表格，标记保护")
            elif is_mermaid and self.config.preserve_flow:
                section.content_type = ContentType.FLOW
                section.protected = True
                logger.debug(f"章节 '{section.title}' 识别为流程图，标记保护")
            elif is_code and self.config.preserve_code:
                section.content_type = ContentType.CODE
                section.protected = True
                logger.debug(f"章节 '{section.title}' 识别为代码块，标记保护")
            
            result.append(section)
        
        return result
    
    # ============================================================
    # 第三层：句子级语义切分（核心）
    # ============================================================
    
    def _layer3_semantic_chunking(self, sections: List[Section]) -> List[Dict]:
        """
        第三层：句子级语义切分
        
        核心特性：
        - 每个章节(Section)独立处理，互不干扰
        - 句子相似度计算和合并只在同一章节内进行
        - 不同章节的句子不会被合并到同一个chunk
        - 保护内容（表格/代码/流程）直接作为一个完整chunk
        
        处理流程:
        1. 遍历每个章节
        2. 保护内容 → 直接作为chunk
        3. 普通文本 → 调用 _semantic_split 进行章节内语义切分
        """
        chunks = []
        chunk_index = 0
        
        logger.info(f"[Layer3] 开始处理 {len(sections)} 个章节，语义合并只在同章节内进行")
        
        for section in sections:
            if section.protected:
                # 保护内容直接作为一个chunk
                chunk = self._create_chunk(
                    content=self._add_section_context(section),
                    section_title=section.title,
                    content_type=section.content_type.value,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                logger.info(f"[Layer3] 章节 '{section.title}' 为保护内容({section.content_type.value})，直接作为chunk")
            else:
                # 普通文本：执行语义切分（只在同章节内合并）
                semantic_chunks = self._semantic_split(section)
                for chunk_content in semantic_chunks:
                    chunk = self._create_chunk(
                        content=chunk_content,
                        section_title=section.title,
                        content_type=ContentType.TEXT.value,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        
        return chunks
    
    def _semantic_split(self, section: Section) -> List[str]:
        """
        语义切分核心算法
        
        重要说明：
        - 本方法对单个章节(Section)进行独立处理
        - 句子的相似度计算和合并只在同一章节内进行
        - 不同章节的句子不会被合并到同一个chunk
        
        流程:
        1. 按句子切分
        2. 计算每个句子的embedding
        3. 计算相邻句子的相似度（只计算同一章节内的相邻句子）
        4. 合并相似度 >= 阈值的句子（确保不跨章节边界）
        5. 检查token限制，超过则分割
        """
        content = section.content
        
        # 添加章节处理日志
        section_title = section.title if section.title else "无标题"
        logger.info(f"[语义切分] 处理章节: '{section_title}' (层级{section.level}), 内容长度: {len(content)}")
        
        # Step 1: 按句子切分
        sentences = self._split_sentences(content)
        if not sentences:
            return []
        
        logger.info(f"[语义切分] 章节 '{section_title}' 切分为 {len(sentences)} 个句子")
        
        # 单句子直接返回
        if len(sentences) == 1:
            return [self._add_section_context(section, sentences[0].text)]
        
        # Step 2: 批量计算embedding
        embeddings = self._batch_encode([s.text for s in sentences])
        for i, sent in enumerate(sentences):
            sent.embedding = embeddings[i]
        
        # Step 3: 计算相邻句子相似度（只在同一章节内）
        similarities = self._calculate_similarities(sentences)
        
        # 输出相似度信息
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            max_sim = max(similarities)
            min_sim = min(similarities)
            logger.info(f"[语义切分] 章节 '{section_title}' 相似度统计: 平均={avg_sim:.3f}, 最大={max_sim:.3f}, 最小={min_sim:.3f}, 阈值={self.config.similarity_threshold}")
        
        # Step 4 & 5: 动态合并（考虑相似度和token限制，确保不跨章节）
        merged_groups = self._merge_by_similarity(sentences, similarities)
        
        logger.info(f"[语义切分] 章节 '{section_title}' 合并为 {len(merged_groups)} 个chunk组")
        
        # 构建结果
        result = []
        for group in merged_groups:
            chunk_text = "".join([s.text for s in group])
            chunk_text = self._add_section_context(section, chunk_text)
            result.append(chunk_text)
        
        return result
    
    def _split_sentences(self, content: str) -> List[Sentence]:
        """
        按句子边界切分
        
        使用中文句子分隔符：。；！？
        保留分隔符在句子末尾
        """
        sentences = []
        
        # 使用正则切分，保留分隔符
        parts = re.split(self.SENTENCE_DELIMITERS, content)
        
        current_text = ""
        current_start = 0
        pos = 0
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # 文本内容
                current_text += part
            else:  # 分隔符
                current_text += part
                
                # 检查是否为有效句子（非空）
                stripped = current_text.strip()
                if stripped:
                    sentences.append(Sentence(
                        text=stripped,
                        start=current_start,
                        end=current_start + len(current_text)
                    ))
                    pos = current_start + len(current_text)
                
                current_start = pos
                current_text = ""
        
        # 处理最后可能剩余的文本
        if current_text.strip():
            sentences.append(Sentence(
                text=current_text.strip(),
                start=current_start,
                end=current_start + len(current_text)
            ))
        
        return sentences
    
    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """批量计算embedding"""
        if not texts:
            return np.array([])
        
        # 检查缓存
        uncached_texts = []
        uncached_indices = []
        cached_embeddings = {}
        
        if self.config.enable_cache:
            for i, text in enumerate(texts):
                cache_key = text[:100]  # 使用前100字符作为缓存key
                if cache_key in self._sentence_cache:
                    cached_embeddings[i] = self._sentence_cache[cache_key]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # 对未缓存的文本进行编码
        if uncached_texts:
            try:
                new_embeddings = self.embedding_service.encode(uncached_texts)
                
                # 更新缓存
                if self.config.enable_cache:
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        cache_key = text[:100]
                        self._sentence_cache[cache_key] = embedding
                        
                        # 限制缓存大小
                        if len(self._sentence_cache) > 10000:
                            # 移除最早的缓存项
                            oldest_key = next(iter(self._sentence_cache))
                            del self._sentence_cache[oldest_key]
            except Exception as e:
                logger.error(f"批量编码失败: {e}")
                # 返回零向量作为fallback
                dim = self.embedding_service.get_dimension() or 768
                new_embeddings = np.zeros((len(uncached_texts), dim), dtype=np.float32)
        else:
            new_embeddings = np.array([])
        
        # 构建完整的结果
        result = [None] * len(texts)
        
        # 填充缓存的结果
        for i, emb in cached_embeddings.items():
            result[i] = emb
        
        # 填充新计算的结果
        for idx, emb in zip(uncached_indices, new_embeddings):
            result[idx] = emb
        
        return np.array(result)
    
    def _calculate_similarities(self, sentences: List[Sentence]) -> List[float]:
        """
        计算相邻句子的余弦相似度
        
        使用余弦相似度衡量句子间的语义相似程度
        """
        similarities = []
        
        for i in range(len(sentences) - 1):
            emb1 = sentences[i].embedding
            emb2 = sentences[i + 1].embedding
            
            if emb1 is not None and emb2 is not None:
                # 余弦相似度
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(emb1, emb2) / (norm1 * norm2)
                    similarities.append(float(sim))
                else:
                    similarities.append(0.0)
            else:
                similarities.append(0.0)
        
        return similarities
    
    def _merge_by_similarity(
        self, 
        sentences: List[Sentence], 
        similarities: List[float]
    ) -> List[List[Sentence]]:
        """
        基于相似度动态合并句子
        
        规则:
        - 相邻句子相似度 >= threshold → 尝试合并
        - 合并后token数 <= max_chunk_tokens → 确认合并
        - 否则 → 分割
        """
        if not sentences:
            return []
        
        groups = []
        current_group = [sentences[0]]
        current_tokens = self.token_counter(sentences[0].text)
        
        for i, sim in enumerate(similarities):
            next_sent = sentences[i + 1]
            next_tokens = self.token_counter(next_sent.text)
            
            # 判断是否可以合并
            can_merge = (
                sim >= self.config.similarity_threshold and
                current_tokens + next_tokens <= self.config.max_chunk_tokens
            )
            
            if can_merge:
                # 合并到当前组
                current_group.append(next_sent)
                current_tokens += next_tokens
            else:
                # 保存当前组，开始新组
                if current_group:
                    groups.append(current_group)
                current_group = [next_sent]
                current_tokens = next_tokens
        
        # 保存最后一组
        if current_group:
            groups.append(current_group)
        
        # 处理过短的组（小于min_chunk_tokens）
        groups = self._merge_short_groups(groups)
        
        return groups
    
    def _merge_short_groups(self, groups: List[List[Sentence]]) -> List[List[Sentence]]:
        """
        合并过短的组
        
        如果某个组的token数小于min_chunk_tokens，
        尝试与前一组或后一组合并
        """
        if not groups or len(groups) == 1:
            return groups
        
        result = []
        i = 0
        
        while i < len(groups):
            current_group = groups[i]
            current_tokens = sum(
                self.token_counter(s.text) for s in current_group
            )
            
            # 当前组过短，尝试与下一组合并
            while (current_tokens < self.config.min_chunk_tokens and 
                   i + 1 < len(groups)):
                next_group = groups[i + 1]
                next_tokens = sum(
                    self.token_counter(s.text) for s in next_group
                )
                
                if current_tokens + next_tokens <= self.config.max_chunk_tokens:
                    current_group = current_group + next_group
                    current_tokens += next_tokens
                    i += 1
                else:
                    break
            
            result.append(current_group)
            i += 1
        
        return result
    
    # ============================================================
    # 第四层：后处理优化
    # ============================================================
    
    def _layer4_post_process(self, chunks: List[Dict]) -> List[Dict]:
        """
        第四层：后处理优化
        
        1. 过滤无效片段（纯分隔线、空白等）
        2. 重建索引
        """
        if not chunks:
            return chunks
        
        # 过滤无效片段
        invalid_patterns = [
            r'^---+$',           # 纯分隔线
            r'^\*\*\*+$',        # 纯星号分隔线
            r'^___+$',           # 纯下划线分隔线
            r'^\s*$',            # 纯空白
        ]
        
        filtered = []
        for chunk in chunks:
            content = chunk['content'].strip()
            
            # 检查是否匹配无效模式
            is_invalid = False
            for pattern in invalid_patterns:
                if re.match(pattern, content):
                    is_invalid = True
                    break
            
            if not is_invalid and content:
                filtered.append(chunk)
        
        # 重建索引
        for i, chunk in enumerate(filtered):
            chunk['metadata']['chunk_index'] = i
        
        logger.info(f"[后处理] 移除了 {len(chunks) - len(filtered)} 个无效片段")
        
        return filtered
    
    # ============================================================
    # 辅助方法
    # ============================================================
    
    def _add_section_context(self, section: Section, content: str = None) -> str:
        """
        添加章节上下文
        
        为chunk内容添加章节标题作为上下文
        """
        text = content or section.content
        
        if section.title and section.level > 0:
            # 如果内容本身就是标题（空内容章节），直接返回带markdown格式的标题
            if text.strip() == section.title.strip():
                return "#" * section.level + " " + section.title
            # 正常情况：添加标题作为前缀
            prefix = "#" * section.level + " " + section.title + "\n"
            return prefix + text
        return text
    
    def _create_chunk(
        self, 
        content: str, 
        section_title: str,
        content_type: str,
        chunk_index: int
    ) -> Dict[str, Any]:
        """创建chunk对象"""
        return {
            "content": content,
            "type": content_type,
            "metadata": {
                "chunk_index": chunk_index,
                "section_title": section_title,
                "content_type": content_type,
                "char_count": len(content),
                "token_count": self.token_counter(content) if content else 0
            }
        }
    
    def clear_cache(self):
        """清空embedding缓存"""
        self._sentence_cache.clear()
        logger.info("[LayeredLLMChunker] Embedding缓存已清空")


# 便捷函数
def layered_llm_chunk(content: str, doc_id: str, **kwargs) -> List[Dict[str, Any]]:
    """
    便捷函数：执行分层LLM切分
    
    Args:
        content: 文档内容
        doc_id: 文档ID
        **kwargs: 配置参数 (max_chunk_tokens, similarity_threshold等)
    
    Returns:
        切分后的chunk列表
    """
    config = LayeredLLMConfig(**kwargs)
    chunker = LayeredLLMChunker(config)
    return chunker.chunk(content, doc_id)
