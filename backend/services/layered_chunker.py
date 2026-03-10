"""
分层智能切分器 - 三层递进式文档处理系统

第一层：文档结构化分层处理
  - 对Markdown格式文档进行结构解析，实现基于章节的第一轮切分
  - 通过正则表达式精确匹配#标记，准确识别并提取标题层级结构
  - 保留原始文档的章节逻辑关系，建立清晰的文档结构树

第二层：内容类型智能识别系统
  - 表格区域：通过Markdown表格特有的竖线分隔符模式进行识别
  - 流程区域：通过关键词匹配、流程箭头符号、Mermaid语法综合判断
  - 代码块：识别 ``` 代码块
  - 列表：识别 - * 或数字列表
  - 普通文本：排除法识别

第三层：差异化内容切分策略
  - 表格区域：完整保留，大表格智能拆分
  - 流程区域：保持完整性，不可分割
  - 代码块：保持完整性
  - 普通文本：按语义边界切分 + overlap重叠区
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from utils.logger import logger


class ContentType(Enum):
    """内容类型枚举"""
    TABLE = "table"        # 表格
    FLOW = "flow"          # 流程
    CODE = "code"          # 代码块
    LIST = "list"          # 列表
    TEXT = "text"          # 普通文本


@dataclass
class StructureNode:
    """结构树节点"""
    level: int                    # 标题层级 (0-6, 0为根节点)
    title: str                    # 标题文本
    content: str                  # 章节内容（不含标题）
    children: List['StructureNode'] = field(default_factory=list)
    content_type: Optional[ContentType] = None
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class LayeredChunkConfig:
    """分层切分配置"""
    max_chunk_size: int = 1500     # 最大chunk大小（字符数）- 增大以减少碎片
    overlap: int = 100             # 重叠区大小（字符数）
    min_chunk_size: int = 50       # 最小chunk大小（字符数）- 减小以减少不当合并
    preserve_table: bool = True    # 保护表格完整性
    preserve_flow: bool = True     # 保护流程完整性
    preserve_code: bool = True     # 保护代码块完整性
    max_table_rows: int = 20       # 大表格拆分时的最大行数
    min_confidence: float = 0.95   # 类型识别置信度阈值（预留）


class LayeredChunker:
    """
    三层递进式智能切分器
    
    使用方法:
        config = LayeredChunkConfig(max_chunk_size=800, overlap=100)
        chunker = LayeredChunker(config)
        chunks = chunker.chunk(document_content, doc_id)
    """
    
    # ==================== 第一层：标题识别模式 ====================
    HEADING_PATTERNS = {
        1: re.compile(r'^#\s+(.+)$', re.MULTILINE),
        2: re.compile(r'^##\s+(.+)$', re.MULTILINE),
        3: re.compile(r'^###\s+(.+)$', re.MULTILINE),
        4: re.compile(r'^####\s+(.+)$', re.MULTILINE),
        5: re.compile(r'^#####\s+(.+)$', re.MULTILINE),
        6: re.compile(r'^######\s+(.+)$', re.MULTILINE),
    }
    
    # ==================== 第二层：内容类型识别模式 ====================
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
    
    # 代码块模式（排除mermaid）
    CODE_BLOCK_PATTERN = re.compile(
        r'```(\w*)\s*([\s\S]*?)```',
        re.MULTILINE
    )
    
    # 列表模式
    LIST_PATTERN = re.compile(
        r'(?:^|\n)(?:[-*+]|\d+\.)\s+.+(?:\n(?:[-*+]|\d+\.)\s+.+)*',
        re.MULTILINE
    )
    
    # 流程关键词
    FLOW_KEYWORDS = [
        "流程", "审批", "步骤", "环节", "程序", "路径",
        "申请", "审核", "批准", "确认", "提交", "验收",
        "办理", "登记", "备案", "签署", "流转"
    ]
    
    # 流程箭头符号
    FLOW_ARROWS = ["→", "->", "➜", "➔", "⇒", "⟶", "➡", "↴"]
    
    def __init__(self, config: LayeredChunkConfig = None):
        self.config = config or LayeredChunkConfig()
    
    def chunk(self, content: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        执行三层递进式切分
        
        Args:
            content: 文档内容
            doc_id: 文档ID
            
        Returns:
            切分后的chunk列表，每个chunk包含:
            - content: 内容文本
            - type: 内容类型(table/flow/code/list/text)
            - metadata: 元数据字典
        """
        logger.info(f"[LayeredChunker] 开始处理文档: {doc_id}, 内容长度: {len(content)}")
        
        # 第一层：结构化分层
        structure_tree, sections = self._layer1_structure_parse(content)
        logger.info(f"[Layer1] 识别到 {len(sections)} 个章节")
        
        # 第二层：内容类型识别
        typed_sections = self._layer2_content_recognition(sections)
        type_stats = {}
        for s in typed_sections:
            t = s.get('content_type', ContentType.TEXT).value
            type_stats[t] = type_stats.get(t, 0) + 1
        logger.info(f"[Layer2] 内容类型分布: {type_stats}")
        
        # 第三层：差异化切分
        chunks = self._layer3_differentiated_chunking(typed_sections, doc_id)
        logger.info(f"[Layer3] 生成 {len(chunks)} 个chunks")
        
        # 第四层：后处理优化
        chunks = self._post_process_chunks(chunks)
        logger.info(f"[后处理] 最终生成 {len(chunks)} 个chunks")
        
        return chunks
    
    # ============================================================
    # 第一层：文档结构化分层处理
    # ============================================================
    
    def _layer1_structure_parse(self, content: str) -> Tuple[StructureNode, List[Dict]]:
        """
        第一层：文档结构解析
        
        对Markdown格式文档进行结构解析，实现基于章节的第一轮切分
        通过正则表达式精确匹配#标记，准确识别并提取标题层级结构
        
        Returns:
            (结构树根节点, 扁平化章节列表)
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
            
            sections.append({
                'level': match['level'],
                'title': match['title'],
                'content': section_content,
                'full_content': f"{'#' * match['level']} {match['title']}\n{section_content}",
                'start_pos': match['start'],
                'end_pos': content_end
            })
        
        # 如果没有识别到标题，整个文档作为一个章节
        if not sections:
            sections.append({
                'level': 0,
                'title': '文档内容',
                'content': content,
                'full_content': content,
                'start_pos': 0,
                'end_pos': len(content)
            })
        
        # 构建结构树（用于可视化或调试）
        root = self._build_structure_tree(sections)
        
        return root, sections
    
    def _build_structure_tree(self, sections: List[Dict]) -> StructureNode:
        """构建层级结构树"""
        if not sections:
            return StructureNode(level=0, title="root", content="")
        
        root = StructureNode(level=0, title="root", content="")
        stack = [root]
        
        for section in sections:
            node = StructureNode(
                level=section['level'],
                title=section['title'],
                content=section['content'],
                start_pos=section['start_pos'],
                end_pos=section['end_pos']
            )
            
            # 找到合适的父节点
            while len(stack) > 1 and stack[-1].level >= node.level:
                stack.pop()
            
            stack[-1].children.append(node)
            stack.append(node)
        
        return root
    
    # ============================================================
    # 第二层：内容类型智能识别系统
    # ============================================================
    
    def _layer2_content_recognition(self, sections: List[Dict]) -> List[Dict]:
        """
        第二层：内容类型智能识别
        
        对每个已切分章节进行深度内容分析，实现内容类型的精准识别
        """
        typed_sections = []
        
        for section in sections:
            content = section['content']
            
            if not content.strip():
                section['content_type'] = ContentType.TEXT
                section['confidence'] = 1.0
                section['regions'] = []
                typed_sections.append(section)
                continue
            
            # 识别内容类型和区域
            content_type, confidence, regions = self._detect_content_type(content)
            
            section['content_type'] = content_type
            section['confidence'] = confidence
            section['regions'] = regions
            
            typed_sections.append(section)
        
        return typed_sections
    
    def _detect_content_type(self, content: str) -> Tuple[ContentType, float, List[Dict]]:
        """
        检测内容类型
        
        Returns:
            (主类型, 置信度, 区域列表)
        """
        regions = []
        total_length = len(content)
        
        if total_length == 0:
            return ContentType.TEXT, 1.0, []
        
        # 1. 检测Mermaid流程图（优先级最高）
        for match in self.MERMAID_PATTERN.finditer(content):
            regions.append({
                'type': ContentType.FLOW,
                'start': match.start(),
                'end': match.end(),
                'content': match.group()
            })
        
        # 2. 检测代码块（排除mermaid）
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            # 检查是否已被mermaid覆盖
            overlap = False
            for region in regions:
                if match.start() >= region['start'] and match.end() <= region['end']:
                    overlap = True
                    break
            if not overlap:
                regions.append({
                    'type': ContentType.CODE,
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group()
                })
        
        # 3. 检测表格
        for match in self.TABLE_PATTERN.finditer(content):
            # 检查是否已被其他类型覆盖
            overlap = False
            for region in regions:
                if match.start() >= region['start'] and match.end() <= region['end']:
                    overlap = True
                    break
            if not overlap:
                regions.append({
                    'type': ContentType.TABLE,
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group()
                })
        
        # 4. 检测文本流程描述
        text_flow_regions = self._detect_text_flow(content, regions)
        regions.extend(text_flow_regions)
        
        # 5. 检测列表
        for match in self.LIST_PATTERN.finditer(content):
            # 检查是否已被其他类型覆盖
            overlap = False
            for region in regions:
                if match.start() >= region['start'] and match.end() <= region['end']:
                    overlap = True
                    break
            if not overlap:
                regions.append({
                    'type': ContentType.LIST,
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group()
                })
        
        # 6. 填充普通文本区域
        regions = self._fill_text_regions(content, regions)
        
        # 计算主类型和置信度
        type_lengths = {t: 0 for t in ContentType}
        for region in regions:
            type_lengths[region['type']] += (region['end'] - region['start'])
        
        main_type = max(type_lengths, key=type_lengths.get)
        confidence = type_lengths[main_type] / total_length if total_length > 0 else 1.0
        
        return main_type, confidence, regions
    
    def _detect_text_flow(self, content: str, existing_regions: List[Dict]) -> List[Dict]:
        """
        检测文本形式的流程描述
        
        通过多维度识别机制：
        - 关键词匹配（"流程"、"审批"、"步骤"等）
        - 流程箭头符号（→、->等）
        - 步骤编号模式
        """
        flow_regions = []
        
        # 按段落分析
        paragraphs = re.split(r'\n\s*\n', content)
        current_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_start = content.find(para, current_pos)
            if para_start == -1:
                continue
            para_end = para_start + len(para)
            current_pos = para_end
            
            # 检查是否已被覆盖
            overlap = False
            for region in existing_regions:
                if para_start >= region['start'] and para_end <= region['end']:
                    overlap = True
                    break
            
            if overlap:
                continue
            
            # 流程特征检测
            is_flow = False
            flow_score = 0
            
            # 特征1: 包含流程关键词
            keyword_count = sum(1 for kw in self.FLOW_KEYWORDS if kw in para)
            if keyword_count >= 2:
                is_flow = True
                flow_score += keyword_count * 2
            
            # 特征2: 包含流程箭头序列
            arrow_count = sum(para.count(arrow) for arrow in self.FLOW_ARROWS)
            if arrow_count >= 2:
                is_flow = True
                flow_score += arrow_count * 3
            
            # 特征3: 步骤编号模式
            step_patterns = [
                (r'\d+\.\s+', '数字编号'),           # 1. 2. 3.
                (r'第[一二三四五六七八九十]+步', '中文步骤'),  # 第一步、第二步
                (r'第\d+步', '数字步骤'),           # 第1步、第2步
                (r'[一二三四五六七八九十]、', '中文序号'),   # 一、二、三、
            ]
            for pattern, name in step_patterns:
                matches = re.findall(pattern, para)
                if len(matches) >= 3:
                    is_flow = True
                    flow_score += len(matches) * 2
            
            if is_flow and flow_score >= 4:
                flow_regions.append({
                    'type': ContentType.FLOW,
                    'start': para_start,
                    'end': para_end,
                    'content': para
                })
        
        return flow_regions
    
    def _fill_text_regions(self, content: str, regions: List[Dict]) -> List[Dict]:
        """填充普通文本区域"""
        # 按位置排序
        regions.sort(key=lambda x: x['start'])
        
        # 找出空白区域
        text_regions = []
        current_pos = 0
        
        for region in regions:
            if region['start'] > current_pos:
                text_content = content[current_pos:region['start']]
                if text_content.strip():
                    text_regions.append({
                        'type': ContentType.TEXT,
                        'start': current_pos,
                        'end': region['start'],
                        'content': text_content
                    })
            current_pos = max(current_pos, region['end'])
        
        # 最后的文本区域
        if current_pos < len(content):
            text_content = content[current_pos:]
            if text_content.strip():
                text_regions.append({
                    'type': ContentType.TEXT,
                    'start': current_pos,
                    'end': len(content),
                    'content': text_content
                })
        
        # 合并并排序
        all_regions = regions + text_regions
        all_regions.sort(key=lambda x: x['start'])
        
        return all_regions
    
    # ============================================================
    # 第三层：差异化内容切分策略
    # ============================================================
    
    def _layer3_differentiated_chunking(
        self, 
        sections: List[Dict], 
        doc_id: str
    ) -> List[Dict[str, Any]]:
        """
        第三层：差异化内容切分
        
        针对不同内容类型采用特定切分方式
        """
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_title = section['title']
            section_level = section['level']
            regions = section.get('regions', [])
            
            if not regions:
                # 没有区域标记，整体作为文本处理
                region_chunks = self._chunk_text(
                    section['full_content'],
                    self.config.max_chunk_size,
                    self.config.overlap
                )
                for content in region_chunks:
                    chunks.append(self._create_chunk(
                        content, chunk_index, doc_id,
                        section_title, section_level, ContentType.TEXT
                    ))
                    chunk_index += 1
            else:
                # 按区域类型分别处理
                # 生成章节标题前缀（用于添加上下文）
                section_heading = f"{'#' * section_level} {section_title}" if section_level > 0 else ""
                
                for region_idx, region in enumerate(regions):
                    region_content = region['content']
                    region_type = region['type']
                    
                    # 为每个区域添加章节上下文
                    if section_heading:
                        if region_idx == 0:
                            # 第一个区域：完整标题
                            region_content = f"{section_heading}\n{region_content}"
                        else:
                            # 后续区域：轻量级上下文（括号注释）
                            region_content = f"（接上节：{section_title}）\n{region_content}"
                    
                    # 根据类型选择切分策略
                    if region_type == ContentType.TABLE:
                        region_chunks = self._chunk_table(region_content, section_title)
                    
                    elif region_type == ContentType.FLOW:
                        region_chunks = self._chunk_flow(region_content, section_title)
                    
                    elif region_type == ContentType.CODE:
                        region_chunks = self._chunk_code(region_content)
                    
                    elif region_type == ContentType.LIST:
                        region_chunks = self._chunk_list(region_content)
                    
                    else:  # TEXT
                        region_chunks = self._chunk_text(
                            region_content,
                            self.config.max_chunk_size,
                            self.config.overlap
                        )
                    
                    for content in region_chunks:
                        chunks.append(self._create_chunk(
                            content, chunk_index, doc_id,
                            section_title, section_level, region_type
                        ))
                        chunk_index += 1
        
        return chunks
    
    def _chunk_table(self, content: str, section_title: str) -> List[str]:
        """
        表格切分策略
        
        - 完整保留表格的原始结构和格式
        - 大表格按行分组拆分，每组附加表头
        """
        lines = content.strip().split('\n')
        
        # 过滤空行
        lines = [l for l in lines if l.strip()]
        
        if len(lines) < 2:
            return [content] if content.strip() else []
        
        # 小表格保持完整（表头+分隔线+数据行）
        if len(lines) <= self.config.max_table_rows + 2:
            return [content]
        
        # 大表格智能拆分
        header = lines[0]
        separator = lines[1]
        data_rows = lines[2:]
        
        chunks = []
        
        for i in range(0, len(data_rows), self.config.max_table_rows):
            chunk_rows = data_rows[i:i + self.config.max_table_rows]
            chunk_content = f"{header}\n{separator}\n" + "\n".join(chunk_rows)
            
            # 添加上下文说明
            if i > 0:
                chunk_content = f"（续表，来自：{section_title}）\n{chunk_content}"
            elif len(data_rows) > self.config.max_table_rows:
                chunk_content = f"（表格拆分，来自：{section_title}）\n{chunk_content}"
            
            chunks.append(chunk_content)
        
        return chunks
    
    def _chunk_flow(self, content: str, section_title: str) -> List[str]:
        """
        流程切分策略
        
        - 将整个流程区域作为一个不可分割的完整chunk
        - 确保流程逻辑的完整性
        """
        # 流程必须保持完整，不进行拆分
        return [content]
    
    def _chunk_code(self, content: str) -> List[str]:
        """
        代码块切分策略
        
        - 保持代码块完整性
        """
        return [content]
    
    def _chunk_list(self, content: str) -> List[str]:
        """
        列表切分策略
        
        - 如果列表不长，保持完整
        - 如果超长，按列表项拆分
        """
        if len(content) <= self.config.max_chunk_size:
            return [content]
        
        # 按列表项拆分
        items = re.split(r'\n(?=[-*+]|\d+\.)', content)
        items = [item for item in items if item.strip()]
        
        chunks = []
        current_chunk = ""
        
        for item in items:
            if len(current_chunk) + len(item) + 1 <= self.config.max_chunk_size:
                current_chunk += "\n" + item if current_chunk else item
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = item
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content]
    
    def _chunk_text(self, content: str, max_size: int, overlap: int) -> List[str]:
        """
        普通文本切分策略
        
        - 按自然段落边界进行切分
        - 实现上下文保留机制：overlap重叠区
        """
        if not content.strip():
            return []
        
        if len(content) <= max_size:
            return [content]
        
        # 按段落边界切分
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果单个段落就超过限制，按句子切分
            if len(para) > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 按句子边界切分
                sentences = re.split(r'([。；！？\n])', para)
                sentence_buffer = ""
                
                for i in range(0, len(sentences) - 1, 2):
                    sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                    if len(sentence_buffer) + len(sentence) <= max_size:
                        sentence_buffer += sentence
                    else:
                        if sentence_buffer:
                            chunks.append(sentence_buffer.strip())
                        sentence_buffer = sentence
                
                # 处理最后一个句子（没有分隔符的情况）
                if len(sentences) % 2 == 1 and sentences[-1]:
                    if len(sentence_buffer) + len(sentences[-1]) <= max_size:
                        sentence_buffer += sentences[-1]
                    else:
                        if sentence_buffer:
                            chunks.append(sentence_buffer.strip())
                        sentence_buffer = sentences[-1]
                
                if sentence_buffer:
                    current_chunk = sentence_buffer
            else:
                if len(current_chunk) + len(para) + 2 <= max_size:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 添加overlap重叠区（使用自然过渡，不添加标记）
        if overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # 从上一个chunk末尾取overlap内容
                    prev_chunk = chunks[i - 1]
                    # 找到语义边界（句子结束）
                    overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                    
                    # 尝试在句子边界截断
                    for j in range(len(overlap_text) - 1, max(0, len(overlap_text) - 50), -1):
                        if overlap_text[j] in "。；！？\n":
                            overlap_text = overlap_text[j+1:].strip()
                            break
                    
                    if overlap_text:
                        # 自然过渡，不添加标记
                        chunk = f"{overlap_text}\n\n{chunk}"
                
                overlapped_chunks.append(chunk)
            chunks = overlapped_chunks
        
        return chunks
    
    def _create_chunk(
        self, 
        content: str, 
        index: int, 
        doc_id: str,
        section_title: str,
        section_level: int,
        content_type: ContentType
    ) -> Dict[str, Any]:
        """创建chunk对象"""
        return {
            "content": content,
            "type": content_type.value,
            "metadata": {
                "chunk_index": index,
                "section_title": section_title,
                "section_level": section_level,
                "content_type": content_type.value,
                "char_count": len(content),
                "doc_id": doc_id
            }
        }
    
    # ============================================================
    # 第四层：后处理优化
    # ============================================================
    
    def _post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        第四层：后处理优化
        
        1. 过滤无效片段（纯分隔线等）
        2. 合并过短片段
        3. 重建索引
        """
        if not chunks:
            return chunks
        
        # 步骤1：过滤无效片段
        filtered_chunks = self._filter_invalid_chunks(chunks)
        
        # 步骤2：合并过短片段
        merged_chunks = self._merge_short_chunks(filtered_chunks)
        
        # 步骤3：重建索引
        for i, chunk in enumerate(merged_chunks):
            chunk['metadata']['chunk_index'] = i
        
        return merged_chunks
    
    def _filter_invalid_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤无效片段"""
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
        
        logger.info(f"[过滤] 移除了 {len(chunks) - len(filtered)} 个无效片段")
        return filtered
    
    def _merge_short_chunks(self, chunks: List[Dict[str, Any]], min_size: int = None) -> List[Dict[str, Any]]:
        """合并过短片段"""
        if not chunks:
            return chunks
        
        # 使用配置中的最小chunk大小
        min_size = min_size or self.config.min_chunk_size
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i].copy()
            current_content = current_chunk['content']
            
            # 如果当前片段过短，尝试与相邻片段合并
            while len(current_content) < min_size and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                next_content = next_chunk['content']
                
                # 检查是否可以合并
                can_merge = self._can_merge_chunks(current_content, next_content)
                
                if can_merge:
                    # 合并内容
                    current_content = current_content + '\n\n' + next_content
                    current_chunk['content'] = current_content
                    current_chunk['metadata']['char_count'] = len(current_content)
                    
                    # 更新内容类型（取更具体的类型）
                    if next_chunk['type'] in ['table', 'flow', 'code']:
                        current_chunk['type'] = next_chunk['type']
                        current_chunk['metadata']['content_type'] = next_chunk['type']
                    
                    i += 1
                else:
                    break
            
            merged.append(current_chunk)
            i += 1
        
        logger.info(f"[合并] 从 {len(chunks)} 个片段合并为 {len(merged)} 个")
        return merged
    
    def _can_merge_chunks(self, current_content: str, next_content: str) -> bool:
        """
        判断两个chunk是否可以合并
        
        核心原则：只有语义上确实相关的内容才合并，默认不合并
        
        合并规则：
        1. 当前chunk是纯标题，下一个chunk是表格/代码/普通文本 → 合并（标题与内容关联）
        2. 当前chunk是纯标题，下一个chunk也是标题 → 不合并（不同章节）
        3. 当前chunk是列表项，下一个chunk也是列表项 → 合并
        4. 当前chunk以标题结尾，下一个chunk是新章节标题 → 不合并
        5. 两个chunk都属于同一内容类型 → 可以合并
        6. 其他情况：不合并（保护语义边界）
        """
        current_stripped = current_content.strip()
        next_stripped = next_content.strip()
        
        # 检测标题
        is_title_current = bool(re.match(r'^#{1,6}\s+', current_stripped))
        is_title_next = bool(re.match(r'^#{1,6}\s+', next_stripped))
        
        # 规则2：两个都是标题 → 不合并（不同章节）
        if is_title_current and is_title_next:
            return False
        
        # 规则1：当前是标题，下一个不是标题 → 合并（标题与内容关联）
        if is_title_current and not is_title_next:
            return True
        
        # 规则4：下一个是新章节标题 → 不合并
        if is_title_next:
            return False
        
        # 检测表格
        is_table_current = '|' in current_stripped and '---' in current_stripped
        is_table_next = '|' in next_stripped and '---' in next_stripped
        
        # 规则5：两个都是表格 → 可以合并（同一表格的不同部分）
        if is_table_current and is_table_next:
            return True
        
        # 表格与非表格内容 → 不合并
        if is_table_current != is_table_next:
            return False
        
        # 检测代码块
        is_code_current = current_stripped.startswith('```')
        is_code_next = next_stripped.startswith('```')
        
        # 代码块与非代码块 → 不合并
        if is_code_current != is_code_next:
            return False
        
        # 规则3：列表项合并
        is_list_current = current_stripped.startswith(('- ', '* ', '+ ')) or re.match(r'^\d+\.\s', current_stripped)
        is_list_next = next_stripped.startswith(('- ', '* ', '+ ')) or re.match(r'^\d+\.\s', next_stripped)
        if is_list_current and is_list_next:
            return True
        
        # 检测加粗标题
        is_bold_title = current_stripped.startswith('**') and current_stripped.endswith('**')
        if is_bold_title and is_table_next:
            return True
        
        # 规则5：都是普通文本 → 检查是否有章节边界
        # 检查是否包含章节分隔（如水平线）
        has_separator = bool(re.search(r'^---+$|^\*\*\*+$|^___+$', current_stripped, re.MULTILINE))
        if has_separator:
            return False
        
        # 默认：不合并（保护语义边界）
        return False


# 便捷函数
def layered_chunk(content: str, doc_id: str, **kwargs) -> List[Dict[str, Any]]:
    """
    便捷函数：执行分层智能切分
    
    Args:
        content: 文档内容
        doc_id: 文档ID
        **kwargs: 配置参数 (max_chunk_size, overlap等)
    
    Returns:
        切分后的chunk列表
    """
    config = LayeredChunkConfig(**kwargs)
    chunker = LayeredChunker(config)
    return chunker.chunk(content, doc_id)
