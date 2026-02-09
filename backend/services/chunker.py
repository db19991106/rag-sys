"""
文档切分器 - RAGFlow 风格的智能切分系统（制度文档增强版）
"""

from typing import List, Tuple, Optional, Dict, Any
import re
import json
from pathlib import Path

# 导入模型类
from models import ChunkInfo, ChunkType, ChunkConfig
from utils.logger import logger
from utils.token_counter import num_tokens_from_string

# 导入财务制度专用切分器
try:
    from services.financial_chunker_v2 import FinancialDocumentChunker

    FINANCIAL_CHUNKER_AVAILABLE = True
except ImportError:
    FINANCIAL_CHUNKER_AVAILABLE = False
    logger.warning("Financial chunker v2 not available")


class RAGFlowChunker:
    """
    RAGFlow 风格的文档切分器（支持财务制度类文档的智能切分）
    """

    def __init__(self):
        # 章标题：支持 "## 第一章 总则" 或 "##第一章..."
        self.chapter_pattern = re.compile(
            r"^#{2}\s*(第[零一二三四五六七八九十百0-9]+章\s*.+)$", re.MULTILINE
        )
        # 节标题：支持 "### 2.5 培训与会议" 或 "###2.5培训..."（数字后可无空格）
        self.section_pattern = re.compile(
            r"^#{3}\s*([0-9]+\.[0-9]+\s*.+)$", re.MULTILINE
        )
        # 文档标题
        self.doc_title_pattern = re.compile(r"^#\s*(.+)$", re.MULTILINE)

        self.table_pattern = re.compile(
            r"((?:^|\n)(?:\|[^\n]+\|\n)(?:\|[-:\|\s]+\|\n)(?:\|[^\n]+\|\n?)+)",
            re.MULTILINE,
        )
        # 用于存储元数据的缓存
        self._chunk_metadata_cache: Dict[str, Dict] = {}

    def chunk(self, content: str, doc_id: str, config: ChunkConfig) -> List[ChunkInfo]:
        """
        根据配置切分文档
        """
        if not content or not content.strip():
            logger.warning("文档内容为空，无法切分")
            return []

        # 清空上一轮切分的元数据缓存
        self._chunk_metadata_cache = {}

        try:
            # 检测文档类型
            is_policy_doc = self._detect_policy_document(content)

            # 智能切分：对于INTELLIGENT类型或制度类文档，使用优化的财务/制度类文档切分策略
            if config.type == ChunkType.INTELLIGENT or (
                is_policy_doc and config.type in [ChunkType.PDF, ChunkType.ENHANCED]
            ):
                logger.info(
                    f"使用智能切分策略 (config.type: {config.type})。文档长度: {len(content)}"
                )

                # 优先使用新版财务制度切分器（如果文档包含明显的制度特征）
                if is_policy_doc and (
                    "报销" in content or "差旅" in content or "审批" in content
                ):
                    logger.info(
                        "检测到财务/报销制度文档，使用 specialized financial chunker"
                    )
                    try:
                        chunk_dicts = self._financial_chunker_v2_chunking(
                            content, doc_id
                        )
                    except Exception as e:
                        logger.warning(f"新版财务切分器失败，回退到标准切分: {e}")
                        chunk_dicts = self._financial_policy_chunking(content)
                else:
                    chunk_dicts = self._financial_policy_chunking(content)
            elif config.type == ChunkType.NAIVE:
                chunk_texts = self._naive_merge(
                    content,
                    chunk_token_num=config.chunk_token_size,
                    delimiter=config.delimiters[0]
                    if config.delimiters
                    else "\n。；！？",
                    overlapped_percent=config.overlapped_percent,
                )
                chunk_dicts = [
                    {"content": t, "type": "text", "metadata": {}} for t in chunk_texts
                ]
            else:
                chunk_texts = self._naive_merge(
                    content,
                    chunk_token_num=config.chunk_token_size,
                    delimiter=config.delimiters[0]
                    if config.delimiters
                    else "\n。；！？",
                    overlapped_percent=config.overlapped_percent,
                )
                chunk_dicts = [
                    {"content": t, "type": "text", "metadata": {}} for t in chunk_texts
                ]

            # 转换为 ChunkInfo 对象
            chunk_infos = self._create_chunk_infos(chunk_dicts, doc_id)

            # 保存结果
            self._save_chunks_to_json(chunk_infos, doc_id, config)

            return chunk_infos

        except Exception as e:
            logger.error(f"文档切分失败: {str(e)}")
            import traceback

            logger.error(f"堆栈跟踪: {traceback.format_exc()}")
            # 失败后尝试使用朴素切分作为fallback
            logger.info("尝试使用朴素切分作为后备方案...")
            try:
                chunk_texts = self._naive_merge(content, chunk_token_num=512)
                chunk_dicts = [
                    {"content": t, "type": "text", "metadata": {}} for t in chunk_texts
                ]
                chunk_infos = self._create_chunk_infos(chunk_dicts, doc_id)
                self._save_chunks_to_json(chunk_infos, doc_id, config)
                return chunk_infos
            except Exception as e2:
                logger.error(f"后备切分也失败: {e2}")
                return []

    def _detect_policy_document(self, content: str) -> bool:
        """检测是否为制度类文档"""
        policy_keywords = [
            "报销",
            "审批",
            "第一章",
            "第.*章",
            "差旅费",
            "发票",
            "借款",
            "违规处理",
        ]
        count = sum(1 for keyword in policy_keywords if re.search(keyword, content))
        return count >= 3

    def _analyze_chapter_structure(self, content: str) -> List[Dict]:
        """分析文档的章节结构（修复正则，确保捕获所有格式）"""
        structure = []

        # 匹配各级标题（放宽格式要求，支持有无空格的情况）
        patterns = [
            (self.doc_title_pattern, 1, "doc_title"),  # 文档标题
            (self.chapter_pattern, 2, "chapter"),  # 章
            (self.section_pattern, 3, "section"),  # 节（如 2.1, 2.2 等）
        ]

        for pattern, level, elem_type in patterns:
            for match in pattern.finditer(content):
                title = match.group(1).strip()
                # 清理标题中可能的脏字符
                title = re.sub(r"\s+", " ", title).strip()
                structure.append(
                    {
                        "pos": match.start(),
                        "end": match.end(),
                        "level": level,
                        "type": elem_type,
                        "title": title,
                        "full_match": match.group(0),
                    }
                )

        # 按位置排序
        structure.sort(key=lambda x: x["pos"])

        # 日志输出识别到的结构
        chapters = [s for s in structure if s["type"] == "chapter"]
        sections = [s for s in structure if s["type"] == "section"]
        logger.info(
            f"文档结构分析：识别到文档标题: {len([s for s in structure if s['type'] == 'doc_title'])} 个"
        )
        logger.info(
            f"文档结构分析：识别到章: {len(chapters)} 个 - {[c['title'] for c in chapters]}"
        )
        logger.info(
            f"文档结构分析：识别到节: {len(sections)} 个 - {[s['title'] for s in sections[:10]]}{'...' if len(sections) > 10 else ''}"
        )

        if not chapters:
            logger.error(
                "未识别到任何章节（第X章），请检查文档格式是否为Markdown（# ## ###）"
            )

        return structure

    def _financial_policy_chunking(self, content: str) -> List[Dict[str, Any]]:
        """
        财务/制度类文档专用切分（修复：确保不遗漏任何章节）
        """
        chunks = []

        # 步骤1：保护表格
        protected_tables = []
        content_with_placeholders = content

        for match in self.table_pattern.finditer(content):
            placeholder = f"__TABLE_{len(protected_tables)}__"
            table_content = match.group(1).strip()
            protected_tables.append(
                {"content": table_content, "placeholder": placeholder}
            )
            content_with_placeholders = content_with_placeholders.replace(
                match.group(1), f"\n{placeholder}\n", 1
            )

        logger.info(f"发现并保护 {len(protected_tables)} 个表格")

        # 步骤2：识别章节结构
        structure = self._analyze_chapter_structure(content_with_placeholders)

        if not structure:
            logger.warning("未识别到文档结构，回退到文本切分")
            return [
                {"content": content, "type": "doc", "metadata": {"type": "full_doc"}}
            ]

        # 步骤3：构建切分边界
        boundaries = self._build_chunk_boundaries(structure, content_with_placeholders)

        logger.info(f"规划生成 {len(boundaries)} 个片段")

        # 步骤4：生成chunk内容
        for idx, (start_pos, end_pos, chunk_type, metadata) in enumerate(boundaries, 1):
            if start_pos >= end_pos:
                logger.warning(
                    f"片段 {idx} 的边界无效（{start_pos} >= {end_pos}），跳过"
                )
                continue

            chunk_content = content_with_placeholders[start_pos:end_pos].strip()

            # 恢复表格
            for table_info in protected_tables:
                if table_info["placeholder"] in chunk_content:
                    chunk_content = chunk_content.replace(
                        table_info["placeholder"], table_info["content"]
                    )

            if not chunk_content:
                continue

            # 完善元数据
            metadata.update(
                {
                    "chunk_id": f"{idx:03d}",
                    "type": chunk_type,
                    "char_length": len(chunk_content),
                    "token_count": num_tokens_from_string(chunk_content),
                }
            )

            # 自动检测表格
            if "|" in chunk_content and "\n|" in chunk_content:
                metadata["has_table"] = True
                metadata["tables_count"] = chunk_content.count("\n|") // 3

            chunks.append(
                {"content": chunk_content, "type": chunk_type, "metadata": metadata}
            )

        logger.info(f"实际生成 {len(chunks)} 个片段")
        return chunks

    def _financial_chunker_v2_chunking(
        self, content: str, doc_id: str
    ) -> List[Dict[str, Any]]:
        """
        使用新版财务制度切分器（V2优化版）
        特点：
        1. 按一级标题(#)和二级标题(##)层次切分
        2. 保留表格完整性
        3. 自动标注职级和费用类型metadata
        4. 适合财务报销制度类文档
        """
        if not FINANCIAL_CHUNKER_AVAILABLE:
            raise ImportError("Financial chunker v2 not available")

        # 使用新版切分器
        chunker = FinancialDocumentChunker(max_chunk_size=1000)
        chunks = chunker.chunk_document(content, doc_id)

        # 转换为统一格式
        result = []
        for chunk in chunks:
            result.append(
                {
                    "content": chunk.content,
                    "type": chunk.chunk_type,
                    "metadata": {
                        **chunk.metadata,
                        "level": chunk.metadata.get("level", ""),
                        "expense_type": chunk.metadata.get("expense_type", ""),
                        "char_count": len(chunk.content),
                    },
                }
            )

        logger.info(f"新版财务切分器生成 {len(result)} 个片段")
        return result

    def _build_chunk_boundaries(
        self, structure: List[Dict], content: str
    ) -> List[Tuple]:
        """
        构建切分边界（修复：确保覆盖所有内容，无遗漏）
        """
        boundaries = []
        content_len = len(content)

        # 分类元素
        doc_title = next((s for s in structure if s["type"] == "doc_title"), None)
        chapters = [s for s in structure if s["type"] == "chapter"]
        all_sections = [s for s in structure if s["type"] == "section"]

        if not chapters:
            # 没有章节，整个文档作为一个chunk
            start = doc_title["pos"] if doc_title else 0
            return [
                (
                    start,
                    content_len,
                    "doc",
                    {"hierarchy_path": [doc_title["title"]] if doc_title else ["文档"]},
                )
            ]

        # 辅助函数：获取某章下的所有节
        def get_sections_in_range(start_pos, end_pos):
            return [s for s in all_sections if start_pos < s["pos"] < end_pos]

        # 处理第1章（通常包含在文档标题chunk中）
        first_ch = chapters[0]
        first_ch_end = chapters[1]["pos"] if len(chapters) > 1 else content_len
        first_ch_sections = get_sections_in_range(first_ch["pos"], first_ch_end)

        # Chunk 1: 文档标题 + 第1章 + 其所有节（第1章通常较短）
        start_pos = doc_title["pos"] if doc_title else first_ch["pos"]
        # 第1章通常在1.2结束，或如果有1.3则到1.3，否则到第2章开始
        if len(first_ch_sections) >= 2:
            # 包含前2节（通常是1.1和1.2）
            end_pos = (
                first_ch_sections[2]["pos"]
                if len(first_ch_sections) > 2
                else first_ch_end
            )
        elif len(first_ch_sections) == 1:
            end_pos = (
                first_ch_sections[0]["end"] + len(first_ch_sections[0]["title"]) * 10
            )  # 估算长度
            end_pos = min(end_pos, first_ch_end)
        else:
            end_pos = first_ch_end

        boundaries.append(
            (
                start_pos,
                end_pos,
                "chapter",
                {
                    "hierarchy_path": [
                        doc_title["title"] if doc_title else "文档",
                        first_ch["title"],
                    ],
                    "chapter_index": 1,
                },
            )
        )

        # 处理第2章及以后
        for i, ch in enumerate(chapters[1:], 2):
            ch_start = ch["pos"]
            ch_end = chapters[i]["pos"] if i < len(chapters) else content_len
            ch_sections = get_sections_in_range(ch_start, ch_end)

            if not ch_sections:
                # 空章节，直接取整章
                boundaries.append(
                    (
                        ch_start,
                        ch_end,
                        "chapter",
                        {"hierarchy_path": [ch["title"]], "chapter_index": i},
                    )
                )
                continue

            # 策略：章节标题 + 第1节合并（如果第1节包含表格或较短）
            first_sec = ch_sections[0]
            first_sec_end = ch_sections[1]["pos"] if len(ch_sections) > 1 else ch_end

            # 检查第1节是否包含表格
            chunk1_content = content[first_sec["pos"] : first_sec_end]
            has_table_in_sec1 = (
                "__TABLE_" in chunk1_content or "|" in chunk1_content[:100]
            )
            is_sec1_short = (first_sec_end - first_sec["pos"]) < 800

            if has_table_in_sec1 or is_sec1_short:
                # 合并章节标题和第1节
                boundaries.append(
                    (
                        ch_start,
                        first_sec_end,
                        "chapter_section",
                        {
                            "hierarchy_path": [ch["title"], first_sec["title"]],
                            "chapter_index": i,
                            "section_index": 1,
                            "has_table": has_table_in_sec1,
                        },
                    )
                )
                sec_start_idx = 1
            else:
                # 章节标题单独（较短，避免孤立）
                boundaries.append(
                    (
                        ch_start,
                        first_sec["pos"],
                        "chapter_title",
                        {
                            "hierarchy_path": [ch["title"]],
                            "chapter_index": i,
                            "is_title_only": True,
                        },
                    )
                )
                sec_start_idx = 0

            # 处理剩余节（每节一个chunk，短节合并）
            for j in range(sec_start_idx, len(ch_sections)):
                sec = ch_sections[j]
                next_pos = (
                    ch_sections[j + 1]["pos"] if j + 1 < len(ch_sections) else ch_end
                )

                # 检查是否是最后一节且较短，尝试与前一个合并
                if j == len(ch_sections) - 1 and boundaries:
                    sec_len = next_pos - sec["pos"]
                    if sec_len < 200 and boundaries[-1][1] == sec["pos"]:
                        # 合并到前一个
                        last_boundary = list(boundaries[-1])
                        last_boundary[1] = next_pos  # 扩展end
                        # 更新metadata
                        if "section_range" in last_boundary[3]:
                            last_boundary[3]["section_range"] += f", {sec['title']}"
                        else:
                            last_boundary[3]["section_range"] = (
                                f"{last_boundary[3].get('section_title', '')} + {sec['title']}"
                            )
                        boundaries[-1] = tuple(last_boundary)
                        continue

                boundaries.append(
                    (
                        sec["pos"],
                        next_pos,
                        "section",
                        {
                            "hierarchy_path": [ch["title"], sec["title"]],
                            "chapter_index": i,
                            "section_title": sec["title"],
                        },
                    )
                )

        # 验证：确保最后一个boundary到达文档末尾
        if boundaries and boundaries[-1][1] < content_len:
            last_end = boundaries[-1][1]
            remaining_content = content[last_end:].strip()
            if remaining_content and remaining_content != "---":  # 忽略分隔线
                logger.warning(
                    f"发现未覆盖的文档尾部（{content_len - last_end} 字符），自动添加为最后一个片段"
                )
                # 扩展最后一个片段到文档末尾
                last = list(boundaries[-1])
                last[1] = content_len
                boundaries[-1] = tuple(last)

        return boundaries

    def _create_chunk_infos(
        self, chunks_data: List[Dict], doc_id: str
    ) -> List[ChunkInfo]:
        """将chunk数据转换为ChunkInfo对象（修复 metadata 存储）"""
        chunk_infos = []

        for i, chunk_data in enumerate(chunks_data, 1):
            chunk_info = ChunkInfo(
                id=f"{doc_id}_chunk_{i}",
                document_id=doc_id,
                num=i,
                content=chunk_data["content"].strip(),
                length=len(chunk_data["content"].strip()),
                embedding_status="pending",
            )

            # 存储元数据到缓存（避免Pydantic字段错误）
            metadata = chunk_data.get("metadata", {})
            self._chunk_metadata_cache[chunk_info.id] = metadata

            chunk_infos.append(chunk_info)
            logger.info(
                f"片段 {i:02d} | 长度: {chunk_info.length:4d} | 类型: {metadata.get('type', 'text'):15s} | 层级: {'/'.join(metadata.get('hierarchy_path', []))[:30]}..."
            )

        return chunk_infos

    def _save_chunks_to_json(
        self, chunks: List[ChunkInfo], doc_id: str, config: ChunkConfig
    ):
        """保存切分结果为JSON（包含完整metadata）"""
        try:
            json_dir = Path(__file__).parent.parent / "data" / "json"
            json_dir.mkdir(parents=True, exist_ok=True)

            chunks_data = {
                "document_id": doc_id,
                "config": {
                    "type": str(config.type),
                    "chunk_token_size": config.chunk_token_size,
                    "overlapped_percent": config.overlapped_percent,
                },
                "total_chunks": len(chunks),
                "chunks": [],
            }

            for chunk in chunks:
                chunk_dict = {
                    "id": chunk.id,
                    "num": chunk.num,
                    "length": chunk.length,
                    "content": chunk.content,
                    "embedding_status": chunk.embedding_status,
                }

                # 从缓存获取元数据
                metadata = self._chunk_metadata_cache.get(chunk.id, {})
                if metadata:
                    chunk_dict["metadata"] = metadata
                    # 展开关键字段便于查看
                    chunk_dict["hierarchy_path"] = metadata.get("hierarchy_path", [])
                    chunk_dict["section_title"] = metadata.get("section_title", "")
                    chunk_dict["has_table"] = metadata.get("has_table", False)
                    chunk_dict["char_length"] = metadata.get("char_length", 0)

                chunks_data["chunks"].append(chunk_dict)

            json_file = json_dir / f"chunks_{doc_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✓ 切分结果已保存: {json_file} (共 {len(chunks)} 个片段)")
        except Exception as e:
            logger.error(f"保存JSON失败: {e}")

    def _naive_merge(
        self,
        content: str,
        chunk_token_num: int = 512,
        delimiter: str = "\n。；！？",
        overlapped_percent: float = 0,
    ) -> List[str]:
        """朴素切分方法（增强版，优先按段落切分）"""
        if not content:
            return []

        # 优先按双换行（段落）切分，保持语义完整
        paragraphs = re.split(r"\n\s*\n", content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [content]

        cks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = num_tokens_from_string(para)

            if para_tokens > chunk_token_num:
                # 单个段落超长，需要内部切分
                if current_chunk:
                    cks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                # 简单按句子切分这个长段落
                sentences = re.split(r"([。！？；\n])", para)
                temp = ""
                for s in sentences:
                    if not s:
                        continue
                    if num_tokens_from_string(temp + s) > chunk_token_num:
                        if temp:
                            cks.append(temp)
                        temp = s
                    else:
                        temp += s
                if temp:
                    cks.append(temp)
            else:
                if current_tokens + para_tokens > chunk_token_num:
                    cks.append(current_chunk)
                    current_chunk = para
                    current_tokens = para_tokens
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
                    current_tokens += para_tokens

        if current_chunk:
            cks.append(current_chunk)

        return cks


class Chunker(RAGFlowChunker):
    """向后兼容的别名"""

    pass
