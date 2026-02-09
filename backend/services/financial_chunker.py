"""
财务制度文档智能切分器
专为财务报销制度类文档设计，支持职级差异表格的智能展开
"""

import re
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """文档片段"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_type: str = "text"  # text, table, procedure
    level: str = ""  # 职级标签
    expense_type: str = ""  # 费用类型


class FinancialDocumentChunker:
    """财务制度文档切分器"""

    # 职级关键词映射
    LEVEL_KEYWORDS = {
        "8-9级": [
            "8-9级",
            "普通员工",
            "软件研发工程师",
            "机械研发工程师",
            "工艺工程师",
            "实施工程师",
        ],
        "10-11级": ["10-11级", "经理级", "经理"],
        "12级及以上": ["12级及以上", "总监", "专家级", "专家"],
    }

    # 费用类型关键词
    EXPENSE_TYPES = {
        "差旅费": ["差旅", "出差", "交通工具", "住宿", "补贴"],
        "业务招待费": ["招待", "宴请", "客户", "礼品"],
        "通讯费": ["通讯", "电话", "手机"],
        "办公费": ["办公", "文具", "书籍", "软件"],
        "培训会议费": ["培训", "会议"],
        "借款": ["借款", "备用金"],
    }

    def __init__(self, max_chunk_size: int = 800, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_document(self, document_text: str, doc_id: str = "") -> List[Chunk]:
        """
        主切分方法

        Args:
            document_text: 文档完整文本
            doc_id: 文档ID

        Returns:
            List[Chunk]: 切分后的文档片段列表
        """
        chunks = []

        # 第一步：按一级标题切分
        sections = self._split_by_headers(document_text)

        for section_title, section_content in sections:
            # 第二步：识别并处理特殊区域
            section_chunks = self._process_section(section_title, section_content)
            chunks.extend(section_chunks)

        # 第三步：添加全局metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {"doc_id": doc_id, "chunk_index": i, "total_chunks": len(chunks)}
            )

        return chunks

    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """按一级标题(#)切分文档"""
        # 匹配 # 开头的标题
        pattern = r"(^|\n)#\s+(.+?)(?=\n#\s|\Z)"
        matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))

        sections = []
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            content = match.group(0).strip()
            sections.append((title, content))

        return sections

    def _process_section(self, title: str, content: str) -> List[Chunk]:
        """处理单个章节"""
        chunks = []

        # 识别章节类型
        if self._is_table_section(content):
            # 包含职级表格的章节
            table_chunks = self._process_table_section(title, content)
            chunks.extend(table_chunks)
        elif self._is_procedure_section(content):
            # 流程类章节
            procedure_chunk = self._create_procedure_chunk(title, content)
            chunks.append(procedure_chunk)
        else:
            # 普通文本章节
            text_chunks = self._split_text_section(title, content)
            chunks.extend(text_chunks)

        return chunks

    def _is_table_section(self, content: str) -> bool:
        """判断是否包含职级差异表格"""
        # 检查是否包含markdown表格且提到职级
        has_table = "|" in content and "---" in content
        has_level = any(
            keyword in content
            for keywords in self.LEVEL_KEYWORDS.values()
            for keyword in keywords
        )
        return has_table and has_level

    def _is_procedure_section(self, content: str) -> bool:
        """判断是否流程类章节"""
        procedure_keywords = ["流程", "步骤", "审批流", "申请人", "审批"]
        return any(keyword in content for keyword in procedure_keywords) and (
            "→" in content or "```" in content
        )

    def _process_table_section(self, title: str, content: str) -> List[Chunk]:
        """处理包含职级表格的章节，按职级展开"""
        chunks = []

        # 提取表格
        tables = self._extract_tables(content)

        for table in tables:
            # 判断表格是否包含职级差异
            if self._contains_level_differences(table):
                # 按职级展开表格
                level_chunks = self._expand_table_by_level(title, table, content)
                chunks.extend(level_chunks)
            else:
                # 普通表格作为一个chunk
                chunk = Chunk(
                    content=f"## {title}\n\n{table}",
                    chunk_type="table",
                    metadata={"section": title},
                )
                chunk.level = self._detect_level(table)
                chunk.expense_type = self._detect_expense_type(table)
                chunks.append(chunk)

        # 处理表格外的文本
        non_table_content = self._remove_tables(content, tables)
        if non_table_content.strip():
            text_chunks = self._split_text_section(title, non_table_content)
            chunks.extend(text_chunks)

        return chunks

    def _extract_tables(self, content: str) -> List[str]:
        """提取markdown表格"""
        tables = []
        # 匹配markdown表格（|开头，包含分隔行|---|）
        pattern = r"\|[^\n]+\|\n\|[-:|\s]+\|\n(?:\|[^\n]+\|\n?)+"
        matches = re.finditer(pattern, content)

        for match in matches:
            tables.append(match.group(0))

        return tables

    def _contains_level_differences(self, table: str) -> bool:
        """判断表格是否包含职级差异"""
        return any(
            level in table
            for level in ["8-9级", "10-11级", "12级及以上", "普通员工", "经理", "总监"]
        )

    def _expand_table_by_level(
        self, title: str, table: str, context: str
    ) -> List[Chunk]:
        """按职级展开表格"""
        chunks = []

        # 解析表格结构
        rows = [row.strip() for row in table.strip().split("\n") if row.strip()]
        if len(rows) < 3:  # 表头+分隔行+至少一行数据
            return [Chunk(content=f"## {title}\n\n{table}", chunk_type="table")]

        header = rows[0]
        separator = rows[1]
        data_rows = rows[2:]

        # 为每个职级创建独立的chunk
        for level_name, keywords in self.LEVEL_KEYWORDS.items():
            level_rows = []
            for row in data_rows:
                if any(keyword in row for keyword in keywords):
                    level_rows.append(row)

            if level_rows:
                # 构建该职级的专属表格
                level_table = f"{header}\n{separator}\n" + "\n".join(level_rows)

                # 添加相关上下文说明
                context_info = self._extract_context_for_level(context, level_name)

                chunk_content = (
                    f"## {title} - {level_name}\n\n{context_info}\n\n{level_table}"
                )

                chunk = Chunk(
                    content=chunk_content,
                    chunk_type="table",
                    level=level_name,
                    metadata={
                        "section": title,
                        "level": level_name,
                        "table_type": "level_specific",
                    },
                )
                chunk.expense_type = self._detect_expense_type(table)
                chunks.append(chunk)

        return chunks

    def _extract_context_for_level(self, content: str, level: str) -> str:
        """提取与特定职级相关的上下文说明"""
        # 提取该职级相关的注释和说明
        context_parts = []

        # 查找包含职级关键词的段落
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            if level.replace("级", "") in para or any(
                keyword in para for keyword in self.LEVEL_KEYWORDS.get(level, [])
            ):
                if "注：" in para or "说明：" in para or "注意" in para:
                    context_parts.append(para.strip())

        return "\n".join(context_parts) if context_parts else ""

    def _create_procedure_chunk(self, title: str, content: str) -> Chunk:
        """创建流程类chunk，保持完整性"""
        chunk = Chunk(
            content=f"## {title}\n\n{content}",
            chunk_type="procedure",
            metadata={"section": title, "type": "procedure"},
        )
        chunk.level = self._detect_level(content)
        chunk.expense_type = self._detect_expense_type(content)
        return chunk

    def _split_text_section(self, title: str, content: str) -> List[Chunk]:
        """切分普通文本章节"""
        chunks = []

        # 按二级标题(##)进一步切分
        subsections = re.split(r"\n##\s+", content)

        for subsection in subsections:
            if not subsection.strip():
                continue

            # 如果内容太长，按段落切分
            if len(subsection) > self.max_chunk_size:
                paragraph_chunks = self._split_by_paragraphs(title, subsection)
                chunks.extend(paragraph_chunks)
            else:
                chunk = Chunk(
                    content=f"## {title}\n\n{subsection.strip()}",
                    chunk_type="text",
                    metadata={"section": title},
                )
                chunk.level = self._detect_level(subsection)
                chunk.expense_type = self._detect_expense_type(subsection)
                chunks.append(chunk)

        return chunks

    def _split_by_paragraphs(self, title: str, content: str) -> List[Chunk]:
        """按段落切分长文本"""
        chunks = []
        paragraphs = content.split("\n\n")

        current_chunk = f"## {title}\n\n"
        current_size = len(current_chunk)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para) + 2  # +2 for \n\n

            if (
                current_size + para_size > self.max_chunk_size
                and current_chunk.strip() != f"## {title}"
            ):
                # 保存当前chunk
                chunk = Chunk(
                    content=current_chunk.strip(),
                    chunk_type="text",
                    metadata={"section": title},
                )
                chunk.level = self._detect_level(current_chunk)
                chunk.expense_type = self._detect_expense_type(current_chunk)
                chunks.append(chunk)

                # 开始新chunk，保留重叠
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = f"## {title}\n\n{overlap_text}\n\n{para}"
                current_size = len(current_chunk)
            else:
                current_chunk += f"\n\n{para}"
                current_size += para_size

        # 保存最后一个chunk
        if current_chunk.strip() != f"## {title}":
            chunk = Chunk(
                content=current_chunk.strip(),
                chunk_type="text",
                metadata={"section": title},
            )
            chunk.level = self._detect_level(current_chunk)
            chunk.expense_type = self._detect_expense_type(current_chunk)
            chunks.append(chunk)

        return chunks

    def _get_overlap(self, text: str) -> str:
        """获取文本末尾作为重叠部分"""
        lines = text.strip().split("\n")
        overlap_lines = (
            lines[-3:] if len(lines) > 3 else lines[-2:] if len(lines) > 1 else []
        )
        return "\n".join(overlap_lines)

    def _remove_tables(self, content: str, tables: List[str]) -> str:
        """从内容中移除表格"""
        result = content
        for table in tables:
            result = result.replace(table, "")
        return result

    def _detect_level(self, text: str) -> str:
        """检测文本涉及的职级"""
        levels = []
        for level_name, keywords in self.LEVEL_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                levels.append(level_name)
        return ",".join(levels) if levels else ""

    def _detect_expense_type(self, text: str) -> str:
        """检测费用类型"""
        types = []
        for type_name, keywords in self.EXPENSE_TYPES.items():
            if any(keyword in text for keyword in keywords):
                types.append(type_name)
        return ",".join(types) if types else ""


class ChunkingService:
    """切分服务封装"""

    @staticmethod
    def chunk_financial_document(file_path: str, doc_id: str = "") -> List[Dict]:
        """
        切分财务制度文档

        Args:
            file_path: 文档路径
            doc_id: 文档ID

        Returns:
            List[Dict]: 切分结果列表
        """
        # 读取文档
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 创建切分器
        chunker = FinancialDocumentChunker(max_chunk_size=800, overlap=100)

        # 执行切分
        chunks = chunker.chunk_document(content, doc_id)

        # 转换为字典格式
        result = []
        for i, chunk in enumerate(chunks):
            result.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{i:03d}",
                    "content": chunk.content,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_type": chunk.chunk_type,
                        "level": chunk.level,
                        "expense_type": chunk.expense_type,
                        "char_count": len(chunk.content),
                    },
                }
            )

        return result

    @staticmethod
    def save_chunks(chunks: List[Dict], output_path: str):
        """保存切分结果到JSON文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ 切分完成，共生成 {len(chunks)} 个片段，已保存到: {output_path}")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例：切分baoxiao.md
    file_path = "/root/autodl-tmp/rag/backend/data/docs/baoxiao.md"
    doc_id = "baoxiao_001"
    output_path = "/root/autodl-tmp/rag/backend/data/chunks/baoxiao_chunks.json"

    # 执行切分
    chunks = ChunkingService.chunk_financial_document(file_path, doc_id)

    # 保存结果
    ChunkingService.save_chunks(chunks, output_path)

    # 打印统计信息
    print("\n📊 切分统计：")
    print(f"总计生成: {len(chunks)} 个chunk")

    # 按类型统计
    type_count = {}
    level_count = {}
    for chunk in chunks:
        chunk_type = chunk["metadata"]["chunk_type"]
        level = chunk["metadata"]["level"]

        type_count[chunk_type] = type_count.get(chunk_type, 0) + 1
        if level:
            for l in level.split(","):
                level_count[l] = level_count.get(l, 0) + 1

    print("\n按类型分布：")
    for t, count in type_count.items():
        print(f"  {t}: {count}个")

    print("\n按职级分布：")
    for l, count in level_count.items():
        print(f"  {l}: {count}个")

    # 打印前3个chunk示例
    print("\n📝 前3个Chunk示例：")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"类型: {chunk['metadata']['chunk_type']}")
        print(f"职级: {chunk['metadata']['level']}")
        print(f"费用类型: {chunk['metadata']['expense_type']}")
        print(f"内容预览: {chunk['content'][:150]}...")
