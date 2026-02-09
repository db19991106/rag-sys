"""
财务制度文档智能切分器 - 优化版
专为财务报销制度类文档设计
"""

import re
import json
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """文档片段"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_type: str = "text"  # text, table, procedure


class FinancialDocumentChunker:
    """财务制度文档切分器"""

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def chunk_document(self, document_text: str, doc_id: str = "") -> List[Chunk]:
        """主切分方法"""
        chunks = []

        # 步骤1：按一级标题(#)切分大章节
        sections = self._split_by_level1_headers(document_text)

        for section_title, section_content in sections:
            # 步骤2：处理每个章节
            section_chunks = self._process_section(section_title, section_content)
            chunks.extend(section_chunks)

        # 添加全局metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {"doc_id": doc_id, "chunk_index": i, "total_chunks": len(chunks)}
            )

        return chunks

    def _split_by_level1_headers(self, text: str) -> List[tuple]:
        """按一级标题(# )切分文档"""
        # 匹配以#开头且后面有空格的标题
        pattern = r"\n#\s+([^\n]+)\n"

        # 找到所有一级标题位置
        matches = list(re.finditer(pattern, "\n" + text))

        sections = []
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append((title, content))

        return sections

    def _process_section(self, title: str, content: str) -> List[Chunk]:
        """处理单个章节"""
        chunks = []

        # 提取该章节的引言（如果有）
        intro_match = re.match(r"^([^#].*?)(?=\n##|\Z)", content, re.DOTALL)
        intro = intro_match.group(1).strip() if intro_match else ""

        # 按二级标题(## )切分
        subsections = self._split_by_level2_headers(content)

        if not subsections:
            # 没有二级标题，整个章节作为一个chunk
            chunk_content = f"# {title}\n\n{content}".strip()
            chunks.append(self._create_chunk(chunk_content, "text", title))
        else:
            for sub_title, sub_content in subsections:
                # 处理子章节
                sub_chunks = self._process_subsection(
                    title, sub_title, sub_content, intro
                )
                chunks.extend(sub_chunks)

        return chunks

    def _split_by_level2_headers(self, text: str) -> List[tuple]:
        """按二级标题(## )切分"""
        pattern = r"\n##\s+([^\n]+)\n"
        matches = list(re.finditer(pattern, "\n" + text))

        subsections = []
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            subsections.append((title, content))

        return subsections

    def _process_subsection(
        self, section_title: str, sub_title: str, content: str, intro: str = ""
    ) -> List[Chunk]:
        """处理二级子章节"""
        # 检查是否包含表格
        if "|" in content and "---" in content:
            return self._process_with_table(section_title, sub_title, content, intro)
        else:
            # 普通文本
            chunk_content = f"# {section_title}\n\n## {sub_title}\n\n{content}"
            return [self._create_chunk(chunk_content, "text", section_title, sub_title)]

    def _process_with_table(
        self, section_title: str, sub_title: str, content: str, intro: str = ""
    ) -> List[Chunk]:
        """处理包含表格的内容"""
        chunks = []

        # 提取表格
        tables = self._extract_tables(content)

        # 提取表格前后的文本
        non_table_text = content
        for table in tables:
            non_table_text = non_table_text.replace(table, "[TABLE]")

        text_parts = [p.strip() for p in non_table_text.split("[TABLE]") if p.strip()]

        # 为每个表格创建一个chunk，附带上下文
        for i, table in enumerate(tables):
            context_before = text_parts[i] if i < len(text_parts) else ""
            context_after = text_parts[i + 1] if i + 1 < len(text_parts) else ""

            # 判断是否是职级差异表格
            if self._is_level_table(table):
                # 按职级展开
                level_chunks = self._expand_by_level(
                    section_title, sub_title, table, context_before, intro
                )
                chunks.extend(level_chunks)
            else:
                # 普通表格
                chunk_content = f"# {section_title}\n\n## {sub_title}\n\n{context_before}\n\n{table}\n\n{context_after}".strip()
                chunks.append(
                    self._create_chunk(chunk_content, "table", section_title, sub_title)
                )

        # 如果没有表格或文本部分还有剩余
        if not tables and text_parts:
            chunk_content = f"# {section_title}\n\n## {sub_title}\n\n{text_parts[0]}"
            chunks.append(
                self._create_chunk(chunk_content, "text", section_title, sub_title)
            )

        return chunks

    def _extract_tables(self, content: str) -> List[str]:
        """提取markdown表格"""
        tables = []
        # 匹配markdown表格
        pattern = r"\|[^\n]+\|\n\|[-:\|\s]+\|\n(?:\|[^\n]+\|\n?)+"
        matches = re.finditer(pattern, content)

        for match in matches:
            tables.append(match.group(0).strip())

        return tables

    def _is_level_table(self, table: str) -> bool:
        """判断是否是包含职级差异的表格"""
        level_keywords = [
            "8-9级",
            "10-11级",
            "12级及以上",
            "普通员工",
            "经理",
            "总监",
            "专家",
        ]
        return any(keyword in table for keyword in level_keywords)

    def _expand_by_level(
        self,
        section_title: str,
        sub_title: str,
        table: str,
        context: str,
        intro: str = "",
    ) -> List[Chunk]:
        """按职级展开表格"""
        chunks = []

        # 解析表格
        lines = [line.strip() for line in table.split("\n") if line.strip()]
        if len(lines) < 3:
            # 表格格式不对，直接返回
            chunk_content = f"# {section_title}\n\n## {sub_title}\n\n{table}"
            return [
                self._create_chunk(chunk_content, "table", section_title, sub_title)
            ]

        header_line = lines[0]
        separator = lines[1]
        data_lines = lines[2:]

        # 职级关键词映射
        level_keywords = {
            "普通员工/8-9级": ["8-9级", "普通员工"],
            "经理/10-11级": ["10-11级", "经理"],
            "总监及以上/12级": ["12级及以上", "总监", "专家"],
        }

        # 为每个职级创建独立chunk
        for level_name, keywords in level_keywords.items():
            level_data = []
            for line in data_lines:
                if any(keyword in line for keyword in keywords):
                    level_data.append(line)

            if level_data:
                # 构建该职级的表格
                level_table = f"{header_line}\n{separator}\n" + "\n".join(level_data)

                # 构建chunk内容
                chunk_parts = [f"# {section_title}"]
                if intro:
                    chunk_parts.append(intro)
                chunk_parts.extend(
                    [f"## {sub_title} - {level_name}", context, level_table]
                )

                chunk_content = "\n\n".join(chunk_parts)
                chunk = self._create_chunk(
                    chunk_content, "table", section_title, sub_title
                )
                chunk.metadata["level"] = level_name
                chunks.append(chunk)

        return chunks

    def _create_chunk(
        self, content: str, chunk_type: str, section: str, subsection: str = ""
    ) -> Chunk:
        """创建Chunk对象"""
        chunk = Chunk(
            content=content.strip(),
            chunk_type=chunk_type,
            metadata={
                "section": section,
                "subsection": subsection,
                "char_count": len(content),
            },
        )

        # 自动检测职级和费用类型
        chunk.metadata["level"] = self._detect_level(content)
        chunk.metadata["expense_type"] = self._detect_expense_type(content)

        return chunk

    def _detect_level(self, text: str) -> str:
        """检测文本涉及的职级"""
        levels = []
        if "8-9级" in text or "普通员工" in text:
            levels.append("8-9级")
        if "10-11级" in text or "经理" in text:
            levels.append("10-11级")
        if "12级及以上" in text or "总监" in text or "专家" in text:
            levels.append("12级及以上")
        return ",".join(levels)

    def _detect_expense_type(self, text: str) -> str:
        """检测费用类型"""
        types = []
        if any(kw in text for kw in ["差旅", "出差", "交通工具", "住宿", "补贴"]):
            types.append("差旅费")
        if any(kw in text for kw in ["招待", "宴请", "客户", "礼品"]):
            types.append("业务招待费")
        if any(kw in text for kw in ["通讯", "电话", "手机"]):
            types.append("通讯费")
        if any(kw in text for kw in ["办公", "文具", "书籍", "软件"]):
            types.append("办公费")
        if any(kw in text for kw in ["培训", "会议"]):
            types.append("培训会议费")
        return ",".join(types)


class ChunkingService:
    """切分服务封装"""

    @staticmethod
    def chunk_financial_document(file_path: str, doc_id: str = "") -> List[Dict]:
        """切分财务制度文档"""
        # 读取文档
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 创建切分器
        chunker = FinancialDocumentChunker(max_chunk_size=1000)

        # 执行切分
        chunks = chunker.chunk_document(content, doc_id)

        # 转换为字典格式
        result = []
        for i, chunk in enumerate(chunks):
            result.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{i:03d}",
                    "content": chunk.content,
                    "metadata": {**chunk.metadata, "chunk_type": chunk.chunk_type},
                }
            )

        return result

    @staticmethod
    def save_chunks(chunks: List[Dict], output_path: str):
        """保存切分结果"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ 切分完成，共生成 {len(chunks)} 个片段")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import os

    file_path = "/root/autodl-tmp/rag/backend/data/docs/baoxiao.md"
    doc_id = "baoxiao_001"
    output_dir = "/root/autodl-tmp/rag/backend/data/chunks"
    output_path = os.path.join(output_dir, "baoxiao_chunks_v2.json")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 执行切分
    chunks = ChunkingService.chunk_financial_document(file_path, doc_id)

    # 保存结果
    ChunkingService.save_chunks(chunks, output_path)

    # 统计信息
    print("\n📊 切分统计：")
    print(f"总计生成: {len(chunks)} 个chunk")

    # 按类型统计
    type_count = {}
    level_chunks = {"8-9级": [], "10-11级": [], "12级及以上": []}

    for chunk in chunks:
        chunk_type = chunk["metadata"]["chunk_type"]
        type_count[chunk_type] = type_count.get(chunk_type, 0) + 1

        level = chunk["metadata"].get("level", "")
        for lv in level_chunks.keys():
            if lv in level:
                level_chunks[lv].append(chunk["chunk_id"])

    print("\n按类型分布：")
    for t, count in type_count.items():
        print(f"  {t}: {count}个")

    print("\n按职级分布：")
    for lv, ids in level_chunks.items():
        print(f"  {lv}: {len(ids)}个chunk")

    # 打印示例
    print("\n📝 示例Chunk（普通员工-差旅费）：")
    for chunk in chunks:
        if "8-9级" in chunk["metadata"].get("level", "") and "差旅" in chunk["content"]:
            print(f"\nChunk ID: {chunk['chunk_id']}")
            print(f"职级: {chunk['metadata']['level']}")
            print(f"费用类型: {chunk['metadata']['expense_type']}")
            print(f"内容:\n{chunk['content'][:300]}...")
            break
