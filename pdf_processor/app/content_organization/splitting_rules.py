# 智能切分规则模块

from typing import List, Dict, Any

from config.config import settings
from utils.utils import logger


class SplittingRules:
    """智能切分规则系统"""
    
    def __init__(self):
        """初始化切分规则系统"""
        # 定义不同文件类型的切分规则
        self.rules = {
            "financial_report": {
                "name": "财务报表切分规则",
                "strategies": [
                    {"type": "section_based", "sections": ["资产负债表", "利润表", "现金流量表", "所有者权益变动表", "财务报表附注"]},
                    {"type": "table_based", "min_rows": 5},
                    {"type": "size_based", "max_chunk_size": 800}
                ],
                "preserve_elements": ["table"]
            },
            "contract": {
                "name": "合同切分规则",
                "strategies": [
                    {"type": "section_based", "sections": ["第一条", "第二条", "第三条", "第四条", "第五条"]},
                    {"type": "paragraph_based", "min_paragraph_length": 200},
                    {"type": "size_based", "max_chunk_size": 600}
                ],
                "preserve_elements": ["title", "text"]
            },
            "invoice": {
                "name": "发票切分规则",
                "strategies": [
                    {"type": "page_based", "max_pages_per_chunk": 1},
                    {"type": "element_based", "element_types": ["text", "table"]},
                    {"type": "size_based", "max_chunk_size": 400}
                ],
                "preserve_elements": ["text", "table"]
            },
            "report": {
                "name": "报告切分规则",
                "strategies": [
                    {"type": "section_based", "sections": ["摘要", "引言", "方法", "结果", "讨论", "结论", "建议", "参考文献"]},
                    {"type": "paragraph_based", "min_paragraph_length": 150},
                    {"type": "size_based", "max_chunk_size": 700}
                ],
                "preserve_elements": ["title", "text", "table"]
            },
            "thesis": {
                "name": "论文切分规则",
                "strategies": [
                    {"type": "section_based", "sections": ["摘要", "关键词", "引言", "第一章", "第二章", "第三章", "结论", "参考文献", "致谢"]},
                    {"type": "paragraph_based", "min_paragraph_length": 200},
                    {"type": "size_based", "max_chunk_size": 900}
                ],
                "preserve_elements": ["title", "text", "table", "image"]
            },
            "baoxiao": {
                "name": "报销单切分规则",
                "strategies": [
                    {"type": "page_based", "max_pages_per_chunk": 1},
                    {"type": "form_based", "form_fields": ["报销人", "部门", "日期", "用途", "金额"]},
                    {"type": "size_based", "max_chunk_size": 500}
                ],
                "preserve_elements": ["text", "table"]
            },
            "general": {
                "name": "通用切分规则",
                "strategies": [
                    {"type": "size_based", "max_chunk_size": settings.content_organization.chunk_size},
                    {"type": "paragraph_based", "min_paragraph_length": 100},
                    {"type": "element_based", "element_types": ["text", "title", "table", "image"]}
                ],
                "preserve_elements": ["text", "title", "table", "image"]
            }
        }
    
    def get_splitting_rules(self, file_type: str) -> Dict[str, Any]:
        """
        获取指定文件类型的切分规则
        
        Args:
            file_type: 文件类型
            
        Returns:
            切分规则
        """
        if file_type in self.rules:
            logger.info(f"使用{self.rules[file_type]['name']}")
            return self.rules[file_type]
        else:
            logger.info(f"文件类型{file_type}未找到对应规则，使用通用切分规则")
            return self.rules["general"]
    
    def apply_splitting_strategy(self, file_type: str, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        应用切分策略
        
        Args:
            file_type: 文件类型
            elements: 排序后的元素列表
            
        Returns:
            切分后的chunk列表
        """
        rules = self.get_splitting_rules(file_type)
        strategies = rules.get("strategies", [])
        preserve_elements = rules.get("preserve_elements", [])
        
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        # 应用策略
        for strategy in strategies:
            strategy_type = strategy.get("type")
            
            if strategy_type == "size_based":
                # 基于大小的切分
                max_chunk_size = strategy.get("max_chunk_size", settings.content_organization.chunk_size)
                chunks, current_chunk, current_chunk_size = self._apply_size_based_strategy(
                    elements, max_chunk_size, chunks, current_chunk, current_chunk_size
                )
            elif strategy_type == "section_based":
                # 基于章节的切分
                sections = strategy.get("sections", [])
                chunks, current_chunk, current_chunk_size = self._apply_section_based_strategy(
                    elements, sections, chunks, current_chunk, current_chunk_size
                )
            elif strategy_type == "page_based":
                # 基于页面的切分
                max_pages_per_chunk = strategy.get("max_pages_per_chunk", 1)
                chunks, current_chunk, current_chunk_size = self._apply_page_based_strategy(
                    elements, max_pages_per_chunk, chunks, current_chunk, current_chunk_size
                )
        
        # 保存最后一个chunk
        if current_chunk:
            chunks.append({
                "elements": current_chunk,
                "size": current_chunk_size
            })
        
        return chunks
    
    def _apply_size_based_strategy(self, elements: List[Dict[str, Any]], max_chunk_size: int, 
                                  chunks: List[Dict[str, Any]], current_chunk: List[Dict[str, Any]], 
                                  current_chunk_size: int) -> tuple:
        """
        应用基于大小的切分策略
        """
        for element in elements:
            element_content = element.get("content", "")
            element_size = len(element_content)
            
            if current_chunk_size + element_size > max_chunk_size:
                if current_chunk:
                    chunks.append({
                        "elements": current_chunk,
                        "size": current_chunk_size
                    })
                    current_chunk = [element]
                    current_chunk_size = element_size
            else:
                current_chunk.append(element)
                current_chunk_size += element_size
        
        return chunks, current_chunk, current_chunk_size
    
    def _apply_section_based_strategy(self, elements: List[Dict[str, Any]], sections: List[str], 
                                     chunks: List[Dict[str, Any]], current_chunk: List[Dict[str, Any]], 
                                     current_chunk_size: int) -> tuple:
        """
        应用基于章节的切分策略
        """
        for element in elements:
            element_content = element.get("content", "")
            element_size = len(element_content)
            
            # 检查是否是章节开头
            is_section_start = any(section in element_content for section in sections)
            
            if is_section_start and current_chunk:
                chunks.append({
                    "elements": current_chunk,
                    "size": current_chunk_size
                })
                current_chunk = [element]
                current_chunk_size = element_size
            else:
                current_chunk.append(element)
                current_chunk_size += element_size
        
        return chunks, current_chunk, current_chunk_size
    
    def _apply_page_based_strategy(self, elements: List[Dict[str, Any]], max_pages_per_chunk: int, 
                                   chunks: List[Dict[str, Any]], current_chunk: List[Dict[str, Any]], 
                                   current_chunk_size: int) -> tuple:
        """
        应用基于页面的切分策略
        """
        current_page = None
        page_count = 0
        
        for element in elements:
            element_page = element.get("page_num", 0)
            element_content = element.get("content", "")
            element_size = len(element_content)
            
            if current_page is None:
                current_page = element_page
                page_count = 1
            elif element_page != current_page:
                page_count += 1
                current_page = element_page
            
            if page_count > max_pages_per_chunk and current_chunk:
                chunks.append({
                    "elements": current_chunk,
                    "size": current_chunk_size
                })
                current_chunk = [element]
                current_chunk_size = element_size
                page_count = 1
            else:
                current_chunk.append(element)
                current_chunk_size += element_size
        
        return chunks, current_chunk, current_chunk_size


# 创建切分规则实例
splitting_rules = SplittingRules()