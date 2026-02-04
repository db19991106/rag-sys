# 文件类型自动识别模块

import os
import re
from typing import Dict, Any, Optional

from config.config import settings
from utils.utils import logger


class FileTypeIdentifier:
    """文件类型自动识别器"""
    
    def __init__(self):
        """初始化文件类型识别器"""
        # 定义文件类型特征
        self.file_type_features = {
            "financial_report": {
                "name": "财务报表",
                "keywords": ["财务报表", "资产负债表", "利润表", "现金流量表", "财务状况", "审计报告", "会计报表", "财务分析"],
                "patterns": [r"\d{4}年.*财务报表", r"资产负债表.*\d{4}", r"利润表.*\d{4}"],
                "confidence_threshold": 0.6
            },
            "contract": {
                "name": "合同",
                "keywords": ["合同", "协议", "条款", "甲方", "乙方", "签署", "生效", "违约责任", "权利义务"],
                "patterns": [r"(合同|协议)\s*编号", r"甲方.*乙方", r"本合同.*生效"],
                "confidence_threshold": 0.5
            },
            "invoice": {
                "name": "发票",
                "keywords": ["发票", "增值税", "开票日期", "发票号码", "购买方", "销售方", "金额"],
                "patterns": [r"发票号码.*\d+", r"开票日期.*\d{4}-\d{2}-\d{2}", r"价税合计.*￥"],
                "confidence_threshold": 0.7
            },
            "report": {
                "name": "报告",
                "keywords": ["报告", "研究", "分析", "调查", "结论", "建议", "摘要", "目录"],
                "patterns": [r"(研究|分析|调查报告)", r"摘要.*\\n", r"目录.*\\n"],
                "confidence_threshold": 0.4
            },
            "thesis": {
                "name": "论文",
                "keywords": ["论文", "摘要", "关键词", "引言", "结论", "参考文献", "致谢"],
                "patterns": [r"摘要.*\\n", r"关键词.*\\n", r"参考文献.*\\n"],
                "confidence_threshold": 0.6
            },
            "baoxiao": {
                "name": "报销单",
                "keywords": ["报销", "费用", "报销单", "审批", "金额", "日期", "部门", "用途"],
                "patterns": [r"报销单", r"费用报销", r"报销金额.*￥"],
                "confidence_threshold": 0.6
            }
        }
    
    def identify_file_type(self, pdf_path: str, elements: list) -> Dict[str, Any]:
        """
        识别文件类型
        
        Args:
            pdf_path: PDF文件路径
            elements: 提取的元素列表
            
        Returns:
            文件类型识别结果
        """
        try:
            # 提取文本内容
            text_content = self._extract_text_from_elements(elements)
            
            # 分析文件类型
            file_type_scores = {}
            
            for file_type, features in self.file_type_features.items():
                score = self._calculate_file_type_score(text_content, features)
                file_type_scores[file_type] = score
            
            # 找出得分最高的文件类型
            best_file_type = None
            best_score = 0
            
            for file_type, score in file_type_scores.items():
                threshold = self.file_type_features[file_type]["confidence_threshold"]
                if score > best_score and score >= threshold:
                    best_score = score
                    best_file_type = file_type
            
            if best_file_type:
                file_type_info = {
                    "type": best_file_type,
                    "name": self.file_type_features[best_file_type]["name"],
                    "confidence": best_score,
                    "scores": file_type_scores
                }
                logger.info(f"文件类型识别结果: {file_type_info['name']} (置信度: {best_score:.2f})")
            else:
                file_type_info = {
                    "type": "general",
                    "name": "通用文档",
                    "confidence": 0,
                    "scores": file_type_scores
                }
                logger.info("文件类型识别结果: 通用文档")
            
            return file_type_info
            
        except Exception as e:
            logger.error(f"文件类型识别失败: {str(e)}")
            return {
                "type": "general",
                "name": "通用文档",
                "confidence": 0,
                "scores": {}
            }
    
    def _extract_text_from_elements(self, elements: list) -> str:
        """
        从元素列表中提取文本内容
        
        Args:
            elements: 元素列表
            
        Returns:
            提取的文本内容
        """
        text_content = ""
        
        for element in elements:
            if element.get("type") == "text" or element.get("type") == "title":
                text_content += element.get("content", "") + " "
        
        return text_content
    
    def _calculate_file_type_score(self, text_content: str, features: Dict[str, Any]) -> float:
        """
        计算文件类型得分
        
        Args:
            text_content: 文本内容
            features: 文件类型特征
            
        Returns:
            得分
        """
        score = 0
        total_weight = 0
        
        # 关键词匹配
        keywords = features.get("keywords", [])
        for keyword in keywords:
            if keyword in text_content:
                score += 0.1
                total_weight += 0.1
        
        # 模式匹配
        patterns = features.get("patterns", [])
        for pattern in patterns:
            if re.search(pattern, text_content):
                score += 0.3
                total_weight += 0.3
        
        # 计算置信度
        if total_weight > 0:
            confidence = score / total_weight
        else:
            confidence = 0
        
        return confidence


# 创建文件类型识别器实例
file_type_identifier = FileTypeIdentifier()