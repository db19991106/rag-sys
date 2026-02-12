"""
文档类型检测器 - 自动识别文档类型
"""

import re
from enum import Enum
from typing import Dict, List, Tuple
from utils.logger import logger


class DocumentCategory(str, Enum):
    """文档类别"""

    FINANCIAL = "financial"  # 财务制度
    PRODUCT = "product"  # 产品文档
    TECHNICAL = "technical"  # 技术规范
    COMPLIANCE = "compliance"  # 合规文件
    HR = "hr"  # HR文档
    PROJECT = "project"  # 项目管理
    UNKNOWN = "unknown"  # 未知类型


class DocumentTypeDetector:
    """文档类型检测器"""

    # 各类文档的特征关键词和权重
    CATEGORY_PATTERNS = {
        DocumentCategory.FINANCIAL: {
            "keywords": [
                ("报销", 3),
                ("差旅费", 3),
                ("发票", 2),
                ("借款", 2),
                ("审批", 2),
                ("费用", 2),
                ("预算", 2),
                ("成本", 2),
                ("财务", 2),
                ("会计", 2),
                ("付款", 1),
                ("收款", 1),
                ("账务", 1),
                ("审计", 1),
                ("票据", 1),
            ],
            "title_patterns": [
                r".*报销.*管理.*",
                r".*费用.*管理.*",
                r".*差旅.*",
                r".*预算.*管理.*",
            ],
        },
        DocumentCategory.PRODUCT: {
            "keywords": [
                ("产品", 2),
                ("功能", 2),
                ("用户", 2),
                ("使用", 1),
                ("介绍", 1),
                ("特性", 2),
                ("模块", 1),
                ("界面", 1),
                ("操作", 1),
                ("配置", 1),
                ("部署", 1),
                ("安装", 1),
                ("系统", 1),
                ("平台", 1),
                ("版本", 1),
                ("手册", 2),
                ("指南", 2),
                ("说明书", 2),
                ("白皮书", 2),
            ],
            "title_patterns": [
                r".*产品.*",
                r".*手册.*",
                r".*指南.*",
                r".*说明书.*",
                r".*介绍.*",
                r".*白皮书.*",
            ],
        },
        DocumentCategory.TECHNICAL: {
            "keywords": [
                ("API", 3),
                ("接口", 2),
                ("规范", 2),
                ("设计", 1),
                ("架构", 2),
                ("代码", 2),
                ("开发", 2),
                ("测试", 1),
                ("数据库", 2),
                ("配置", 1),
                ("协议", 2),
                ("标准", 2),
                ("约定", 1),
                ("命名", 1),
                ("格式", 1),
                ("RESTful", 3),
                ("HTTP", 2),
                ("JSON", 1),
                ("SQL", 2),
                ("微服务", 2),
                ("DevOps", 2),
                ("CI/CD", 2),
            ],
            "title_patterns": [
                r".*规范.*",
                r".*标准.*",
                r".*API.*",
                r".*接口.*",
                r".*设计.*",
                r".*架构.*",
                r".*开发.*",
                r".*数据库.*",
            ],
        },
        DocumentCategory.COMPLIANCE: {
            "keywords": [
                ("隐私", 3),
                ("政策", 2),
                ("安全", 2),
                ("合规", 3),
                ("法律", 2),
                ("制度", 2),
                ("规定", 2),
                ("条款", 2),
                ("同意", 1),
                ("授权", 1),
                ("数据保护", 3),
                ("信息安全", 3),
                ("保密", 2),
                ("协议", 1),
                ("许可", 2),
                ("版权", 2),
                ("知识产权", 2),
                ("免责", 1),
            ],
            "title_patterns": [
                r".*隐私.*",
                r".*政策.*",
                r".*安全.*",
                r".*制度.*",
                r".*合规.*",
                r".*协议.*",
            ],
        },
        DocumentCategory.HR: {
            "keywords": [
                ("员工", 3),
                ("绩效", 3),
                ("考核", 2),
                ("薪酬", 2),
                ("福利", 2),
                ("入职", 2),
                ("离职", 2),
                ("请假", 1),
                ("考勤", 2),
                ("培训", 1),
                ("晋升", 2),
                ("调薪", 2),
                ("招聘", 1),
                ("面试", 1),
                ("劳动合同", 2),
                ("手册", 2),
                ("制度", 1),
                ("规定", 1),
                ("管理", 1),
            ],
            "title_patterns": [
                r".*员工.*",
                r".*绩效.*",
                r".*考核.*",
                r".*手册.*",
                r".*薪酬.*",
                r".*HR.*",
            ],
        },
        DocumentCategory.PROJECT: {
            "keywords": [
                ("项目", 2),
                ("流程", 2),
                ("管理", 1),
                ("开发", 1),
                ("发布", 2),
                ("版本", 2),
                ("迭代", 2),
                ("Sprint", 2),
                ("需求", 1),
                ("测试", 1),
                ("部署", 2),
                ("上线", 2),
                ("回滚", 2),
                ("灰度", 2),
                ("评审", 1),
                ("里程碑", 1),
                ("进度", 1),
                ("风险", 1),
                ("质量", 1),
            ],
            "title_patterns": [
                r".*流程.*",
                r".*管理.*",
                r".*开发.*",
                r".*发布.*",
                r".*版本.*",
                r".*项目.*",
            ],
        },
    }

    @classmethod
    def detect(cls, content: str, title: str = "") -> Tuple[DocumentCategory, float]:
        """
        检测文档类型

        Args:
            content: 文档内容
            title: 文档标题（可选，提高准确性）

        Returns:
            (文档类别, 置信度)
        """
        if not content:
            return DocumentCategory.UNKNOWN, 0.0

        # 计算各类别的得分
        scores = {}

        for category, patterns in cls.CATEGORY_PATTERNS.items():
            score = 0

            # 1. 关键词匹配
            for keyword, weight in patterns["keywords"]:
                # 使用正则表达式匹配，支持变体
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                matches = len(pattern.findall(content))
                if matches > 0:
                    # 前3次匹配计分，后续匹配权重递减
                    effective_matches = min(matches, 3)
                    score += effective_matches * weight

            # 2. 标题模式匹配（如果有标题）
            if title:
                for title_pattern in patterns["title_patterns"]:
                    if re.search(title_pattern, title, re.IGNORECASE):
                        score += 5  # 标题匹配权重较高

            scores[category] = score

        # 找出得分最高的类别
        if not scores:
            return DocumentCategory.UNKNOWN, 0.0

        max_category = max(scores, key=scores.get)
        max_score = scores[max_category]

        # 计算置信度
        total_score = sum(scores.values())
        if total_score == 0:
            confidence = 0.0
        else:
            # 置信度 = 最高得分 / 总得分，但考虑绝对得分
            confidence = min(max_score / max(total_score * 0.5, 1), 1.0)

        # 如果最高得分太低，标记为未知
        if max_score < 3:
            return DocumentCategory.UNKNOWN, confidence

        logger.info(
            f"文档类型检测: {max_category.value}, 置信度: {confidence:.2f}, 得分: {scores}"
        )

        return max_category, confidence

    @classmethod
    def detect_from_filename(cls, filename: str) -> DocumentCategory:
        """
        从文件名推断文档类型

        Args:
            filename: 文件名

        Returns:
            文档类别
        """
        filename_lower = filename.lower()

        # 财务相关
        if any(
            kw in filename_lower
            for kw in ["报销", "差旅", "费用", "预算", "成本", "财务"]
        ):
            return DocumentCategory.FINANCIAL

        # 产品相关
        if any(
            kw in filename_lower
            for kw in ["产品", "手册", "指南", "介绍", "白皮书", "说明书"]
        ):
            return DocumentCategory.PRODUCT

        # 技术相关
        if any(
            kw in filename_lower
            for kw in ["规范", "api", "接口", "设计", "架构", "技术", "开发"]
        ):
            return DocumentCategory.TECHNICAL

        # 合规相关
        if any(
            kw in filename_lower
            for kw in ["隐私", "政策", "安全", "合规", "制度", "协议"]
        ):
            return DocumentCategory.COMPLIANCE

        # HR相关
        if any(
            kw in filename_lower
            for kw in ["员工", "绩效", "考核", "手册", "薪酬", "hr"]
        ):
            return DocumentCategory.HR

        # 项目相关
        if any(
            kw in filename_lower
            for kw in ["流程", "管理", "开发", "发布", "版本", "项目"]
        ):
            return DocumentCategory.PROJECT

        return DocumentCategory.UNKNOWN


def detect_document_type(
    content: str, title: str = "", filename: str = ""
) -> DocumentCategory:
    """
    检测文档类型的便捷函数

    Args:
        content: 文档内容
        title: 文档标题
        filename: 文件名

    Returns:
        文档类别
    """
    # 优先使用内容检测
    category, confidence = DocumentTypeDetector.detect(content, title)

    # 如果置信度低，尝试用文件名补充
    if confidence < 0.5 and filename:
        category_from_name = DocumentTypeDetector.detect_from_filename(filename)
        if category_from_name != DocumentCategory.UNKNOWN:
            logger.info(f"文件名辅助检测: {category_from_name.value}")
            return category_from_name

    return category
