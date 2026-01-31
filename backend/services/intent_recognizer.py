"""
意图识别服务 - 基于规则和向量相似度的混合方案
支持动态使用当前选择的嵌入模型
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np
from utils.logger import logger


class IntentType(str, Enum):
    """意图类型枚举"""
    QUESTION = "question"      # 问题咨询
    SEARCH = "search"          # 信息搜索
    SUMMARY = "summary"        # 内容总结
    COMPARISON = "comparison"  # 对比分析
    PROCEDURE = "procedure"    # 操作流程
    DEFINITION = "definition"  # 定义说明
    GREETING = "greeting"      # 问候
    OTHER = "other"            # 其他


class IntentConfig:
    """意图对应的检索配置"""
    
    CONFIGS = {
        IntentType.QUESTION: {
            "top_k": 5,
            "similarity_threshold": 0.7,
            "description": "问题咨询",
            "prompt_template": "根据以下信息回答问题："
        },
        IntentType.SEARCH: {
            "top_k": 10,
            "similarity_threshold": 0.6,
            "description": "信息搜索",
            "prompt_template": "以下是搜索到的相关信息："
        },
        IntentType.SUMMARY: {
            "top_k": 15,
            "similarity_threshold": 0.5,
            "description": "内容总结",
            "prompt_template": "请总结以下内容："
        },
        IntentType.COMPARISON: {
            "top_k": 8,
            "similarity_threshold": 0.65,
            "description": "对比分析",
            "prompt_template": "对比以下信息："
        },
        IntentType.PROCEDURE: {
            "top_k": 5,
            "similarity_threshold": 0.7,
            "description": "操作流程",
            "prompt_template": "以下操作步骤："
        },
        IntentType.DEFINITION: {
            "top_k": 5,
            "similarity_threshold": 0.7,
            "description": "定义说明",
            "prompt_template": "定义如下："
        },
        IntentType.GREETING: {
            "top_k": 3,
            "similarity_threshold": 0.6,
            "description": "问候",
            "prompt_template": "您好！"
        },
        IntentType.OTHER: {
            "top_k": 5,
            "similarity_threshold": 0.7,
            "description": "其他",
            "prompt_template": "根据以下信息回答："
        }
    }
    
    @classmethod
    def get_config(cls, intent: IntentType) -> Dict:
        """获取意图对应的配置"""
        return cls.CONFIGS.get(intent, cls.CONFIGS[IntentType.OTHER])


class IntentRecognizer:
    """意图识别器"""
    
    # 规则匹配的关键词
    RULE_BASED_KEYWORDS = {
        IntentType.GREETING: [
            "你好", "您好", "hello", "hi", "嗨", "早上好", "下午好", "晚上好",
            "在吗", "在不在", "有人吗"
        ],
        IntentType.QUESTION: [
            "什么", "如何", "怎么", "为什么", "哪", "多少", "是否", "能不能",
            "可不可以", "有没有", "会吗", "需要吗"
        ],
        IntentType.SEARCH: [
            "搜索", "查找", "找一下", "查查", "搜一搜", "寻找", "查询",
            "帮我找", "我想找", "有没有关于"
        ],
        IntentType.SUMMARY: [
            "总结", "概括", "概述", "总结一下", "简要总结", "概括一下",
            "总的来说", "归纳", "摘要", "归纳一下"
        ],
        IntentType.COMPARISON: [
            "对比", "区别", "差异", "不同", "比较", "有什么区别",
            "区别在哪里", "有什么不同", "差异是什么"
        ],
        IntentType.PROCEDURE: [
            "步骤", "流程", "方法", "怎么做", "如何操作", "怎么弄",
            "操作步骤", "流程是什么", "方法是什么"
        ],
        IntentType.DEFINITION: [
            "定义", "含义", "是什么意思", "什么是", "解释", "说明",
            "意思是什么", "指的是", "解释一下", "说明一下", "介绍一下"
        ]
    }
    
    # 向量相似度匹配的意图模板
    INTENT_TEMPLATES = {
        IntentType.QUESTION: [
            "这是什么意思", "如何解决这个问题", "为什么会出现这种情况",
            "这个问题怎么处理", "这是什么情况", "为什么会这样",
            "请问这是什么", "需要了解什么信息"
        ],
        IntentType.SEARCH: [
            "搜索相关内容", "查找相关信息", "帮我找一下资料",
            "搜索关于这个的内容", "查找相关文档", "搜索相关信息"
        ],
        IntentType.SUMMARY: [
            "请总结一下", "概括主要内容", "简要概述",
            "总结这段内容", "概括这些信息", "概述总体情况"
        ],
        IntentType.COMPARISON: [
            "对比一下", "有什么区别", "差异在哪里",
            "比较两者的不同", "分析差异", "对比分析"
        ],
        IntentType.PROCEDURE: [
            "操作步骤是什么", "如何进行操作", "流程是怎样的",
            "具体步骤", "操作方法", "实施流程"
        ],
        IntentType.DEFINITION: [
            "这个概念的定义", "解释这个术语", "是什么意思",
            "定义是什么", "含义是什么", "这个指什么"
        ],
        IntentType.GREETING: [
            "你好啊", "您好", "大家好", "早上好",
            "下午好", "晚上好", "欢迎"
        ]
    }
    
    def __init__(self):
        self.embedding_service = None
        self.intent_template_embeddings: Dict[IntentType, np.ndarray] = {}
        self._initialized = False
    
    def initialize(self, embedding_service):
        """
        初始化意图识别器
        
        Args:
            embedding_service: 嵌入服务实例
        """
        self.embedding_service = embedding_service
        
        # 预计算意图模板的向量
        self._precompute_template_embeddings()
        
        self._initialized = True
        logger.info("意图识别器初始化完成")
    
    def _precompute_template_embeddings(self):
        """预计算意图模板的向量"""
        if not self.embedding_service or not self.embedding_service.is_loaded():
            logger.warning("嵌入模型未加载，无法预计算意图模板向量")
            return
        
        try:
            for intent, templates in self.INTENT_TEMPLATES.items():
                # 计算每个模板的向量
                template_vectors = self.embedding_service.encode(templates)
                # 计算平均向量作为该意图的表示
                avg_vector = np.mean(template_vectors, axis=0)
                self.intent_template_embeddings[intent] = avg_vector
            
            logger.info(f"已预计算 {len(self.intent_template_embeddings)} 个意图的模板向量")
        except Exception as e:
            logger.error(f"预计算意图模板向量失败: {str(e)}")
    
    def recognize(self, query: str) -> Tuple[IntentType, float, Dict]:
        """
        识别用户查询的意图
        
        Args:
            query: 用户查询文本
            
        Returns:
            (意图类型, 置信度, 详细信息)
        """
        if not query or not query.strip():
            return IntentType.OTHER, 0.0, {"method": "empty_query"}
        
        query = query.strip()
        
        # 步骤1: 规则匹配（快速通道）
        rule_intent, rule_confidence = self._rule_based_match(query)
        if rule_confidence > 0.8:
            logger.info(f"规则匹配识别意图: {rule_intent.value} (置信度: {rule_confidence:.2f})")
            return rule_intent, rule_confidence, {
                "method": "rule_based",
                "matched_keywords": self._get_matched_keywords(query, rule_intent)
            }
        
        # 步骤2: 向量相似度匹配
        if self._initialized and self.embedding_service and self.embedding_service.is_loaded():
            vector_intent, vector_confidence = self._vector_based_match(query)
            if vector_confidence > 0.6:  # 提高阈值，确保向量匹配更准确
                logger.info(f"向量匹配识别意图: {vector_intent.value} (置信度: {vector_confidence:.2f})")
                return vector_intent, vector_confidence, {
                    "method": "vector_based",
                    "similarities": self._get_all_similarities(query)
                }
        
        # 步骤3: 回退到规则匹配结果
        if rule_confidence > 0.3:
            return rule_intent, rule_confidence, {
                "method": "rule_based_fallback",
                "matched_keywords": self._get_matched_keywords(query, rule_intent)
            }
        
        # 默认返回其他意图
        logger.info(f"无法识别意图，返回默认意图: {IntentType.OTHER.value}")
        return IntentType.OTHER, 0.0, {"method": "default"}
    
    def _rule_based_match(self, query: str) -> Tuple[IntentType, float]:
        """
        基于规则的意图匹配
        
        Args:
            query: 用户查询
            
        Returns:
            (意图类型, 置信度)
        """
        best_intent = IntentType.OTHER
        best_score = 0.0
        
        for intent, keywords in self.RULE_BASED_KEYWORDS.items():
            score = 0.0
            for keyword in keywords:
                if keyword in query:
                    # 根据关键词长度和位置计算分数
                    score += len(keyword) / len(query)
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # 归一化分数
        confidence = min(best_score * 2, 1.0)
        
        return best_intent, confidence
    
    def _vector_based_match(self, query: str) -> Tuple[IntentType, float]:
        """
        基于向量相似度的意图匹配
        
        Args:
            query: 用户查询
            
        Returns:
            (意图类型, 置信度)
        """
        if not self.intent_template_embeddings:
            return IntentType.OTHER, 0.0
        
        try:
            # 计算查询的向量
            query_vector = self.embedding_service.encode([query])[0]
            
            # 计算与每个意图模板的相似度
            similarities = {}
            for intent, template_vector in self.intent_template_embeddings.items():
                # 使用余弦相似度
                similarity = self._cosine_similarity(query_vector, template_vector)
                similarities[intent] = similarity
            
            # 找到最相似的意图
            best_intent = max(similarities, key=similarities.get)
            best_score = similarities[best_intent]
            
            return best_intent, float(best_score)
            
        except Exception as e:
            logger.error(f"向量匹配失败: {str(e)}")
            return IntentType.OTHER, 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if vec1.ndim > 1:
            vec1 = vec1.flatten()
        if vec2.ndim > 1:
            vec2 = vec2.flatten()
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_matched_keywords(self, query: str, intent: IntentType) -> List[str]:
        """获取匹配的关键词"""
        keywords = self.RULE_BASED_KEYWORDS.get(intent, [])
        matched = [kw for kw in keywords if kw in query]
        return matched
    
    def _get_all_similarities(self, query: str) -> Dict[str, float]:
        """获取与所有意图的相似度"""
        try:
            query_vector = self.embedding_service.encode([query])[0]
            similarities = {}
            for intent, template_vector in self.intent_template_embeddings.items():
                similarity = self._cosine_similarity(query_vector, template_vector)
                similarities[intent.value] = float(similarity)
            return similarities
        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            return {}
    
    def update_embedding_service(self, embedding_service):
        """
        更新嵌入服务（当模型切换时调用）
        
        Args:
            embedding_service: 新的嵌入服务实例
        """
        self.embedding_service = embedding_service
        self.intent_template_embeddings = {}
        self._precompute_template_embeddings()
        logger.info("已更新意图识别器的嵌入服务")


# 全局意图识别器实例
intent_recognizer = IntentRecognizer()