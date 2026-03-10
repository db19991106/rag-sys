"""
查询增强器 (Query Enhancer)
实现功能：指代消解、HyDE(假设文档嵌入)、问题分解
"""

import re
import hashlib
import os
import torch
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from utils.logger import logger
from config import settings


class EnhancementType(str, Enum):
    """增强类型"""

    COREFERENCE_RESOLUTION = "coreference_resolution"  # 指代消解
    HYDE_SINGLE = "hyde_single"  # 单假设HyDE
    HYDE_MULTI = "hyde_multi"  # 多假设HyDE
    QUERY_DECOMPOSITION = "query_decomposition"  # 问题分解
    TEMPORAL_INJECTION = "temporal_injection"  # 时间约束注入


@dataclass
class SubQuery:
    """子查询"""

    id: str
    text: str
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他子查询ID
    query_type: str = "independent"  # independent | dependent
    expected_answer_type: str = "text"


@dataclass
class HypothesisDoc:
    """假设文档"""

    text: str
    entity: Optional[str] = None
    confidence: float = 0.0


@dataclass
class EnhancedQuery:
    """增强后的查询"""

    original_query: str
    main_query: str  # 主查询(改写后)
    sub_queries: List[SubQuery] = field(default_factory=list)
    hypotheses: List[HypothesisDoc] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    scene_tags: List[str] = field(default_factory=list)
    execution_plan: Dict = field(default_factory=dict)
    enhancements_applied: List[str] = field(default_factory=list)


class QueryEnhancer:
    """查询增强器"""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        
        # 指代消解专用 LLM 客户端（使用轻量级模型）
        self.coref_llm = None
        self.coref_tokenizer = None
        self._coref_llm_initialized = False
    
    def _init_coref_llm(self):
        """延迟初始化指代消解专用 LLM"""
        if self._coref_llm_initialized:
            return
            
        try:
            # 设置环境变量以抑制日志和进度条
            os.environ['TQDM_DISABLE'] = '1'
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            os.environ['TRANSFORMERS_SILENCE_DEPRECATION_WARNINGS'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_path = settings.coref_llm_model_path
            device = settings.coref_llm_device
            
            logger.info(f"初始化指代消解专用 LLM: {model_path}")

            # 临时重定向stdout和stderr以抑制进度条
            import sys
            from io import StringIO
            
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                
                # 加载 tokenizer
                self.coref_tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )

                # 加载模型
                model_kwargs = {
                    "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                    "device_map": "auto" if device == "cuda" else None,
                }

                self.coref_llm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    **model_kwargs
                )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            if device == "cpu":
                self.coref_llm = self.coref_llm.to(device)

            self._coref_llm_initialized = True
            logger.info(f"指代消解专用 LLM 初始化完成，设备: {device}")

        except Exception as e:
            logger.error(f"初始化指代消解专用 LLM 失败: {str(e)}")
            raise
    
    def _coref_llm_chat(self, prompt: str, temperature: float = 0.0) -> str:
        """调用指代消解专用 LLM"""
        if not self._coref_llm_initialized:
            self._init_coref_llm()
            
        try:
            # 构建 messages 格式
            messages = [
                {"role": "user", "content": prompt}
            ]

            # 应用 chat template
            text = self.coref_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.coref_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.coref_llm.device)

            # 生成
            with torch.no_grad():
                do_sample = True
                if temperature <= 0.0:
                    do_sample = False
                    temperature = 1.0
                
                outputs = self.coref_llm.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=do_sample,
                    pad_token_id=self.coref_tokenizer.eos_token_id
                )

            # 解码
            response = self.coref_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            logger.error(f"指代消解 LLM 调用失败: {str(e)}")
            return ""

    def enhance(
        self, query: str, session_context: Dict, scene_tags: List[str]
    ) -> EnhancedQuery:
        """
        查询增强主入口

        Args:
            query: 原始查询
            session_context: 会话上下文
            scene_tags: 场景标签

        Returns:
            EnhancedQuery: 增强后的查询包
        """
        enhanced = EnhancedQuery(
            original_query=query, main_query=query, scene_tags=scene_tags
        )

        # 1. 指代消解 (如果存在HistoryRef标签)
        if "HistoryRef" in scene_tags:
            resolved_query, entities_used = self._resolve_coreference(
                query, session_context
            )
            if resolved_query != query:
                enhanced.main_query = resolved_query
                enhanced.enhancements_applied.append("coreference_resolution")
                enhanced.filters["entities_inherited"] = entities_used
                logger.info(f"Coreference resolved: '{query}' -> '{resolved_query}'")

        # 2. HyDE增强 (如果存在Ambiguous标签)
        if "Ambiguous" in scene_tags:
            hypotheses = self._generate_hyde_hypotheses(
                enhanced.main_query, session_context, max_hypotheses=3
            )
            enhanced.hypotheses = hypotheses
            enhanced.enhancements_applied.append(
                "hyde_multi" if len(hypotheses) > 1 else "hyde_single"
            )

        # 3. 问题分解 (如果存在MultiQuestion标签)
        if "MultiQuestion" in scene_tags:
            sub_queries, execution_plan = self._decompose_query(enhanced.main_query)
            enhanced.sub_queries = sub_queries
            enhanced.execution_plan = execution_plan
            enhanced.enhancements_applied.append("query_decomposition")

        # 4. 时间约束注入 (如果存在Temporal标签)
        if "Temporal" in scene_tags:
            enhanced.filters["temporal_boost"] = True
            enhanced.filters["recency_weight"] = 0.3
            enhanced.enhancements_applied.append("temporal_injection")

        # 5. 对比处理 (如果存在Comparative标签)
        if "Comparative" in scene_tags:
            comparative_queries = self._handle_comparative(enhanced.main_query)
            if comparative_queries:
                enhanced.sub_queries.extend(comparative_queries)
                enhanced.enhancements_applied.append("comparative_expansion")

        return enhanced

    def _resolve_coreference(
        self, query: str, session_context: Dict
    ) -> Tuple[str, List[str]]:
        """
        使用 LLM 进行指代消解

        Args:
            query: 包含指代词的查询
            session_context: 包含历史实体和对话历史的会话上下文

        Returns:
            (消解后的查询, 使用的实体列表)
        """
        # 获取对话历史
        history = session_context.get("history", [])
        entities = session_context.get("entities", [])
        
        if not history:
            return query, []

        # 格式化对话历史
        history_str = self._format_history_for_coref(history)
        
        # 格式化实体信息
        entities_str = ""
        if entities:
            entity_names = [e.get("name", "") for e in entities if e.get("name")]
            if entity_names:
                entities_str = "已知实体：" + "、".join(entity_names[:5])
        
        # 构建 LLM prompt
        prompt = f"""任务：将用户查询中的指代词替换为具体内容。

对话历史：
{history_str}

{entities_str}

用户当前查询：{query}

规则：
1. 把"它/这/那/该/此"等代词替换成历史中提到的具体事物
2. 不要回答问题，只改写查询
3. 只输出改写后的查询，不要其他文字

改写后："""

        try:
            resolved_query = self._coref_llm_chat(prompt, temperature=0.0).strip()
            
            # 清理可能的格式残留
            resolved_query = (
                resolved_query.replace("【改写后查询】", "")
                .replace("改写后查询：", "")
                .strip()
            )
            
            if resolved_query and len(resolved_query) > 3:
                # 提取使用的实体（简化处理）
                entities_used = []
                for entity in entities:
                    entity_name = entity.get("name", "")
                    if entity_name and entity_name in resolved_query:
                        entities_used.append(entity_name)
                
                logger.info(f"LLM 指代消解: '{query}' -> '{resolved_query}'")
                return resolved_query, entities_used
            else:
                logger.warning(f"LLM 返回空或太短，使用原始查询: {query}")
                return query, []
                
        except Exception as e:
            logger.error(f"LLM 指代消解失败: {str(e)}")
            return query, []
    
    def _format_history_for_coref(self, history: List[Dict]) -> str:
        """格式化对话历史用于指代消解"""
        # 最多保留最近5轮对话
        recent_history = history[-5:] if len(history) > 5 else history
        history_lines = []
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            role_cn = "用户" if role == "user" else "助手"
            history_lines.append(f"{role_cn}：{content}")
        return "\n".join(history_lines)

    def _generate_hyde_hypotheses(
        self, query: str, session_context: Dict, max_hypotheses: int = 3
    ) -> List[HypothesisDoc]:
        """
        生成HyDE假设文档

        HyDE (Hypothetical Document Embedding):
        使用LLM生成假设的理想回答文档，然后基于这些文档进行检索

        Args:
            query: 查询
            session_context: 会话上下文
            max_hypotheses: 最大假设数量

        Returns:
            List[HypothesisDoc]: 假设文档列表
        """
        logger.info(f"[HyDE] ========== 开始 HyDE 处理 ==========")
        logger.info(f"[HyDE] 原始查询: {query}")
        logger.info(f"[HyDE] 最大假设数量: {max_hypotheses}")
        
        hypotheses = []

        # 检查是否有歧义实体
        logger.info(f"[HyDE] 检测歧义实体...")
        ambiguous_entities = self._detect_ambiguous_entities(query)
        logger.info(f"[HyDE] 检测到 {len(ambiguous_entities)} 个歧义实体: {[e['entity'] for e in ambiguous_entities]}")

        if ambiguous_entities and len(ambiguous_entities) <= max_hypotheses:
            # 为每个歧义实体生成假设
            logger.info(f"[HyDE] 进入多假设模式（歧义消解）")
            for i, entity_info in enumerate(ambiguous_entities):
                entity = entity_info["entity"]
                meanings = entity_info["meanings"]
                logger.info(f"[HyDE] 处理歧义实体 [{i+1}/{len(ambiguous_entities)}]: '{entity}' -> 可能含义: {meanings}")

                for meaning in meanings[:1]:  # 每个实体取最可能的含义
                    # 构建特化查询
                    specialized_query = query.replace(entity, meaning)
                    logger.info(f"[HyDE] 特化查询: '{query}' -> '{specialized_query}'")

                    # 生成假设文档
                    hypothesis_text = self._generate_hypothesis_text(
                        specialized_query, meaning
                    )

                    hypotheses.append(
                        HypothesisDoc(
                            text=hypothesis_text, entity=meaning, confidence=0.7
                        )
                    )
                    logger.info(f"[HyDE] 假设文档 #{len(hypotheses)} 已生成，实体: {meaning}，置信度: 0.7")
        else:
            # 生成单一假设
            logger.info(f"[HyDE] 进入单假设模式（无歧义或歧义过多）")
            hypothesis_text = self._generate_hypothesis_text(query)
            hypotheses.append(HypothesisDoc(text=hypothesis_text, confidence=0.8))
            logger.info(f"[HyDE] 单一假设文档已生成，置信度: 0.8")

        result_count = len(hypotheses[:max_hypotheses])
        logger.info(f"[HyDE] ========== HyDE 处理完成 ==========")
        logger.info(f"[HyDE] 最终生成 {result_count} 个假设文档")
        for i, h in enumerate(hypotheses[:max_hypotheses]):
            logger.info(f"[HyDE] 假设 {i+1}: 实体={h.entity or 'N/A'}, 置信度={h.confidence}, 内容长度={len(h.text)}字")

        return hypotheses[:max_hypotheses]

    def _generate_hypothesis_text(self, query: str, context: str = None) -> str:
        """
        使用 LLM 生成假设文档文本
        
        HyDE 核心思想：用 LLM 生成一个假设的理想回答文档，
        然后用这个假设文档的向量去检索，缩小查询-文档语义差距
        
        Args:
            query: 用户查询
            context: 额外上下文（如歧义消解后的实体）
            
        Returns:
            假设文档文本
        """
        logger.info(f"[HyDE] 开始生成假设文档，查询: {query[:50]}...")
        if context:
            logger.info(f"[HyDE] 上下文实体: {context}")
        
        # 构建 LLM prompt
        context_hint = f"\n上下文提示：这可能关于「{context}」。" if context else ""
        
        prompt = f"""请为以下问题生成一个假设的理想回答文档。
这个文档应该包含可能相关的关键信息、概念和细节，用于语义检索匹配。{context_hint}

问题：{query}

要求：
1. 生成一段完整的回答文档（100-200字）
2. 包含可能相关的关键词、概念、实体名称
3. 不需要完全准确，但要语义相关
4. 只输出假设文档内容，不要其他解释

假设回答："""

        try:
            # 使用指代消解专用的 LLM 生成假设文档
            hypothesis = self._coref_llm_chat(prompt, temperature=0.7).strip()
            
            # 清理可能的格式残留
            hypothesis = (
                hypothesis.replace("假设回答：", "")
                .replace("【假设回答】", "")
                .strip()
            )
            
            if hypothesis and len(hypothesis) > 20:
                logger.info(f"[HyDE] 假设文档生成完成 ({len(hypothesis)} 字): {hypothesis[:100]}...")
                return hypothesis
            else:
                logger.warning(f"[HyDE] LLM 返回内容过短，使用备用方案")
                # 备用方案：关键词拼接
                keywords = self._extract_keywords(query)
                fallback = f"关于「{query}」的相关信息，涉及{', '.join(keywords[:5])}等方面。"
                logger.info(f"[HyDE] 使用备用假设文档: {fallback}")
                return fallback
                
        except Exception as e:
            logger.error(f"[HyDE] LLM 生成假设文档失败: {str(e)}")
            # 备用方案
            keywords = self._extract_keywords(query)
            fallback = f"关于「{query}」的相关信息，涉及{', '.join(keywords[:5])}等方面。"
            logger.info(f"[HyDE] 使用备用假设文档: {fallback}")
            return fallback

    def _detect_ambiguous_entities(self, query: str) -> List[Dict]:
        """检测歧义实体"""
        ambiguous_dict = {
            "苹果": {
                "meanings": ["苹果公司", "苹果水果"],
                "context_hints": ["手机", "股价", "吃", "水果"],
            },
            "java": {
                "meanings": ["Java编程语言", "爪哇岛"],
                "context_hints": ["编程", "代码", "印度尼西亚", "旅游"],
            },
            "python": {
                "meanings": ["Python编程语言", "蟒蛇"],
                "context_hints": ["编程", "代码", "动物", "蛇"],
            },
            " Aurora": {
                "meanings": ["极光现象", "欧若拉(罗马女神)"],
                "context_hints": ["天文", "北欧", "神话", "女神"],
            },
            "亚马逊": {
                "meanings": ["亚马逊公司", "亚马逊雨林", "亚马逊河"],
                "context_hints": ["电商", "购物", "森林", "河流"],
            },
        }

        results = []
        query_lower = query.lower()

        for entity, info in ambiguous_dict.items():
            # 跳过空字符串实体
            if not entity:
                continue
            if entity in query_lower or entity.lower() in query_lower:
                results.append({"entity": entity, **info})

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取
        words = re.findall(r"[\w\u4e00-\u9fa5]+", text)
        # 过滤停用词
        stop_words = {
            "的",
            "了",
            "在",
            "是",
            "我",
            "有",
            "和",
            "就",
            "不",
            "人",
            "都",
            "一",
            "一个",
            "上",
            "也",
            "很",
            "到",
            "说",
            "要",
            "去",
            "你",
            "会",
            "着",
            "没有",
            "看",
            "好",
            "自己",
            "这",
        }
        keywords = [w for w in words if len(w) > 1 and w not in stop_words]
        return keywords[:10]

    def _decompose_query(self, query: str) -> Tuple[List[SubQuery], Dict]:
        """
        问题分解

        将复合问题分解为多个子问题，构建DAG执行计划

        Returns:
            (子查询列表, 执行计划)
        """
        sub_queries = []

        # 基于标点分割句子
        sentences = re.split(r"[。！？；\n]", query)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 基于连接词识别子问题
        if len(sentences) == 1:
            # 尝试基于连接词分割
            sub_parts = self._split_by_connectors(query)
        else:
            sub_parts = sentences

        # 构建子查询
        for i, part in enumerate(sub_parts):
            sub_q = SubQuery(
                id=f"sub_{i}", text=part, dependencies=[], query_type="independent"
            )
            sub_queries.append(sub_q)

        # 识别依赖关系
        for i, sq in enumerate(sub_queries):
            # 检查是否依赖前面子问题的答案
            dep_keywords = ["上述", "前面提到", "之前", "上面"]
            for keyword in dep_keywords:
                if keyword in sq.text:
                    # 依赖所有前面的子查询
                    sq.dependencies = [f"sub_{j}" for j in range(i)]
                    sq.query_type = "dependent"
                    break

        # 构建执行计划 (DAG)
        execution_plan = self._build_execution_plan(sub_queries)

        return sub_queries, execution_plan

    def _split_by_connectors(self, query: str) -> List[str]:
        """基于连接词分割查询"""
        connectors = ["并且", "而且", "同时", "另外", "还有", "以及"]

        parts = [query]
        for connector in connectors:
            new_parts = []
            for part in parts:
                if connector in part:
                    new_parts.extend(part.split(connector))
                else:
                    new_parts.append(part)
            parts = new_parts

        return [p.strip() for p in parts if p.strip()]

    def _build_execution_plan(self, sub_queries: List[SubQuery]) -> Dict:
        """构建执行计划 (DAG)"""
        # 拓扑排序
        in_degree = {sq.id: 0 for sq in sub_queries}
        adj_list = {sq.id: [] for sq in sub_queries}

        for sq in sub_queries:
            for dep in sq.dependencies:
                adj_list[dep].append(sq.id)
                in_degree[sq.id] += 1

        # Kahn算法
        stages = []
        current_stage = [sq_id for sq_id, degree in in_degree.items() if degree == 0]

        while current_stage:
            stages.append(current_stage)
            next_stage = []

            for sq_id in current_stage:
                for neighbor in adj_list[sq_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_stage.append(neighbor)

            current_stage = next_stage

        return {
            "type": "dag",
            "stages": stages,
            "total_subqueries": len(sub_queries),
            "parallel_groups": len(stages),
        }

    def _handle_comparative(self, query: str) -> List[SubQuery]:
        """处理对比型查询"""
        entities = self._extract_comparative_entities(query)

        if len(entities) >= 2:
            sub_queries = []
            for i, entity in enumerate(entities[:2]):
                sq = SubQuery(
                    id=f"comp_{i}",
                    text=f"{entity}的相关信息",
                    dependencies=[],
                    query_type="independent",
                    expected_answer_type="entity_info",
                )
                sub_queries.append(sq)

            # 添加对比分析子查询
            analysis_sq = SubQuery(
                id="comp_analysis",
                text=f"对比分析{'和'.join(entities[:2])}的异同",
                dependencies=[f"comp_{i}" for i in range(len(entities[:2]))],
                query_type="dependent",
                expected_answer_type="comparison",
            )
            sub_queries.append(analysis_sq)

            return sub_queries

        return []

    def _extract_comparative_entities(self, query: str) -> List[str]:
        """提取对比实体"""
        patterns = [
            r"([\w\u4e00-\u9fa5]+)[和与跟].*?([\w\u4e00-\u9fa5]+).*?[区别差异不同对比]",
            r"([\w\u4e00-\u9fa5]+)\s*vs\s*([\w\u4e00-\u9fa5]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return list(match.groups())

        return []


# 单例
query_enhancer = QueryEnhancer()

__all__ = [
    "QueryEnhancer",
    "EnhancedQuery",
    "SubQuery",
    "HypothesisDoc",
    "query_enhancer",
]
