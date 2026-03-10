"""
意图识别服务 - 纯 LLM 模式

意图类型：
- HR: 人力资源相关事务
- FINANCE: 财务管理相关事务
- ADMIN: 行政制度相关事务
- COMPLIANCE: 合规安全相关事务
- PROCESS: 流程管理相关事务
- TECH_REPORT: 专业技术相关事务
- CASUAL_CHAT: 闲聊（不属于以上任何类别）
"""

from typing import Dict, List, Tuple, Optional
import json
from utils.logger import logger
from models import IntentType


# 意图到文档的映射
INTENT_DOCUMENT_MAPPING = {
    IntentType.HR: [
        "员工管理制度.md",
        "公司员工手册.md",
        "人力资源部绩效考核方案.md",
        "研发部门绩效考核制度.md",
        "生产部绩效考核方案.md",
        "行政部绩效考核方案.md",
        "销售部绩效考核方案.md",
        "绩效考核管理制度总则.md",
        "发展晋升管理制度.md",
        "考勤休假管理制度.md",
    ],
    IntentType.FINANCE: [
        "薪酬福利管理制度.md",
        "财务报销标准.md",
        "城市分类地区报销差异.md",
        "酒店级别标准.md",
    ],
    IntentType.ADMIN: [
        "办公室管理制度.md",
        "企业管理制度汇编索引.md",
    ],
    IntentType.COMPLIANCE: [
        "合规安全管理制度.md",
    ],
    IntentType.PROCESS: [
        "企业管理制度流程规范.md",
        "企业管理制度汇编索引.md",
    ],
    IntentType.TECH_REPORT: [],  # 技术报告类文档
    IntentType.CASUAL_CHAT: [],  # 闲聊不对应任何文档
}


class IntentConfig:
    """意图对应的检索配置"""

    CONFIGS = {
        IntentType.HR: {
            "top_k": 6,
            "similarity_threshold": 0.25,
            "description": "人力资源咨询",
            "prompt_template": "根据公司人力资源管理制度回答：",
            "use_knowledge_base": True,
        },
        IntentType.FINANCE: {
            "top_k": 5,
            "similarity_threshold": 0.25,
            "description": "财务管理咨询",
            "prompt_template": "根据公司财务制度回答：",
            "use_knowledge_base": True,
        },
        IntentType.ADMIN: {
            "top_k": 5,
            "similarity_threshold": 0.25,
            "description": "行政制度咨询",
            "prompt_template": "根据公司行政制度回答：",
            "use_knowledge_base": True,
        },
        IntentType.COMPLIANCE: {
            "top_k": 5,
            "similarity_threshold": 0.25,
            "description": "合规安全咨询",
            "prompt_template": "根据公司合规安全制度回答：",
            "use_knowledge_base": True,
        },
        IntentType.PROCESS: {
            "top_k": 8,
            "similarity_threshold": 0.2,
            "description": "流程管理咨询",
            "prompt_template": "根据公司制度流程回答：",
            "use_knowledge_base": True,
        },
        IntentType.TECH_REPORT: {
            "top_k": 6,
            "similarity_threshold": 0.2,
            "description": "技术报告咨询",
            "prompt_template": "根据技术文档回答：",
            "use_knowledge_base": True,
        },
        IntentType.CASUAL_CHAT: {
            "top_k": 0,  # 不检索知识库
            "similarity_threshold": 0.0,
            "description": "闲聊",
            "prompt_template": "请友好地回答用户的问题：",
            "use_knowledge_base": False,  # 不使用知识库
        },
    }

    @classmethod
    def get_config(cls, intent: IntentType) -> Dict:
        """获取意图对应的配置"""
        return cls.CONFIGS.get(intent, cls.CONFIGS[IntentType.CASUAL_CHAT])

    @classmethod
    def should_use_knowledge_base(cls, intent: IntentType) -> bool:
        """判断是否应该使用知识库"""
        config = cls.get_config(intent)
        return config.get("use_knowledge_base", True)

    @classmethod
    def get_target_documents(cls, intent: IntentType) -> List[str]:
        """获取意图对应的目标文档列表"""
        return INTENT_DOCUMENT_MAPPING.get(intent, [])


class IntentRecognizer:
    """
    意图识别器 - 纯 LLM 模式

    直接使用 LLM 进行意图识别，无需规则匹配
    """

    def __init__(self):
        self._initialized = False
        self._config = None

    def initialize_with_config(self, config):
        """使用配置初始化"""
        self._config = config
        self._initialized = True
        logger.info("意图识别器已初始化（纯 LLM 模式）")

    def recognize(self, query: str) -> Tuple[IntentType, float, Dict]:
        """
        识别用户查询的意图

        Args:
            query: 用户查询文本

        Returns:
            Tuple of (意图类型, 置信度, 详细信息)
        """
        # 直接使用 LLM 进行意图识别
        try:
            return self._llm_based_recognize(query)
        except Exception as e:
            logger.error(f"LLM 意图识别失败: {e}")
            # Fallback: 返回 tech_report，使用通用知识库检索
            return IntentType.TECH_REPORT, 0.5, {"method": "fallback", "error": str(e)}

    def _llm_based_recognize(self, query: str) -> Tuple[IntentType, float, Dict]:
        """
        基于LLM的意图识别（使用 7B 模型）
        """
        from openai import OpenAI

        # 构建提示词 - 精准分析与判定
        prompt = f'''对用户提交的查询进行意图类别的精准分析与判定。

【类别定义】

- hr: 人力资源相关事务（招聘、员工关系、薪酬福利、绩效考核、培训发展等）
- finance: 财务管理相关事务（报销、预算、费用核算、财务报表、税务等）
- admin: 行政制度相关事务（办公环境、行政流程、办公用品、会务、差旅等）
- compliance: 合规安全相关事务（法律法规、公司政策、信息安全、数据保护、风险管控等）
- process: 流程管理相关事务（业务流程优化、规范制定、执行监督、效率提升等）
- tech_report: 专业技术相关事务（各行业的技术原理、专业设备操作、技术规范、工艺流程、故障排查、技术概念等）
- casual_chat: 闲聊，完全不符合以上任何专业类别

【判断原则】

1. **主题优先**：根据问题**最核心**主题判断所属领域；若涉及多领域，选择**最直接相关的单一类别**
2. **技术判定**：涉及专业技术原理、专业设备/系统操作、技术规范、故障排查、技术概念的，归为 tech_report（跨行业适用）
3. **职能归属**：涉及人力资源、财务管理、行政制度、合规安全、流程管理的制度、政策、管理方法，分别归属对应类别
4. **闲聊**：仅当查询不属于以上任何一类时，归类为 casual_chat
5. **置信度标准**：confidence 取值 0.0-1.0，明确归属时≥0.8，边界模糊时 0.6-0.8，难以判断时≤0.6

【输出格式】

返回JSON，字段要求：

- intent: string，枚举值 [hr, finance, admin, compliance, process, tech_report, casual_chat]
- confidence: number，范围 0.0-1.0，保留两位小数
- reason: string，长度 10-40字，简明说明核心判断依据

示例：

- 高置信度：{{"intent": "finance", "confidence": 0.95, "reason": "核心主题为差旅费用报销标准"}}
- 中置信度：{{"intent": "tech_report", "confidence": 0.75, "reason": "涉及系统操作但可能是流程问题"}}
- 低置信度：{{"intent": "casual_chat", "confidence": 0.55, "reason": "内容模糊难以判断专业类别"}}

用户查询: "{query}"

请返回JSON结果：'''

        try:
            from config import settings

            # 使用意图识别专用 vLLM 服务 (0.5B 模型，端口 8002)
            if settings.intent_vllm_enabled:
                client = OpenAI(
                    api_key="EMPTY",
                    base_url=f"http://{settings.intent_vllm_host}:{settings.intent_vllm_port}/v1",
                )
                
                response = client.chat.completions.create(
                    model=settings.intent_vllm_model_path,  # Qwen2.5-0.5B-Instruct
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                )
                
                response_text = response.choices[0].message.content
            else:
                # 本地模型回退
                raise RuntimeError("意图识别 vLLM 服务 (8002) 未启用，无法进行意图识别")

            logger.info(f"意图识别 LLM 原始响应: {response_text}")

            # 预处理：去除 markdown 代码块包装
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # 去除开头的 ```json 或 ```
                import re
                response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
                # 去除结尾的 ```
                response_text = re.sub(r'\s*```$', '', response_text)
                logger.debug(f"去除 markdown 包装后: {response_text}")

            # 尝试解析JSON
            result = json.loads(response_text.strip())

            intent_str = result.get("intent", "tech_report")
            confidence = result.get("confidence", 0.5)
            reason = result.get("reason", "")

            # 转换字符串为IntentType
            intent_map = {
                "hr": IntentType.HR,
                "finance": IntentType.FINANCE,
                "admin": IntentType.ADMIN,
                "compliance": IntentType.COMPLIANCE,
                "process": IntentType.PROCESS,
                "tech_report": IntentType.TECH_REPORT,
                "casual_chat": IntentType.CASUAL_CHAT,
            }
            intent = intent_map.get(intent_str, IntentType.TECH_REPORT)

            logger.info(f"意图识别结果: intent={intent.value}, confidence={confidence}, reason={reason}")

            return (
                intent,
                confidence,
                {"method": "llm", "reason": reason, "raw_response": response_text},
            )

        except json.JSONDecodeError as e:
            logger.error(f"LLM意图识别JSON解析失败: {e}, response: {response_text}")
            return IntentType.TECH_REPORT, 0.5, {"method": "llm", "error": str(e), "raw_response": response_text}
        except Exception as e:
            logger.error(f"LLM意图识别失败: {e}")
            return IntentType.TECH_REPORT, 0.5, {"method": "llm", "error": str(e)}

    def recognize_intent(self, query: str) -> Dict:
        """
        识别意图（兼容旧接口）

        Args:
            query: 用户查询文本

        Returns:
            Dict: 包含intent、confidence和details的字典
        """
        intent, confidence, details = self.recognize(query)
        return {"intent": intent.value, "confidence": confidence, "details": details}


# 全局意图识别器实例
intent_recognizer = IntentRecognizer()