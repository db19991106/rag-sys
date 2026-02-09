from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from services.conversation_manager import LLMClient
from utils.logger import logger
import re
import json

router = APIRouter(prefix="/summary", tags=["摘要生成"])

class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待摘要文本")

class SummaryResponse(BaseModel):
    summary: str = Field(..., min_length=2, max_length=15, description="生成的摘要标题")

# 加载制度文件作为上下文（实际部署时建议预加载到内存）
REGULATION_CONTEXT = """
财务报销管理制度要点：
- 职级：总监及以上、经理级、主管及以下
- 费用类型：差旅费（交通、住宿、补贴）、业务招待费（人均300-500元）、通讯费（150-800元/月）、办公费、培训会议费
- 关键数字：事前审批>5000元，招待费>2000元需申请，报销时限30天，跨季度原则上不报销
- 审批流：直属经理→部门总监→财务→CEO（按金额分级）
"""

# 创建专门用于摘要生成的LLM客户端
summary_llm_client = LLMClient(model_path="./data/models/Qwen2.5-7B-Instruct")

def extract_entities(text: str) -> dict:
    """
    提取关键实体：职级、费用类型、动作意图
    """
    text_lower = text.lower()
    
    # 职级识别
    ranks = {
        "总监": ["总监", "director", "chief"],
        "经理": ["经理", "manager", "主管"], 
        "员工": ["员工", "职员", "staff", "我"]
    }
    detected_rank = None
    for rank, keywords in ranks.items():
        if any(kw in text for kw in keywords):
            detected_rank = rank
            break
    
    # 费用类型识别
    expense_types = {
        "差旅费": ["差旅", "出差", "机票", "酒店", "住宿", "火车", "交通"],
        "招待费": ["招待", "宴请", "客户", "吃饭", "礼品", "吃饭"],
        "通讯费": ["通讯", "电话", "手机", "话费", "流量"],
        "报销额度": ["报销", "报多少", "额度", "标准", "限额", "能报"]
    }
    detected_type = None
    for exp_type, keywords in expense_types.items():
        if any(kw in text for kw in keywords):
            detected_type = exp_type
            break
    
    # 意图识别
    intentions = {
        "查询标准": ["多少", "标准", "额度", "限额", "能报", "可以报", "怎么算"],
        "申请流程": ["怎么申请", "流程", "审批", "需要谁批", "找谁"],
        "材料要求": ["需要什么", "材料", "发票", "单据", "凭证"],
        "违规处理": ["违规", "处罚", "罚款", "假发票"]
    }
    detected_intent = None
    for intent, keywords in intentions.items():
        if any(kw in text for kw in keywords):
            detected_intent = intent
            break
    
    return {
        "rank": detected_rank,
        "expense_type": detected_type,
        "intention": detected_intent
    }

def smart_truncate(text: str, max_len: int = 15) -> str:
    """
    智能截断：优先保留完整词语，避免截断在词中间
    """
    if len(text) <= max_len:
        return text
    
    # 在标点或空格处截断
    for i in range(max_len, max_len//2, -1):
        if text[i] in "，。、；：！？ ,.;:!?":
            return text[:i]
    
    # 在中英文边界截断
    for i in range(max_len, 2, -1):
        # 避免"英文单词被截断"或"中文词被截断"
        if re.search(r'[a-zA-Z]$', text[:i]) and re.search(r'^[^a-zA-Z]', text[i:]):
            # 回退到前一个空格或中文
            for j in range(i-1, 0, -1):
                if text[j] == ' ' or re.search(r'[\u4e00-\u9fff]', text[j]):
                    return text[:j+1]
        elif re.search(r'[\u4e00-\u9fff]$', text[:i]) and re.search(r'^[a-zA-Z]', text[i:]):
            return text[:i]
    
    return text[:max_len]

def build_prompt(text: str, entities: dict) -> str:
    """
    根据提取的实体动态构建提示词，提升针对性
    """
    # 基础指令
    base_prompt = f"""你是企业财务系统的对话标题生成专家。请根据用户的财务报销咨询，生成5-12字的精准标题。

【核心要求】
1. 标题必须包含：职级（如有）+ 费用类型 + 意图
2. 禁止出现："你好""请问""咨询""问题"等客套词
3. 禁止出现：标点符号、引号、"关于""如何"等前缀
4. 直接输出标题，不要解释

【制度背景】
{REGULATION_CONTEXT}

【用户输入】
{text}

【提取信息】
- 涉及职级：{entities['rank'] or '未明确'}
- 费用类型：{entities['expense_type'] or '未明确'}
- 用户意图：{entities['intention'] or '查询标准'}

【参考示例】
输入："总监出差能住多少钱的酒店？" → 总监差旅住宿标准
输入："经理级招待客户吃饭能报多少" → 经理招待费标准
输入："我出差坐飞机能选商务舱吗" → 员工差旅交通标准
输入："报销需要哪些材料" → 报销材料清单
输入："假发票会被发现吗" → 违规处理规定

【输出标题】"""
    
    return base_prompt

def post_process(raw_summary: str, entities: dict, original_text: str) -> str:
    """
    后处理：清理格式 + 实体校验 + 长度控制
    """
    summary = raw_summary.strip()
    
    # 1. 去除常见前缀和无关内容
    prefixes = ["标题：", "摘要：", "输出：", "结果：", "Title:", "Summary:", "→", "- ", "1.", "1、"]
    for prefix in prefixes:
        if summary.startswith(prefix):
            summary = summary[len(prefix):].strip()
    
    # 去除引号和括号
    summary = summary.strip('"\'"\'「」[]【】()（）')
    
    # 2. 实体补全：如果模型漏了关键实体，尝试补充
    if entities['rank'] and entities['rank'] not in summary and len(summary) < 12:
        summary = entities['rank'] + summary
    
    if entities['expense_type'] and entities['expense_type'] not in summary and len(summary) < 12:
        if "标准" not in summary and "额度" not in summary:
            summary = summary + "标准"
    
    # 3. 关键词替换：统一术语
    replacements = {
        "飞机": "差旅",
        "酒店": "住宿",
        "吃饭": "招待",
        "电话费": "通讯费",
        "能报多少": "标准",
        "怎么申请": "申请流程",
        "需要什么": "材料清单"
    }
    for old, new in replacements.items():
        if old in summary and new not in summary and len(summary.replace(old, new)) <= 15:
            summary = summary.replace(old, new)
    
    # 4. 长度控制
    summary = smart_truncate(summary, 12)  # 留3字余量
    
    # 5. 兜底校验
    if len(summary) < 4 or summary in ["新对话", "财务报销"]:
        # 使用实体拼接兜底
        parts = [p for p in [entities['rank'], entities['expense_type'], "咨询"] if p]
        summary = "".join(parts) if parts else "报销标准咨询"
    
    return summary[:15]

@router.post("/generate", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """
    生成财务报销场景的对话标题
    """
    text = request.text.strip()
    
    # 前置过滤：无意义输入
    if len(text) < 3 or text in ["你好", "您好", "在吗", "在"]:
        return SummaryResponse(summary="新对话")
    
    # 提取实体
    entities = extract_entities(text)
    logger.info(f"实体提取: {entities}, 输入: {text[:30]}...")
    
    try:
        # 动态构建提示词
        prompt = build_prompt(text, entities)
        
        # 调用LLM，参数调优
        raw_output = summary_llm_client.chat(
            prompt, 
            temperature=0.05,  # 更低温度，减少随机性
            max_tokens=25,     # 限制输出长度
            stop=["\n", "。", "，", " "]  # 遇到这些字符停止生成
        ).strip()
        
        # 后处理
        summary = post_process(raw_output, entities, text)
        
        logger.info(f"生成结果: '{summary}' (原始: '{raw_output[:20]}...')")
        
        return SummaryResponse(summary=summary)
        
    except Exception as e:
        logger.error(f"LLM失败: {str(e)}")
        # 异常时实体兜底
        fallback = "".join([p for p in [entities['rank'], entities['expense_type'], "咨询"] if p]) or "报销咨询"
        return SummaryResponse(summary=fallback[:15])