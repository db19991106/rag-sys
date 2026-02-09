#!/usr/bin/env python3
"""
优化版批量测试 - 改进评估标准
"""

import json
import sys
from pathlib import Path

# 同义词映射表
SYNONYMS = {
    "8-9级": ["8-9级", "普通员工", "员工", "软件研发", "机械研发", "实施工程师"],
    "10-11级": ["10-11级", "经理", "经理级"],
    "12级": ["12级", "总监", "总监级", "专家"],
    "上海": ["上海", "一线城市", "北上广深"],
    "深圳": ["深圳", "一线城市"],
    "北京": ["北京", "一线城市"],
    "广州": ["广州", "一线城市"],
    "成都": ["成都", "新一线", "省会城市"],
    "500": ["500", "500元"],
    "300": ["300", "300元"],
    "180": ["180", "180元"],
    "经济舱": ["经济舱", "二等座"],
    "一等座": ["一等座", "高铁一等座"],
    "商务舱": ["商务舱", "头等舱"],
}

def check_keywords_with_synonyms(text, expected_keywords):
    """检查关键词，考虑同义词"""
    if not expected_keywords:
        return 1.0, 0, 0
    
    hit = 0
    total = len(expected_keywords)
    
    for keyword in expected_keywords:
        # 获取同义词列表
        synonyms = SYNONYMS.get(keyword, [keyword])
        
        # 检查是否命中任意同义词
        if any(syn in text for syn in synonyms):
            hit += 1
    
    return hit / total, hit, total

# 使用示例
if __name__ == "__main__":
    # 测试同义词匹配
    test_cases = [
        ("普通员工在上海出差", ["8-9级", "上海"], 1.0),
        ("经理级去深圳", ["10-11级", "深圳"], 1.0),
    ]
    
    for text, keywords, expected in test_cases:
        rate, hit, total = check_keywords_with_synonyms(text, keywords)
        print(f"文本: {text}")
        print(f"关键词: {keywords}")
        print(f"命中率: {hit}/{total} = {rate*100:.0f}%")
        print()
