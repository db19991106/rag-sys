"""
修复IntentConfig相似度阈值 - 降低以获得更多检索结果
"""

import re

# 读取文件
with open("/root/autodl-tmp/rag/backend/services/intent_recognizer.py", "r") as f:
    content = f.read()

# 修复所有意图的similarity_threshold为0.4
old_thresholds = [
    ("IntentType.QUESTION", "0.7"),
    ("IntentType.SEARCH", "0.6"),
    ("IntentType.SUMMARY", "0.5"),
    ("IntentType.COMPARISON", "0.65"),
    ("IntentType.PROCEDURE", "0.7"),
    ("IntentType.DEFINITION", "0.7"),
    ("IntentType.GREETING", "0.6"),
    ("IntentType.OTHER", "0.7"),
]

for intent_type, old_val in old_thresholds:
    pattern = f'({intent_type}): {{\n            "top_k": (\d+),\n            "similarity_threshold": {old_val},'
    replacement = (
        f'\1: {{\n            "top_k": \2,\n            "similarity_threshold": 0.4,'
    )
    content = re.sub(pattern, replacement, content)

# 写回文件
with open("/root/autodl-tmp/rag/backend/services/intent_recognizer.py", "w") as f:
    f.write(content)

print("✅ 已修复所有意图的similarity_threshold为0.4")
print("\n修改后的配置:")
import re

matches = re.findall(
    r'IntentType\.\w+: \{\n            "top_k": (\d+),\n            "similarity_threshold": ([\d.]+),',
    content,
)
for i, (topk, threshold) in enumerate(matches, 1):
    intent_name = [
        "QUESTION",
        "SEARCH",
        "SUMMARY",
        "COMPARISON",
        "PROCEDURE",
        "DEFINITION",
        "GREETING",
        "OTHER",
    ][i - 1]
    print(f"   {intent_name}: top_k={topk}, threshold={threshold}")
