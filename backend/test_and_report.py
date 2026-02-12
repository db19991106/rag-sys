#!/usr/bin/env python3
"""
实时测试并生成可视化报告
"""

import sys

sys.path.insert(0, "/root/autodl-tmp/rag/backend")

import json
import time
import asyncio
from pathlib import Path
from datetime import datetime

# 导入测试套件
from tests.core.rag_test_suite import RAGTestSuite

print("=" * 80)
print("🎯 RAG系统实时测试 + 报告生成")
print("=" * 80)
print()

# 运行完整测试套件
print("⏳ 正在运行完整测试套件（约需5-10分钟）...")
print()

test_suite = RAGTestSuite()
results = test_suite.run_all_tests(mode="full")

# 保存测试结果
results_dir = Path("/root/autodl-tmp/rag/backend/tests/data/reports")
results_dir.mkdir(parents=True, exist_ok=True)

timestamp = int(time.time())
results_file = results_dir / f"test_results_{timestamp}.json"

with open(results_file, "w", encoding="utf-8") as f:
    json.dump(
        {
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "results": results,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"✅ 测试结果已保存: {results_file}")
print()

# 生成报告
print("⏳ 正在生成可视化报告...")

from tests.core.test_report_generator import TestReportGenerator

generator = TestReportGenerator(results)
html_report = generator.generate_html_report()

# 保存HTML报告
report_file = Path("/root/autodl-tmp/rag/backend/evaluation_report.html")
with open(report_file, "w", encoding="utf-8") as f:
    f.write(html_report)

print(f"✅ 报告已生成: {report_file}")
print()

# 打印控制台摘要
console_report = generator.generate_console_report()
print("\n" + "=" * 80)
print("📊 测试摘要")
print("=" * 80)
print(console_report)

print("\n" + "=" * 80)
print("🎉 完成！")
print(f"📄 HTML报告: {report_file}")
print(f"📊 数据文件: {results_file}")
print("=" * 80)
