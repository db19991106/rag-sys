#!/usr/bin/env python3
"""
RAG系统测试报告生成器
生成可视化测试报告
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))


class TestReportGenerator:
    """测试报告生成器"""

    def __init__(self, results_data: Dict = None):
        self.results = results_data or {}
        self.report_lines = []

    def load_from_file(self, filepath: str):
        """从文件加载测试结果"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.results = data.get("results", {})
        return self

    def generate_console_report(self) -> str:
        """生成控制台报告"""
        lines = []

        # 标题
        lines.append("=" * 80)
        lines.append("RAG系统测试报告".center(80))
        lines.append("=" * 80)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 功能测试结果
        if "document_parser" in self.results:
            lines.append("📄 文档解析测试")
            lines.append("-" * 80)
            result = self.results["document_parser"]
            lines.append(f"  成功率: {result.get('success_rate', 0) * 100:.1f}%")
            if "details" in result:
                for detail in result["details"]:
                    status = "✅" if detail.get("success") else "❌"
                    lines.append(
                        f"  {status} {detail.get('format', 'unknown')}: "
                        f"{detail.get('content_length', 0)}字符"
                    )
            lines.append("")

        # 切分测试
        if "chunking" in self.results:
            lines.append("✂️  文档切分测试")
            lines.append("-" * 80)
            result = self.results["chunking"]
            if "details" in result:
                for detail in result["details"]:
                    status = "✅" if detail.get("success") else "❌"
                    lines.append(
                        f"  {status} {detail.get('strategy', 'unknown')}: "
                        f"{detail.get('chunk_count', 0)}个chunks"
                    )
            lines.append("")

        # 嵌入测试
        if "embedding" in self.results:
            lines.append("🔢 嵌入服务测试")
            lines.append("-" * 80)
            result = self.results["embedding"]
            if result.get("success"):
                lines.append(f"  ✅ 编码成功")
                lines.append(f"  向量维度: {result.get('dimension', 0)}")
                lines.append(f"  编码耗时: {result.get('encode_time', 0) * 1000:.2f}ms")
                lines.append(
                    f"  平均/文本: {result.get('avg_time_per_text', 0) * 1000:.2f}ms"
                )
            else:
                lines.append(f"  ❌ 编码失败: {result.get('error', 'unknown')}")
            lines.append("")

        # 向量数据库测试
        if "vector_db" in self.results:
            lines.append("💾 向量数据库测试")
            lines.append("-" * 80)
            result = self.results["vector_db"]
            if result.get("success"):
                lines.append(f"  ✅ 测试通过")
                lines.append(f"  向量总数: {result.get('total_vectors', 0)}")
                lines.append(f"  搜索结果数: {result.get('search_results_count', 0)}")
            else:
                lines.append(f"  ❌ 测试失败: {result.get('error', 'unknown')}")
            lines.append("")

        # 检索质量测试
        if "retrieval_quality" in self.results:
            lines.append("🔍 检索质量测试")
            lines.append("-" * 80)
            result = self.results["retrieval_quality"]
            hit_rate = result.get("avg_hit_rate", 0)
            lines.append(f"  平均关键词命中率: {hit_rate * 100:.1f}%")

            if hit_rate >= 0.7:
                lines.append(f"  评级: 🟢 优秀")
            elif hit_rate >= 0.5:
                lines.append(f"  评级: 🟡 良好")
            else:
                lines.append(f"  评级: 🔴 需优化")

            if "details" in result:
                lines.append("  详细结果:")
                for detail in result["details"]:
                    if "hit_rate" in detail:
                        lines.append(
                            f"    - {detail['query'][:40]}... "
                            f"命中率: {detail['hit_rate'] * 100:.0f}%"
                        )
            lines.append("")

        # 端到端测试
        if "end_to_end" in self.results:
            lines.append("🎯 端到端测试")
            lines.append("-" * 80)
            result = self.results["end_to_end"]
            success_rate = result.get("success_rate", 0)
            lines.append(f"  成功率: {success_rate * 100:.1f}%")

            if "details" in result:
                total_time = sum(
                    d.get("total_time", 0)
                    for d in result["details"]
                    if "total_time" in d
                )
                avg_time = (
                    total_time / len(result["details"]) if result["details"] else 0
                )
                lines.append(f"  平均响应时间: {avg_time * 1000:.0f}ms")

                for detail in result["details"]:
                    status = "✅" if detail.get("success") else "❌"
                    lines.append(f"  {status} {detail.get('query', 'unknown')[:40]}...")
            lines.append("")

        # 性能测试
        if "retrieval_performance" in self.results:
            lines.append("⚡ 检索性能测试")
            lines.append("-" * 80)
            result = self.results["retrieval_performance"]
            lines.append(f"  平均响应时间: {result.get('avg_time_ms', 0):.2f}ms")
            lines.append(f"  P95响应时间: {result.get('p95_time_ms', 0):.2f}ms")
            lines.append(f"  P99响应时间: {result.get('p99_time_ms', 0):.2f}ms")
            lines.append(f"  吞吐量: {result.get('throughput_qps', 0):.1f} QPS")

            avg_time = result.get("avg_time_ms", 0)
            if avg_time < 200:
                lines.append(f"  评级: 🟢 优秀")
            elif avg_time < 500:
                lines.append(f"  评级: 🟡 良好")
            else:
                lines.append(f"  评级: 🔴 需优化")
            lines.append("")

        # 并发测试
        if "concurrent_performance" in self.results:
            lines.append("👥 并发性能测试")
            lines.append("-" * 80)
            result = self.results["concurrent_performance"]
            lines.append(f"  并发用户数: {result.get('concurrent_users', 0)}")
            lines.append(f"  总请求数: {result.get('total_requests', 0)}")
            lines.append(f"  吞吐量: {result.get('throughput_qps', 0):.1f} QPS")
            lines.append(
                f"  平均响应时间: {result.get('avg_response_time_ms', 0):.2f}ms"
            )
            lines.append("")

        # 总结
        lines.append("=" * 80)
        lines.append("测试总结".center(80))
        lines.append("=" * 80)

        total_tests = len(self.results)
        passed_tests = sum(
            1
            for r in self.results.values()
            if isinstance(r, dict)
            and (r.get("success") or r.get("success_rate", 0) > 0.5)
        )

        lines.append(f"总测试项: {total_tests}")
        lines.append(f"通过: {passed_tests}")
        lines.append(f"失败: {total_tests - passed_tests}")
        lines.append(
            f"通过率: {passed_tests / total_tests * 100:.1f}%"
            if total_tests > 0
            else "通过率: N/A"
        )
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_html_report(self, output_file: str = "test_report.html"):
        """生成HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RAG系统测试报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header .timestamp {{
            opacity: 0.9;
            margin-top: 10px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metric {{
            display: inline-block;
            background: #f0f0f0;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px;
            font-size: 14px;
        }}
        .metric-label {{
            color: #666;
            font-size: 12px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .status-good {{
            color: #22c55e;
        }}
        .status-warning {{
            color: #f59e0b;
        }}
        .status-bad {{
            color: #ef4444;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .summary .metric {{
            background: rgba(255,255,255,0.2);
            color: white;
        }}
        .summary .metric-label {{
            color: rgba(255,255,255,0.8);
        }}
        .summary .metric-value {{
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 RAG系统测试报告</h1>
        <div class="timestamp">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>
"""

        # 添加功能测试卡片
        if "document_parser" in self.results:
            result = self.results["document_parser"]
            success_rate = result.get("success_rate", 0) * 100
            status_class = (
                "status-good"
                if success_rate >= 80
                else "status-warning"
                if success_rate >= 50
                else "status-bad"
            )

            html_content += f"""
    <div class="card">
        <h2>📄 文档解析测试</h2>
        <div class="metric">
            <div class="metric-label">成功率</div>
            <div class="metric-value {status_class}">{success_rate:.1f}%</div>
        </div>
        <table>
            <tr>
                <th>格式</th>
                <th>状态</th>
                <th>内容长度</th>
            </tr>
"""
            for detail in result.get("details", []):
                status = "✅ 成功" if detail.get("success") else "❌ 失败"
                html_content += f"""
            <tr>
                <td>{detail.get("format", "unknown").upper()}</td>
                <td>{status}</td>
                <td>{detail.get("content_length", 0)} 字符</td>
            </tr>
"""
            html_content += "</table></div>"

        # 添加性能测试卡片
        if "retrieval_performance" in self.results:
            result = self.results["retrieval_performance"]
            avg_time = result.get("avg_time_ms", 0)
            status_class = (
                "status-good"
                if avg_time < 200
                else "status-warning"
                if avg_time < 500
                else "status-bad"
            )

            html_content += f"""
    <div class="card">
        <h2>⚡ 检索性能测试</h2>
        <div class="metric">
            <div class="metric-label">平均响应时间</div>
            <div class="metric-value {status_class}">{avg_time:.2f}ms</div>
        </div>
        <div class="metric">
            <div class="metric-label">P95响应时间</div>
            <div class="metric-value">{result.get("p95_time_ms", 0):.2f}ms</div>
        </div>
        <div class="metric">
            <div class="metric-label">P99响应时间</div>
            <div class="metric-value">{result.get("p99_time_ms", 0):.2f}ms</div>
        </div>
        <div class="metric">
            <div class="metric-label">吞吐量</div>
            <div class="metric-value">{result.get("throughput_qps", 0):.1f} QPS</div>
        </div>
    </div>
"""

        # 添加总结卡片
        total_tests = len(self.results)
        passed_tests = sum(
            1
            for r in self.results.values()
            if isinstance(r, dict)
            and (r.get("success") or r.get("success_rate", 0) > 0.5)
        )
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        html_content += f"""
    <div class="card summary">
        <h2 style="border-bottom-color: rgba(255,255,255,0.3);">📊 测试总结</h2>
        <div class="metric">
            <div class="metric-label">总测试项</div>
            <div class="metric-value">{total_tests}</div>
        </div>
        <div class="metric">
            <div class="metric-label">通过</div>
            <div class="metric-value" style="color: #86efac;">{passed_tests}</div>
        </div>
        <div class="metric">
            <div class="metric-label">失败</div>
            <div class="metric-value" style="color: #fca5a5;">{total_tests - passed_tests}</div>
        </div>
        <div class="metric">
            <div class="metric-label">通过率</div>
            <div class="metric-value">{pass_rate:.1f}%</div>
        </div>
    </div>
</body>
</html>
"""

        # 保存HTML文件
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTML报告已生成: {output_path.absolute()}")
        return str(output_path)

    def print_summary(self):
        """打印简要总结"""
        total_tests = len(self.results)
        passed_tests = sum(
            1
            for r in self.results.values()
            if isinstance(r, dict)
            and (r.get("success") or r.get("success_rate", 0) > 0.5)
        )

        print("\n" + "=" * 60)
        print("🧪 RAG系统测试总结")
        print("=" * 60)
        print(f"总测试项: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {total_tests - passed_tests} ❌")
        print(
            f"通过率: {passed_tests / total_tests * 100:.1f}%"
            if total_tests > 0
            else "通过率: N/A"
        )
        print("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG系统测试报告生成器")
    parser.add_argument("--input", "-i", help="测试结果JSON文件路径")
    parser.add_argument(
        "--output", "-o", default="test_report.html", help="HTML报告输出路径"
    )
    parser.add_argument("--console", "-c", action="store_true", help="仅输出控制台报告")
    args = parser.parse_args()

    generator = TestReportGenerator()

    if args.input:
        generator.load_from_file(args.input)
    else:
        # 尝试加载最新的测试结果
        test_data_dir = Path(__file__).parent / "test_data"
        result_files = sorted(test_data_dir.glob("test_report_*.json"))
        if result_files:
            latest_file = result_files[-1]
            print(f"📂 加载测试结果: {latest_file}")
            generator.load_from_file(str(latest_file))
        else:
            print("⚠️ 未找到测试结果文件")
            return

    if args.console:
        print(generator.generate_console_report())
    else:
        print(generator.generate_console_report())
        generator.generate_html_report(args.output)

    generator.print_summary()


if __name__ == "__main__":
    main()
