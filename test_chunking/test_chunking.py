#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG文档切分效果验证测试

测试维度：
1. 格式验证：检查输出JSON的字段完整性、字段类型正确性
2. 内容验证：核对chunk的content字段、tables字段
3. 语义验证：抽样验证avg_similarity是否符合规则，chunking_reason是否匹配
4. 唯一性验证：检查所有chunk_id是否全局唯一
5. 路径验证：检查输出文件是否保存在正确目录

使用方法：
    python test_chunking.py

输出结果：
    - 测试通过/失败状态
    - 详细的错误信息和统计
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    message: str
    details: List[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = []


class ChunkingValidator:
    """切分结果验证器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.all_chunks: List[Dict] = []
        self.all_files: List[Path] = []
        self.test_results: List[TestResult] = []
        
        # 必需字段定义
        self.required_fields = {
            "chunk_id": str,
            "chapter": str,
            "subtitle": str,
            "articles": list,
            "content": str,
            "tables": list,
            "metadata": dict
        }
        
        # metadata必需字段
        self.required_metadata_fields = {
            "char_count": int,
            "article_count": int,
            "avg_similarity": (int, float),
            "has_table": bool,
            "chunking_reason": str
        }
        
        # 加载所有数据
        self._load_all_chunks()
    
    def _load_all_chunks(self):
        """加载所有输出文件中的chunks"""
        if not self.output_dir.exists():
            return
        
        # 查找所有JSON文件
        json_files = list(self.output_dir.glob("*_chunks.json"))
        self.all_files = json_files
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.all_chunks.extend(data)
            except Exception as e:
                print(f"警告: 无法加载文件 {file_path}: {e}")
    
    # ==================== 测试方法 ====================
    
    def test_path_validation(self) -> TestResult:
        """测试5：路径验证 - 检查输出文件是否保存在正确目录"""
        test_name = "路径验证"
        
        if not self.output_dir.exists():
            return TestResult(
                name=test_name,
                passed=False,
                message=f"输出目录不存在: {self.output_dir}",
                details=[]
            )
        
        # 检查文件命名格式
        invalid_files = []
        for file_path in self.all_files:
            if not file_path.name.endswith("_chunks.json"):
                invalid_files.append(file_path.name)
        
        if invalid_files:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"发现 {len(invalid_files)} 个命名格式错误的文件",
                details=[f"  - {name}" for name in invalid_files[:5]]
            )
        
        if not self.all_files:
            return TestResult(
                name=test_name,
                passed=False,
                message="输出目录中未找到任何 *_chunks.json 文件",
                details=[]
            )
        
        return TestResult(
            name=test_name,
            passed=True,
            message=f"✓ 路径验证通过，发现 {len(self.all_files)} 个有效输出文件",
            details=[f"  - {f.name}" for f in self.all_files]
        )
    
    def test_format_validation(self) -> TestResult:
        """测试1：格式验证 - 检查字段完整性和类型正确性"""
        test_name = "格式验证"
        
        if not self.all_chunks:
            return TestResult(
                name=test_name,
                passed=False,
                message="没有可验证的chunks",
                details=[]
            )
        
        errors = []
        
        for i, chunk in enumerate(self.all_chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            
            # 检查必需字段
            for field, expected_type in self.required_fields.items():
                if field not in chunk:
                    errors.append(f"[{chunk_id}] 缺少必需字段: {field}")
                elif not isinstance(chunk[field], expected_type):
                    errors.append(
                        f"[{chunk_id}] 字段类型错误: {field} 应为 {expected_type.__name__}, "
                        f"实际为 {type(chunk[field]).__name__}"
                    )
            
            # 检查metadata字段
            if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                metadata = chunk["metadata"]
                for field, expected_type in self.required_metadata_fields.items():
                    if field not in metadata:
                        errors.append(f"[{chunk_id}.metadata] 缺少必需字段: {field}")
                    elif not isinstance(metadata[field], expected_type):
                        errors.append(
                            f"[{chunk_id}.metadata] 字段类型错误: {field} 应为 {expected_type}, "
                            f"实际为 {type(metadata[field]).__name__}"
                        )
            
            # 限制错误数量
            if len(errors) >= 20:
                errors.append("... (更多错误已省略)")
                break
        
        if errors:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"✗ 发现 {len(errors)} 个格式错误",
                details=errors[:15]
            )
        
        return TestResult(
            name=test_name,
            passed=True,
            message=f"✓ 格式验证通过，检查了 {len(self.all_chunks)} 个chunks的所有字段",
            details=[]
        )
    
    def test_content_validation(self) -> TestResult:
        """测试2：内容验证 - 核对content字段和tables字段"""
        test_name = "内容验证"
        
        if not self.all_chunks:
            return TestResult(
                name=test_name,
                passed=False,
                message="没有可验证的chunks",
                details=[]
            )
        
        errors = []
        warnings_list = []
        
        for i, chunk in enumerate(self.all_chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            content = chunk.get("content", "")
            tables = chunk.get("tables", [])
            metadata = chunk.get("metadata", {})
            has_table_flag = metadata.get("has_table", False)
            char_count = metadata.get("char_count", 0)
            
            # 检查content非空
            if not content or not content.strip():
                errors.append(f"[{chunk_id}] content字段为空")
            
            # 检查char_count与content长度一致
            actual_char_count = len(content)
            if abs(char_count - actual_char_count) > 10:  # 允许10字符误差
                errors.append(
                    f"[{chunk_id}] char_count不匹配: metadata显示 {char_count}, "
                    f"实际 {actual_char_count}"
                )
            
            # 检查tables字段
            if has_table_flag and not tables:
                errors.append(f"[{chunk_id}] has_table为true但tables为空")
            
            if tables and not has_table_flag:
                errors.append(f"[{chunk_id}] tables非空但has_table为false")
            
            # 验证表格结构
            for j, table in enumerate(tables):
                if not isinstance(table, dict):
                    errors.append(f"[{chunk_id}] tables[{j}] 不是字典类型")
                elif "rows" not in table:
                    errors.append(f"[{chunk_id}] tables[{j}] 缺少rows字段")
                elif not isinstance(table.get("rows"), list):
                    errors.append(f"[{chunk_id}] tables[{j}].rows 不是列表类型")
            
            # 警告：content过短
            if actual_char_count < 50:
                warnings_list.append(f"[{chunk_id}] content长度仅 {actual_char_count} 字符，可能过短")
            
            if len(errors) >= 15:
                errors.append("... (更多错误已省略)")
                break
        
        details = errors[:15]
        if warnings_list:
            details.extend(["", "警告:"] + warnings_list[:5])
        
        if errors:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"✗ 发现 {len(errors)} 个内容错误",
                details=details
            )
        
        # 统计信息
        total_tables = sum(len(c.get("tables", [])) for c in self.all_chunks)
        avg_content_len = sum(len(c.get("content", "")) for c in self.all_chunks) / len(self.all_chunks)
        
        return TestResult(
            name=test_name,
            passed=True,
            message=f"✓ 内容验证通过，平均内容长度 {avg_content_len:.0f} 字符，共 {total_tables} 个表格",
            details=[]
        )
    
    def test_semantic_validation(self) -> TestResult:
        """测试3：语义验证 - 检查avg_similarity和chunking_reason"""
        test_name = "语义验证"
        
        if not self.all_chunks:
            return TestResult(
                name=test_name,
                passed=False,
                message="没有可验证的chunks",
                details=[]
            )
        
        errors = []
        similarity_stats = []
        
        for i, chunk in enumerate(self.all_chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            metadata = chunk.get("metadata", {})
            avg_similarity = metadata.get("avg_similarity", 0)
            chunking_reason = metadata.get("chunking_reason", "")
            
            # 检查similarity范围
            if not (0 <= avg_similarity <= 1):
                errors.append(f"[{chunk_id}] avg_similarity {avg_similarity} 超出有效范围 [0, 1]")
            
            # 检查chunking_reason格式
            if not chunking_reason:
                errors.append(f"[{chunk_id}] chunking_reason为空")
            else:
                # 验证reason与similarity匹配（使用0.95作为阈值，与处理器保持一致）
                threshold = 0.95
                if avg_similarity >= threshold and "强语义关联" not in chunking_reason:
                    errors.append(
                        f"[{chunk_id}] chunking_reason不匹配: 相似度 {avg_similarity} ≥ {threshold} "
                        f"但reason不包含'强语义关联'"
                    )
                elif avg_similarity < threshold and "弱语义关联" not in chunking_reason:
                    errors.append(
                        f"[{chunk_id}] chunking_reason不匹配: 相似度 {avg_similarity} < {threshold} "
                        f"但reason不包含'弱语义关联'"
                    )
            
            similarity_stats.append(avg_similarity)
            
            if len(errors) >= 15:
                errors.append("... (更多错误已省略)")
                break
        
        # 统计
        if similarity_stats:
            avg_sim = sum(similarity_stats) / len(similarity_stats)
            high_sim_count = sum(1 for s in similarity_stats if s >= 0.8)
            low_sim_count = len(similarity_stats) - high_sim_count
        else:
            avg_sim = 0
            high_sim_count = low_sim_count = 0
        
        if errors:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"✗ 发现 {len(errors)} 个语义验证错误",
                details=errors[:15]
            )
        
        return TestResult(
            name=test_name,
            passed=True,
            message=f"✓ 语义验证通过，平均相似度 {avg_sim:.3f} "
                   f"(强关联: {high_sim_count}, 弱关联: {low_sim_count})",
            details=[]
        )
    
    def test_uniqueness_validation(self) -> TestResult:
        """测试4：唯一性验证 - 检查chunk_id是否全局唯一"""
        test_name = "唯一性验证"
        
        if not self.all_chunks:
            return TestResult(
                name=test_name,
                passed=False,
                message="没有可验证的chunks",
                details=[]
            )
        
        chunk_ids = []
        duplicate_ids = []
        
        for chunk in self.all_chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id:
                if chunk_id in chunk_ids:
                    duplicate_ids.append(chunk_id)
                chunk_ids.append(chunk_id)
        
        if duplicate_ids:
            unique_duplicates = list(set(duplicate_ids))
            return TestResult(
                name=test_name,
                passed=False,
                message=f"✗ 发现 {len(unique_duplicates)} 个重复的chunk_id",
                details=[f"  - {cid} (重复 {duplicate_ids.count(cid)} 次)" for cid in unique_duplicates[:10]]
            )
        
        # 检查chunk_id格式
        format_errors = []
        pattern = r"^CH\d+_\d{3}$"
        for chunk_id in chunk_ids:
            if not re.match(pattern, chunk_id):
                format_errors.append(chunk_id)
        
        if format_errors:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"✗ 发现 {len(format_errors)} 个格式错误的chunk_id",
                details=[f"  - {cid}" for cid in format_errors[:10]]
            )
        
        return TestResult(
            name=test_name,
            passed=True,
            message=f"✓ 唯一性验证通过，所有 {len(chunk_ids)} 个chunk_id均唯一且格式正确",
            details=[]
        )
    
    def test_articles_validation(self) -> TestResult:
        """额外测试：条款验证 - 检查articles字段与article_count的一致性"""
        test_name = "条款验证"
        
        if not self.all_chunks:
            return TestResult(
                name=test_name,
                passed=False,
                message="没有可验证的chunks",
                details=[]
            )
        
        errors = []
        
        for i, chunk in enumerate(self.all_chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            articles = chunk.get("articles", [])
            metadata = chunk.get("metadata", {})
            article_count = metadata.get("article_count", 0)
            
            if len(articles) != article_count:
                errors.append(
                    f"[{chunk_id}] article_count不匹配: metadata显示 {article_count}, "
                    f"实际articles数量 {len(articles)}"
                )
        
        if errors:
            return TestResult(
                name=test_name,
                passed=False,
                message=f"✗ 发现 {len(errors)} 个条款计数错误",
                details=errors[:10]
            )
        
        return TestResult(
            name=test_name,
            passed=True,
            message=f"✓ 条款验证通过，所有chunks的article_count与articles列表一致",
            details=[]
        )
    
    # ==================== 运行所有测试 ====================
    
    def run_all_tests(self) -> bool:
        """运行所有验证测试"""
        print("\n" + "="*70)
        print("RAG文档切分效果验证")
        print("="*70)
        print(f"输出目录: {self.output_dir}")
        print(f"待验证文件数: {len(self.all_files)}")
        print(f"待验证chunks数: {len(self.all_chunks)}")
        print("="*70)
        
        # 执行所有测试
        tests = [
            self.test_path_validation,
            self.test_format_validation,
            self.test_content_validation,
            self.test_semantic_validation,
            self.test_uniqueness_validation,
            self.test_articles_validation
        ]
        
        all_passed = True
        
        for test_func in tests:
            result = test_func()
            self.test_results.append(result)
            
            # 打印结果
            status = "✅ 通过" if result.passed else "❌ 失败"
            print(f"\n【{result.name}】{status}")
            print(f"  {result.message}")
            
            if result.details:
                for detail in result.details[:10]:  # 最多显示10条详情
                    print(f"  {detail}")
                if len(result.details) > 10:
                    print(f"  ... (还有 {len(result.details) - 10} 条详情)")
            
            if not result.passed:
                all_passed = False
        
        # 打印总结
        print("\n" + "="*70)
        print("测试总结")
        print("="*70)
        
        passed_count = sum(1 for r in self.test_results if r.passed)
        failed_count = len(self.test_results) - passed_count
        
        if all_passed:
            print(f"✅ 全部 {len(self.test_results)} 项测试通过！")
        else:
            print(f"❌ 测试完成: {passed_count} 项通过, {failed_count} 项失败")
        
        print("="*70)
        
        return all_passed


def main():
    """主函数"""
    output_dir = "/root/autodl-tmp/rag/test_chunking/output"
    
    # 创建验证器并运行测试
    validator = ChunkingValidator(output_dir)
    
    if not validator.all_files:
        print("\n⚠️ 未找到任何输出文件，请先运行文档处理器")
        print(f"   期望目录: {output_dir}")
        print(f"   期望文件: *_chunks.json")
        sys.exit(1)
    
    success = validator.run_all_tests()
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
