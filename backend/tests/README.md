# 🧪 RAG系统测试框架

本目录包含完整的RAG系统测试框架，所有测试文件已按功能分类整理。

## 📁 目录结构

```
tests/
├── core/                    # ⭐ 核心测试框架
│   ├── rag_test_suite.py           # 主测试套件
│   ├── test_report_generator.py    # 测试报告生成器
│   └── testing_guide.py            # 测试指南
│
├── unit/                    # 🔬 单元测试
│   ├── test_functionality.py       # 功能测试
│   └── test_performance.py         # 性能测试
│
├── chunking/               # ✂️ 文档切分测试
│   ├── test_intelligent_chunking.py
│   ├── test_enhanced_chunking.py
│   ├── test_enhanced_chunking_comprehensive.py
│   ├── test_direct_secondary_split.py
│   ├── test_secondary_split.py
│   ├── test_financial_report_chunking.py
│   └── test_pdf_chunking.py
│
├── evaluation/             # 📊 评估和RAGAS
│   ├── offline_ragas_eval.py
│   ├── optimized_test_evaluator.py
│   ├── setup_local_eval.py
│   └── batch_test.py
│
├── integration/            # 🔗 集成测试
│   ├── test_retrieval_system.py
│   ├── test_local_llm.py
│   ├── test_context.py
│   └── test_log_rotation.py
│
├── tools/                  # 🛠️ 工具脚本
│   ├── quick_verify.py            # 快速验证
│   ├── quick_fix.py               # 快速修复
│   └── batch_embed.py             # 批量嵌入
│
├── diagnostics/            # 🔧 诊断修复
│   ├── diagnose_issue.py
│   ├── diagnose_and_fix.py
│   ├── fix_metadata.py
│   └── fix_intent_config.py
│
└── data/                   # 📁 测试数据
    ├── test_dataset.json
    └── reports/           # 测试报告存储
```

## 🚀 快速开始

### 1. 运行完整测试套件

```bash
cd /root/autodl-tmp/rag/backend
python tests/core/rag_test_suite.py --mode full
```

### 2. 快速功能验证

```bash
# 使用快速验证工具
python tests/tools/quick_verify.py
```

### 3. 测试文档切分

```bash
# 运行所有chunking测试
python tests/chunking/test_intelligent_chunking.py
python tests/chunking/test_enhanced_chunking.py
```

### 4. RAGAS评估

```bash
# 批量RAGAS评估
python tests/evaluation/batch_test.py

# 离线RAGAS评估
python tests/evaluation/offline_ragas_eval.py --mode batch
```

### 5. 生成测试报告

```bash
# 生成HTML报告
python tests/core/test_report_generator.py

# 查看测试指南
python tests/core/testing_guide.py --help
```

## 📊 测试类别说明

### 🔴 Core - 核心测试框架
- **rag_test_suite.py**: 一站式测试框架，涵盖功能、性能、效果测试
- **test_report_generator.py**: 生成可视化测试报告（HTML）
- **testing_guide.py**: 测试指南和快速参考

### 🔵 Unit - 单元测试
- **test_functionality.py**: 文档解析、嵌入、检索等核心功能测试
- **test_performance.py**: 响应时间、吞吐量、并发性能测试

### 🟢 Chunking - 文档切分测试
测试各种文档切分策略的效果：
- 智能切分 (Intelligent)
- 增强切分 (Enhanced)
- 财务报告切分
- PDF切分
- 二级切分策略

### 🟡 Evaluation - 评估工具
- **offline_ragas_eval.py**: 离线RAGAS评估（Faithfulness, Relevance等）
- **batch_test.py**: 批量测试所有用例
- **setup_local_eval.py**: 本地评估环境设置

### 🟠 Integration - 集成测试
- **test_retrieval_system.py**: 检索系统端到端测试
- **test_local_llm.py**: 本地LLM测试
- **test_log_rotation.py**: 日志轮转测试

### ⚪ Tools - 工具脚本
- **quick_verify.py**: 快速验证系统状态
- **quick_fix.py**: 常见问题快速修复
- **batch_embed.py**: 批量文档嵌入

### ⚫ Diagnostics - 诊断修复
- **diagnose_issue.py**: 系统诊断
- **diagnose_and_fix.py**: 诊断并修复
- **fix_metadata.py**: 修复元数据问题
- **fix_intent_config.py**: 修复意图配置

## 🎯 测试层次

| 层次 | 测试类型 | 目标 | 命令 |
|------|---------|------|------|
| L1 | 功能测试 | 核心功能正常 | `python tests/core/rag_test_suite.py --mode quick` |
| L2 | 性能测试 | 响应时间、吞吐量达标 | `python tests/core/rag_test_suite.py --mode benchmark` |
| L3 | 效果测试 | 检索准确率、生成质量 | `python tests/evaluation/batch_test.py` |
| L4 | 端到端测试 | 完整RAG流程 | `python tests/integration/test_retrieval_system.py` |

## 📝 添加新测试

### 添加单元测试

在 `tests/unit/` 目录下创建新文件：

```python
# tests/unit/test_new_feature.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestNewFeature:
    def test_feature_x(self):
        assert True

if __name__ == "__main__":
    test = TestNewFeature()
    test.test_feature_x()
```

### 添加Chunking测试

在 `tests/chunking/` 目录下创建新文件，参考现有测试文件格式。

## 🐛 故障排查

### 测试失败诊断流程

1. **运行系统诊断**
   ```bash
   python tests/diagnostics/diagnose_issue.py
   ```

2. **快速修复常见问题**
   ```bash
   python tests/tools/quick_fix.py
   ```

3. **查看详细日志**
   ```bash
   tail -f logs/rag_system.log
   ```

4. **验证检索系统**
   ```bash
   python tests/integration/test_retrieval_system.py
   ```

## 🔗 相关文档

- [详细测试指南](./core/testing_guide.py) - 完整的测试方法论
- [RAG测试指南](../RAG_TESTING_GUIDE.md) - 原始测试文档
- [Chunking指南](../CHUNKING_GUIDE.md) - 文档切分策略

## 💡 最佳实践

1. **开发阶段**: 每次代码变更后运行快速测试
   ```bash
   python tests/tools/quick_verify.py
   ```

2. **发布前**: 运行完整测试套件
   ```bash
   python tests/core/rag_test_suite.py --mode full
   ```

3. **定期评估**: 每周运行RAGAS评估
   ```bash
   python tests/evaluation/offline_ragas_eval.py --mode batch
   ```

4. **性能监控**: 每月运行性能基准测试
   ```bash
   python tests/core/rag_test_suite.py --mode benchmark
   ```

---

**版本**: 2.0  
**更新日期**: 2026-02-09  
**分类整理**: 核心测试 | 单元测试 | Chunking测试 | 评估工具 | 集成测试 | 工具脚本 | 诊断修复
