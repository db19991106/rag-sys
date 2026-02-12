# RAG测评模块

## 📁 文件说明

### 🔧 核心文件
- **`enhanced_eval.py`** - 增强版测评脚本（推荐使用）
- **`mrr_debug.py`** - MRR修复工具
- **`quick_eval.py`** - 快速测评脚本

### 📊 使用方法

#### 1. 应用MRR修复（必需）
```bash
python mrr_debug.py
```

#### 2. 运行完整测评
```bash
# 使用修复版数据集
python -m evaluation.enhanced_eval --test-file simple_working_dataset.json

# 使用扩展数据集
python -m evaluation.enhanced_eval --test-file test_dataset_extended.json

# 快速测试
python -m evaluation.enhanced_eval --limit 10
```

#### 3. 运行快速测评
```bash
python quick_eval.py
```

### 📋 已清理的文件
以下文件已被删除，功能已集成到核心文件中：
- `batch_test.py`
- `comprehensive_evaluation.py`
- `eval_test.py`
- `evaluator_fix.py`
- `extended_eval.py`
- `fixed_eval.py`
- `mrr_complete_fix.py`
- `mrr_final_test.py`
- `mrr_simple_fix.py`
- `ultimate_mrr_fix.py`
- `offline_ragas_eval.py`
- `optimized_test_evaluator.py`
- `setup_local_eval.py`

### 🎯 推荐工作流

```bash
# 1. 修复MRR问题
python mrr_debug.py

# 2. 运行增强测评
python -m evaluation.enhanced_eval --test-file simple_working_dataset.json --limit 10

# 3. 查看结果
cat ../test_reports/rag_evaluation_summary_*.md
```

### 📈 支持的测评模式

| 模式 | 文件 | 说明 |
|------|------|------|
| **增强测评** | `enhanced_eval.py` | 完整功能，支持MRR修复 |
| **快速测评** | `quick_eval.py` | 基础测试，快速验证 |

### 🔧 功能特性

- ✅ MRR计算修复
- ✅ 多种测试数据集支持
- ✅ 综合评分系统
- ✅ JSON和Markdown报告
- ✅ 问题用例识别
- ✅ 按难度和类别分析

---

**最后更新**: 2026-02-09  
**版本**: 1.0