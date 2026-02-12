# RAG系统测评文档

## 📁 目录结构

```
test_reports/
├── README.md                 # 本说明文档
├── rag_evaluation_report_*.json    # 详细测试报告（JSON格式）
├── rag_evaluation_summary_*.md     # 测评摘要报告（Markdown格式）
└── historical/             # 历史测评记录（可选）
    └── archive_*.json
```

## 🚀 快速开始

### 1. 运行测评

```bash
# 基础测评（使用默认数据集）
cd /root/autodl-tmp/rag/backend
python -m tests.evaluation.enhanced_eval

# 指定数据集测评
python -m tests.evaluation.enhanced_eval --test-file test_dataset.json

# 限制测试数量（快速测试）
python -m tests.evaluation.enhanced_eval --limit 10

# 指定输出目录
python -m tests.evaluation.enhanced_eval --output-dir custom_reports
```

### 2. 测评参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--test-file` | `test_dataset_extended.json` | 测试数据文件名 |
| `--limit` | 无限制 | 限制测试用例数量 |
| `--output-dir` | `test_reports` | 报告输出目录 |

## 📊 报告文件说明

### JSON详细报告 (`rag_evaluation_report_*.json`)

完整的测评数据，包含：

```json
{
  "evaluation_info": {
    "timestamp": "2026-02-09T20:54:28.123",
    "test_file": "test_dataset_extended.json",
    "evaluator": "enhanced_eval.py",
    "version": "1.0"
  },
  "dataset_info": {
    "version": "2.0",
    "description": "RAG系统测试数据集 - 扩展版",
    "total_test_cases": 80
  },
  "score_info": {
    "total_score": 70,
    "max_score": 100,
    "grade": "🟡 良好",
    "grade_descriptions": ["🟡 P@1 精确率良好 (+20)", ...]
  },
  "analysis": {
    "statistics": { /* 各项指标统计 */ },
    "by_difficulty": { /* 按难度分组统计 */ },
    "by_category": { /* 按类别分组统计 */ },
    "problem_cases": [ /* 问题用例列表 */ ]
  },
  "detailed_results": [ /* 每个测试用例的详细结果 */ ]
}
```

### Markdown摘要报告 (`rag_evaluation_summary_*.md`)

简明的测评概览，包含：
- 测评概览和综合评分
- 关键指标表格
- 按难度分析
- 问题用例分析
- 优化建议

## 🔍 查看和分析报告

### 1. 快速查看摘要

```bash
# 查看最新的Markdown报告
ls -la test_reports/*.md | tail -1 | awk '{print $9}' | xargs cat

# 或者直接打开最新报告
cat test_reports/rag_evaluation_summary_*.md | head -50
```

### 2. 查看详细数据

```bash
# 查看JSON报告的关键指标
jq '.analysis.statistics' test_reports/rag_evaluation_report_*.json

# 查看问题用例
jq '.analysis.problem_cases[]' test_reports/rag_evaluation_report_*.json

# 查看按难度分组的统计
jq '.analysis.by_difficulty' test_reports/rag_evaluation_report_*.json
```

### 3. 分析趋势（如果有多份报告）

```bash
# 提取所有报告的评分
for file in test_reports/rag_evaluation_report_*.json; do
  echo "$(basename $file): $(jq -r '.score_info.total_score' $file)"
done

# 比较不同时间的关键指标
for file in test_reports/rag_evaluation_report_*.json; do
  echo "$(basename $file): P@1=$(jq -r '.analysis.statistics.avg_precision_at_1' $file), 命中率=$(jq -r '.analysis.statistics.avg_keyword_hit_rate' $file)"
done
```

## 📈 评估指标说明

### 检索性能指标

| 指标 | 含义 | 计算方式 | 优秀标准 |
|------|------|----------|----------|
| **P@1** | Precision@1，第一个结果的准确率 | 第一个结果是否相关 | ≥0.7 |
| **P@3** | Precision@3，前3个结果的准确率 | 前3个结果中相关的比例 | ≥0.7 |
| **P@5** | Precision@5，前5个结果的准确率 | 前5个结果中相关的比例 | ≥0.8 |
| **MRR** | Mean Reciprocal Rank，平均倒数排名 | 相关结果的排名倒数的平均值 | ≥0.5 |
| **关键词命中率** | 预期关键词的命中比例 | 命中关键词数/总关键词数 | ≥80% |
| **响应时间** | 检索耗时 | 向量检索时间 | ≤100ms |

### 综合评分规则

- **P@1 精确率** (30分): ≥0.7(优秀), ≥0.5(良好), ≥0.3(一般)
- **关键词命中率** (35分): ≥0.8(优秀), ≥0.6(良好), ≥0.4(一般)
- **MRR** (20分): ≥0.5(优秀), ≥0.3(良好)
- **响应速度** (15分): ≤100ms(优秀), ≤500ms(良好)

### 评级标准

| 分数范围 | 评级 | 说明 |
|----------|------|------|
| 80-100 | 🟢 优秀 | RAG系统表现良好，可投入使用 |
| 60-79 | 🟡 良好 | 基本满足需求，有优化空间 |
| 40-59 | 🟠 一般 | 存在明显问题，需要调优 |
| 0-39 | 🔴 需改进 | 严重不足，建议重构 |

## 🛠️ 故障排查

### 常见问题

1. **向量数据库为空**
   ```bash
   # 检查向量库状态
   python -c "
   from services.vector_db import vector_db_manager
   status = vector_db_manager.get_status()
   print(f'向量数: {status.total_vectors}')
   "
   ```

2. **嵌入模型未加载**
   ```bash
   # 检查模型状态
   python -c "
   from services.embedding import embedding_service
   print(f'模型已加载: {embedding_service.is_loaded()}')
   "
   ```

3. **测试数据文件不存在**
   ```bash
   # 查看可用测试文件
   ls -la test_data/test_dataset*.json
   ```

4. **权限问题**
   ```bash
   # 检查输出目录权限
   ls -la test_reports/
   chmod 755 test_reports/
   ```

### 调试模式

```bash
# 运行单个测试用例调试
python -m tests.evaluation.enhanced_eval --limit 1

# 查看详细日志
python -m tests.evaluation.enhanced_eval 2>&1 | tee debug.log
```

## 📝 自定义测试数据

### 创建测试数据文件格式

```json
{
  "metadata": {
    "version": "1.0",
    "description": "自定义测试数据",
    "total_test_cases": 5
  },
  "retrieval_test_cases": [
    {
      "id": "custom_001",
      "category": "自定义类别",
      "query": "测试查询语句",
      "description": "测试描述",
      "expected_keywords": ["关键词1", "关键词2"],
      "expected_topics": ["主题1", "主题2"],
      "difficulty": "medium"
    }
  ]
}
```

### 最佳实践

1. **查询多样性**: 包含不同长度和复杂度的查询
2. **关键词准确**: 确保预期关键词在文档中存在
3. **难度分级**: 合理分配easy/medium/hard比例
4. **类别覆盖**: 覆盖主要业务领域

## 🔄 持续集成

### 自动化测评脚本

```bash
#!/bin/bash
# ci_eval.sh - CI/CD自动化测评

echo "🚀 开始RAG系统测评..."

# 运行测评
cd /root/autodl-tmp/rag/backend
python -m tests.evaluation.enhanced_eval --test-file test_dataset_extended.json

# 检查评分
SCORE=$(jq -r '.score_info.total_score' test_reports/rag_evaluation_report_*.json | tail -1)

if [ "$SCORE" -lt 60 ]; then
    echo "❌ 测评不通过: 分数 $SCORE < 60"
    exit 1
else
    echo "✅ 测评通过: 分数 $SCORE"
fi
```

### 定期测评计划

- **每日**: 运行基础测评（5-10个用例）
- **每周**: 运行完整测评（全部用例）
- **版本发布**: 运行扩展测评（多数据集）

## 📞 技术支持

如果遇到问题，请：

1. 检查本文档的故障排查部分
2. 查看日志文件 `logs/app.log`
3. 查看测评生成的详细报告
4. 联系技术团队提供错误信息和报告文件

---

**最后更新**: 2026-02-09  
**版本**: 1.0  
**维护者**: RAG开发团队