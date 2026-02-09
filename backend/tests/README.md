# 🧪 RAG系统测试方案

本目录包含完整的RAG系统测试框架，帮助你全面评估系统质量。

## 📁 文件说明

```
backend/
├── rag_test_suite.py              # 主测试套件 ⭐核心文件
├── test_report_generator.py       # 测试报告生成器
├── RAG_TESTING_GUIDE.md          # 详细测试指南
├── test_data/
│   ├── test_dataset.json         # 测试数据集
│   └── daily_reports/            # 每日测试报告存储
└── tests/
    ├── test_functionality.py     # 现有功能测试
    └── test_performance.py       # 现有性能测试
```

## 🚀 快速开始

### 1. 运行完整测试（推荐）

```bash
cd /root/autodl-tmp/rag/backend

# 运行所有测试
python rag_test_suite.py --mode full
```

**输出示例：**
```
======================================================================
RAG系统综合测试套件
======================================================================
开始时间: 2026-02-08 14:30:00
======================================================================

======================================================================
【测试1】文档解析功能测试
======================================================================
✅ TXT解析: 成功
✅ MD解析: 成功
✅ JSON解析: 成功

解析成功率: 100.0%

======================================================================
【测试2】文档切分策略测试
======================================================================
✅ INTELLIGENT策略: 生成5个chunk
✅ NAIVE策略: 生成8个chunk
...
```

### 2. 快速测试（只测核心功能）

```bash
python rag_test_suite.py --mode quick
```

适合日常快速验证，只测试：
- 文档解析
- 嵌入服务
- 向量数据库
- 检索质量

### 3. 性能基准测试

```bash
python rag_test_suite.py --mode benchmark
```

重点测试：
- 检索性能（响应时间、吞吐量）
- 并发性能

## 📊 生成测试报告

### 控制台报告

```bash
python test_report_generator.py --console
```

### HTML可视化报告

```bash
# 自动加载最新的测试结果
python test_report_generator.py

# 或者指定结果文件
python test_report_generator.py --input test_data/test_report_1234567890.json --output my_report.html
```

生成的HTML报告包含：
- 📄 文档解析成功率
- ✂️ 文档切分效果
- 🔢 嵌入服务性能
- 💾 向量数据库状态
- 🔍 检索质量指标
- ⚡ 性能测试结果
- 📊 综合评分

## 🎯 测试层次

我们的测试分为4个层次：

### Level 1: 功能测试 ✅
- 文档解析（PDF、Word、Markdown等）
- 文档切分（智能切分、朴素切分）
- 嵌入服务（编码、缓存）
- 向量数据库（添加、搜索）

**目标**：所有核心功能正常工作

### Level 2: 性能测试 ⚡
- 响应时间（P50、P95、P99）
- 吞吐量（QPS）
- 并发性能

**基准**：
| 指标 | 优秀 | 良好 | 需优化 |
|------|------|------|--------|
| 平均响应 | <200ms | <500ms | >500ms |
| 吞吐量 | >50 QPS | >20 QPS | <20 QPS |

### Level 3: 效果测试 🔍
- 检索准确性（关键词命中率）
- 排序质量（MRR、NDCG）
- 生成质量（相关性、准确性）

**目标**：
- 关键词命中率 > 60%
- Top-5召回率 > 80%

### Level 4: 端到端测试 🎯
- 完整RAG流程
- 真实用户场景
- 多轮对话

## 📝 使用测试数据集

### 预设测试用例

查看 `test_data/test_dataset.json`，包含：

1. **检索测试用例**（12个）
   - 按职级查询（8-9级、经理、总监）
   - 按费用类型查询（住宿、交通、补贴）
   - 复杂查询（多条件组合）

2. **端到端测试用例**（5个）
   - 场景化查询
   - 预期回答内容
   - 性能要求

3. **性能测试查询**（10个）
   - 用于压力测试

### 自定义测试用例

编辑 `test_data/test_dataset.json`，添加：

```json
{
  "retrieval_test_cases": [
    {
      "id": "your_test_001",
      "query": "你的测试查询",
      "expected_keywords": ["关键词1", "关键词2"]
    }
  ]
}
```

## ⚙️ 自动化测试

### 设置每日自动测试

```bash
# 编辑crontab
crontab -e

# 添加以下行（每天凌晨2点执行）
0 2 * * * cd /root/autodl-tmp/rag/backend && python rag_test_suite.py --mode quick >> /var/log/rag_test.log 2>&1
```

### 查看历史报告

```bash
ls -la test_data/daily_reports/

# 生成指定日期的报告
python test_report_generator.py --input test_data/daily_reports/report_20260208.json
```

## 🐛 故障排查

### 测试失败怎么办？

#### 1. 文档解析失败

```bash
# 检查依赖
pip install pypdf python-docx html2text

# 单独测试解析
python -c "
from services.document_parser import DocumentParser
parser = DocumentParser()
result = parser.parse('your_file.pdf')
print('解析成功:', len(result), '字符')
"
```

#### 2. 嵌入服务失败

```bash
# 检查模型是否已下载
ls -la data/models/

# 手动初始化
python -c "
from services.embedding import embedding_service
from models import EmbeddingConfig, EmbeddingModelType

config = EmbeddingConfig(
    model_type=EmbeddingModelType.BGE,
    model_name='BAAI/bge-small-zh-v1.5',
    device='cpu'
)
response = embedding_service.load_model(config)
print(response)
"
```

#### 3. 检索质量差

```bash
# 检查切分质量
python test_intelligent_chunking.py

# 检查向量数量
python -c "
from services.vector_db import vector_db_manager
status = vector_db_manager.get_status()
print(f'向量数量: {status.total_vectors}')
print(f'维度: {status.dimension}')
"
```

#### 4. 性能不达标

```bash
# 检查索引类型（应该使用HNSW）
# 检查缓存命中率
python -c "
from services.embedding import embedding_service
stats = embedding_service.get_cache_stats()
print(f'缓存大小: {stats[\"cache_size\"]}')
print(f'平均编码时间: {stats[\"average_encode_time\"]:.4f}s')
"
```

## 📈 性能优化建议

### 检索性能优化

1. **使用HNSW索引**（默认已启用）
2. **增大嵌入缓存**
   ```python
   embedding_service.cache_size = 50000
   ```
3. **批处理优化**
   ```python
   config.batch_size = 64  # 根据内存调整
   ```

### 生成性能优化

1. **限制生成长度**
   ```python
   generation_config.max_tokens = 300
   ```
2. **使用更快的LLM**
3. **启用流式生成**

## 🔗 相关文档

- [详细测试指南](./RAG_TESTING_GUIDE.md) - 完整的测试方法论
- [数据准备分析](../数据准备阶段分析.md) - 数据准备阶段详解
- [Chunking指南](../CHUNKING_GUIDE.md) - 文档切分策略

## 💡 最佳实践

1. **开发阶段**：每次代码变更后运行快速测试
2. **发布前**：运行完整测试套件，确保通过率>90%
3. **生产环境**：设置每日自动测试，监控关键指标
4. **定期评估**：每月人工评估生成质量，更新测试用例

## 📞 获取帮助

遇到问题？

1. 查看详细日志：`tail -f logs/rag_system.log`
2. 运行诊断脚本：`python diagnose_issue.py`
3. 检查系统状态：`python test_retrieval_system.py`

---

**版本**: 1.0  
**更新日期**: 2026-02-08
