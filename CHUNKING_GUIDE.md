# RAGFlow 风格切分功能使用指南

## 概述

本项目已成功集成 RAGFlow 风格的文档切分功能，支持 11 种切分策略和高级特性。

## 安装依赖

```bash
cd /root/autodl-tmp/rag/backend
pip install -r requirements.txt
```

主要依赖：
- `tiktoken`: 用于精确的 Token 计算
- `beautifulsoup4`: 用于 HTML 解析
- `lxml`: BeautifulSoup 的依赖

## 切分策略

### 1. Naive（朴素切分）⭐ 推荐

**适用场景**：通用文档、技术文档、普通文本

**特点**：
- 基于自定义分隔符的智能切分
- 支持 Token 数量限制
- 支持重叠切分
- 自动过滤小于 8 tokens 的片段

**配置示例**：
```
切分方式: 朴素切分
Token数量: 512
重叠百分比: 10%
主分隔符: \n, 。, ；, ！, ？
```

**自定义分隔符示例**：
```
主分隔符: `\n\n`, `。`, `；`, `！`, `？`
```
（使用反引号包裹多个分隔符，会按顺序匹配）

### 2. QA（问答对切分）

**适用场景**：问答集、FAQ 文档、考试题库

**识别格式**：
- `第X问`
- `第1问`
- `QUESTION ONE`
- `问题：`
- `Q1:`

**配置示例**：
```
切分方式: 问答对切分
Token数量: 256
重叠百分比: 0%
```

### 3. Paper（论文切分）

**适用场景**：学术论文、研究报告、技术白皮书

**识别格式**：
- Markdown 标题：`#`, `##`, `###`
- 中文标题：`第X章`, `第X节`
- 数字编号：`1.1`, `2.3.4`
- 英文标题：`CHAPTER`, `SECTION`, `ABSTRACT`, `INTRODUCTION`

**配置示例**：
```
切分方式: 论文切分
Token数量: 1024
重叠百分比: 15%
```

### 4. Laws（法律文档切分）

**适用场景**：法律条文、法规、合同

**识别格式**：
- `第X条`
- `第X节`
- `第X章`
- `(1)`, `(2)`, `一、`, `二、`

**配置示例**：
```
切分方式: 法律文档切分
Token数量: 256
重叠百分比: 10%
主分隔符: \n, 。
```

### 5. Book（书籍切分）

**适用场景**：小说、教材、书籍

**识别格式**：
- `第X章`
- `第X节`
- `PART ONE`
- `Chapter I`
- `Section 1`

**配置示例**：
```
切分方式: 书籍切分
Token数量: 512
重叠百分比: 10%
```

### 6. Table（表格切分）

**适用场景**：包含大量表格的文档、报表、数据手册

**支持格式**：
- Markdown 表格
- HTML 表格

**特点**：
- 自动提取表格
- 表格单独作为一个 chunk
- 非表格内容按段落切分

**配置示例**：
```
切分方式: 表格切分
```

### 7. Custom（自定义切分）

**适用场景**：需要精细控制切分的文档

**特点**：
- 支持主分隔符
- 支持子分隔符（细粒度切分）
- 完全自定义切分逻辑

**配置示例**：
```
切分方式: 自定义切分
Token数量: 512
重叠百分比: 5%
主分隔符: \n\n
启用子分隔符: 是
子分隔符: \n
```

### 8. Char（按字符切分）

**适用场景**：代码文件、日志文件、无结构的文本

**特点**：
- 按固定字符数切分
- 支持重叠

**配置示例**：
```
切分方式: 按字符切分
切分长度: 500
重叠长度: 50
```

### 9. Sentence（按句子切分）

**适用场景**：对话、叙述性文本、新闻文章

**特点**：
- 按句子边界切分
- 支持重叠
- 识别中英文标点

**配置示例**：
```
切分方式: 按句子切分
切分长度: 10  # 句子数量
重叠长度: 2
```

### 10. Paragraph（按段落切分）

**适用场景**：邮件、博客、普通文章

**特点**：
- 按段落边界切分（空行分隔）
- 最简单直接的切分方式

**配置示例**：
```
切分方式: 按段落切分
```

### 11. Picture（图片切分）

**适用场景**：包含图片的文档（预留接口）

**特点**：
- 为图片处理预留
- 未来版本支持图片内容提取

## 参数说明

### Token 数量（chunkTokenSize）

控制每个 chunk 的大小，单位是 token（不是字符）

**推荐值**：
- 短文档：128-256
- 中等文档：512（默认）
- 长文档：1024-2048

**注意**：
- 1 个中文字符 ≈ 1.5-2 tokens
- 1 个英文单词 ≈ 1.3 tokens

### 重叠百分比（overlappedPercent）

控制相邻 chunk 之间的重叠程度，范围 0-0.5（0-50%）

**推荐值**：
- 一般文档：0-10%
- 连续性强的文档：10-20%
- 独立段落：0%

**示例**：
- `overlappedPercent=0.1` 表示相邻 chunk 有 10% 的重叠

### 主分隔符（delimiters）

用于主要切分点的字符序列

**常见配置**：
```
中文文档: \n, 。, ；, ！, ？
英文文档: \n, ., !, ?, ;
代码文档: \n
Markdown: \n## 或 \n###
```

**自定义分隔符**：
使用反引号包裹多个分隔符：
```
主分隔符: `\n\n`, `。`, `；`
```

### 子分隔符（childrenDelimiters）

用于在主 chunk 内部进行细粒度切分

**使用场景**：
- 主 chunk 仍然太大时
- 需要更细粒度的切分时

**示例**：
```
主分隔符: \n\n  # 按段落切分
子分隔符: \n    # 段落内按行切分
```

### 启用子分隔符（enableChildren）

是否启用子分隔符功能

- `true`: 启用，会对主 chunk 进一步切分
- `false`: 禁用，只使用主分隔符

### 表格上下文大小（tableContextSize）

为表格块附加周围的文本上下文（预留功能）

### 图片上下文大小（imageContextSize）

为图片块附加周围的文本上下文（预留功能）

## 使用示例

### 示例 1：技术文档

```python
config = ChunkConfig(
    type=ChunkType.NAIVE,
    chunkTokenSize=512,
    overlappedPercent=0.1,
    delimiters=["\n", "。", "；", "！", "？"]
)
```

### 示例 2：法律文档

```python
config = ChunkConfig(
    type=ChunkType.LAWS,
    chunkTokenSize=256,
    overlappedPercent=0.1,
    delimiters=["\n", "。"]
)
```

### 示例 3：问答集

```python
config = ChunkConfig(
    type=ChunkType.QA,
    chunkTokenSize=256,
    overlappedPercent=0
)
```

### 示例 4：论文

```python
config = ChunkConfig(
    type=ChunkType.PAPER,
    chunkTokenSize=1024,
    overlappedPercent=0.15
)
```

### 示例 5：自定义切分（带子分隔符）

```python
config = ChunkConfig(
    type=ChunkType.CUSTOM,
    chunkTokenSize=512,
    overlappedPercent=0.05,
    delimiters=["\n\n", "。"],
    enableChildren=True,
    childrenDelimiters=["\n", "；"]
)
```

## 最佳实践

### 1. 选择合适的切分策略

| 文档类型 | 推荐策略 | Token 数量 | 重叠 |
|---------|---------|-----------|------|
| 技术文档 | Naive | 512 | 5-10% |
| 法律文档 | Laws | 256 | 10% |
| 问答对 | QA | 128-256 | 0% |
| 论文 | Paper | 1024 | 15% |
| 表格密集 | Table | 256 | 0% |
| 书籍 | Book | 512 | 10% |
| 代码 | Char | 500 | 5% |
| 对话 | Sentence | 10 句 | 2 句 |

### 2. Token 数量选择

- **过小**（<128）：信息碎片化，语义不完整
- **适中**（128-512）：适合大多数场景
- **过大**（>1024）：检索精度下降，相关性降低

### 3. 重叠设置

- **连续性强的文本**（小说、技术文档）：10-20%
- **独立内容**（问答对、法律条文）：0%
- **一般文档**：5-10%

### 4. 分隔符选择

- **避免**：使用过于通用的分隔符（如单个空格）
- **推荐**：使用语义明确的分隔符（如句号、段落分隔）
- **注意**：中文和英文的分隔符可能不同

## 核心算法说明

### Naive 切分算法（RAGFlow 核心）

```python
def naive_merge(content, chunk_token_num, delimiter, overlapped_percent):
    1. 解析自定义分隔符
    2. 按分隔符切分文本
    3. 逐段合并，直到达到 token 限制
    4. 如果启用重叠，保留前一个 chunk 的部分内容
    5. 过滤小于 8 tokens 的片段
```

### Token 计算

使用 `tiktoken` 库进行精确计算：

```python
from utils.token_counter import num_tokens_from_string

tokens = num_tokens_from_string("这是一段中文文本")
# 返回精确的 token 数量
```

如果 `tiktoken` 不可用，自动降级为估算方案：
- 中文字符：约 1.5-2 个字符 = 1 token
- 英文单词：约 1 个单词 = 1.3 tokens

## 故障排除

### 问题 1：切分后 chunk 太少

**可能原因**：
- Token 数量设置过大
- 分隔符选择不当

**解决方案**：
- 减小 `chunkTokenSize`
- 调整分隔符，使用更细致的分隔符

### 问题 2：切分后语义不完整

**可能原因**：
- 分隔符选择不当
- 重叠百分比过低

**解决方案**：
- 使用语义明确的分隔符（如句号、段落）
- 增加重叠百分比（5-15%）

### 问题 3：切分速度慢

**可能原因**：
- Token 计算开销大
- 正则表达式复杂

**解决方案**：
- 减小文档大小
- 简化分隔符正则表达式

## 与 RAGFlow 的兼容性

本实现完全兼容 RAGFlow 的切分逻辑：

- ✅ 支持所有 RAGFlow 的切分策略
- ✅ 使用相同的 Token 计算方法
- ✅ 支持自定义分隔符语法
- ✅ 支持重叠切分
- ✅ 支持子分隔符

## 未来扩展

计划添加的功能：
- [ ] 图片内容提取（使用 OCR）
- [ ] 表格结构识别
- [ ] 语义感知切分（基于 BERT）
- [ ] 自适应切分（自动选择最佳策略）
- [ ] 切分质量评估

## 技术支持

如有问题，请检查：
1. 依赖是否正确安装：`pip list | grep tiktoken`
2. 日志文件：`backend/logs/app.log`
3. 前端控制台是否有错误信息

## 参考资料

- [RAGFlow 官方文档](https://github.com/infiniflow/ragflow)
- [tiktoken 文档](https://github.com/openai/tiktoken)
- [Chunking Strategies for RAG](https://www.llamaindex.ai/blog/chunking-strategies-for-rag)