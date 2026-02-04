# PDF文档智能处理系统

## 系统概述

本系统是一个完整的PDF文档智能处理系统，能够对PDF文件进行类型检测、版面分析、内容提取、内容组织和标准化输出。系统支持处理文本型PDF和扫描型PDF，能够提取文本、表格和图片内容，并生成符合指定格式的结构化输出。

## 系统功能

1. **PDF类型检测与预处理**
   - 区分文本型PDF和扫描型PDF
   - 对于扫描型PDF，使用OCR技术进行文字识别
   - 对于文本型PDF，直接提取原生文本流

2. **版面分析**
   - 检测文本块、表格和图片的坐标信息
   - 对检测到的元素进行分类标记

3. **内容提取**
   - 提取文本内容并保留字体属性
   - 提取表格内容并输出结构化表格数据
   - 提取图片内容并进行OCR识别

4. **内容组织**
   - 恢复阅读顺序
   - 智能切分内容
   - 标准化输出

5. **输出格式**
   - 生成JSON格式的chunk数组
   - 生成Markdown格式的内容

## 系统架构

```
pdf_processor/
├── app/
│   ├── preprocessing/        # PDF类型检测与预处理模块
│   ├── layout_analysis/      # 版面分析模块
│   ├── content_extraction/   # 内容提取模块
│   ├── content_organization/ # 内容组织模块
│   └── output_formatting/    # 输出格式处理模块
├── config/                   # 配置文件
├── utils/                    # 工具函数
├── tests/                    # 测试脚本
├── main.py                   # 主入口文件
├── requirements.txt          # 依赖项
└── README.md                 # 说明文档
```

## 环境要求

- Python 3.8+
- 依赖项详见requirements.txt

## 安装步骤

1. 克隆项目代码

2. 安装依赖项
   ```bash
   pip install -r requirements.txt
   ```

3. 安装PaddleOCR（用于OCR识别）
   ```bash
   pip install paddleocr paddlepaddle
   ```

## 使用方法

### 命令行方式

```bash
python main.py <pdf_file_path>
```

例如：
```bash
python main.py sample.pdf
```

### 代码调用方式

```python
from main import PDFProcessor

# 创建PDF处理器
processor = PDFProcessor()

# 处理PDF文件
result = processor.process_pdf("sample.pdf")

# 检查处理结果
if result.get("status") == "success":
    print(f"处理成功！生成 {result.get('total_chunks', 0)} 个chunk")
    print(f"输出文件:")
    for file_type, file_path in result.get('output_files', {}).items():
        print(f"{file_type}: {file_path}")
else:
    print(f"处理失败: {result.get('message', '未知错误')}")
```

## 输出格式

系统最终输出为包含多个chunk对象的JSON数组，每个chunk对象包含以下字段：

- `chunk_id`: 唯一标识符，格式为"{类型}_{序号}"
- `type`: 元素类型，包括"title"、"text"、"table"、"image"等
- `content`: 元素的具体内容，根据类型不同采用相应格式
- `metadata`: 元数据对象，包含但不限于页面编号、元素类型、所属章节等信息

示例输出：

```json
[
  {
    "chunk_id": "title_0",
    "type": "title",
    "content": "## 第二章 休假政策",
    "metadata": {
      "page": 5,
      "element_type": "title",
      "level": "h2"
    }
  },
  {
    "chunk_id": "text_1",
    "type": "text",
    "content": "根据《劳动法》规定，员工享有以下带薪假期...",
    "metadata": {
      "page": 5,
      "element_type": "text"
    }
  }
]
```

## 测试

运行测试脚本：

```bash
python tests/test_system.py
```

测试脚本会使用当前目录下的`test_sample.pdf`文件进行测试，并验证系统的各个功能模块是否正常工作。

## 注意事项

1. 系统处理大型PDF文件可能需要较长时间，请耐心等待
2. 对于扫描型PDF，OCR识别的准确率取决于PDF的质量和清晰度
3. 系统默认将提取的图片保存到`./extracted_images`目录
4. 系统默认将输出文件保存到`./output`目录

## 故障排除

### 常见错误及解决方案

1. **PDF类型检测失败**
   - 检查PDF文件是否损坏
   - 确保PDF文件可以正常打开

2. **OCR识别失败**
   - 检查PaddleOCR是否正确安装
   - 确保PDF文件的扫描质量良好

3. **内容提取失败**
   - 检查PDF文件是否受密码保护
   - 确保PDF文件包含可提取的内容

4. **输出文件生成失败**
   - 检查输出目录是否存在且有写入权限
   - 确保磁盘空间充足

## 版本历史

- v1.0.0: 初始版本，实现了基本功能

## 许可证

本项目采用MIT许可证。
