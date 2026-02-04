#!/usr/bin/env python3
# PDF文件分析脚本

import fitz
import os


def analyze_pdf(pdf_path):
    """
    分析PDF文件的结构和内容
    
    Args:
        pdf_path: PDF文件路径
    """
    try:
        # 打开PDF文件
        doc = fitz.open(pdf_path)
        
        # 基本信息
        print(f"=== PDF文件分析结果 ===")
        print(f"文件路径: {pdf_path}")
        print(f"总页数: {len(doc)}")
        print(f"文件大小: {os.path.getsize(pdf_path) / 1024:.2f} KB")
        
        # 检查是否有书签
        print(f"\n=== 书签信息 ===")
        try:
            toc = doc.get_toc()
            if toc:
                print(f"有书签: 是")
                print(f"书签数量: {len(toc)}")
                # 打印前几个书签
                print("前5个书签:")
                for i, bookmark in enumerate(toc[:5]):
                    level, title, page = bookmark
                    print(f"  {i+1}. 级别: {level}, 标题: {title}, 页码: {page+1}")
            else:
                print(f"有书签: 否")
        except Exception as e:
            print(f"书签检查失败: {str(e)}")
            print(f"有书签: 否")
        
        # 分析前几页的内容
        print(f"\n=== 内容分析 ===")
        max_pages = min(5, len(doc))
        for page_num in range(max_pages):
            page = doc.load_page(page_num)
            
            # 提取文本
            text = page.get_text()
            text_length = len(text)
            
            # 检查是否有图片
            images = page.get_images(full=True)
            has_images = len(images) > 0
            
            # 检查是否有表格（简单检测）
            has_tables = "|" in text and "---" in text
            
            print(f"第 {page_num+1} 页:")
            print(f"  文本长度: {text_length} 字符")
            print(f"  有图片: {'是' if has_images else '否'}")
            print(f"  有表格: {'是' if has_tables else '否'}")
            
            # 打印前100个字符的文本预览
            if text_length > 0:
                preview = text[:100] + "..." if text_length > 100 else text
                print(f"  文本预览: {preview}")
            print()
        
        # 分析页面大小
        print(f"=== 页面信息 ===")
        if len(doc) > 0:
            page = doc.load_page(0)
            rect = page.rect
            print(f"页面大小: {rect.width:.2f} x {rect.height:.2f} 点")
            print(f"页面方向: {'横向' if rect.width > rect.height else '纵向'}")
        
        # 关闭PDF文件
        doc.close()
        
        print(f"=== 分析完成 ===")
        
    except Exception as e:
        print(f"分析失败: {str(e)}")


if __name__ == "__main__":
    # PDF文件路径
    pdf_path = "/root/autodl-tmp/rag/backend/data/docs/baoxiao.pdf"
    analyze_pdf(pdf_path)
