#!/usr/bin/env python3
"""
自动生成 Ground Truth 测试数据

使用 Qwen2.5-7B-Instruct 为所有文档生成问答对测试数据
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加 backend 目录到路径
BACKEND_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from config import settings
from services.vector_db import vector_db_manager


class GroundTruthGenerator:
    """Ground Truth 生成器"""
    
    def __init__(self, model_path: str = None):
        """初始化生成器
        
        Args:
            model_path: LLM 模型路径，默认使用配置中的路径
        """
        self.model_path = model_path or settings.local_llm_model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载 LLM 模型"""
        print(f"正在加载模型: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("模型加载完成")
    
    def generate_qa_pairs(self, content: str, doc_name: str, num_pairs: int = 6) -> List[Dict]:
        """为文档内容生成问答对
        
        Args:
            content: 文档内容
            doc_name: 文档名称
            num_pairs: 生成的问答对数量
            
        Returns:
            问答对列表
        """
        prompt = f"""你是一个专业的问答数据生成助手。请根据以下文档内容，生成 {num_pairs} 个高质量的问答对。

要求：
1. 问题应该多样化，包括：
   - 具体数值查询（如金额、天数、比例等）
   - 流程步骤查询
   - 条件判断查询
   - 对比查询
2. 问题应该自然、口语化，模拟用户真实提问
3. 答案应该准确，直接从文档中提取
4. 每个问答对单独一行，格式为 JSON: {{"question": "问题", "answer": "答案"}}

文档名称：{doc_name}

文档内容：
{content[:6000]}

请生成 {num_pairs} 个问答对（只输出 JSON 数组，不要其他内容）：
"""

        messages = [
            {"role": "system", "content": "你是一个专业的问答数据生成助手，擅长从文档中提取关键信息并生成高质量的问答对。"},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # 解析生成的 JSON
        try:
            # 尝试直接解析
            qa_pairs = json.loads(generated_text)
            if isinstance(qa_pairs, list):
                return qa_pairs
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 数组
        import re
        json_match = re.search(r'\[.*\]', generated_text, re.DOTALL)
        if json_match:
            try:
                qa_pairs = json.loads(json_match.group())
                if isinstance(qa_pairs, list):
                    return qa_pairs
            except json.JSONDecodeError:
                pass
        
        # 尝试逐行解析
        qa_pairs = []
        lines = generated_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    qa = json.loads(line)
                    if 'question' in qa and 'answer' in qa:
                        qa_pairs.append(qa)
                except json.JSONDecodeError:
                    continue
        
        return qa_pairs
    
    def _get_all_chunks(self) -> List[Dict]:
        """获取所有 chunks"""
        try:
            return vector_db_manager.get_all_metadata()
        except Exception as e:
            print(f"获取 chunks 失败: {e}")
            return []
    
    def _filter_doc_chunks(self, all_chunks: List[Dict], doc_name: str) -> List[Dict]:
        """筛选指定文档的 chunks"""
        doc_chunks = [
            chunk for chunk in all_chunks 
            if chunk.get('doc_id') == doc_name or doc_name in chunk.get('chunk_id', '')
        ]
        
        if not doc_chunks:
            # 尝试使用文档名的关键词匹配
            doc_key = doc_name.replace('.md', '').replace('管理制度', '').replace('绩效考核方案', '')
            doc_chunks = [
                chunk for chunk in all_chunks 
                if doc_key in chunk.get('chunk_id', '') or doc_key in chunk.get('doc_id', '')
            ]
        
        return doc_chunks
    
    def find_relevant_chunks(self, query: str, doc_name: str, top_k: int = 3) -> List[str]:
        """查找与问题相关的 chunk_id
        
        Args:
            query: 查询问题
            doc_name: 文档名称
            top_k: 返回的 chunk 数量
            
        Returns:
            相关 chunk_id 列表
        """
        try:
            # 获取该文档的所有 chunks
            all_chunks = self._get_all_chunks()
            doc_chunks = self._filter_doc_chunks(all_chunks, doc_name)
            
            if not doc_chunks:
                print(f"  未找到文档 {doc_name} 的 chunks")
                return []
            
            # 简单的关键词匹配来找到相关 chunk
            query_keywords = set(query)
            scored_chunks = []
            
            for chunk in doc_chunks:
                content = chunk.get('content', '')
                chunk_id = chunk.get('chunk_id', chunk.get('id', ''))
                # 计算关键词重叠得分
                overlap = sum(1 for kw in query_keywords if kw in content)
                # 检查问题中的关键词是否在内容中
                query_words = query.replace('？', '').replace('吗', '').replace('什么', '').replace('多少', '').split()
                word_match = sum(1 for word in query_words if len(word) > 1 and word in content)
                
                score = overlap + word_match * 2
                scored_chunks.append((chunk_id, score, content))
            
            # 按得分排序
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # 返回得分最高的 chunks
            return [chunk[0] for chunk in scored_chunks[:top_k] if chunk[0]]
            
        except Exception as e:
            print(f"查找 chunks 失败: {e}")
            return []
    
    def find_relevant_content(self, query: str, doc_name: str) -> str:
        """查找与问题相关的内容片段
        
        Args:
            query: 查询问题
            doc_name: 文档名称
            
        Returns:
            相关内容
        """
        try:
            all_chunks = self._get_all_chunks()
            doc_chunks = self._filter_doc_chunks(all_chunks, doc_name)
            
            if not doc_chunks:
                return ""
            
            # 关键词匹配
            query_keywords = [w for w in query if len(w) > 1]
            best_chunk = None
            best_score = -1
            
            for chunk in doc_chunks:
                content = chunk.get('content', '')
                score = sum(1 for kw in query_keywords if kw in content)
                if score > best_score:
                    best_score = score
                    best_chunk = content
            
            return best_chunk or ""
            
        except Exception as e:
            print(f"查找内容失败: {e}")
            return ""
    
    def generate_for_document(self, doc_path: str, num_pairs: int = 6) -> List[Dict]:
        """为单个文档生成 Ground Truth
        
        Args:
            doc_path: 文档路径
            num_pairs: 生成的问答对数量
            
        Returns:
            Ground Truth 条目列表
        """
        doc_name = Path(doc_path).stem
        print(f"\n处理文档: {doc_name}")
        
        # 读取文档内容
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            print(f"  文档为空，跳过")
            return []
        
        # 生成问答对
        qa_pairs = self.generate_qa_pairs(content, doc_name, num_pairs)
        
        if not qa_pairs:
            print(f"  未能生成问答对")
            return []
        
        print(f"  生成了 {len(qa_pairs)} 个问答对")
        
        # 构建 Ground Truth 条目
        ground_truth_entries = []
        
        for i, qa in enumerate(qa_pairs):
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            
            if not question or not answer:
                continue
            
            # 查找相关 chunks
            relevant_chunks = self.find_relevant_chunks(question, doc_name)
            
            # 查找相关内容
            relevant_content = self.find_relevant_content(question, doc_name)
            if not relevant_content:
                relevant_content = answer
            
            entry = {
                "query": question,
                "relevant_chunks": relevant_chunks,
                "relevant_content": relevant_content
            }
            
            ground_truth_entries.append(entry)
            print(f"  - {question[:50]}...")
        
        return ground_truth_entries
    
    def generate_for_all_documents(self, docs_dir: str = None, num_pairs_per_doc: int = 6) -> Dict:
        """为所有文档生成 Ground Truth
        
        Args:
            docs_dir: 文档目录
            num_pairs_per_doc: 每个文档生成的问答对数量
            
        Returns:
            完整的 Ground Truth 数据
        """
        if docs_dir is None:
            docs_dir = PROJECT_ROOT / "backend" / "data" / "docs"
        
        docs_path = Path(docs_dir)
        
        # 获取所有 markdown 文档
        doc_files = list(docs_path.glob("*.md"))
        print(f"找到 {len(doc_files)} 个文档")
        
        ground_truth = {}
        case_id = 1
        
        for doc_file in doc_files:
            entries = self.generate_for_document(str(doc_file), num_pairs_per_doc)
            
            for entry in entries:
                case_key = f"ret_{case_id:03d}"
                ground_truth[case_key] = entry
                case_id += 1
        
        return {"ground_truth": ground_truth}


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自动生成 Ground Truth 测试数据")
    parser.add_argument("--model-path", type=str, default=None, help="LLM 模型路径")
    parser.add_argument("--docs-dir", type=str, default=None, help="文档目录")
    parser.add_argument("--num-pairs", type=int, default=6, help="每个文档生成的问答对数量")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = GroundTruthGenerator(model_path=args.model_path)
    
    # 生成 Ground Truth
    ground_truth = generator.generate_for_all_documents(
        docs_dir=args.docs_dir,
        num_pairs_per_doc=args.num_pairs
    )
    
    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "retrieval_test_cases_ground_truth.json"
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    
    print(f"\n生成完成！")
    print(f"总计 {len(ground_truth['ground_truth'])} 个测试用例")
    print(f"保存到: {output_path}")


if __name__ == "__main__":
    main()
