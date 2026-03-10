"""
Ground Truth生成脚本 v2
基于当前chunk数据生成测试用例，确保100% chunk_id匹配
"""
import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入嵌入服务
from services.embedding import embedding_service
from models import EmbeddingConfig, EmbeddingModelType
from config import settings


def load_all_chunks(chunks_dir: Path) -> Dict[str, List[Dict]]:
    """加载所有chunk数据"""
    all_chunks = {}
    
    # 从优化后的chunks目录加载
    for chunk_file in chunks_dir.glob("*.json"):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            doc_id = data.get('document_id', chunk_file.stem)
            chunks = data.get('chunks', [])
            if chunks:
                all_chunks[doc_id] = chunks
    
    # 如果优化目录不存在，从原始chunks目录加载
    if not all_chunks:
        chunks_dir = Path(__file__).parent.parent.parent / "data" / "chunks"
        for chunk_file in chunks_dir.glob("*.json"):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                doc_id = data.get('document_id', chunk_file.stem)
                chunks = data.get('chunks', [])
                if chunks:
                    all_chunks[doc_id] = chunks
    
    return all_chunks


def extract_key_info(content: str) -> List[str]:
    """从内容中提取关键信息"""
    key_info = []
    
    # 提取数字相关的信息
    import re
    numbers = re.findall(r'\d+(?:\.\d+)?(?:元|天|小时|次|个|人|月|年|%)', content)
    key_info.extend(numbers[:5])
    
    # 提取关键短语
    keywords = re.findall(r'[\u4e00-\u9fa5]{2,10}(?:制度|规定|要求|标准|流程|条件)', content)
    key_info.extend(keywords[:5])
    
    return key_info


def generate_question_for_chunk(chunk: Dict, doc_name: str) -> Tuple[str, str]:
    """为chunk生成测试问题和相关内容"""
    content = chunk.get('content', '')
    chunk_id = chunk.get('id', '')
    
    # 提取关键信息
    key_info = extract_key_info(content)
    
    # 根据内容类型生成不同类型的问题
    questions = []
    
    # 1. 数值类问题
    import re
    numbers = re.findall(r'(\d+(?:\.\d+)?)(元|天|小时|次|个|人|月|年|%)', content)
    for num, unit in numbers[:2]:
        if '元' in unit:
            questions.append(f"{doc_name}中规定的金额{num}元是用于什么情况？")
        elif '天' in unit:
            questions.append(f"{doc_name}中提到的{num}天是指什么期限？")
        elif '%' in unit:
            questions.append(f"{doc_name}中{num}%的比例是如何规定的？")
        elif '月' in unit:
            questions.append(f"{doc_name}中{num}个月的规定是什么？")
    
    # 2. 条件类问题
    conditions = re.findall(r'(?:如果|若|当|凡)[\u4e00-\u9fa5，、；]+', content)
    for cond in conditions[:2]:
        questions.append(f"{cond.strip()}，按照{doc_name}应如何处理？")
    
    # 3. 标准类问题
    standards = re.findall(r'(?:标准|要求|规定|条件)[：:][\u4e00-\u9fa5]+', content)
    for std in standards[:2]:
        questions.append(f"{doc_name}中的{std}是什么？")
    
    # 4. 流程类问题
    if '流程' in content or '步骤' in content:
        questions.append(f"根据{doc_name}，相关流程是怎样的？")
    
    # 5. 通用问题
    if len(content) > 50:
        # 从内容中提取一个关键句子作为问题基础
        sentences = re.split(r'[。；]', content)
        for sent in sentences:
            if len(sent) > 10 and len(sent) < 100:
                questions.append(f"{sent.strip()}的相关规定是什么？")
                break
    
    # 选择一个问题
    if questions:
        question = random.choice(questions)
    else:
        # 使用内容的前50个字符生成问题
        prefix = content[:50].strip()
        question = f"关于{prefix}...，{doc_name}是如何规定的？"
    
    return question, content


def generate_test_cases(all_chunks: Dict[str, List[Dict]], target_count: int = 200) -> Dict:
    """生成测试用例"""
    test_cases = {}
    case_id = 1
    
    # 为每个文档生成测试用例
    for doc_name, chunks in all_chunks.items():
        # 每个文档生成多个测试用例
        num_cases = max(5, len(chunks) // 3)  # 每个chunk约1/3概率被选中
        
        selected_chunks = random.sample(chunks, min(num_cases, len(chunks)))
        
        for chunk in selected_chunks:
            question, relevant_content = generate_question_for_chunk(chunk, doc_name)
            
            # 查找相似的其他chunks作为补充
            similar_chunks = []
            chunk_id = chunk.get('id', '')
            
            # 简单的内容相似度：基于文档名和关键词
            for other_chunk in chunks:
                if other_chunk.get('id') != chunk_id:
                    other_content = other_chunk.get('content', '')
                    # 简单的关键词重叠检测
                    common_words = set(question) & set(other_content[:200])
                    if len(common_words) > 5:
                        similar_chunks.append(other_chunk.get('id'))
            
            test_cases[f"ret_{case_id:03d}"] = {
                "query": question,
                "relevant_chunks": [chunk_id] + similar_chunks[:2],  # 主chunk + 最多2个相似chunk
                "relevant_content": relevant_content[:500]  # 截取相关内容
            }
            case_id += 1
            
            if case_id > target_count:
                break
        
        if case_id > target_count:
            break
    
    return test_cases


def generate_additional_cases(all_chunks: Dict[str, List[Dict]], existing_cases: Dict, target_count: int = 250) -> Dict:
    """生成额外的测试用例（跨文档、复杂查询）"""
    additional_cases = {}
    case_id = len(existing_cases) + 1
    
    # 1. 生成跨文档关联问题
    doc_names = list(all_chunks.keys())
    
    cross_doc_patterns = [
        "试用期",  # 涉及员工管理和劳动合同
        "请假",    # 涉及考勤和休假
        "报销",    # 涉及财务制度
        "考核",    # 涉及绩效管理
        "离职",    # 涉及员工管理
        "培训",    # 涉及发展和培训
    ]
    
    for pattern in cross_doc_patterns:
        related_chunks = []
        for doc_name, chunks in all_chunks.items():
            for chunk in chunks:
                content = chunk.get('content', '')
                if pattern in content:
                    related_chunks.append({
                        'id': chunk.get('id'),
                        'doc': doc_name,
                        'content': content[:300]
                    })
        
        if len(related_chunks) >= 2:
            # 生成跨文档问题
            question = f"关于{pattern}的规定有哪些？"
            relevant_chunk_ids = [c['id'] for c in related_chunks[:3]]
            relevant_content = " ".join([c['content'] for c in related_chunks[:2]])
            
            additional_cases[f"ret_{case_id:03d}"] = {
                "query": question,
                "relevant_chunks": relevant_chunk_ids,
                "relevant_content": relevant_content[:500]
            }
            case_id += 1
            
            if case_id > target_count:
                break
    
    # 2. 生成数值查询问题
    for doc_name, chunks in all_chunks.items():
        for chunk in chunks[:3]:  # 每个文档检查前3个chunk
            content = chunk.get('content', '')
            
            # 提取数值
            import re
            numbers = re.findall(r'(\d+(?:\.\d+)?)(元|天|小时|次|个|人|月|年|%)', content)
            
            for num, unit in numbers[:1]:
                question = f"{doc_name}中涉及{num}{unit}的具体规定是什么？"
                
                additional_cases[f"ret_{case_id:03d}"] = {
                    "query": question,
                    "relevant_chunks": [chunk.get('id')],
                    "relevant_content": content[:500]
                }
                case_id += 1
                
                if case_id > target_count:
                    break
            
            if case_id > target_count:
                break
        
        if case_id > target_count:
            break
    
    return additional_cases


def main():
    print("=" * 70)
    print("Ground Truth 生成器 v2")
    print("=" * 70)
    
    # 加载嵌入模型
    print("\n[1/4] 加载嵌入模型...")
    if not embedding_service.is_loaded():
        embedding_service.load_model(
            EmbeddingConfig(
                model_type=EmbeddingModelType.BGE,
                model_name=settings.embedding_model_name,
                device="cpu",
            )
        )
    print(f"  ✅ 模型: {settings.embedding_model_name}")
    
    # 加载所有chunk
    print("\n[2/4] 加载chunk数据...")
    chunks_dir = Path(__file__).parent.parent.parent / "data" / "chunks_optimized"
    all_chunks = load_all_chunks(chunks_dir)
    
    total_chunks = sum(len(chunks) for chunks in all_chunks.values())
    print(f"  ✅ 加载 {len(all_chunks)} 个文档，{total_chunks} 个chunks")
    
    # 生成基础测试用例
    print("\n[3/4] 生成基础测试用例...")
    test_cases = generate_test_cases(all_chunks, target_count=180)
    print(f"  ✅ 生成 {len(test_cases)} 个基础用例")
    
    # 生成额外测试用例
    print("\n[4/4] 生成额外测试用例...")
    additional_cases = generate_additional_cases(all_chunks, test_cases, target_count=250)
    test_cases.update(additional_cases)
    print(f"  ✅ 总计 {len(test_cases)} 个测试用例")
    
    # 验证chunk_id存在性
    print("\n[验证] 检查chunk_id...")
    all_chunk_ids = set()
    for chunks in all_chunks.values():
        for chunk in chunks:
            all_chunk_ids.add(chunk.get('id'))
    
    missing_count = 0
    for case_id, case_data in test_cases.items():
        for chunk_id in case_data.get('relevant_chunks', []):
            if chunk_id not in all_chunk_ids:
                missing_count += 1
                print(f"  ⚠️ 缺失: {chunk_id}")
    
    if missing_count == 0:
        print("  ✅ 所有chunk_id验证通过")
    
    # 保存结果
    output = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_cases": len(test_cases),
            "total_chunks": total_chunks,
            "documents": list(all_chunks.keys()),
            "chunk_id_validation": "passed" if missing_count == 0 else f"failed_{missing_count}_missing"
        },
        "ground_truth": test_cases
    }
    
    output_file = Path(__file__).parent.parent.parent.parent / "retrieval_test_cases_ground_truth_v2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Ground Truth已保存到: {output_file}")
    
    # 统计信息
    print("\n" + "=" * 70)
    print("统计信息")
    print("=" * 70)
    
    # 按文档统计
    doc_stats = {}
    for case_id, case_data in test_cases.items():
        for chunk_id in case_data.get('relevant_chunks', []):
            doc_name = chunk_id.rsplit('_chunk_', 1)[0]
            doc_stats[doc_name] = doc_stats.get(doc_name, 0) + 1
    
    print("\n各文档测试用例数:")
    for doc_name, count in sorted(doc_stats.items(), key=lambda x: -x[1])[:10]:
        print(f"  {doc_name}: {count}")


if __name__ == "__main__":
    main()
