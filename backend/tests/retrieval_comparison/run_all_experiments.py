"""
检索对比实验主运行脚本

按顺序执行两个实验：
1. 实验一：纯向量检索 vs RRF融合检索
2. 实验二：无精排 vs BGE Reranker精排

每个实验内部串行执行，防止内存溢出
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.retrieval_comparison.evaluation_utils import clear_gpu_memory, wait_for_memory


def run_experiment_1():
    """运行实验一：纯向量检索 vs RRF融合检索"""
    print("\n" + "#"*70)
    print("# 实验一：纯向量检索 vs RRF融合检索准确率对比")
    print("#"*70 + "\n")
    
    from tests.retrieval_comparison.experiment1_vector_vs_rrf import main as exp1_main
    
    return exp1_main()


def run_experiment_2():
    """运行实验二：无精排 vs BGE Reranker精排"""
    print("\n" + "#"*70)
    print("# 实验二：无精排 vs BGE Reranker精排准确率对比")
    print("#"*70 + "\n")
    
    from tests.retrieval_comparison.experiment2_reranker_comparison import main as exp2_main
    
    return exp2_main()


def main():
    """主函数"""
    print("="*70)
    print("RAG检索系统准确率对比实验")
    print("="*70)
    
    start_time = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"实验开始时间: {start_datetime}")
    
    # ==================== 实验一 ====================
    print("\n" + "="*70)
    print("开始执行实验一...")
    print("="*70)
    
    try:
        exp1_results = run_experiment_1()
        print("\n实验一执行成功！")
    except Exception as e:
        print(f"\n实验一执行失败: {e}")
        import traceback
        traceback.print_exc()
        exp1_results = None
    
    # 等待资源释放
    print("\n实验一完成，等待资源释放...")
    clear_gpu_memory()
    wait_for_memory(10)  # 等待10秒
    
    # ==================== 实验二 ====================
    print("\n" + "="*70)
    print("开始执行实验二...")
    print("="*70)
    
    try:
        exp2_results = run_experiment_2()
        print("\n实验二执行成功！")
    except Exception as e:
        print(f"\n实验二执行失败: {e}")
        import traceback
        traceback.print_exc()
        exp2_results = None
    
    # ==================== 总结 ====================
    total_time = time.time() - start_time
    end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "="*70)
    print("所有实验完成！")
    print("="*70)
    print(f"结束时间: {end_datetime}")
    print(f"总耗时: {total_time/60:.2f} 分钟")
    
    # 打印报告位置
    print("\n报告位置:")
    print(f"  - 实验一: {project_root.parent}/test_reports/纯向量与融合向量准确率对比.md")
    print(f"  - 实验二: {project_root.parent}/test_reports/使用二次精排准确率对比.md")
    
    return {
        'experiment_1': exp1_results,
        'experiment_2': exp2_results,
        'total_time': total_time
    }


if __name__ == "__main__":
    main()
