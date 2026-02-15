#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业绩效考核文档标准化处理脚本 - V2
功能：自动识别文档类型、提取元数据、生成标准RAG Chunk
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any


class PerformanceDocProcessor:
    """绩效考核文档处理器"""
    
    # 文档类型识别规则
    DOC_TYPE_RULES = {
        "财务型": {
            "keywords": ["财务部", "岗位业绩考核指标", "优", "良", "中", "差"],
            "structure": "岗位 × 指标矩阵，四档评分"
        },
        "HR型": {
            "keywords": ["人力资源部", "薪酬", "KPI", "权重"],
            "structure": "岗位序列 + 定量指标 + 扣分规则"
        },
        "生产型": {
            "keywords": ["生产部", "计分制", "质量奖", "现场奖"],
            "structure": "工序/工种 + 多维度奖金 + 行为细则"
        },
        "销售型": {
            "keywords": ["销售部", "业绩任务", "提成", "开发客户"],
            "structure": "业绩指标 + 行为指标 + 系数计算"
        },
        "研发型": {
            "keywords": ["研发部", "项目考核", "EVT", "DVT", "MP"],
            "structure": "项目阶段 + 部门协作 + 强制分布"
        },
        "行政型": {
            "keywords": ["行政部", "支持服务", "满意度"],
            "structure": "岗位分类 + 定性评价 + 360度反馈"
        }
    }
    
    # 部门映射
    DEPT_MAPPING = {
        "财务部": "财务",
        "人力资源部": "人力资源",
        "生产部": "生产",
        "销售部": "销售",
        "研发部": "研发",
        "行政部": "行政"
    }
    
    # 预定义的岗位列表（按部门）
    DEPT_POSITIONS = {
        "财务": ["会计核算", "出纳", "财务分析", "财务主管", "财务经理"],
        "人力资源": ["招聘专员", "培训专员", "薪酬专员", "HRBP", "人力资源主管", "人力资源经理"],
        "生产": ["操作工", "班组长", "车间主任", "质检员", "设备维护员"],
        "销售": ["销售代表", "高级销售", "销售主管", "区域经理", "销售总监"],
        "研发": ["初级工程师", "中级工程师", "高级工程师", "技术专家", "项目经理", "研发总监"],
        "行政": ["前台", "行政专员", "后勤管理员", "行政主管", "行政经理"]
    }
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir.parent / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.chunks = []
        
    def identify_doc_type(self, content: str, filename: str) -> Dict[str, str]:
        """识别文档类型和部门"""
        content_lower = content.lower()
        scores = {}
        
        for doc_type, rules in self.DOC_TYPE_RULES.items():
            score = 0
            for keyword in rules["keywords"]:
                if keyword.lower() in content_lower or keyword.lower() in filename.lower():
                    score += 1
            scores[doc_type] = score / len(rules["keywords"])
        
        # 选择得分最高的类型
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # 提取部门
        dept = "未知"
        for dept_name, dept_code in self.DEPT_MAPPING.items():
            if dept_name in content or dept_name in filename:
                dept = dept_code
                break
        
        return {
            "doc_type": best_type,
            "confidence": confidence,
            "department": dept
        }
    
    def extract_positions(self, content: str, dept: str) -> List[Dict[str, Any]]:
        """提取岗位信息"""
        positions = []
        
        # 首先尝试从预定义列表匹配
        if dept in self.DEPT_POSITIONS:
            for pos_name in self.DEPT_POSITIONS[dept]:
                # 在文档中查找该岗位
                if pos_name in content:
                    level = self.identify_position_level(pos_name)
                    positions.append({
                        "name": pos_name,
                        "level": level
                    })
        
        # 如果没找到，使用正则提取
        if not positions:
            positions = self.extract_positions_by_pattern(content)
        
        return positions if positions else [{"name": "通用岗位", "level": "未知"}]
    
    def extract_positions_by_pattern(self, content: str) -> List[Dict[str, Any]]:
        """使用正则表达式提取岗位"""
        positions = []
        
        # 匹配模式：中文序号 + 岗位名称 + "岗位"或"职位"
        # 例如：（一）会计核算岗位
        pattern1 = r'[（(]([一二三四五六七八九十]+)[）)]\s*([^\n]+?)(?:岗位|职位)'
        matches1 = re.findall(pattern1, content)
        
        for _, pos_name in matches1:
            pos_name = pos_name.strip()
            # 清理岗位名称
            pos_name = re.sub(r'[:：].*', '', pos_name).strip()
            if pos_name and len(pos_name) < 20:
                level = self.identify_position_level(pos_name)
                if not any(p["name"] == pos_name for p in positions):
                    positions.append({
                        "name": pos_name,
                        "level": level
                    })
        
        return positions
    
    def identify_position_level(self, position_name: str) -> str:
        """识别岗位级别"""
        levels = {
            "初级": ["初级", "助理", "专员", "操作工", "前台", "代表"],
            "中级": ["中级", "高级专员", "资深"],
            "高级": ["高级", "工程师", "分析师", "专家"],
            "主管": ["主管", "组长", "主任", "班长"],
            "经理": ["经理", "总监", "VP"]
        }
        
        for level, keywords in levels.items():
            for kw in keywords:
                if kw in position_name:
                    return level
        
        return "未知"
    
    def extract_assessment_type(self, content: str) -> str:
        """提取考核类型"""
        content = content.lower()
        
        # 优先级排序
        if any(kw in content for kw in ["evt", "dvt", "pvt", "mp", "项目考核", "项目阶段"]):
            return "项目"
        elif any(kw in content for kw in ["业绩", "kpi", "业绩指标", "销售额", "业绩任务"]):
            return "业绩"
        elif any(kw in content for kw in ["态度", "满意度", "360度"]):
            return "态度"
        elif any(kw in content for kw in ["行为", "行为指标", "协作", "配合"]):
            return "行为"
        elif any(kw in content for kw in ["能力", "胜任力", "专业技能"]):
            return "能力"
        
        return "综合"
    
    def extract_frequency(self, content: str) -> str:
        """提取考核周期"""
        # 项目周期优先级最高
        if any(kw in content for kw in ["项目周期", "项目阶段", "项目节点", "evt", "dvt"]):
            return "项目周期"
        elif any(kw in content for kw in ["月度", "每月", "月考核"]):
            return "月度"
        elif any(kw in content for kw in ["季度", "每季度", "季考核"]):
            return "季度"
        elif any(kw in content for kw in ["半年", "半年度"]):
            return "半年"
        elif any(kw in content for kw in ["年度", "每年", "年考核"]):
            return "年度"
        
        return "月度"
    
    def extract_grade_scale(self, content: str) -> str:
        """提取评分等级"""
        if re.search(r'优.*良.*中.*差', content):
            return "优/良/中/差"
        elif re.search(r'S级.*A级.*B级.*C级.*D级', content):
            return "S/A/B/C/D"
        elif re.search(r'[SABCD][级]', content):
            return "S/A/B/C/D"
        elif re.search(r'[12345][级]', content):
            return "五级制"
        else:
            return "百分制"
    
    def extract_evaluator(self, content: str) -> str:
        """提取考评人"""
        if "360度" in content or ("360" in content and "度" in content):
            return "360度"
        elif "跨部门" in content:
            return "跨部门"
        elif "部门负责人" in content:
            return "部门负责人"
        elif "直接上级" in content:
            return "直接上级"
        else:
            return "直接上级"
    
    def split_into_chunks(self, content: str, doc_info: Dict, positions: List[Dict]) -> List[Dict]:
        """将文档切分为chunks"""
        chunks = []
        
        # 按岗位切分
        for idx, position in enumerate(positions, 1):
            chunk_id = f"{doc_info['department']}_{idx:03d}"
            
            # 提取该岗位相关内容
            position_content = self.extract_position_content(content, position["name"])
            
            # 如果提取内容太短，使用通用内容
            if len(position_content) < 200:
                position_content = self.extract_generic_content(content, position["name"])
            
            # 清理内容
            cleaned_content = self.clean_content(position_content)
            
            chunk = {
                "metadata": {
                    "doc_source": doc_info["department"],
                    "position": position["name"],
                    "position_level": position["level"],
                    "assessment_type": self.extract_assessment_type(content),
                    "assessment_scope": "个人",
                    "weight_total": "100%",
                    "evaluator": self.extract_evaluator(content),
                    "frequency": self.extract_frequency(content),
                    "grade_scale": self.extract_grade_scale(content),
                    "chunk_id": chunk_id
                },
                "chunk_content": cleaned_content
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def extract_position_content(self, content: str, position_name: str) -> str:
        """提取特定岗位的内容"""
        # 查找岗位开始位置
        patterns = [
            rf'[（(][一二三四五六七八九十]+[）)]\s*{re.escape(position_name)}(?:岗位|职位)?',
            rf'\n\s*{re.escape(position_name)}(?:岗位|职位)?[:：]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                start = match.start()
                # 查找下一个岗位或章节开始位置
                next_pattern = r'\n\s*[（(][一二三四五六七八九十]+[）)]'
                next_match = re.search(next_pattern, content[start+10:])
                if next_match:
                    end = start + 10 + next_match.start()
                    return content[start:end]
                else:
                    # 查找下一个空行作为结束
                    following_content = content[start+500:]
                    if '\n\n' in following_content[:1000]:
                        end = start + 500 + following_content[:1000].find('\n\n')
                        return content[start:end]
                    return content[start:start+1500]
        
        # 如果找不到特定岗位内容，返回通用内容
        return self.extract_generic_content(content, position_name)
    
    def extract_generic_content(self, content: str, position_name: str) -> str:
        """提取通用内容（包含岗位名称的部分）"""
        lines = content.split('\n')
        result_lines = []
        capture = False
        
        for i, line in enumerate(lines):
            if position_name in line and len(line) < 100:
                capture = True
                result_lines.append(line)
            elif capture:
                # 检查是否是下一个岗位
                if re.match(r'\s*[（(][一二三四五六七八九十]+[）)]', line) and len(line) < 50:
                    break
                result_lines.append(line)
                # 限制长度
                if len('\n'.join(result_lines)) > 2000:
                    break
        
        if result_lines:
            return '\n'.join(result_lines)
        
        # 最后返回包含岗位名称的段落
        idx = content.find(position_name)
        if idx >= 0:
            start = max(0, idx - 200)
            end = min(len(content), idx + 1500)
            return content[start:end]
        
        return content[:1500]
    
    def clean_content(self, content: str) -> str:
        """清理内容，去除噪音"""
        # 移除多余空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        # 移除页眉页脚类内容
        content = re.sub(r'\d+/\d+页', '', content)
        content = re.sub(r'第\s*\d+\s*页', '', content)
        # 移除特殊字符
        content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', content)
        # 移除多余空格
        content = re.sub(r'[ \t]+', ' ', content)
        return content.strip()
    
    def process_file(self, filepath: Path) -> List[Dict]:
        """处理单个文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 识别文档类型
        doc_info = self.identify_doc_type(content, filepath.name)
        print(f"处理文件: {filepath.name}")
        print(f"  识别类型: {doc_info['doc_type']} (置信度: {doc_info['confidence']:.2f})")
        print(f"  所属部门: {doc_info['department']}")
        
        # 提取岗位
        positions = self.extract_positions(content, doc_info['department'])
        print(f"  提取岗位数: {len(positions)}")
        for pos in positions:
            print(f"    - {pos['name']} ({pos['level']})")
        
        # 切分chunks
        chunks = self.split_into_chunks(content, doc_info, positions)
        
        return chunks
    
    def process_all(self):
        """处理所有文档"""
        all_chunks = []
        
        for filepath in sorted(self.data_dir.glob("*.txt")):
            chunks = self.process_file(filepath)
            all_chunks.extend(chunks)
            print()
        
        self.chunks = all_chunks
        
        # 保存结果
        output_file = self.output_dir / "standardized_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 处理完成！")
        print(f"   共生成 {len(all_chunks)} 个标准Chunk")
        print(f"   输出文件: {output_file}")
        
        return all_chunks


def main():
    """主函数"""
    data_dir = "/root/autodl-tmp/rag/test_chunking/data"
    
    processor = PerformanceDocProcessor(data_dir)
    chunks = processor.process_all()
    
    # 打印第一个chunk作为示例
    if chunks:
        print("\n" + "="*80)
        print("示例Chunk（第一个）：")
        print("="*80)
        print(json.dumps(chunks[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
