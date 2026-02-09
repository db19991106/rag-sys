#!/usr/bin/env python3
"""
日志轮转测试脚本
"""
import logging
import os
import time
from utils.logger import setup_logger


def test_log_rotation():
    """
    测试日志轮转功能
    """
    print("=== 开始日志轮转测试 ===")
    
    # 设置日志记录器
    logger = setup_logger("test_logger")
    
    # 获取日志文件路径
    log_file = None
    for handler in logger.handlers:
        if hasattr(handler, 'baseFilename'):
            log_file = getattr(handler, 'baseFilename')
            break
    
    if not log_file:
        print("错误: 找不到日志文件路径")
        return
    
    print(f"测试日志文件: {log_file}")
    
    # 记录初始日志文件大小
    if os.path.exists(log_file):
        initial_size = os.path.getsize(log_file)
        print(f"初始文件大小: {initial_size} 字节")
    else:
        initial_size = 0
        print("初始文件大小: 0 字节")
    
    # 测试1: 正常日志输出
    print("\n测试1: 正常日志输出")
    for i in range(10):
        logger.info(f"测试日志 {i+1}: 这是一条测试消息，用于验证日志系统的正常输出功能")
        time.sleep(0.1)
    
    # 检查文件大小
    current_size = os.path.getsize(log_file)
    print(f"测试1后文件大小: {current_size} 字节")
    print(f"增加了: {current_size - initial_size} 字节")
    
    # 测试2: 生成大量日志以触发大小轮转
    print("\n测试2: 生成大量日志以触发大小轮转")
    print("开始生成大量日志...")
    
    start_time = time.time()
    log_count = 0
    
    # 生成足够的日志来触发轮转（假设5MB限制）
    large_message = "X" * 1000  # 每条日志1000字节
    
    while time.time() - start_time < 30:  # 最多运行30秒
        for i in range(100):
            logger.info(f"大量测试日志 {log_count + i + 1}: {large_message}")
        log_count += 100
        
        # 检查文件大小
        current_size = os.path.getsize(log_file)
        print(f"已生成 {log_count} 条日志，文件大小: {current_size/1024/1024:.2f} MB")
        
        # 检查是否有备份文件生成
        log_dir = os.path.dirname(log_file)
        backup_files = [f for f in os.listdir(log_dir) if f.startswith('app.log.')]
        if backup_files:
            print(f"发现备份文件: {backup_files}")
            break
        
        time.sleep(1)
    
    # 测试3: 检查备份文件数量
    print("\n测试3: 检查备份文件数量")
    log_dir = os.path.dirname(log_file)
    backup_files = [f for f in os.listdir(log_dir) if f.startswith('app.log.')]
    print(f"备份文件数量: {len(backup_files)}")
    print(f"备份文件: {backup_files}")
    
    # 测试4: 检查日志文件格式
    print("\n测试4: 检查日志文件格式")
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:5]  # 读取前5行
    
    print("日志格式样例:")
    for line in lines:
        print(line.strip())
    
    # 测试5: 检查错误处理
    print("\n测试5: 检查错误处理")
    try:
        # 测试异常情况下的日志记录
        logger.error("测试错误日志: 这是一条错误消息")
        logger.warning("测试警告日志: 这是一条警告消息")
        logger.debug("测试调试日志: 这是一条调试消息")
        print("错误处理测试成功: 各种级别的日志都能正常记录")
    except Exception as e:
        print(f"错误处理测试失败: {str(e)}")
    
    print("\n=== 日志轮转测试完成 ===")


if __name__ == "__main__":
    test_log_rotation()
