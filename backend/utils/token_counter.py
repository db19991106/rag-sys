"""
Token 计数工具模块
参考 RAGFlow 的实现，使用 tiktoken 进行精确的 Token 计算
"""

try:
    import tiktoken
    HAS_TIKTOKEN = True
    # 使用 cl100k_base 编码器（GPT-3.5-turbo, GPT-4 使用的编码器）
    ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    HAS_TIKTOKEN = False
    ENCODER = None


def num_tokens_from_string(string: str) -> int:
    """
    计算文本字符串的 Token 数量
    参考 RAGFlow 的实现
    
    Args:
        string: 输入文本
        
    Returns:
        Token 数量
    """
    if not string:
        return 0
    
    try:
        if HAS_TIKTOKEN:
            code_list = ENCODER.encode(string)
            return len(code_list)
    except Exception:
        pass
    
    # 降级方案：估算中英文 Token 数量
    return estimate_tokens(string)


def estimate_tokens(string: str) -> int:
    """
    估算 Token 数量的降级方案（当 tiktoken 不可用时）
    粗略估计：
    - 中文字符：约 1.5-2 个字符 = 1 token
    - 英文单词：约 1 个单词 = 1.3 tokens
    - 标点符号：约 1 个字符 = 0.5 token
    
    Args:
        string: 输入文本
        
    Returns:
        估算的 Token 数量
    """
    if not string:
        return 0
    
    chinese_count = 0
    english_word_count = 0
    punctuation_count = 0
    
    # 简单的分词
    words = string.split()
    for word in words:
        # 检查是否为中文
        chinese_chars = [c for c in word if '\u4e00' <= c <= '\u9fff']
        if chinese_chars:
            chinese_count += len(chinese_chars)
            # 检查是否包含英文
            english_chars = [c for c in word if 'a' <= c <= 'z' or 'A' <= c <= 'Z']
            if english_chars:
                english_word_count += 1
        else:
            # 检查是否为英文单词
            if any(c.isalpha() for c in word):
                english_word_count += 1
    
    # 估算标点符号
    for char in string:
        if char in '.,;:!?，。；！？、""''()[]{}<>':
            punctuation_count += 1
    
    # 估算 Token 数量
    chinese_tokens = chinese_count / 1.5  # 中文字符约 1.5 个 = 1 token
    english_tokens = english_word_count * 1.3  # 英文单词约 1.3 tokens
    punctuation_tokens = punctuation_count * 0.5  # 标点约 0.5 个 = 1 token
    
    return int(chinese_tokens + english_tokens + punctuation_tokens)