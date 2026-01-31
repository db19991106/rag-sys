from typing import Optional, Dict, Any, List
import re
import html
from utils.logger import logger


class DocumentCleaner:
    """文档数据清洗器 - 提供多种数据清洗方法"""

    def __init__(self):
        pass

    def clean(
        self,
        content: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        清洗文档内容

        Args:
            content: 原始文档内容
            config: 清洗配置，包含以下选项：
                - remove_whitespace: 是否移除多余空白 (默认: True)
                - remove_special_chars: 是否移除特殊字符 (默认: False)
                - normalize_quotes: 是否规范化引号 (默认: True)
                - remove_html_tags: 是否移除HTML标签 (默认: False)
                - remove_urls: 是否移除URL (默认: False)
                - remove_emails: 是否移除邮箱 (默认: False)
                - remove_numbers: 是否移除数字 (默认: False)
                - remove_chinese_punctuation: 是否移除中文标点 (默认: False)
                - remove_english_punctuation: 是否移除英文标点 (默认: False)
                - normalize_whitespace: 是否规范化空白字符 (默认: True)
                - remove_duplicate_lines: 是否移除重复行 (默认: False)
                - trim_lines: 是否去除每行首尾空白 (默认: True)
                - custom_regex: 自定义正则表达式替换规则 [{"pattern": "", "replacement": ""}]
                - remove_empty_lines: 是否移除空行 (默认: True)
                - convert_full_to_half: 是否全角转半角 (默认: True)

        Returns:
            清洗后的文档内容
        """
        if not content:
            return content

        if config is None:
            config = {}

        cleaned = content

        # 1. 移除HTML标签
        if config.get('remove_html_tags', False):
            cleaned = self._remove_html_tags(cleaned)

        # 2. 移除多余空白
        if config.get('remove_whitespace', True):
            cleaned = self._remove_extra_whitespace(cleaned)

        # 3. 规范化空白字符
        if config.get('normalize_whitespace', True):
            cleaned = self._normalize_whitespace(cleaned)

        # 4. 规范化引号
        if config.get('normalize_quotes', True):
            cleaned = self._normalize_quotes(cleaned)

        # 5. 全角转半角
        if config.get('convert_full_to_half', True):
            cleaned = self._convert_full_to_half(cleaned)

        # 6. 移除特殊字符
        if config.get('remove_special_chars', False):
            cleaned = self._remove_special_chars(cleaned)

        # 7. 移除URL
        if config.get('remove_urls', False):
            cleaned = self._remove_urls(cleaned)

        # 8. 移除邮箱
        if config.get('remove_emails', False):
            cleaned = self._remove_emails(cleaned)

        # 9. 移除数字
        if config.get('remove_numbers', False):
            cleaned = self._remove_numbers(cleaned)

        # 10. 移除中文标点
        if config.get('remove_chinese_punctuation', False):
            cleaned = self._remove_chinese_punctuation(cleaned)

        # 11. 移除英文标点
        if config.get('remove_english_punctuation', False):
            cleaned = self._remove_english_punctuation(cleaned)

        # 12. 去除每行首尾空白
        if config.get('trim_lines', True):
            cleaned = self._trim_lines(cleaned)

        # 13. 移除空行
        if config.get('remove_empty_lines', True):
            cleaned = self._remove_empty_lines(cleaned)

        # 14. 移除重复行
        if config.get('remove_duplicate_lines', False):
            cleaned = self._remove_duplicate_lines(cleaned)

        # 15. 自定义正则替换
        if config.get('custom_regex'):
            cleaned = self._apply_custom_regex(cleaned, config['custom_regex'])

        logger.info(f"文档清洗完成，原长度: {len(content)}，清洗后长度: {len(cleaned)}")
        return cleaned

    def _remove_html_tags(self, content: str) -> str:
        """移除HTML标签"""
        # 使用正则表达式移除HTML标签
        clean = re.sub(r'<[^>]+>', '', content)
        # 解码HTML实体
        clean = html.unescape(clean)
        return clean

    def _remove_extra_whitespace(self, content: str) -> str:
        """移除多余的空白字符"""
        # 将多个空格替换为一个空格
        clean = re.sub(r' +', ' ', content)
        # 将多个制表符替换为一个空格
        clean = re.sub(r'\t+', ' ', content)
        return clean

    def _normalize_whitespace(self, content: str) -> str:
        """规范化空白字符"""
        # 将各种空白字符统一为空格
        clean = re.sub(r'[\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]', ' ', content)
        return clean

    def _normalize_quotes(self, content: str) -> str:
        """规范化引号"""
        # 中文引号转英文引号
        clean = content.replace('"', '"').replace('"', '"')
        clean = clean.replace(''', "'").replace(''', "'")
        # 统一为英文引号
        clean = clean.replace('"', '"').replace('"', '"')
        clean = clean.replace(''', "'").replace(''', "'")
        return clean

    def _convert_full_to_half(self, content: str) -> str:
        """全角字符转半角字符"""
        clean = content
        # 全角数字
        full_to_half_digits = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9'
        }
        for full, half in full_to_half_digits.items():
            clean = clean.replace(full, half)

        # 全角字母
        clean = clean.replace('Ａ', 'A').replace('Ｂ', 'B').replace('Ｃ', 'C')
        clean = clean.replace('Ｄ', 'D').replace('Ｅ', 'E').replace('Ｆ', 'F')
        clean = clean.replace('Ｇ', 'G').replace('Ｈ', 'H').replace('Ｉ', 'I')
        clean = clean.replace('Ｊ', 'J').replace('Ｋ', 'K').replace('Ｌ', 'L')
        clean = clean.replace('Ｍ', 'M').replace('Ｎ', 'N').replace('Ｏ', 'O')
        clean = clean.replace('Ｐ', 'P').replace('Ｑ', 'Q').replace('Ｒ', 'R')
        clean = clean.replace('Ｓ', 'S').replace('Ｔ', 'T').replace('Ｕ', 'U')
        clean = clean.replace('Ｖ', 'V').replace('Ｗ', 'W').replace('Ｘ', 'X')
        clean = clean.replace('Ｙ', 'Y').replace('Ｚ', 'Z')
        clean = clean.replace('ａ', 'a').replace('ｂ', 'b').replace('ｃ', 'c')
        clean = clean.replace('ｄ', 'd').replace('ｅ', 'e').replace('ｆ', 'f')
        clean = clean.replace('ｇ', 'g').replace('ｈ', 'h').replace('ｉ', 'i')
        clean = clean.replace('ｊ', 'j').replace('ｋ', 'k').replace('ｌ', 'l')
        clean = clean.replace('ｍ', 'm').replace('ｎ', 'n').replace('ｏ', 'o')
        clean = clean.replace('ｐ', 'p').replace('ｑ', 'q').replace('ｒ', 'r')
        clean = clean.replace('ｓ', 's').replace('ｔ', 't').replace('ｕ', 'u')
        clean = clean.replace('ｖ', 'v').replace('ｗ', 'w').replace('ｘ', 'x')
        clean = clean.replace('ｙ', 'y').replace('ｚ', 'z')

        # 全角标点
        clean = clean.replace('，', ',').replace('。', '.')
        clean = clean.replace('；', ';').replace('：', ':')
        clean = clean.replace('？', '?').replace('！', '!')
        clean = clean.replace('（', '(').replace('）', ')')
        clean = clean.replace('【', '[').replace('】', ']')
        clean = clean.replace('｛', '{').replace('｝', '}')

        return clean

    def _remove_special_chars(self, content: str) -> str:
        """移除特殊字符，保留中文、英文、数字和基本标点"""
        clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,!?;:()\[\]{}"\'-]', '', content)
        return clean

    def _remove_urls(self, content: str) -> str:
        """移除URL"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        clean = re.sub(url_pattern, '', content)
        return clean

    def _remove_emails(self, content: str) -> str:
        """移除邮箱地址"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        clean = re.sub(email_pattern, '', content)
        return clean

    def _remove_numbers(self, content: str) -> str:
        """移除数字"""
        clean = re.sub(r'\d+', '', content)
        return clean

    def _remove_chinese_punctuation(self, content: str) -> str:
        """移除中文标点符号"""
        chinese_punctuation = '，。！？；：""''（）【】《》〈〉「」『』【】｛｝、·…—'
        for char in chinese_punctuation:
            content = content.replace(char, '')
        return content

    def _remove_english_punctuation(self, content: str) -> str:
        """移除英文标点符号"""
        english_punctuation = '.,!?;:"\'()[]{}-'
        for char in english_punctuation:
            content = content.replace(char, '')
        return content

    def _trim_lines(self, content: str) -> str:
        """去除每行首尾空白"""
        lines = content.split('\n')
        trimmed_lines = [line.strip() for line in lines]
        return '\n'.join(trimmed_lines)

    def _remove_empty_lines(self, content: str) -> str:
        """移除空行"""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        return '\n'.join(non_empty_lines)

    def _remove_duplicate_lines(self, content: str) -> str:
        """移除重复行"""
        lines = content.split('\n')
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        return '\n'.join(unique_lines)

    def _apply_custom_regex(self, content: str, rules: List[Dict[str, str]]) -> str:
        """应用自定义正则替换规则"""
        clean = content
        for rule in rules:
            pattern = rule.get('pattern', '')
            replacement = rule.get('replacement', '')
            if pattern:
                try:
                    clean = re.sub(pattern, replacement, clean)
                except re.error as e:
                    logger.error(f"自定义正则替换失败: {pattern}, 错误: {str(e)}")
        return clean

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认清洗配置"""
        return {
            'remove_whitespace': True,
            'remove_special_chars': False,
            'normalize_quotes': True,
            'remove_html_tags': False,
            'remove_urls': False,
            'remove_emails': False,
            'remove_numbers': False,
            'remove_chinese_punctuation': False,
            'remove_english_punctuation': False,
            'normalize_whitespace': True,
            'remove_duplicate_lines': False,
            'trim_lines': True,
            'custom_regex': None,
            'remove_empty_lines': True,
            'convert_full_to_half': True
        }

    def get_preset_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取预设的清洗配置"""
        return {
            'minimal': {
                'remove_whitespace': True,
                'remove_special_chars': False,
                'normalize_quotes': True,
                'remove_html_tags': False,
                'remove_urls': False,
                'remove_emails': False,
                'remove_numbers': False,
                'remove_chinese_punctuation': False,
                'remove_english_punctuation': False,
                'normalize_whitespace': True,
                'remove_duplicate_lines': False,
                'trim_lines': True,
                'custom_regex': None,
                'remove_empty_lines': True,
                'convert_full_to_half': True
            },
            'aggressive': {
                'remove_whitespace': True,
                'remove_special_chars': True,
                'normalize_quotes': True,
                'remove_html_tags': True,
                'remove_urls': True,
                'remove_emails': True,
                'remove_numbers': False,
                'remove_chinese_punctuation': False,
                'remove_english_punctuation': False,
                'normalize_whitespace': True,
                'remove_duplicate_lines': True,
                'trim_lines': True,
                'custom_regex': None,
                'remove_empty_lines': True,
                'convert_full_to_half': True
            },
            'text_only': {
                'remove_whitespace': True,
                'remove_special_chars': True,
                'normalize_quotes': True,
                'remove_html_tags': True,
                'remove_urls': True,
                'remove_emails': True,
                'remove_numbers': True,
                'remove_chinese_punctuation': False,
                'remove_english_punctuation': False,
                'normalize_whitespace': True,
                'remove_duplicate_lines': True,
                'trim_lines': True,
                'custom_regex': None,
                'remove_empty_lines': True,
                'convert_full_to_half': True
            },
            'html_clean': {
                'remove_whitespace': True,
                'remove_special_chars': False,
                'normalize_quotes': True,
                'remove_html_tags': True,
                'remove_urls': False,
                'remove_emails': False,
                'remove_numbers': False,
                'remove_chinese_punctuation': False,
                'remove_english_punctuation': False,
                'normalize_whitespace': True,
                'remove_duplicate_lines': False,
                'trim_lines': True,
                'custom_regex': None,
                'remove_empty_lines': True,
                'convert_full_to_half': True
            }
        }


# 全局文档清洗器实例
document_cleaner = DocumentCleaner()