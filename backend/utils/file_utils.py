import os
import hashlib
import magic
from pathlib import Path
from typing import Optional, Tuple
from config import settings


def safe_join(base_path: str, *paths) -> str:
    """
    安全的路径拼接，防止路径遍历攻击
    
    Args:
        base_path: 基础路径
        *paths: 要拼接的路径组件
        
    Returns:
        安全的绝对路径
        
    Raises:
        ValueError: 如果路径试图跳出基础目录
    """
    # 将基础路径转换为绝对路径
    base = os.path.abspath(base_path)
    
    # 拼接所有路径组件
    full_path = os.path.abspath(os.path.join(base, *paths))
    
    # 验证最终路径是否在基础路径下
    if not full_path.startswith(base):
        raise ValueError(f"路径遍历检测: 路径 '{os.path.join(*paths)}' 试图跳出基础目录")
    
    return full_path


def safe_filename(filename: str) -> str:
    """
    安全的文件名处理，移除危险字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        安全的文件名
    """
    # 移除路径分隔符
    filename = filename.replace('/', '').replace('\\', '')
    
    # 移除路径遍历序列
    filename = filename.replace('..', '')
    
    # 只保留安全的字符（字母、数字、下划线、短横线、点）
    import re
    filename = re.sub(r'[^\w\-.]', '_', filename)
    
    # 如果文件名为空，使用默认名称
    if not filename or filename == '.':
        filename = "unnamed_file"
    
    return filename


def get_file_extension(filename: str) -> str:
    """获取文件扩展名"""
    return Path(filename).suffix.lower()


def is_allowed_file(filename: str) -> bool:
    """检查文件是否允许上传"""
    ext = get_file_extension(filename)
    return ext in settings.allowed_extensions


def get_file_size(file_path: str) -> int:
    """获取文件大小（字节）"""
    return os.path.getsize(file_path)


def generate_file_id(filename: str) -> str:
    """生成文件唯一ID"""
    content = f"{filename}_{os.urandom(8).hex()}"
    return hashlib.md5(content.encode()).hexdigest()


def get_mime_type(file_path: str) -> str:
    """获取文件MIME类型"""
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
    except Exception:
        return "application/octet-stream"


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def ensure_upload_dir() -> Path:
    """确保上传目录存在"""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def save_upload_file(file_content: bytes, filename: str) -> Tuple[str, str]:
    """保存上传的文件"""
    upload_dir = ensure_upload_dir()
    file_id = generate_file_id(filename)

    # 使用原始文件名保存
    file_path = upload_dir / filename

    with open(file_path, 'wb') as f:
        f.write(file_content)

    return str(file_id), str(file_path)


def delete_file(filename: str) -> bool:
    """删除文件"""
    upload_dir = ensure_upload_dir()
    file_path = upload_dir / filename

    if file_path.exists():
        try:
            file_path.unlink()
            return True
        except Exception:
            return False
    return False