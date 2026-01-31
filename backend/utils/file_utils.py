import os
import hashlib
import magic
from pathlib import Path
from typing import Optional, Tuple
from config import settings


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