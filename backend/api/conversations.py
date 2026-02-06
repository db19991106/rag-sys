from fastapi import APIRouter, HTTPException
from utils.logger import logger

router = APIRouter(prefix="/conversations", tags=["对话管理"])

# 导入conversation_manager
from services.conversation_manager import conversation_manager

@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    删除指定对话
    
    Args:
        conversation_id: 对话ID
    """
    try:
        success = conversation_manager.delete_conversation(conversation_id)
        if success:
            logger.info(f"对话已删除: {conversation_id}")
            return {
                "success": True,
                "message": "对话删除成功"
            }
        else:
            logger.info(f"尝试删除不存在的对话: {conversation_id}")
            # 对于不存在的对话，返回200但success为false
            return {
                "success": False,
                "message": "对话不存在"
            }
    except Exception as e:
        logger.error(f"删除对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除对话失败: {str(e)}")

@router.get("/")
async def list_conversations():
    """
    获取当前用户的对话列表
    """
    try:
        # 这里需要用户认证，暂时返回所有对话
        conversations = list(conversation_manager.conversations.values())
        return {
            "success": True,
            "data": [
                {
                    "id": conv.id,
                    "title": conv.title,
                    "message_count": len(conv.messages),
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat()
                }
                for conv in conversations
            ]
        }
    except Exception as e:
        logger.error(f"获取对话列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取对话列表失败: {str(e)}")