import React from 'react';
import type { Conversation } from '../contexts/ChatContext';

interface ConversationListProps {
  conversations: Conversation[];
  currentConversationId: string;
  onSelect: (id: string) => void;
  onCreateNew: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}

export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  currentConversationId,
  onSelect,
  onCreateNew,
  onRename,
  onDelete,
}) => {
  const [editingId, setEditingId] = React.useState<string | null>(null);
  const [editTitle, setEditTitle] = React.useState('');

  const handleStartEdit = (e: React.MouseEvent, conv: Conversation) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };

  const handleSaveEdit = () => {
    if (editingId && editTitle.trim()) {
      onRename(editingId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveEdit();
    } else if (e.key === 'Escape') {
      setEditingId(null);
      setEditTitle('');
    }
  };

  return (
    <div className="chat-sidebar">
      <div className="sidebar-header">
        <h3>历史对话</h3>
        <button className="new-conversation-btn" onClick={onCreateNew} title="创建新对话">
          <i className="fas fa-plus"></i>
        </button>
      </div>
      
      <div className="conversation-list">
        {conversations.map((conversation) => (
          <div
            key={conversation.id}
            className={`conversation-item ${conversation.id === currentConversationId ? 'active' : ''}`}
            onClick={() => onSelect(conversation.id)}
          >
            <div className="conversation-info">
              {editingId === conversation.id ? (
                <input
                  type="text"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onBlur={handleSaveEdit}
                  onKeyDown={handleKeyDown}
                  onClick={(e) => e.stopPropagation()}
                  className="conversation-edit-input"
                  autoFocus
                />
              ) : (
                <div className="conversation-title">{conversation.title}</div>
              )}
            </div>
            <div className="conversation-actions">
              <button
                className="action-btn rename-btn"
                onClick={(e) => handleStartEdit(e, conversation)}
                title="重命名"
              >
                <i className="fas fa-edit"></i>
              </button>
              <button
                className="action-btn delete-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(conversation.id);
                }}
                title="删除"
              >
                <i className="fas fa-trash"></i>
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ConversationList;
