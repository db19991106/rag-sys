import React, { useState, useCallback } from 'react';
import { ChatProvider, useChat } from '../contexts/ChatContext';
import ConversationList from '../components/ConversationList';
import MessageList from '../components/MessageList';
import ChatInput from '../components/ChatInput';
import './Chat.css';

// 确认对话框组件
interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onClose: () => void;
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({ isOpen, title, message, onConfirm, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog-container dialog-confirm" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <h3>{title}</h3>
        </div>
        <div className="dialog-body">
          <p>{message}</p>
        </div>
        <div className="dialog-actions">
          <button type="button" className="dialog-button dialog-button-secondary" onClick={onClose}>
            取消
          </button>
          <button
            type="button"
            className="dialog-button dialog-button-danger"
            onClick={() => {
              onConfirm();
              onClose();
            }}
          >
            确定
          </button>
        </div>
      </div>
    </div>
  );
};

// Chat 主内容组件
const ChatContent: React.FC = () => {
  const {
    conversations,
    currentConversationId,
    isLoading,
    currentConversation,
    currentInputValue,
    createNewConversation,
    updateConversation,
    setCurrentConversationId,
    sendMessage,
    deleteConversation,
    renameConversation,
  } = useChat();

  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  const [deletingConversationId, setDeletingConversationId] = useState<string>('');

  // 侧边栏切换
  const handleToggleSidebar = () => {
    setSidebarHidden(!sidebarHidden);
  };

  // 输入值变化
  const handleInputChange = useCallback((value: string) => {
    if (currentConversationId) {
      updateConversation(currentConversationId, { inputValue: value });
    }
  }, [currentConversationId, updateConversation]);

  // 清空当前对话
  const handleClearChat = useCallback(() => {
    if (!currentConversationId) return;
    setDeletingConversationId(currentConversationId);
    setIsDeleteConfirmOpen(true);
  }, [currentConversationId]);

  // 删除对话确认
  const handleDeleteConfirm = useCallback(() => {
    if (!deletingConversationId) return;
    deleteConversation(deletingConversationId);
    setDeletingConversationId('');
  }, [deletingConversationId, deleteConversation]);

  // 示例问题点击
  const handleExampleClick = useCallback((question: string) => {
    if (currentConversationId) {
      updateConversation(currentConversationId, { inputValue: question });
    }
  }, [currentConversationId, updateConversation]);

  return (
    <div className={`chat-container ${sidebarHidden ? 'sidebar-hidden' : ''}`}>
      {/* 侧边栏悬停显示区域 */}
      <div 
        className="sidebar-hover-area"
        onMouseEnter={() => setSidebarHidden(false)}
      />
      
      {/* 侧边栏 */}
      <div 
        className={`chat-sidebar ${sidebarHidden ? 'hidden' : ''}`}
        onMouseLeave={sidebarHidden ? () => setSidebarHidden(true) : undefined}
      >
        <ConversationList
          conversations={conversations}
          currentConversationId={currentConversationId}
          onSelect={setCurrentConversationId}
          onCreateNew={createNewConversation}
          onRename={renameConversation}
          onDelete={(id) => {
            setDeletingConversationId(id);
            setIsDeleteConfirmOpen(true);
          }}
        />
      </div>

      {/* 主内容区 */}
      <div className="chat-main">
        {/* 头部 */}
        <div className="chat-header">
          <div className="chat-header-content">
            <button 
              className={`sidebar-toggle ${sidebarHidden ? 'active' : ''}`}
              onClick={handleToggleSidebar}
              title={sidebarHidden ? '显示侧边栏' : '隐藏侧边栏'}
            >
              <i className={`fas ${sidebarHidden ? 'fa-arrow-right' : 'fa-arrow-left'}`}></i>
            </button>
            <div className="chat-header-text">
              <h2>{currentConversation?.title || '智能对话'}</h2>
              <p>内容由 AI 生成</p>
            </div>
          </div>
          <div className="chat-header-actions">
            <button className="clear-chat-btn" onClick={handleClearChat} title="清空对话">
              <i className="fas fa-trash"></i>
            </button>
          </div>
        </div>

        {/* 消息区域 */}
        {currentConversation && currentConversation.messages.length === 0 ? (
          <div className="chat-messages">
            <div className="chat-empty">
              <div className="empty-icon">
                <i className="fas fa-robot"></i>
              </div>
              <h3>开始对话</h3>
              <p>请输入您的问题，我会基于知识库为您解答</p>
              <div className="example-questions">
                <p>示例问题：</p>
                <div className="example-question" onClick={() => handleExampleClick('RAG的核心流程是什么？')}>
                  RAG的核心流程是什么？
                </div>
                <div className="example-question" onClick={() => handleExampleClick('如何进行文档切分？')}>
                  如何进行文档切分？
                </div>
                <div className="example-question" onClick={() => handleExampleClick('向量数据库有什么作用？')}>
                  向量数据库有什么作用？
                </div>
              </div>
            </div>
          </div>
        ) : (
          <MessageList 
            messages={currentConversation?.messages || []} 
            isLoading={isLoading} 
          />
        )}

        {/* 输入区域 */}
        <ChatInput
          value={currentInputValue}
          onChange={handleInputChange}
          onSend={sendMessage}
          isLoading={isLoading}
          disabled={!currentConversationId}
        />
      </div>

      {/* 删除确认对话框 */}
      <ConfirmDialog
        isOpen={isDeleteConfirmOpen}
        title="确认删除"
        message="确定要删除这个对话吗？此操作不可撤销。"
        onConfirm={handleDeleteConfirm}
        onClose={() => {
          setIsDeleteConfirmOpen(false);
          setDeletingConversationId('');
        }}
      />
    </div>
  );
};

// Chat 主组件（带 Provider）
const Chat: React.FC = () => {
  return (
    <ChatProvider>
      <ChatContent />
    </ChatProvider>
  );
};

export default Chat;