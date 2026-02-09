import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ragApi, documentApi } from '../services/api';
import './Chat.css';

// 添加console.log的别名logger
const logger = console;

interface RenameDialogProps {
  isOpen: boolean;
  currentTitle: string;
  onClose: () => void;
  onConfirm: (newTitle: string) => void;
}

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onClose: () => void;
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({ isOpen, title, message, onConfirm, onClose }) => {
  const confirmButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => {
        confirmButtonRef.current?.focus();
      }, 100);
    }
  }, [isOpen]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="dialog-overlay" onClick={onClose} onKeyDown={handleKeyDown}>
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
            ref={confirmButtonRef}
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

const RenameDialog: React.FC<RenameDialogProps> = ({ isOpen, currentTitle, onClose, onConfirm }) => {
  const [title, setTitle] = useState(currentTitle);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setTitle(currentTitle);
      setTimeout(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      }, 100);
    }
  }, [isOpen, currentTitle]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (title.trim()) {
      onConfirm(title.trim());
      onClose();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog-container" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <h3>重命名对话</h3>
        </div>
        <form onSubmit={handleSubmit} className="dialog-content">
          <input
            ref={inputRef}
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="请输入新的对话标题"
            className="dialog-input"
            maxLength={100}
            autoFocus
          />
          <div className="dialog-actions">
            <button type="button" className="dialog-button dialog-button-secondary" onClick={onClose}>
              取消
            </button>
            <button type="submit" className="dialog-button dialog-button-primary" disabled={!title.trim()}>
              确定
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  intent?: {
    type: string;
    confidence: number;
    description?: string;
  };
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  inputValue: string;
  createdAt: Date;
  updatedAt: Date;
}

const STORAGE_KEY = 'chat_conversations';
const CURRENT_CONVERSATION_KEY = 'current_conversation_id';

const Chat: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    if (typeof window === 'undefined') return [];
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        return parsed.map((conv: any) => ({
          ...conv,
          createdAt: new Date(conv.createdAt),
          updatedAt: new Date(conv.updatedAt),
          messages: conv.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }))
        }));
      }
    } catch (e) {
      console.error('加载对话失败:', e);
    }
    return [];
  });

  const [currentConversationId, setCurrentConversationId] = useState<string>(() => {
    if (typeof window === 'undefined') return '';
    return localStorage.getItem(CURRENT_CONVERSATION_KEY) || '';
  });

  const [isLoading, setIsLoading] = useState(false);
  const [isRenameDialogOpen, setIsRenameDialogOpen] = useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  const [deleteConfirmTitle, setDeleteConfirmTitle] = useState('');
  const [deleteConfirmMessage, setDeleteConfirmMessage] = useState('');
  const [deletingConversationId, setDeletingConversationId] = useState<string>('');
  const [renamingConversationId, setRenamingConversationId] = useState<string>('');
  const [sidebarHidden, setSidebarHidden] = useState(false);
  
  // 文件上传相关
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const inputWrapperRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const conversationsRef = useRef<Conversation[]>(conversations);
  const currentConversationIdRef = useRef<string>(currentConversationId);

  const getCurrentConversation = useCallback(() => {
    return conversations.find(conv => conv.id === currentConversationId) || null;
  }, [conversations, currentConversationId]);

  const currentMessages = getCurrentConversation()?.messages || [];
  const currentInputValue = getCurrentConversation()?.inputValue || '';

  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);
  
  useEffect(() => {
    currentConversationIdRef.current = currentConversationId;
  }, [currentConversationId]);

  useEffect(() => {
    if (conversations.length === 0) {
      const newConversation: Conversation = {
        id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        title: '新对话',
        messages: [],
        inputValue: '',
        createdAt: new Date(),
        updatedAt: new Date(),
      };
      setConversations(prev => [...prev, newConversation]);
      setCurrentConversationId(newConversation.id);
    } else if (!currentConversationId) {
      setCurrentConversationId(conversations[0].id);
    }
  }, [conversations, currentConversationId]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      // 只保存有消息的对话
      const conversationsToSave = conversations.filter(conv => conv.messages.length > 0);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversationsToSave));
    } catch (error) {
      console.error('保存对话列表失败:', error);
    }
  }, [conversations]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      localStorage.setItem(CURRENT_CONVERSATION_KEY, currentConversationId);
    } catch (error) {
      console.error('保存当前对话ID失败:', error);
    }
  }, [currentConversationId]);

  useEffect(() => {
    return () => {
      if (typeof window === 'undefined') return;
      try {
        // 只保存有消息的对话
        const conversationsToSave = conversationsRef.current.filter(conv => conv.messages.length > 0);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(conversationsToSave));
        localStorage.setItem(CURRENT_CONVERSATION_KEY, currentConversationIdRef.current);
      } catch (error) {
        console.error('卸载时保存状态失败:', error);
      }
    };
  }, []);

  // 滚动到最新消息
  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: 'smooth',
        block: 'end',
        inline: 'nearest'
      });
    }
  }, []);

  // 页面加载完成后滚动到最底部
  useEffect(() => {
    // 延迟执行，确保DOM已经完全渲染
    const timer = setTimeout(() => {
      scrollToBottom();
    }, 100);

    return () => clearTimeout(timer);
  }, [scrollToBottom]);

  // 新消息添加时滚动到最底部
  useEffect(() => {
    // 延迟执行，确保新消息已经完全渲染到DOM中
    const timer = setTimeout(() => {
      scrollToBottom();
    }, 50);

    return () => clearTimeout(timer);
  }, [currentMessages, scrollToBottom]);

  const createNewConversation = useCallback(() => {
    // 检查当前对话是否为空
    if (currentConversationId) {
      const currentConv = conversations.find(conv => conv.id === currentConversationId);
      if (currentConv && currentConv.messages.length === 0) {
        // 当前对话为空，直接返回它
        return currentConv;
      }
    }
    
    // 创建新对话
    const newConversation: Conversation = {
      id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title: '新对话',
      messages: [],
      inputValue: '',
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    setConversations(prev => [...prev, newConversation]);
    setCurrentConversationId(newConversation.id);
    return newConversation;
  }, [currentConversationId, conversations]);

  const updateConversation = useCallback((conversationId: string, updates: Partial<Conversation> | ((prevConversation: Conversation) => Partial<Conversation>)) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === conversationId) {
        let updatesObject: Partial<Conversation>;
        if (typeof updates === 'function') {
          updatesObject = updates(conv);
        } else {
          updatesObject = updates;
        }
        return {
          ...conv,
          ...updatesObject,
          updatedAt: new Date(),
        };
      }
      return conv;
    }));
  }, []);

  const handleSendMessage = useCallback(async () => {
    const trimmedInput = currentInputValue.trim();
    if (!trimmedInput || isLoading) return;

    if (!currentConversationId) {
      createNewConversation();
      return;
    }

    setIsLoading(true);

    const userMessage: Message = {
      id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
    };

    // 检查当前对话是否为空
    const currentConv = getCurrentConversation();
    const isFirstMessage = currentConv && currentConv.messages.length === 0;
    
    // 如果是第一条消息，生成对话标题
    if (isFirstMessage) {
      try {
        // 调用后端API生成摘要
        const summaryResponse = await ragApi.generateSummary(trimmedInput);
        const newTitle = summaryResponse.summary;
        
        // 更新对话标题
        updateConversation(currentConversationId, {
          title: newTitle,
        });
      } catch (error) {
        console.warn('生成对话标题失败，使用默认标题:', error);
        // 如果API调用失败，使用简单的前端摘要算法
        let newTitle = trimmedInput;
        if (newTitle.length > 15) {
          // 简单摘要算法：取前几个关键词
          const words = newTitle.split(/[\s，。！？；：,.;:!?]/).filter(word => word.length > 0);
          if (words.length > 1) {
            // 取前两个关键词
            newTitle = words.slice(0, 2).join(' ');
          } else {
            // 取前15个字
            newTitle = newTitle.substring(0, 15);
          }
        }
        // 确保标题长度在5-15字之间
        if (newTitle.length < 5) {
          newTitle = newTitle.padEnd(5, ' ');
        }
        newTitle = newTitle.trim();
        
        // 更新对话标题
        updateConversation(currentConversationId, {
          title: newTitle,
        });
      }
    }

    // 添加用户消息
    updateConversation(currentConversationId, (prevConversation) => ({
      messages: [...prevConversation.messages, userMessage],
      inputValue: '',
    }));

    try {
      let intentInfo: { type: string; confidence: number; description: string } | undefined = undefined;
      try {
        const intentResult = await ragApi.recognizeIntent(trimmedInput);
        intentInfo = {
          type: intentResult.intent,
          confidence: intentResult.confidence,
          description: getIntentDescription(intentResult.intent)
        };
        updateConversation(currentConversationId, (prevConversation) => {
          const messages = [...prevConversation.messages];
          const lastMessage = messages[messages.length - 1];
          if (lastMessage && lastMessage.id === userMessage.id) {
            messages[messages.length - 1] = { ...lastMessage, intent: intentInfo };
          }
          return { messages };
        });
      } catch (e) {
        console.warn('意图识别失败:', e);
      }

      const data = await ragApi.generate({
        query: trimmedInput,
        retrieval_config: { top_k: 5, similarity_threshold: 0.4, algorithm: 'cosine', enable_rerank: false, reranker_type: 'none', reranker_model: '', reranker_top_k: 5, reranker_threshold: 0.5 },
        generation_config: {
          llm_provider: 'local',
          llm_model: 'Qwen2.5-7B-Instruct',
          temperature: 0.7,
          max_tokens: 2000,
          top_p: 0.9,
          frequency_penalty: 0.0,
          presence_penalty: 0.0,
        },
        conversation_id: currentConversationId,
      });

      const assistantMessage: Message = {
        id: `${Date.now() + 1}_${Math.random().toString(36).substr(2, 9)}`,
        role: 'assistant',
        content: data.answer || '抱歉，我无法回答这个问题。',
        timestamp: new Date(),
      };

      // 添加AI回复
      updateConversation(currentConversationId, (prevConversation) => ({
        messages: [...prevConversation.messages, assistantMessage],
      }));
      // 发送消息后滚动到底部
      scrollToBottom();
    } catch (error) {
      console.error('发送消息失败:', error);
      const errorMessage: Message = {
        id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        role: 'assistant',
        content: `服务暂时不可用: ${error instanceof Error ? error.message : '未知错误'}`,
        timestamp: new Date(),
      };
      updateConversation(currentConversationId, (prevConversation) => ({
        messages: [...prevConversation.messages, errorMessage],
      }));
      // 发送消息后滚动到底部
      scrollToBottom();
    } finally {
      setIsLoading(false);
    }
  }, [currentInputValue, currentConversationId, currentMessages, createNewConversation, updateConversation, getCurrentConversation, isLoading, scrollToBottom]);

  const getIntentDescription = (intentType: string): string => {
    const descriptions: Record<string, string> = {
      'question': '问题咨询',
      'search': '信息搜索',
      'summary': '内容总结',
      'comparison': '对比分析',
      'procedure': '操作流程',
      'definition': '定义说明',
      'greeting': '问候',
      'other': '其他'
    };
    return descriptions[intentType] || '未知';
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    if (currentConversationId) {
      updateConversation(currentConversationId, {
        inputValue: newValue,
      });
    }
    const target = e.target;
    target.style.height = 'auto';
    target.style.height = Math.min(target.scrollHeight, 150) + 'px';
  };

  const handleClearChat = useCallback(() => {
    if (!currentConversationId) return;
    setDeletingConversationId(currentConversationId);
    setIsDeleteConfirmOpen(true);
  }, [currentConversationId]);

  const handleClearChatConfirm = useCallback(() => {
    if (!deletingConversationId) return;
    updateConversation(deletingConversationId, {
      messages: [],
      inputValue: '',
    });
  }, [deletingConversationId, updateConversation]);

  const handleDeleteConversation = useCallback((conversationId?: string) => {
    const targetId = conversationId || currentConversationId;
    if (!targetId) return;
    setDeletingConversationId(targetId);
    setIsDeleteConfirmOpen(true);
  }, [currentConversationId]);

  const handleDeleteConversationConfirm = useCallback(async () => {
    if (!deletingConversationId) return;

    try {
      // 调用后端API删除对话
      await ragApi.deleteConversation(deletingConversationId);
      logger.info(`对话已删除: ${deletingConversationId}`);
    } catch (error) {
      console.error('删除对话失败:', error);
      // 即使后端删除失败，也从本地移除（保持数据一致性）
    }

    const conversationIndex = conversations.findIndex(conv => conv.id === deletingConversationId);
    setConversations(prev => prev.filter(conv => conv.id !== deletingConversationId));

    if (conversations.length === 1) {
      createNewConversation();
    } else {
      const newIndex = Math.max(0, conversationIndex - 1);
      if (conversations[newIndex]) {
        setCurrentConversationId(conversations[newIndex].id);
      } else if (conversations[conversationIndex + 1]) {
        setCurrentConversationId(conversations[conversationIndex + 1].id);
      }
    }
  }, [deletingConversationId, conversations, createNewConversation]);

  const handleRenameConversation = useCallback((conversationId: string, newTitle: string) => {
    if (!conversationId || !newTitle.trim()) return;
    updateConversation(conversationId, {
      title: newTitle.trim(),
    });
  }, [updateConversation]);

  const openRenameDialog = useCallback((conversationId: string, conversationTitle: string) => {
    setRenamingConversationId(conversationId);
    setIsRenameDialogOpen(true);
  }, []);

  const closeRenameDialog = useCallback(() => {
    setIsRenameDialogOpen(false);
    setRenamingConversationId('');
  }, []);

  // 文件上传处理函数
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const supportExt = ['.txt', '.pdf', '.docx', '.md'];

    for (const file of Array.from(files)) {
      const ext = '.' + file.name.split('.').pop();
      if (!ext) continue;
      const extLower = ext.toLowerCase();
      if (!supportExt.includes(extLower)) {
        alert(`文件${file.name}格式不支持,仅支持TXT/PDF/DOCX/MD`);
        continue;
      }

      try {
        await documentApi.upload(file);
        alert(`文档 ${file.name} 上传成功!`);
      } catch (error) {
        console.error('上传失败:', error);
        alert(`文档 ${file.name} 上传失败: ${error instanceof Error ? error.message : '未知错误'}`);
      }
    }

    if (fileInputRef.current?.value) {
      fileInputRef.current.value = '';
    }
  };

  // 触发文件选择
  const handleAddButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleExampleClick = (question: string) => {
    if (currentConversationId) {
      updateConversation(currentConversationId, {
        inputValue: question,
      });
    }
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 150) + 'px';
    }
  };

  // 侧边栏切换处理函数
  const handleToggleSidebar = () => {
    setSidebarHidden(!sidebarHidden);
  };

  // 侧边栏悬停显示处理函数
  const handleSidebarHover = () => {
    setSidebarHidden(false);
  };

  // 侧边栏离开隐藏处理函数
  const handleSidebarLeave = () => {
    setSidebarHidden(true);
  };

  // 监听输入区域高度变化，调整消息区域滚动
  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (messagesRef.current) {
          // 当输入区域高度变化时，保持消息区域的滚动位置
          // 确保最新的消息始终可见
          messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
      }
    });

    if (inputWrapperRef.current) {
      observer.observe(inputWrapperRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, []);

  return (
    <div className={`chat-container ${sidebarHidden ? 'sidebar-hidden' : ''}`}>
      {/* 侧边栏悬停显示区域 */}
      <div 
        className="sidebar-hover-area"
        onMouseEnter={handleSidebarHover}
      />
      
      {/* 侧边栏 */}
      <div 
        className={`chat-sidebar ${sidebarHidden ? 'hidden' : ''}`}
        onMouseLeave={sidebarHidden ? handleSidebarLeave : undefined}
      >
        <div className="sidebar-header">
          <h3>历史对话</h3>
          <button 
            className="new-conversation-btn"
            onClick={createNewConversation}
            title="创建新对话"
          >
            <i className="fas fa-plus"></i>
          </button>
        </div>
        
        <div className="conversation-list">
          {conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`conversation-item ${conversation.id === currentConversationId ? 'active' : ''}`}
              onClick={() => setCurrentConversationId(conversation.id)}
            >
              <div className="conversation-info">
                <div className="conversation-title">
                  {conversation.title}
                </div>
                {/* <div className="conversation-meta"> */}
                  {/* <span className="message-count">
                    {conversation.messages.length} 条消息
                  </span>
                  <span className="last-updated">
                    {conversation.updatedAt.toLocaleTimeString('zh-CN', {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </span> */}
                {/* </div> */}
              </div>
              <div className="conversation-actions">
                <button 
                  className="action-btn rename-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    openRenameDialog(conversation.id, conversation.title);
                  }}
                  title="重命名对话"
                >
                  <i className="fas fa-edit"></i>
                </button>
                <button
                  className="action-btn delete-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteConversation(conversation.id);
                  }}
                  title="删除对话"
                >
                  <i className="fas fa-trash"></i>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="chat-main">
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
              <h2>{getCurrentConversation()?.title || '智能对话'}</h2>
              <p>内容由 AI 生成</p>
            </div>
          </div>
          <div className="chat-header-actions">
            <button className="clear-chat-btn" onClick={handleClearChat} title="清空对话">
              <i className="el-icon-delete"></i>
            </button>
          </div>
        </div>

        <div className="chat-messages" ref={messagesRef}>
          {currentMessages.length === 0 ? (
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
          ) : (
            <>
              {currentMessages.map((message) => (
                <div key={message.id} className={`message ${message.role}`}>
                  <div className="message-avatar">
                    {message.role === 'user' ? (
                      <i className="fas fa-user"></i>
                    ) : (
                      <i className="fas fa-robot"></i>
                    )}
                  </div>
                  <div className="message-content">
                    {message.intent && (
                      <div className="message-intent">
                        <span className="intent-label">
                          <i className="fas fa-bullseye"></i>
                          意图: {message.intent.description}
                        </span>
                        <span className="intent-confidence">
                          置信度: {(message.intent.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    )}
                    <div className="message-text">
                      {message.content.split('\n').map((line, index) => (
                        <p key={index}>{line || '\u00A0'}</p>
                      ))}
                    </div>
                    <div className="message-time">
                      {message.timestamp.toLocaleTimeString('zh-CN', {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="message assistant">
                  <div className="message-avatar">
                    <i className="fas fa-robot"></i>
                  </div>
                  <div className="message-content">
                    <div className="message-loading">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        <div className="chat-input-area">
          <div className="input-container">
            <div className="input-wrapper" ref={inputWrapperRef}>
              <textarea
                ref={inputRef}
                className="chat-input"
                placeholder="问点难的，让我多想一步"
                value={currentInputValue}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                disabled={isLoading}
                rows={1}
              />
              <button className="add-button" title="添加附件" onClick={handleAddButtonClick}>
                <i className="fas fa-plus"></i>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileUpload}
                multiple
                accept=".txt,.pdf,.docx,.md"
                style={{ display: 'none' }}
              />
              <div className="button-container">
                <div className="think-dropdown" title="思考模式">
                  <span>K2.5思考</span>
                  <i className="fas fa-chevron-down"></i>
                </div>
                <button
                  className="send-button"
                  onClick={handleSendMessage}
                  disabled={!currentInputValue.trim() || isLoading}
                  title="发送消息"
                >
                  <i className={`fas ${isLoading ? 'fa-spinner fa-spin' : 'fa-paper-plane'}`}></i>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <RenameDialog
        isOpen={isRenameDialogOpen}
        currentTitle={conversations.find(c => c.id === renamingConversationId)?.title || ''}
        onClose={closeRenameDialog}
        onConfirm={(newTitle) => handleRenameConversation(renamingConversationId, newTitle)}
      />

      <ConfirmDialog
        isOpen={isDeleteConfirmOpen}
        title="确认删除"
        message="确定要删除这个对话吗？此操作不可撤销。"
        onConfirm={handleDeleteConversationConfirm}
        onClose={() => setIsDeleteConfirmOpen(false)}
      />
    </div>
  );
};

export default Chat;
