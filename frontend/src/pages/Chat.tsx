import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ragApi } from '../services/api';
import './Chat.css';

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
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
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
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
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
        localStorage.setItem(STORAGE_KEY, JSON.stringify(conversationsRef.current));
        localStorage.setItem(CURRENT_CONVERSATION_KEY, currentConversationIdRef.current);
      } catch (error) {
        console.error('卸载时保存状态失败:', error);
      }
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentMessages]);

  const createNewConversation = useCallback(() => {
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
  }, []);

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

      const currentConv = getCurrentConversation();
      if (currentConv && currentConv.messages.length === 1) {
        const newTitle = trimmedInput.length > 20 
          ? trimmedInput.substring(0, 20) + '...' 
          : trimmedInput;
        updateConversation(currentConversationId, {
          title: newTitle,
        });
      }

      updateConversation(currentConversationId, (prevConversation) => ({
        messages: [...prevConversation.messages, assistantMessage],
      }));
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
    } finally {
      setIsLoading(false);
    }
  }, [currentInputValue, currentConversationId, currentMessages, createNewConversation, updateConversation, getCurrentConversation, isLoading]);

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
    if (confirm('确定要清空当前对话记录吗？')) {
      updateConversation(currentConversationId, {
        messages: [],
        inputValue: '',
      });
    }
  }, [currentConversationId, updateConversation]);

  const handleDeleteConversation = useCallback(() => {
    if (!currentConversationId) return;
    if (confirm('确定要删除当前对话吗？此操作不可恢复。')) {
      const conversationIndex = conversations.findIndex(conv => conv.id === currentConversationId);
      setConversations(prev => prev.filter(conv => conv.id !== currentConversationId));
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
    }
  }, [currentConversationId, conversations, createNewConversation]);

  const handleRenameConversation = useCallback((newTitle: string) => {
    if (!currentConversationId || !newTitle.trim()) return;
    updateConversation(currentConversationId, {
      title: newTitle.trim(),
    });
  }, [currentConversationId, updateConversation]);

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

  return (
    <div className="chat-container">
      <div className="chat-sidebar">
        <div className="sidebar-header">
          <h3>对话</h3>
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
                <div className="conversation-meta">
                  <span className="message-count">
                    {conversation.messages.length} 条消息
                  </span>
                  <span className="last-updated">
                    {conversation.updatedAt.toLocaleTimeString('zh-CN', {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </span>
                </div>
              </div>
              <div className="conversation-actions">
                <button 
                  className="action-btn rename-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    const newTitle = prompt('请输入新的对话标题:', conversation.title);
                    if (newTitle) {
                      handleRenameConversation(newTitle);
                    }
                  }}
                  title="重命名对话"
                >
                  <i className="fas fa-edit"></i>
                </button>
                <button 
                  className="action-btn delete-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (conversation.id === currentConversationId) {
                      handleDeleteConversation();
                    } else {
                      if (confirm('确定要删除此对话吗？')) {
                        setConversations(prev => prev.filter(conv => conv.id !== conversation.id));
                      }
                    }
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
            <div className="chat-header-icon">
              <i className="fas fa-comments"></i>
            </div>
            <div className="chat-header-text">
              <h2>{getCurrentConversation()?.title || '智能对话'}</h2>
              <p>基于 RAG 技术的智能问答助手</p>
            </div>
          </div>
          <div className="chat-header-actions">
            <button className="clear-chat-btn" onClick={handleClearChat} title="清空对话">
              <i className="fas fa-trash-alt"></i>
            </button>
            <button className="delete-conversation-btn" onClick={handleDeleteConversation} title="删除对话">
              <i className="fas fa-times"></i>
            </button>
          </div>
        </div>

        <div className="chat-messages">
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
          <div className="input-hint">
            <span>提示: 上传文档后，我才能基于知识库回答问题</span>
          </div>
          <div className="input-container">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder="输入您的问题... (按 Enter 发送，Shift + Enter 换行)"
              value={currentInputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows={1}
            />
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
  );
};

export default Chat;
