import React, { createContext, useContext, useState, useRef, useCallback, useEffect } from 'react';
import type { ReactNode } from 'react';
import { ragApi } from '../services/api';

// 类型定义
export interface Message {
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

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  inputValue: string;
  createdAt: Date;
  updatedAt: Date;
}

interface ChatContextType {
  // 状态
  conversations: Conversation[];
  currentConversationId: string;
  isLoading: boolean;
  currentConversation: Conversation | null;
  currentMessages: Message[];
  currentInputValue: string;
  
  // 操作
  createNewConversation: () => Conversation;
  updateConversation: (id: string, updates: Partial<Conversation> | ((prev: Conversation) => Partial<Conversation>)) => void;
  setCurrentConversationId: (id: string) => void;
  setIsLoading: (loading: boolean) => void;
  sendMessage: () => Promise<void>;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

const STORAGE_KEY = 'chat_conversations';
const CURRENT_CONVERSATION_KEY = 'current_conversation_id';

// 获取意图描述
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

export const ChatProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
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
  
  const conversationsRef = useRef(conversations);
  const currentConversationIdRef = useRef(currentConversationId);

  // 获取当前对话
  const currentConversation = conversations.find(conv => conv.id === currentConversationId) || null;
  const currentMessages = currentConversation?.messages || [];
  const currentInputValue = currentConversation?.inputValue || '';

  // 保持 ref 同步
  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);

  useEffect(() => {
    currentConversationIdRef.current = currentConversationId;
  }, [currentConversationId]);

  // 初始化：如果没有对话，创建一个新对话
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
      setConversations([newConversation]);
      setCurrentConversationId(newConversation.id);
    } else if (!currentConversationId) {
      setCurrentConversationId(conversations[0].id);
    }
  }, [conversations, currentConversationId]);

  // 持久化
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
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

  // 创建新对话
  const createNewConversation = useCallback((): Conversation => {
    const currentConv = conversations.find(conv => conv.id === currentConversationId);
    if (currentConv && currentConv.messages.length === 0) {
      return currentConv;
    }

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

  // 更新对话
  const updateConversation = useCallback((id: string, updates: Partial<Conversation> | ((prev: Conversation) => Partial<Conversation>)) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === id) {
        const updatesObject = typeof updates === 'function' ? updates(conv) : updates;
        return { ...conv, ...updatesObject, updatedAt: new Date() };
      }
      return conv;
    }));
  }, []);

  // 发送消息
  const sendMessage = useCallback(async () => {
    const trimmedInput = currentInputValue.trim();
    if (!trimmedInput || isLoading || !currentConversationId) return;

    setIsLoading(true);

    const userMessage: Message = {
      id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
    };

    const isFirstMessage = currentMessages.length === 0;

    // 如果是第一条消息，生成对话标题
    if (isFirstMessage) {
      try {
        const summaryResponse = await ragApi.generateSummary(trimmedInput);
        updateConversation(currentConversationId, { title: summaryResponse.summary });
      } catch (error) {
        console.warn('生成对话标题失败:', error);
        let newTitle = trimmedInput.substring(0, 15);
        updateConversation(currentConversationId, { title: newTitle || '新对话' });
      }
    }

    // 添加用户消息
    updateConversation(currentConversationId, prev => ({
      messages: [...prev.messages, userMessage],
      inputValue: '',
    }));

    try {
      // 意图识别
      try {
        const intentResult = await ragApi.recognizeIntent(trimmedInput);
        const intentInfo = {
          type: intentResult.intent,
          confidence: intentResult.confidence,
          description: getIntentDescription(intentResult.intent)
        };
        updateConversation(currentConversationId, prev => {
          const messages = [...prev.messages];
          const lastMessage = messages[messages.length - 1];
          if (lastMessage && lastMessage.id === userMessage.id) {
            messages[messages.length - 1] = { ...lastMessage, intent: intentInfo };
          }
          return { messages };
        });
      } catch (e) {
        console.warn('意图识别失败:', e);
      }

      // 添加空的 AI 消息
      const assistantMessageId = `${Date.now() + 1}_${Math.random().toString(36).substr(2, 9)}`;
      updateConversation(currentConversationId, prev => ({
        messages: [...prev.messages, {
          id: assistantMessageId,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
        }],
      }));

      // 流式生成
      await ragApi.generateStream({
        query: trimmedInput,
        retrieval_config: { top_k: 5, similarity_threshold: 0.4, algorithm: 'cosine', enable_rerank: false, reranker_type: 'none', reranker_model: '', reranker_top_k: 5, reranker_threshold: 0.5 },
        generation_config: { temperature: 0.7, max_tokens: 2000, top_p: 0.9, frequency_penalty: 0.0, presence_penalty: 0.0 },
        conversation_id: currentConversationId,
      }, {
        onToken: (token) => {
          updateConversation(currentConversationId, prev => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            if (lastMsg && lastMsg.id === assistantMessageId) {
              messages[messages.length - 1] = { ...lastMsg, content: lastMsg.content + token };
            }
            return { messages };
          });
        },
        onComplete: (fullResponse) => {
          if (!fullResponse) {
            updateConversation(currentConversationId, prev => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              if (lastMsg && lastMsg.id === assistantMessageId) {
                messages[messages.length - 1] = { ...lastMsg, content: '抱歉，我无法回答这个问题。' };
              }
              return { messages };
            });
          }
          setIsLoading(false);
        },
        onError: (error) => {
          console.error('流式输出错误:', error);
          updateConversation(currentConversationId, prev => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            if (lastMsg && lastMsg.id === assistantMessageId) {
              messages[messages.length - 1] = { ...lastMsg, content: `服务暂时不可用: ${error.message}` };
            }
            return { messages };
          });
          setIsLoading(false);
        }
      });
    } catch (error) {
      console.error('发送消息失败:', error);
      updateConversation(currentConversationId, prev => ({
        messages: [...prev.messages, {
          id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          role: 'assistant',
          content: `服务暂时不可用: ${error instanceof Error ? error.message : '未知错误'}`,
          timestamp: new Date(),
        }],
      }));
      setIsLoading(false);
    }
  }, [currentInputValue, currentConversationId, currentMessages, isLoading, updateConversation]);

  // 删除对话
  const deleteConversation = useCallback((id: string) => {
    setConversations(prev => prev.filter(conv => conv.id !== id));
    if (id === currentConversationId) {
      const remaining = conversations.filter(conv => conv.id !== id);
      if (remaining.length > 0) {
        setCurrentConversationId(remaining[0].id);
      }
    }
  }, [currentConversationId, conversations]);

  // 重命名对话
  const renameConversation = useCallback((id: string, title: string) => {
    updateConversation(id, { title });
  }, [updateConversation]);

  return (
    <ChatContext.Provider value={{
      conversations,
      currentConversationId,
      isLoading,
      currentConversation,
      currentMessages,
      currentInputValue,
      createNewConversation,
      updateConversation,
      setCurrentConversationId,
      setIsLoading,
      sendMessage,
      deleteConversation,
      renameConversation,
    }}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};
