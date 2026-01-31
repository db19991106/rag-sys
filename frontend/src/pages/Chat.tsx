import React, { useState, useRef, useEffect } from 'react';
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

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // 自动滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 处理发送消息
  const handleSendMessage = async () => {
    const trimmedInput = inputValue.trim();
    if (!trimmedInput || isLoading) return;

    // 先识别意图
    let intentInfo = undefined;
    try {
      const intentResult = await ragApi.recognizeIntent(trimmedInput);
      intentInfo = {
        type: intentResult.intent,
        confidence: intentResult.confidence,
        description: getIntentDescription(intentResult.intent)
      };
    } catch (error) {
      console.warn('意图识别失败，继续处理:', error);
    }

    // 添加用户消息
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
      intent: intentInfo
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // 调用后端 RAG API
      const data = await ragApi.generate({
        query: trimmedInput,
        retrieval_config: {
          top_k: 5,
          similarity_threshold: 0.7,
          algorithm: 'cosine',
        },
        generation_config: {
          llm_provider: 'local',
          llm_model: 'Qwen2.5-0.5B-Instruct',
          temperature: 0.7,
          max_tokens: 2000,
          top_p: 0.9,
          frequency_penalty: 0.0,
          presence_penalty: 0.0,
        },
      });

      // 添加助手消息
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer || '抱歉，我无法回答这个问题。',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('发送消息失败:', error);

      // 添加错误消息
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `抱歉，智能对话服务暂时不可用。\n\n错误详情: ${error instanceof Error ? error.message : '未知错误'}\n\n请检查：\n1. 后端服务是否正常运行\n2. 网络连接是否正常\n3. 稍后重试或联系管理员`,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

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

  // 处理键盘事件
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // 清空对话
  const handleClearChat = () => {
    if (confirm('确定要清空对话记录吗？')) {
      setMessages([]);
    }
  };

  // 调整输入框高度
  const handleInputHeight = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const target = e.target;
    target.style.height = 'auto';
    target.style.height = Math.min(target.scrollHeight, 150) + 'px';
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div className="chat-header-content">
          <div className="chat-header-icon">
            <i className="fas fa-comments"></i>
          </div>
          <div className="chat-header-text">
            <h2>智能对话</h2>
            <p>基于 RAG 技术的智能问答助手</p>
          </div>
        </div>
        <button className="clear-chat-btn" onClick={handleClearChat} title="清空对话">
          <i className="fas fa-trash-alt"></i>
        </button>
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <div className="empty-icon">
              <i className="fas fa-robot"></i>
            </div>
            <h3>开始对话</h3>
            <p>请输入您的问题，我会基于知识库为您解答</p>
            <div className="example-questions">
              <p>示例问题：</p>
              <div className="example-question" onClick={() => setInputValue('RAG的核心流程是什么？')}>
                RAG的核心流程是什么？
              </div>
              <div className="example-question" onClick={() => setInputValue('如何进行文档切分？')}>
                如何进行文档切分？
              </div>
              <div className="example-question" onClick={() => setInputValue('向量数据库有什么作用？')}>
                向量数据库有什么作用？
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
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
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder="输入您的问题... (按 Enter 发送，Shift + Enter 换行)"
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value);
              handleInputHeight(e);
            }}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            rows={1}
          />
          <button
            className="send-button"
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            title="发送消息"
          >
            <i className={`fas ${isLoading ? 'fa-spinner fa-spin' : 'fa-paper-plane'}`}></i>
          </button>
        </div>
        <div className="input-hint">
          <span>💡 提示: 上传文档后，我才能基于知识库回答问题</span>
        </div>
      </div>
    </div>
  );
};

export default Chat;