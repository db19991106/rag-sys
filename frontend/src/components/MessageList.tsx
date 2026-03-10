import React, { useRef, useEffect } from 'react';
import type { Message } from '../contexts/ChatContext';

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
}

export const MessageList: React.FC<MessageListProps> = ({ messages, isLoading }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="chat-empty">
        <div className="empty-icon">
          <i className="fas fa-robot"></i>
        </div>
        <h3>开始对话</h3>
        <p>请输入您的问题，我会基于知识库为您解答</p>
      </div>
    );
  }

  return (
    <div className="chat-messages">
      {messages.map((message) => (
        <div key={message.id} className={`message ${message.role}`}>
          <div className="message-avatar">
            <i className={`fas ${message.role === 'user' ? 'fa-user' : 'fa-robot'}`}></i>
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
              {message.timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
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
    </div>
  );
};

export default MessageList;
