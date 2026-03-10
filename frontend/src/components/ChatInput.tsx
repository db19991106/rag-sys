import React, { useRef, useCallback } from 'react';
import { documentApi } from '../services/api';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  isLoading: boolean;
  disabled: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  value,
  onChange,
  onSend,
  isLoading,
  disabled,
}) => {
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
    // 自动调整高度
    const target = e.target;
    target.style.height = 'auto';
    target.style.height = Math.min(target.scrollHeight, 150) + 'px';
  };

  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
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
  }, []);

  return (
    <div className="chat-input-area">
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder="问点难的，让我多想一步"
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            disabled={isLoading || disabled}
            rows={1}
          />
          <button 
            className="add-button" 
            title="添加附件" 
            onClick={() => fileInputRef.current?.click()}
            type="button"
          >
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
            <button
              className="send-button"
              onClick={onSend}
              disabled={!value.trim() || isLoading}
              title="发送消息"
              type="button"
            >
              <i className={`fas ${isLoading ? 'fa-spinner fa-spin' : 'fa-paper-plane'}`}></i>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInput;
