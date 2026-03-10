import { useState, useCallback, useRef } from 'react';

interface StreamingMessage {
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
}

interface UseStreamingChatOptions {
  apiUrl?: string;
  onComplete?: (message: StreamingMessage) => void;
  onError?: (error: Error) => void;
}

interface StreamEvent {
  type: 'token' | 'done' | 'error' | 'metadata';
  content?: string;
  metadata?: {
    query?: string;
    context_chunks?: Array<{
      chunk_id: string;
      document_name: string;
      similarity: number;
    }>;
    retrieval_time_ms?: number;
    generation_time_ms?: number;
    total_time_ms?: number;
  };
}

/**
 * 流式聊天Hook - 支持SSE流式输出
 * 
 * 使用方法：
 * ```tsx
 * const { sendMessage, isStreaming, currentMessage } = useStreamingChat({
 *   onComplete: (msg) => console.log('完成:', msg)
 * });
 * 
 * await sendMessage("你好");
 * ```
 */
export function useStreamingChat(options: UseStreamingChatOptions = {}) {
  const {
    apiUrl = '/api/rag/generate/stream',
    onComplete,
    onError
  } = options;

  const [isStreaming, setIsStreaming] = useState(false);
  const [currentMessage, setCurrentMessage] = useState('');
  const [metadata, setMetadata] = useState<StreamEvent['metadata']>();
  const [error, setError] = useState<Error | null>(null);
  
  const abortControllerRef = useRef<AbortController | null>(null);
  // 使用 ref 追踪累积的消息，避免闭包问题
  const accumulatedMessageRef = useRef<string>('');

  /**
   * 发送消息并流式接收回复
   */
  const sendMessage = useCallback(async (
    query: string,
    conversationId?: string
  ): Promise<void> => {
    // 取消之前的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    
    setIsStreaming(true);
    setCurrentMessage('');
    setError(null);
    setMetadata(undefined);
    accumulatedMessageRef.current = ''; // 重置累积消息

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          query,
          conversation_id: conversationId,
          retrieval_config: {
            top_k: 5,
            similarity_threshold: 0.2,
            enable_rerank: true,
          },
          generation_config: {
            temperature: 0.7,
            max_tokens: 1024,
          },
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // 解析SSE事件
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // 保留最后一个不完整的行

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            
            if (data === '[DONE]') {
              // 流式传输完成
              setIsStreaming(false);
              const finalMessage: StreamingMessage = {
                role: 'assistant',
                content: accumulatedMessageRef.current,
              };
              onComplete?.(finalMessage);
              continue;
            }

            try {
              const event: StreamEvent = JSON.parse(data);
              
              switch (event.type) {
                case 'token':
                  if (event.content) {
                    accumulatedMessageRef.current += event.content;
                    setCurrentMessage(accumulatedMessageRef.current);
                  }
                  break;
                  
                case 'metadata':
                  setMetadata(event.metadata);
                  break;
                  
                case 'done':
                  setIsStreaming(false);
                  const doneMessage: StreamingMessage = {
                    role: 'assistant',
                    content: accumulatedMessageRef.current,
                  };
                  onComplete?.(doneMessage);
                  break;
                  
                case 'error':
                  throw new Error(event.content || 'Stream error');
              }
            } catch (parseError) {
              // 忽略解析错误，可能是不完整的数据
              console.warn('Failed to parse SSE event:', data);
            }
          }
        }
      }

    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      
      // 忽略中止错误
      if (error.name === 'AbortError') {
        return;
      }
      
      setError(error);
      setIsStreaming(false);
      onError?.(error);
    }
  }, [apiUrl, onComplete, onError]);

  /**
   * 停止流式传输
   */
  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  /**
   * 重置状态
   */
  const reset = useCallback(() => {
    setCurrentMessage('');
    setMetadata(undefined);
    setError(null);
    setIsStreaming(false);
  }, []);

  return {
    sendMessage,
    stopStreaming,
    reset,
    isStreaming,
    currentMessage,
    metadata,
    error,
  };
}

export default useStreamingChat;
