// API 客户端服务

import type {
  Document,
  Chunk,
  ChunkConfig,
  EmbeddingConfig,
  VectorDBConfig,
  RetrievalConfig,
  RetrievalResult,
  RAGRequest,
  RAGResponse,
  StreamEvent,
  StreamMetadata,
  VectorDocument,
  VectorDBStatus,
  LocalDocItem,
  Conversation,
  IntentRecognitionResult,
  SummaryResponse,
  SimilarChunkResult,
} from '../types';

// 修改点：改为 /api 前缀，通过 Vite 代理转发到后端
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

// 通用请求函数
async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  // 确保 endpoint 以 / 开头
  const normalizedEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const url = `${API_BASE_URL}${normalizedEndpoint}`;

  const defaultOptions: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  try {
    const response = await fetch(url, { ...defaultOptions, ...options });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: '请求失败' }));
      throw new Error(error.message || error.detail || '请求失败');
    }

    return response.json();
  } catch (error) {
    // 处理网络错误
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('网络连接失败，请检查：\n1. 后端服务是否已启动\n2. 网络连接是否正常');
    }
    throw error;
  }
}

// ========== 文档管理 API ==========
export const documentApi = {
  // 上传文档（注意：FormData 不需要设置 Content-Type）
  upload: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
        // 不要手动设置 Content-Type，让浏览器自动设置（包含 boundary）
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ message: '上传失败' }));
        throw new Error(error.message || error.detail || '上传失败');
      }

      return response.json();
    } catch (error) {
      // 处理网络错误
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('网络连接失败，请检查后端服务是否已启动');
      }
      throw error;
    }
  },

  list: async () => {
    return request<Document[]>('/documents/list');
  },

  get: async (docId: string) => {
    return request<Document>(`/documents/${docId}`);
  },

  getContent: async (docId: string) => {
    return request<{ content: string }>(`/documents/${docId}/content`);
  },

  delete: async (docId: string) => {
    return request<{ success: boolean; message: string }>(`/documents/${docId}`, {
      method: 'DELETE',
    });
  },

  batchDelete: async (docIds: string[]) => {
    return request<{ success: boolean; message: string }>('/documents/batch-delete', {
      method: 'POST',
      body: JSON.stringify(docIds),
    });
  },

  // 获取本地 data/docs 目录中的文档列表（树形结构）
  listLocalDocs: async () => {
    return request<LocalDocItem[]>('/documents/local-docs');
  },

  // 获取本地文档内容（docId 为完整路径，自动 URL 编码）
  getLocalDocContent: async (docId: string) => {
    const encodedPath = encodeURIComponent(docId);
    return request<{ content: string; type: string; extension?: string; path?: string }>(`/documents/local-docs/${encodedPath}/content`);
  },
};

// ========== 文档切分 API ==========
export const chunkingApi = {
  split: async (docId: string, config: ChunkConfig) => {
    return request<{ chunks: Chunk[]; total: number }>(`/chunking/split?doc_id=${docId}`, {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  embed: async (docId: string) => {
    return request<{ success: boolean; message: string }>(`/chunking/embed?doc_id=${docId}`, {
      method: 'POST',
    });
  },

  // 批量切分文档
  batchSplit: async (docIds: string[], config: ChunkConfig, autoEmbed: boolean = true) => {
    return request<{
      success: boolean;
      message: string;
      data: {
        total: number;
        success: number;
        failed: number;
        total_chunks: number;
        details: Array<{
          doc_id: string;
          doc_name: string;
          status: string;
          chunk_count?: number;
          error?: string;
        }>;
      };
    }>('/chunking/batch-split', {
      method: 'POST',
      body: JSON.stringify({ doc_ids: docIds, config, auto_embed: autoEmbed }),
    });
  },
};

// ========== 向量嵌入 API ==========
export const embeddingApi = {
  load: async (config: EmbeddingConfig) => {
    return request<{
      model_name: string;
      dimension: number;
      batch_size: number;
      status: string;
      message: string;
    }>('/embedding/load', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  getStatus: async () => {
    return request<{ success: boolean; message: string; data: any }>('/embedding/status');
  },
};

// ========== 向量数据库 API ==========
export const vectorDbApi = {
  init: async (config: VectorDBConfig) => {
    return request<{ success: boolean; message: string }>('/vector-db/init', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  getStatus: async () => {
    return request<VectorDBStatus>('/vector-db/status');
  },

  save: async () => {
    return request<{ success: boolean; message: string }>('/vector-db/save', {
      method: 'POST',
    });
  },

  getDocuments: async () => {
    return request<{
      success: boolean;
      message: string;
      data: {
        total_documents: number;
        total_chunks: number;
        documents: VectorDocument[];
      };
    }>('/vector-db/documents');
  },

  deleteDocument: async (documentId: string) => {
    return request<{ success: boolean; message: string; data?: any }>(`/vector-db/documents/${documentId}`, {
      method: 'DELETE',
    });
  },

  deleteChunk: async (vectorId: string) => {
    return request<{ success: boolean; message: string; data?: any }>(`/vector-db/chunks/${vectorId}`, {
      method: 'DELETE',
    });
  },

  clearVectorDb: async () => {
    return request<{ success: boolean; message: string }>('/vector-db/clear', {
      method: 'DELETE',
    });
  },
};

// ========== 重排序器 API ==========
export const rerankerApi = {
  initialize: async (params: {
    reranker_type: 'cross_encoder' | 'colbert' | 'mmr';
    model_name: string;
    top_k?: number;
    threshold?: number;
  }) => {
    return request<{
      success: boolean;
      message: string;
      type: string;
      model: string;
    }>('/retrieval/reranker/initialize', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  },

  getStatus: async () => {
    return request<{
      is_loaded: boolean;
      type: string;
      model: string;
      top_k: number;
      threshold: number;
    }>('/retrieval/reranker/status');
  },
};

// ========== 系统设置 API ==========
export const settingsApi = {
  get: async () => {
    return request<{
      embedding_model_type: string;
      embedding_model_name: string;
      embedding_device: string;
      embedding_batch_size: number;
      enable_rerank: boolean;
      reranker_type: string;
      reranker_model: string;
      reranker_top_k: number;
      reranker_threshold: number;
      vector_db_type: string;
      vector_db_dimension: number;
      vector_db_index_type: string;
      vector_db_host: string | null;
      vector_db_port: number | null;
      vector_db_collection_name: string | null;
    }>('/settings/');
  },

  update: async (settings: any) => {
    return request('/settings/', {
      method: 'POST',
      body: JSON.stringify(settings),
    });
  },

  reset: async () => {
    return request<{ success: boolean; message: string }>('/settings/reset', {
      method: 'POST',
    });
  },
};

// ========== 检索 API ==========
export const retrievalApi = {
  search: async (query: string, config: RetrievalConfig) => {
    return request<{
      query: string;
      results: RetrievalResult[];
      total: number;
      latency_ms: number;
    }>('/retrieval/search', {
      method: 'POST',
      body: JSON.stringify({ query, config }),
    });
  },

  findSimilarChunks: async (chunkId: string, content: string, similarityThreshold: number, topK: number = 5) => {
    return request<{
      chunk_id: string;
      similar_chunks: SimilarChunkResult[];
      total: number;
    }>('/retrieval/similar-chunks', {
      method: 'POST',
      body: JSON.stringify({
        chunk_id: chunkId,
        content: content,
        similarity_threshold: similarityThreshold,
        top_k: topK
      }),
    });
  },
};

// ========== RAG 生成 API ==========
export const ragApi = {
  generate: async (requestData: RAGRequest) => {
    return request<RAGResponse>('/rag/generate', {
      method: 'POST',
      body: JSON.stringify(requestData),
    });
  },

  /**
   * 流式生成回答 - SSE流式输出
   */
  generateStream: async (
    requestData: RAGRequest,
    callbacks: {
      onToken?: (token: string) => void;
      onMetadata?: (metadata: StreamMetadata) => void;
      onComplete?: (fullResponse: string) => void;
      onError?: (error: Error) => void;
    }
  ): Promise<AbortController> => {
    const controller = new AbortController();
    
    try {
      const response = await fetch(`${API_BASE_URL}/rag/generate/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(requestData),
        signal: controller.signal,
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
      let fullResponse = '';

      // 异步读取流
      const readStream = async () => {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          
          // 解析SSE事件
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              
              if (data === '[DONE]') {
                callbacks.onComplete?.(fullResponse);
                return;
              }

              try {
                const event: StreamEvent = JSON.parse(data);
                
                switch (event.type) {
                  case 'token':
                    if (event.content) {
                      fullResponse += event.content;
                      callbacks.onToken?.(event.content);
                    }
                    break;
                    
                  case 'metadata':
                    callbacks.onMetadata?.(event.metadata || {});
                    break;
                    
                  case 'done':
                    callbacks.onComplete?.(fullResponse);
                    return;
                    
                  case 'error':
                    throw new Error(event.content || 'Stream error');
                }
              } catch (parseError) {
                console.warn('Failed to parse SSE event:', data);
              }
            }
          }
        }
      };

      readStream().catch((err) => {
        if (err.name !== 'AbortError') {
          callbacks.onError?.(err);
        }
      });

    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      if (error.name !== 'AbortError') {
        callbacks.onError?.(error);
      }
    }

    return controller;
  },

  recognizeIntent: async (query: string) => {
    return request<IntentRecognitionResult>('/rag/recognize-intent', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  },

  generateSummary: async (text: string) => {
    return request<SummaryResponse>('/summary/generate', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  },

  deleteConversation: async (conversationId: string) => {
    return request<{ success: boolean; message: string }>(`/conversations/${conversationId}`, {
      method: 'DELETE',
    });
  },

  listConversations: async () => {
    return request<{
      success: boolean;
      data: Conversation[];
    }>('/conversations/', {
      method: 'GET',
    });
  },
};

// Re-export types for convenience
export type {
  Document,
  Chunk,
  ChunkConfig,
  EmbeddingConfig,
  VectorDBConfig,
  RetrievalConfig,
  RetrievalResult,
  RAGRequest,
  RAGResponse,
  StreamMetadata,
  VectorDocument,
  LocalDocItem,
  Conversation,
};
