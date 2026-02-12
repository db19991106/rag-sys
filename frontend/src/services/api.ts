// API 客户端服务

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

  const response = await fetch(url, { ...defaultOptions, ...options });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: '请求失败' }));
    throw new Error(error.message || error.detail || '请求失败');
  }

  return response.json();
}

// ========== 文档管理 API ==========
export const documentApi = {
  // 上传文档（注意：FormData 不需要设置 Content-Type）
  upload: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

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
  },

  // 其他方法保持不变...
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

  // 获取本地 data/docs 目录中的文档列表
  listLocalDocs: async () => {
    return request<Array<{ id: string; name: string; size: string; path: string }>>('/documents/local-docs');
  },

  // 获取本地文档内容
  getLocalDocContent: async (docId: string) => {
    return request<{ content: string }>(`/documents/local-docs/${docId}/content`);
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
    return request<{
      db_type: string;
      total_vectors: number;
      dimension: number;
      status: string;
    }>('/vector-db/status');
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
    return request<{
      query: string;
      answer: string;
      context_chunks: RetrievalResult[];
      generation_time_ms: number;
      retrieval_time_ms: number;
      total_time_ms: number;
      tokens_used?: number;
    }>('/rag/generate', {
      method: 'POST',
      body: JSON.stringify(requestData),
    });
  },

  recognizeIntent: async (query: string) => {
    return request<{
      intent: string;
      confidence: number;
      details: any;
    }>('/rag/recognize-intent', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  },

  generateSummary: async (text: string) => {
    return request<{
      summary: string;
    }>('/summary/generate', {
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
      data: Array<{
        id: string;
        title: string;
        message_count: number;
        created_at: string;
        updated_at: string;
      }>;
    }>('/conversations/', {
      method: 'GET',
    });
  },
};

// ========== 类型定义 ==========
export interface Document {
  id: string;
  name: string;
  size: number;
  status: string;
  upload_time: string;
  chunk_count?: number;
  category?: string;
  tags?: string[];
}

export interface Chunk {
  id: string;
  document_id: string;
  num: number;
  content: string;
  length: number;
  embedding_status: string;
  embedding_dimension?: number;
}

export interface ChunkConfig {
  type: 'naive' | 'intelligent' | 'enhanced' | 'char' | 'sentence' | 'paragraph' | 'qa' | 'paper' | 'laws' | 'book' | 'table' | 'custom' | 'product' | 'technical' | 'compliance' | 'hr' | 'project';
  chunkTokenSize: number;
  delimiters: string[];
  childrenDelimiters: string[];
  enableChildren: boolean;
  overlappedPercent: number;
  tableContextSize: number;
  imageContextSize: number;
  length: number;
  overlap: number;
  customRule: string;
}

export interface EmbeddingConfig {
  model_type: 'sentence-transformers' | 'bge' | 'openai';
  model_name: string;
  batch_size: number;
  device: string;
}

export interface VectorDBConfig {
  db_type: 'faiss' | 'milvus' | 'qdrant';
  dimension: number;
  index_type: string;
  host?: string;
  port?: number;
  collection_name?: string;
}

export interface RetrievalConfig {
  top_k: number;
  similarity_threshold: number;
  algorithm: 'cosine' | 'euclidean' | 'dot';
  enable_rerank?: boolean;
  reranker_type?: 'cross_encoder' | 'colbert' | 'mmr' | 'none';
  reranker_model?: string;
  reranker_top_k?: number;
  reranker_threshold?: number;
}

export interface RetrievalResult {
  chunk_id: string;
  document_id: string;
  document_name: string;
  chunk_num: number;
  content: string;
  similarity: number;
  match_keywords: string[];
}

export interface SimilarChunkResult {
  chunk_id: string;
  document_id: string;
  document_name: string;
  chunk_num: number;
  content: string;
  similarity: number;
}

export interface GenerationConfig {
  llm_provider: string;
  llm_model: string;
  temperature: number;
  max_tokens: number;
  top_p: number;
  frequency_penalty: number;
  presence_penalty: number;
}

export interface RAGRequest {
  query: string;
  retrieval_config: RetrievalConfig;
  generation_config: GenerationConfig;
  conversation_id?: string;  // 多轮对话ID，用于保持上下文
}

export interface VectorDocument {
  document_id: string;
  document_name: string;
  chunk_count: number;
  chunks: VectorChunk[];
}

export interface VectorChunk {
  vector_id: string;
  chunk_num: number;
  content: string;
  similarity: number;
}

export interface VectorDocument {
  document_id: string;
  document_name: string;
  chunk_count: number;
  chunks: VectorChunk[];
}