// API 客户端服务

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// 通用请求函数
async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

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
  // 上传文档
  upload: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: '上传失败' }));
      throw new Error(error.message || error.detail || '上传失败');
    }

    return response.json();
  },

  // 获取文档列表
  list: async () => {
    return request<Document[]>('/documents/list');
  },

  // 获取文档详情
  get: async (docId: string) => {
    return request<Document>(`/documents/${docId}`);
  },

  // 获取文档内容
  getContent: async (docId: string) => {
    return request<{ content: string }>(`/documents/${docId}/content`);
  },

  // 删除文档
  delete: async (docId: string) => {
    return request<{ success: boolean; message: string }>(`/documents/${docId}`, {
      method: 'DELETE',
    });
  },

  // 批量删除文档
  batchDelete: async (docIds: string[]) => {
    return request<{ success: boolean; message: string }>('/documents/batch-delete', {
      method: 'POST',
      body: JSON.stringify(docIds),
    });
  },
};

// ========== 文档切分 API ==========
export const chunkingApi = {
  // 切分文档
  split: async (docId: string, config: ChunkConfig) => {
    return request<{ chunks: Chunk[]; total: number }>(`/chunking/split?doc_id=${docId}`, {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  // 向量化片段
  embed: async (docId: string) => {
    return request<{ success: boolean; message: string }>(`/chunking/embed?doc_id=${docId}`, {
      method: 'POST',
    });
  },
};

// ========== 向量嵌入 API ==========
export const embeddingApi = {
  // 加载嵌入模型
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

  // 获取嵌入模型状态
  getStatus: async () => {
    return request<{ success: boolean; message: string; data: any }>('/embedding/status');
  },
};

// ========== 向量数据库 API ==========
export const vectorDbApi = {
  // 初始化向量数据库
  init: async (config: VectorDBConfig) => {
    return request<{ success: boolean; message: string }>('/vector-db/init', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  // 获取向量数据库状态
  getStatus: async () => {
    return request<{
      db_type: string;
      total_vectors: number;
      dimension: number;
      status: string;
    }>('/vector-db/status');
  },

  // 保存向量数据库
  save: async () => {
    return request<{ success: boolean; message: string }>('/vector-db/save', {
      method: 'POST',
    });
  },

  // 获取向量库文档列表
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

  // 删除向量库文档
  deleteDocument: async (documentId: string) => {
    return request<{ success: boolean; message: string; data?: any }>(`/vector-db/documents/${documentId}`, {
      method: 'DELETE',
    });
  },

  // 删除单个向量片段
  deleteChunk: async (vectorId: string) => {
    return request<{ success: boolean; message: string; data?: any }>(`/vector-db/chunks/${vectorId}`, {
      method: 'DELETE',
    });
  },
};

// ========== 重排序器 API ==========
export const rerankerApi = {
  // 初始化重排序器
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

  // 获取重排序器状态
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
  // 获取系统设置
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

  // 更新系统设置
  update: async (settings: any) => {
    return request('/settings/', {
      method: 'POST',
      body: JSON.stringify(settings),
    });
  },

  // 重置系统设置
  reset: async () => {
    return request<{ success: boolean; message: string }>('/settings/reset', {
      method: 'POST',
    });
  },
};

// ========== 检索 API ==========
export const retrievalApi = {
  // 执行检索
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

  // 查找相似片段
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
  // 生成回答
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

  // 识别意图
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
  type: 'char' | 'sentence' | 'paragraph' | 'custom';
  length: number;
  overlap: number;
  custom_rule: string;
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
  chunks: {
    vector_id: string;
    chunk_num: number;
    content: string;
  }[];
}