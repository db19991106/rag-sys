// =========================================
// RAG System - Unified Type Definitions
// =========================================

// ========== Document Types ==========
export interface Document {
  id: string;
  name: string;
  size: number | string;  // number for raw bytes, string for formatted display
  status: DocumentStatus;
  upload_time: string;
  time?: string;  // Optional formatted display time
  chunk_count?: number;
  category?: string;
  tags?: string[];
  preview?: string;
}

export type DocumentStatus = 'pending' | 'split' | 'indexed' | 'error' | 'processing';

// ========== Chunk Types ==========
export interface Chunk {
  id: string;
  document_id: string;
  num: number;
  content: string;
  length: number;
  embedding_status: EmbeddingStatus;
  embedding_dimension?: number;
  vecStatus?: EmbeddingStatus;
  vecDim?: number;
  vecLength?: number;
  highlightKeywords?: string[];
}

export type EmbeddingStatus = 'pending' | 'processing' | 'success' | 'error';

// ========== Retrieval Types ==========
export interface RetrievalResult {
  chunk_id: string;
  document_id: string;
  document_name: string;
  chunk_num: number;
  content: string;
  similarity: number;
  match_keywords: string[];
  id?: string;
  num?: number;
  sim?: number;
  matchKeywords?: string[];
  vecStatus?: EmbeddingStatus;
}

export interface SimilarChunkResult {
  chunk_id: string;
  document_id: string;
  document_name: string;
  chunk_num: number;
  content: string;
  similarity: number;
}

// ========== User Types ==========
export interface UserInfo {
  id: string;
  username: string;
  email: string;
  isAuthenticated: boolean;
  permissions: string[];
  lastLogin?: string;
  created_at?: string;
}

// ========== Configuration Types ==========
export interface ChunkConfig {
  type: ChunkType;
  chunkTokenSize: number;
  delimiters: string[];
  childrenDelimiters: string[];
  enableChildren: boolean;
  overlappedPercent: number;
  tableContextSize: number;
  imageContextSize: number;
  length?: number;
  overlap?: number;
  customRule?: string;
}

export type ChunkType = 
  | 'naive' 
  | 'char' 
  | 'sentence' 
  | 'paragraph' 
  | 'qa' 
  | 'table' 
  | 'picture'
  | 'resume' 
  | 'manual' 
  | 'paper' 
  | 'book' 
  | 'laws' 
  | 'custom' 
  | 'intelligent' 
  | 'enhanced'
  | 'product' 
  | 'technical' 
  | 'compliance' 
  | 'hr' 
  | 'project' 
  | 'hybrid' 
  | 'layered' 
  | 'layered_llm';

export interface EmbeddingConfig {
  model_type: EmbeddingModelType;
  model_name: string;
  batch_size: number;
  device: string;
}

export type EmbeddingModelType = 'sentence-transformers' | 'bge' | 'openai';

export interface VectorDBConfig {
  db_type: VectorDBType;
  dimension: number;
  index_type: string;
  host?: string;
  port?: number;
  collection_name?: string;
}

export type VectorDBType = 'faiss' | 'milvus' | 'milvus_lite' | 'qdrant';

export interface RetrievalConfig {
  top_k: number;
  similarity_threshold: number;
  algorithm: SimilarityAlgorithm;
  enable_rerank?: boolean;
  reranker_type?: RerankerType;
  reranker_model?: string;
  reranker_top_k?: number;
  reranker_threshold?: number;
  // Legacy fields for backward compatibility
  topK?: number;
  simThreshold?: number;
  simAlgo?: SimilarityAlgorithm;
  retrievalMode?: 'vector' | 'hybrid';
}

export type SimilarityAlgorithm = 'cosine' | 'euclidean' | 'dot' | 'manhattan';
export type RerankerType = 'cross_encoder' | 'colbert' | 'mmr' | 'none';

export interface GenerationConfig {
  llm_provider?: string;
  llm_model?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
}

// ========== RAG Types ==========
export interface RAGRequest {
  query: string;
  retrieval_config: RetrievalConfig;
  generation_config: GenerationConfig;
  conversation_id?: string;
}

export interface RAGResponse {
  query: string;
  answer: string;
  context_chunks: RetrievalResult[];
  generation_time_ms: number;
  retrieval_time_ms: number;
  total_time_ms: number;
  tokens_used?: number;
}

// ========== Stream Types ==========
export interface StreamEvent {
  type: 'token' | 'done' | 'error' | 'metadata';
  content?: string;
  metadata?: StreamMetadata;
}

export interface StreamMetadata {
  query?: string;
  context_chunks?: ContextChunk[];
  retrieval_time_ms?: number;
  generation_time_ms?: number;
  total_time_ms?: number;
}

export interface ContextChunk {
  chunk_id: string;
  document_name: string;
  similarity: number;
}

// ========== Vector DB Types ==========
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

export interface VectorDBStatus {
  db_type: string;
  total_vectors: number;
  dimension: number;
  status: string;
  db_size: number;
}

// ========== Local Document Types ==========
export interface LocalDocItem {
  type: 'folder' | 'file';
  id?: string;
  name: string;
  path: string;
  size?: string;
  extension?: string;
  children?: LocalDocItem[];
}

// ========== Conversation Types ==========
export interface Conversation {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  metadata?: MessageMetadata;
}

export interface MessageMetadata {
  intent?: string;
  confidence?: number;
  context_chunks?: ContextChunk[];
  generation_time_ms?: number;
}

// ========== Settings Types ==========
export interface SystemSettings {
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
}

// ========== API Response Types ==========
export interface ApiResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// ========== Intent Recognition Types ==========
export interface IntentRecognitionResult {
  intent: string;
  confidence: number;
  details: IntentDetails;
}

export interface IntentDetails {
  category?: string;
  sub_intent?: string;
  entities?: Record<string, any>;
}

// ========== Summary Types ==========
export interface SummaryResponse {
  summary: string;
  key_points?: string[];
  word_count?: number;
}
