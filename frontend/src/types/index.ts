// 文档类型
export interface Document {
  id: number;
  name: string;
  size: string;
  time: string;
  status: 'pending' | 'split' | 'index' | 'error';
  category?: string;
  tags?: string[];
  preview?: string;
}

// 片段类型
export interface Chunk {
  id: string;
  num: number;
  content: string;
  length: number;
  vecStatus?: 'pending' | 'processing' | 'success' | 'error';
  vecDim?: number;
  vecLength?: number;
  vecSim?: number;
  matchKeywords?: string[];
  highlightKeywords?: string[];
  createTime?: string;
}

// 检索结果类型
export interface RetrievalResult {
  id: string;
  num: number;
  content: string;
  sim: number;
  matchKeywords: string[];
  vecStatus: 'pending' | 'processing' | 'success' | 'error';
}

// 用户信息类型
export interface UserInfo {
  username: string;
  isAuthenticated: boolean;
}

// 检索配置类型
export interface RetrievalConfig {
  topK: number;
  simThreshold: number;
  simAlgo: 'cosine' | 'euclidean' | 'manhattan';
  retrievalMode: 'vector' | 'hybrid';
}

// 切分配置类型
export interface ChunkConfig {
  // 切分方式
  type: 'naive' | 'char' | 'sentence' | 'paragraph' | 'qa' | 'table' | 'picture' |
       'resume' | 'manual' | 'paper' | 'book' | 'laws' | 'custom';

  // Token配置
  chunkTokenSize: number;  // 每个chunk的token数量（128-2048）

  // 分隔符配置
  delimiters: string[];  // 主分隔符列表
  childrenDelimiters: string[];  // 子分隔符列表
  enableChildren: boolean;  // 是否启用子分隔符

  // 重叠配置
  overlappedPercent: number;  // 重叠百分比（0-0.5）

  // 上下文配置
  tableContextSize: number;  // 表格上下文大小（token）
  imageContextSize: number;  // 图片上下文大小（token）

  // 兼容旧版本（已废弃）
  length?: number;
  overlap?: number;
  customRule?: string;
}

// 切分模板类型
export interface ChunkTemplate {
  id: string;
  name: string;
  config: ChunkConfig;
  createTime: string;
}

// 向量模型配置类型
export interface VectorModelConfig {
  id: string;
  name: string;
  type: 'BGE' | 'text2vec' | 'ernie' | 'openai' | 'custom';
  dimension: number;
  apiKey?: string;
  endpoint?: string;
}

// 向量数据库配置类型
export interface VectorDBConfig {
  id: string;
  name: string;
  type: 'FAISS' | 'Milvus' | 'Pinecone' | 'Qdrant' | 'Chroma';
  host?: string;
  port?: number;
  apiKey?: string;
  indexType?: 'HNSW' | 'IVF' | 'PQ' | 'IVF_PQ';
}

// Prompt模板类型
export interface PromptTemplate {
  id: string;
  name: string;
  template: string;
  createTime: string;
  version: number;
}

// 操作日志类型
export interface OperationLog {
  id: string;
  timestamp: string;
  operation: string;
  module: string;
  details: Record<string, any>;
  user: string;
}

// 导出配置类型
export interface ExportConfig {
  format: 'JSON' | 'CSV' | 'Excel' | 'TXT' | 'PDF';
  fields: string[];
  includeMetadata: boolean;
}

// 检索历史类型
export interface RetrievalHistory {
  id: string;
  query: string;
  timestamp: string;
  results: RetrievalResult[];
  config: RetrievalConfig;
}