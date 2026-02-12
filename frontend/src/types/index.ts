// 文档类型
export interface Document {
  id: string;
  name: string;
  size: string;
  time: string;
  status: 'pending' | 'split' | 'index' | 'error';
  preview?: string;
  category?: string;
  tags?: string[];
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
  highlightKeywords?: string[];
}

// 检索结果类型
export interface RetrievalResult {
  id: string;
  num: number;
  content: string;
  sim: number;
  matchKeywords?: string[];
  vecStatus?: 'pending' | 'processing' | 'success' | 'error';
}

// 用户信息类型
export interface UserInfo {
  id: string;
  username: string;
  email: string;
  isAuthenticated: boolean;
  permissions: string[];
  lastLogin?: string;
  created_at?: string;
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
  type: 'naive' | 'char' | 'sentence' | 'paragraph' | 'qa' | 'table' | 'picture' |
       'resume' | 'manual' | 'paper' | 'book' | 'laws' | 'custom' | 'intelligent' | 'enhanced' |
       'product' | 'technical' | 'compliance' | 'hr' | 'project';
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