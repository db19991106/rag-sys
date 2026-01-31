import React, { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import type {
  Document,
  Chunk,
  OperationLog,
  ChunkTemplate,
  VectorModelConfig,
  PromptTemplate,
  RetrievalHistory
} from '../types';

interface AppDataContextType {
  documents: Document[];
  addDocument: (doc: Document) => void;
  addDocuments: (docs: Document[]) => void;
  updateDocument: (id: number, updates: Partial<Document>) => void;
  deleteDocument: (id: number) => void;
  batchDeleteDocuments: (ids: number[]) => void;
  chunks: Chunk[];
  setChunks: (chunks: Chunk[]) => void;
  selectedDocument: Document | null;
  setSelectedDocument: (doc: Document | null) => void;
  selectedDocuments: Set<number>;
  setSelectedDocuments: (ids: Set<number>) => void;
  operationLogs: OperationLog[];
  addOperationLog: (log: Omit<OperationLog, 'id' | 'timestamp'>) => void;
  chunkTemplates: ChunkTemplate[];
  addChunkTemplate: (template: Omit<ChunkTemplate, 'id' | 'createTime'>) => void;
  updateChunkTemplate: (id: string, updates: Partial<ChunkTemplate>) => void;
  deleteChunkTemplate: (id: string) => void;
  exportChunkTemplate: (id: string) => void;
  vectorModelConfig: VectorModelConfig | null;
  setVectorModelConfig: (config: VectorModelConfig | null) => void;
  vectorDBConfig: VectorModelConfig | null;
  setVectorDBConfig: (config: VectorModelConfig | null) => void;
  promptTemplates: PromptTemplate[];
  addPromptTemplate: (template: Omit<PromptTemplate, 'id' | 'createTime' | 'version'>) => void;
  deletePromptTemplate: (id: string) => void;
  retrievalHistory: RetrievalHistory[];
  addRetrievalHistory: (history: Omit<RetrievalHistory, 'id' | 'timestamp'>) => void;
}

const AppDataContext = createContext<AppDataContextType | undefined>(undefined);

export const AppDataProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<Set<number>>(new Set());
  const [operationLogs, setOperationLogs] = useState<OperationLog[]>([]);
  const [chunkTemplates, setChunkTemplates] = useState<ChunkTemplate[]>([]);
  const [vectorModelConfig, setVectorModelConfig] = useState<VectorModelConfig | null>(null);
  const [vectorDBConfig, setVectorDBConfig] = useState<VectorModelConfig | null>(null);
  const [promptTemplates, setPromptTemplates] = useState<PromptTemplate[]>([]);
  const [retrievalHistory, setRetrievalHistory] = useState<RetrievalHistory[]>([]);

  // 将 addOperationLog 移到最前面声明，因为它被其他函数调用
  const addOperationLog = useCallback((log: Omit<OperationLog, 'id' | 'timestamp'>) => {
    const newLog: OperationLog = {
      ...log,
      id: Date.now().toString(),
      timestamp: new Date().toLocaleString('zh-CN')
    };
    setOperationLogs(prev => [newLog, ...prev].slice(0, 100)); // 保留最近100条
  }, []);

  const addDocument = useCallback((doc: Document) => {
    setDocuments(prev => [...prev, doc]);
    addOperationLog({
      operation: '上传文档',
      module: '文档管理',
      details: { fileName: doc.name, fileSize: doc.size },
      user: 'admin'
    });
  }, [addOperationLog]);

  const addDocuments = useCallback((docs: Document[]) => {
    setDocuments(prev => [...prev, ...docs]);
    addOperationLog({
      operation: '批量导入文档',
      module: '文档管理',
      details: { count: docs.length },
      user: 'admin'
    });
  }, [addOperationLog]);

  const updateDocument = useCallback((id: number, updates: Partial<Document>) => {
    setDocuments(prev =>
      prev.map(doc => (doc.id === id ? { ...doc, ...updates } : doc))
    );
    addOperationLog({
      operation: '更新文档',
      module: '文档管理',
      details: { documentId: id, updates },
      user: 'admin'
    });
  }, [addOperationLog]);

  const deleteDocument = useCallback((id: number) => {
    setDocuments(prev => prev.filter(doc => doc.id !== id));
    addOperationLog({
      operation: '删除文档',
      module: '文档管理',
      details: { documentId: id },
      user: 'admin'
    });
  }, [addOperationLog]);

  const batchDeleteDocuments = useCallback((ids: number[]) => {
    setDocuments(prev => prev.filter(doc => !ids.includes(doc.id)));
    addOperationLog({
      operation: '批量删除文档',
      module: '文档管理',
      details: { count: ids.length, ids },
      user: 'admin'
    });
  }, [addOperationLog]);

  const addChunkTemplate = useCallback((template: Omit<ChunkTemplate, 'id' | 'createTime'>) => {
    const newTemplate: ChunkTemplate = {
      ...template,
      id: Date.now().toString(),
      createTime: new Date().toLocaleString('zh-CN')
    };
    setChunkTemplates(prev => [...prev, newTemplate]);
    addOperationLog({
      operation: '保存切分模板',
      module: '文档切分',
      details: { templateName: template.name },
      user: 'admin'
    });
  }, [addOperationLog]);

  const deleteChunkTemplate = useCallback((id: string) => {
    setChunkTemplates(prev => prev.filter(t => t.id !== id));
    addOperationLog({
      operation: '删除切分模板',
      module: '文档切分',
      details: { templateId: id },
      user: 'admin'
    });
  }, [addOperationLog]);

  const updateChunkTemplate = useCallback((id: string, updates: Partial<ChunkTemplate>) => {
    setChunkTemplates(prev =>
      prev.map(t => (t.id === id ? { ...t, ...updates } : t))
    );
    addOperationLog({
      operation: '更新切分模板',
      module: '文档切分',
      details: { templateId: id, updates },
      user: 'admin'
    });
  }, [addOperationLog]);

  const exportChunkTemplate = useCallback((id: string) => {
    const template = chunkTemplates.find(t => t.id === id);
    if (template) {
      const exportData = JSON.stringify(template, null, 2);
      const blob = new Blob([exportData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `chunk-template-${template.name}-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      addOperationLog({
        operation: '导出切分模板',
        module: '文档切分',
        details: { templateName: template.name },
        user: 'admin'
      });
    }
  }, [chunkTemplates, addOperationLog]);

  const addPromptTemplate = useCallback((template: Omit<PromptTemplate, 'id' | 'createTime' | 'version'>) => {
    const newTemplate: PromptTemplate = {
      ...template,
      id: Date.now().toString(),
      createTime: new Date().toLocaleString('zh-CN'),
      version: 1
    };
    setPromptTemplates(prev => [...prev, newTemplate]);
    addOperationLog({
      operation: '保存Prompt模板',
      module: '上下文构建',
      details: { templateName: template.name },
      user: 'admin'
    });
  }, [addOperationLog]);

  const deletePromptTemplate = useCallback((id: string) => {
    setPromptTemplates(prev => prev.filter(t => t.id !== id));
    addOperationLog({
      operation: '删除Prompt模板',
      module: '上下文构建',
      details: { templateId: id },
      user: 'admin'
    });
  }, [addOperationLog]);

  const addRetrievalHistory = useCallback((history: Omit<RetrievalHistory, 'id' | 'timestamp'>) => {
    const newHistory: RetrievalHistory = {
      ...history,
      id: Date.now().toString(),
      timestamp: new Date().toLocaleString('zh-CN')
    };
    setRetrievalHistory(prev => [newHistory, ...prev].slice(0, 50)); // 保留最近50条
  }, []);

  return (
    <AppDataContext.Provider
      value={{
        documents,
        addDocument,
        addDocuments,
        updateDocument,
        deleteDocument,
        batchDeleteDocuments,
        chunks,
        setChunks,
        selectedDocument,
        setSelectedDocument,
        selectedDocuments,
        setSelectedDocuments,
        operationLogs,
        addOperationLog,
        chunkTemplates,
        addChunkTemplate,
        updateChunkTemplate,
        deleteChunkTemplate,
        exportChunkTemplate,
        vectorModelConfig,
        setVectorModelConfig,
        vectorDBConfig,
        setVectorDBConfig,
        promptTemplates,
        addPromptTemplate,
        deletePromptTemplate,
        retrievalHistory,
        addRetrievalHistory
      }}
    >
      {children}
    </AppDataContext.Provider>
  );
};

export const useAppData = () => {
  const context = useContext(AppDataContext);
  if (context === undefined) {
    throw new Error('useAppData must be used within an AppDataProvider');
  }
  return context;
};