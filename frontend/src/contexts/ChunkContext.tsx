import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import type { ReactNode } from 'react';
import { chunkingApi, documentApi, retrievalApi } from '../services/api';
import type { Chunk, ChunkConfig } from '../types';
import type { LocalDocItem } from '../services/api';

interface ChunkContextType {
  // 文档状态
  localDocs: LocalDocItem[];
  loadingStatus: string;
  selectedLocalDoc: string;
  docContent: string;
  
  // 切分状态
  config: ChunkConfig;
  chunks: Chunk[];
  selectedChunks: Set<string>;
  expandedChunks: Set<string>;
  similarityThreshold: number;
  
  // 批量模式状态
  isBatchMode: boolean;
  selectedBatchDocs: Set<string>;
  batchProcessing: boolean;
  batchProgress: { current: number; total: number; currentDoc: string } | null;
  expandedFolders: Set<string>;
  
  // 模态框状态
  showSimilarModal: boolean;
  showOriginalDocModal: boolean;
  showDocSelector: boolean;
  similarChunks: any[];
  currentSimilarChunkId: string;
  currentSimilarChunkContent: string;
  
  // 操作方法
  setSelectedLocalDoc: (id: string) => void;
  setDocContent: (content: string) => void;
  setConfig: React.Dispatch<React.SetStateAction<ChunkConfig>>;
  setChunks: React.Dispatch<React.SetStateAction<Chunk[]>>;
  setSelectedChunks: React.Dispatch<React.SetStateAction<Set<string>>>;
  setSimilarityThreshold: (threshold: number) => void;
  setIsBatchMode: (mode: boolean) => void;
  setSelectedBatchDocs: React.Dispatch<React.SetStateAction<Set<string>>>;
  setExpandedFolders: React.Dispatch<React.SetStateAction<Set<string>>>;
  setShowSimilarModal: (show: boolean) => void;
  setShowOriginalDocModal: (show: boolean) => void;
  setShowDocSelector: (show: boolean) => void;
  
  // 业务方法
  handleSelectLocalDoc: (docId: string) => Promise<void>;
  handleChunk: () => Promise<void>;
  handleFindSimilar: (chunkId: string, chunkContent: string) => Promise<void>;
  handleReSearchSimilar: () => Promise<void>;
  handleToggleExpand: (chunkId: string) => void;
  handleSelectAll: (checked: boolean) => void;
  handleSelectChunk: (id: string, checked: boolean) => void;
  handleBatchDelete: () => void;
  handleBatchMerge: () => void;
  handleReset: () => void;
  handleBatchChunk: () => Promise<void>;
  toggleFolder: (path: string) => void;
  toggleFolderSelection: (item: LocalDocItem) => void;
  collectFileIds: (item: LocalDocItem) => string[];
  isFolderFullySelected: (item: LocalDocItem) => boolean;
  isFolderPartiallySelected: (item: LocalDocItem) => boolean;
  handleToggleBatchDoc: (docId: string, checked: boolean) => void;
  handleSelectAllBatchDocs: (checked: boolean) => void;
}

const ChunkContext = createContext<ChunkContextType | undefined>(undefined);

export const ChunkProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // 文档状态
  const [localDocs, setLocalDocs] = useState<LocalDocItem[]>([]);
  const [loadingStatus, setLoadingStatus] = useState<string>('');
  const [selectedLocalDoc, setSelectedLocalDoc] = useState<string>('');
  const [docContent, setDocContent] = useState('');
  
  // 切分状态
  const [config, setConfig] = useState<ChunkConfig>({
    type: 'intelligent',
    chunkTokenSize: 512,
    delimiters: ['\n', '。', '；', '！', '？'],
    childrenDelimiters: [],
    enableChildren: false,
    overlappedPercent: 0.1,
    tableContextSize: 0,
    imageContextSize: 0,
    length: 500,
    overlap: 50,
    customRule: ''
  });
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [selectedChunks, setSelectedChunks] = useState<Set<string>>(new Set());
  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set());
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  
  // 批量模式状态
  const [isBatchMode, setIsBatchMode] = useState(false);
  const [selectedBatchDocs, setSelectedBatchDocs] = useState<Set<string>>(new Set());
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number; currentDoc: string } | null>(null);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  
  // 模态框状态
  const [showSimilarModal, setShowSimilarModal] = useState(false);
  const [showOriginalDocModal, setShowOriginalDocModal] = useState(false);
  const [showDocSelector, setShowDocSelector] = useState(false);
  const [similarChunks, setSimilarChunks] = useState<any[]>([]);
  const [currentSimilarChunkId, setCurrentSimilarChunkId] = useState<string>('');
  const [currentSimilarChunkContent, setCurrentSimilarChunkContent] = useState<string>('');
  


  // 加载本地文档列表
  useEffect(() => {
    const loadLocalDocs = async () => {
      try {
        setLoadingStatus('加载中...');
        const docs = await documentApi.listLocalDocs();
        if (docs && Array.isArray(docs)) {
          setLocalDocs(docs);
          const countFiles = (items: LocalDocItem[]): number => {
            return items.reduce((acc, item) => {
              if (item.type === 'file') return acc + 1;
              return acc + countFiles(item.children || []);
            }, 0);
          };
          const fileCount = countFiles(docs);
          setLoadingStatus(`已加载 ${fileCount} 个文件`);
        } else {
          setLocalDocs([]);
          setLoadingStatus('暂无文档');
        }
      } catch (error) {
        console.error('加载本地文档列表失败:', error);
        setLocalDocs([]);
        setLoadingStatus('加载失败');
      }
    };
    loadLocalDocs();
  }, []);

  // 选择本地文档
  const handleSelectLocalDoc = useCallback(async (docId: string) => {
    const findDoc = (items: LocalDocItem[]): LocalDocItem | undefined => {
      for (const item of items) {
        if (item.id === docId) return item;
        if (item.children) {
          const found = findDoc(item.children);
          if (found) return found;
        }
      }
      return undefined;
    };

    const doc = findDoc(localDocs);
    if (doc && doc.type === 'file') {
      setSelectedLocalDoc(docId);
      setDocContent(`# ${doc.name}\n\n加载中...`);

      try {
        const contentResponse = await documentApi.getLocalDocContent(docId);
        if (contentResponse && contentResponse.content) {
          setDocContent(contentResponse.content);
        } else {
          setDocContent(`# ${doc.name}\n\n## 文档内容预览\n\n无法加载文档内容。`);
        }
      } catch (error) {
        console.error('加载本地文档内容失败:', error);
        setDocContent(`# ${doc.name}\n\n## 文档内容预览\n\n加载失败，请重试。`);
      }
    }
  }, [localDocs]);

  // 执行切分
  const handleChunk = useCallback(async () => {
    if (!selectedLocalDoc) {
      alert('请先选择一个文档');
      return;
    }

    try {
      const backendConfig = {
        type: config.type,
        chunk_token_size: config.chunkTokenSize,
        delimiters: config.delimiters,
        children_delimiters: config.childrenDelimiters,
        enable_children: config.enableChildren,
        overlapped_percent: config.overlappedPercent,
        table_context_size: config.tableContextSize,
        image_context_size: config.imageContextSize,
        length: config.length,
        overlap: config.overlap,
        custom_rule: config.customRule
      };

      const response = await chunkingApi.split(selectedLocalDoc, backendConfig as any);

      const newChunks: Chunk[] = response.chunks.map((chunk: any) => ({
        id: chunk.id,
        document_id: selectedLocalDoc,
        num: chunk.num,
        content: chunk.content,
        length: chunk.length,
        embedding_status: 'pending' as const
      }));

      setChunks(newChunks);
      setSelectedChunks(new Set());

      try {
        await chunkingApi.embed(selectedLocalDoc);
        alert(`✅ 切分完成！共生成 ${newChunks.length} 个片段，并已自动向量化完成。`);
      } catch (embedError) {
        console.error('自动向量化失败:', embedError);
        alert(`⚠️ 切分完成！共生成 ${newChunks.length} 个片段，但自动向量化失败。`);
      }
    } catch (error) {
      console.error('切分失败:', error);
      alert(`❌ 切分失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  }, [selectedLocalDoc, config]);

  // 查找相似片段
  const handleFindSimilar = useCallback(async (chunkId: string, chunkContent: string) => {
    try {
      const response = await retrievalApi.findSimilarChunks(chunkId, chunkContent, similarityThreshold, 5);
      setCurrentSimilarChunkId(chunkId);
      setCurrentSimilarChunkContent(chunkContent);
      setSimilarChunks(response.similar_chunks);
      setShowSimilarModal(true);
    } catch (error) {
      console.error('查找相似片段失败:', error);
      alert(`❌ 查找相似片段失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  }, [similarityThreshold]);

  // 重新搜索相似片段
  const handleReSearchSimilar = useCallback(async () => {
    if (!currentSimilarChunkId || !currentSimilarChunkContent) return;
    try {
      const response = await retrievalApi.findSimilarChunks(currentSimilarChunkId, currentSimilarChunkContent, similarityThreshold, 5);
      setSimilarChunks(response.similar_chunks);
    } catch (error) {
      console.error('重新查找相似片段失败:', error);
      alert(`重新查找相似片段失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  }, [currentSimilarChunkId, currentSimilarChunkContent, similarityThreshold]);

  // 展开/折叠片段
  const handleToggleExpand = useCallback((chunkId: string) => {
    setExpandedChunks(prev => {
      const newSet = new Set(prev);
      if (newSet.has(chunkId)) {
        newSet.delete(chunkId);
      } else {
        newSet.add(chunkId);
      }
      return newSet;
    });
  }, []);

  // 全选
  const handleSelectAll = useCallback((checked: boolean) => {
    if (checked) {
      setSelectedChunks(new Set(chunks.map(c => c.id)));
    } else {
      setSelectedChunks(new Set());
    }
  }, [chunks]);

  // 选择片段
  const handleSelectChunk = useCallback((id: string, checked: boolean) => {
    setSelectedChunks(prev => {
      const newSet = new Set(prev);
      if (checked) {
        newSet.add(id);
      } else {
        newSet.delete(id);
      }
      return newSet;
    });
  }, []);

  // 批量删除
  const handleBatchDelete = useCallback(() => {
    if (selectedChunks.size === 0) {
      alert('请先选择要删除的片段');
      return;
    }
    if (confirm(`确定要删除选中的${selectedChunks.size}个片段吗?`)) {
      setChunks(prev => prev.filter(c => !selectedChunks.has(c.id)));
      setSelectedChunks(new Set());
      alert('批量删除成功!');
    }
  }, [selectedChunks]);

  // 批量合并
  const handleBatchMerge = useCallback(() => {
    if (selectedChunks.size < 2) {
      alert('请至少选择2个片段进行合并');
      return;
    }

    const selectedChunksList = chunks.filter(c => selectedChunks.has(c.id));
    const mergedContent = selectedChunksList.map(c => c.content).join('\n\n');

    const mergedChunk: Chunk = {
      id: `merged_${Date.now()}`,
      document_id: selectedLocalDoc,
      num: Math.min(...selectedChunksList.map(c => c.num)),
      content: mergedContent,
      length: mergedContent.length,
      embedding_status: 'pending'
    };

    setChunks(prev => [...prev.filter(c => !selectedChunks.has(c.id)), mergedChunk]);
    setSelectedChunks(new Set());
    alert('片段合并成功!');
  }, [selectedChunks, chunks, selectedLocalDoc]);

  // 重置
  const handleReset = useCallback(() => {
    setChunks([]);
    setSelectedChunks(new Set());
  }, []);

  // 批量切分
  const handleBatchChunk = useCallback(async () => {
    if (selectedBatchDocs.size === 0) {
      alert('请至少选择一个文档进行批量切分');
      return;
    }

    const confirmed = confirm(`确定要对选中的 ${selectedBatchDocs.size} 个文档进行批量切分吗？`);
    if (!confirmed) return;

    setBatchProcessing(true);
    setBatchProgress({ current: 0, total: selectedBatchDocs.size, currentDoc: '' });

    try {
      const docIds = Array.from(selectedBatchDocs);
      const response = await chunkingApi.batchSplit(docIds, config, true);

      if (response.success) {
        alert(`✅ 批量切分完成！\n\n成功: ${response.data.success}/${response.data.total} 个文档\n共生成: ${response.data.total_chunks} 个片段`);
      } else {
        alert(`⚠️ 批量切分部分完成\n\n成功: ${response.data.success}/${response.data.total}\n失败: ${response.data.failed}`);
      }

      setSelectedBatchDocs(new Set());
    } catch (error) {
      console.error('批量切分失败:', error);
      alert(`❌ 批量切分失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setBatchProcessing(false);
      setBatchProgress(null);
    }
  }, [selectedBatchDocs, config]);

  // 展开/折叠文件夹
  const toggleFolder = useCallback((path: string) => {
    setExpandedFolders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(path)) {
        newSet.delete(path);
      } else {
        newSet.add(path);
      }
      return newSet;
    });
  }, []);

  // 收集文件夹下所有文件ID
  const collectFileIds = useCallback((item: LocalDocItem): string[] => {
    if (item.type === 'file') {
      return item.id ? [item.id] : [];
    }
    const ids: string[] = [];
    if (item.children) {
      for (const child of item.children) {
        ids.push(...collectFileIds(child));
      }
    }
    return ids;
  }, []);

  // 检查文件夹是否全部选中
  const isFolderFullySelected = useCallback((item: LocalDocItem): boolean => {
    const fileIds = collectFileIds(item);
    return fileIds.length > 0 && fileIds.every(id => selectedBatchDocs.has(id));
  }, [collectFileIds, selectedBatchDocs]);

  // 检查文件夹是否部分选中
  const isFolderPartiallySelected = useCallback((item: LocalDocItem): boolean => {
    const fileIds = collectFileIds(item);
    const selectedCount = fileIds.filter(id => selectedBatchDocs.has(id)).length;
    return selectedCount > 0 && selectedCount < fileIds.length;
  }, [collectFileIds, selectedBatchDocs]);

  // 切换文件夹选择状态
  const toggleFolderSelection = useCallback((item: LocalDocItem) => {
    const fileIds = collectFileIds(item);
    const allSelected = isFolderFullySelected(item);

    setSelectedBatchDocs(prev => {
      const newSet = new Set(prev);
      if (allSelected) {
        fileIds.forEach(id => newSet.delete(id));
      } else {
        fileIds.forEach(id => newSet.add(id));
      }
      return newSet;
    });
  }, [collectFileIds, isFolderFullySelected]);

  // 切换单个文档选择
  const handleToggleBatchDoc = useCallback((docId: string, checked: boolean) => {
    setSelectedBatchDocs(prev => {
      const newSet = new Set(prev);
      if (checked) {
        newSet.add(docId);
      } else {
        newSet.delete(docId);
      }
      return newSet;
    });
  }, []);

  // 全选所有文档
  const handleSelectAllBatchDocs = useCallback((checked: boolean) => {
    if (checked) {
      const allFileIds: string[] = [];
      const collectAllFileIds = (items: LocalDocItem[]) => {
        for (const item of items) {
          if (item.type === 'file' && item.id) {
            allFileIds.push(item.id);
          } else if (item.children) {
            collectAllFileIds(item.children);
          }
        }
      };
      collectAllFileIds(localDocs);
      setSelectedBatchDocs(new Set(allFileIds));
    } else {
      setSelectedBatchDocs(new Set());
    }
  }, [localDocs]);

  return (
    <ChunkContext.Provider value={{
      localDocs,
      loadingStatus,
      selectedLocalDoc,
      docContent,
      config,
      chunks,
      selectedChunks,
      expandedChunks,
      similarityThreshold,
      isBatchMode,
      selectedBatchDocs,
      batchProcessing,
      batchProgress,
      expandedFolders,
      showSimilarModal,
      showOriginalDocModal,
      showDocSelector,
      similarChunks,
      currentSimilarChunkId,
      currentSimilarChunkContent,
      setSelectedLocalDoc,
      setDocContent,
      setConfig,
      setChunks,
      setSelectedChunks,
      setSimilarityThreshold,
      setIsBatchMode,
      setSelectedBatchDocs,
      setExpandedFolders,
      setShowSimilarModal,
      setShowOriginalDocModal,
      setShowDocSelector,
      handleSelectLocalDoc,
      handleChunk,
      handleFindSimilar,
      handleReSearchSimilar,
      handleToggleExpand,
      handleSelectAll,
      handleSelectChunk,
      handleBatchDelete,
      handleBatchMerge,
      handleReset,
      handleBatchChunk,
      toggleFolder,
      toggleFolderSelection,
      collectFileIds,
      isFolderFullySelected,
      isFolderPartiallySelected,
      handleToggleBatchDoc,
      handleSelectAllBatchDocs,
    }}>
      {children}
    </ChunkContext.Provider>
  );
};

export const useChunk = () => {
  const context = useContext(ChunkContext);
  if (context === undefined) {
    throw new Error('useChunk must be used within a ChunkProvider');
  }
  return context;
};
