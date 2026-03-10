import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ChunkProvider, useChunk } from '../contexts/ChunkContext';
import ChunkFileTree from '../components/ChunkFileTree';
import ChunkList from '../components/ChunkList';
import ChunkConfigPanel from '../components/ChunkConfigPanel';
import type { LocalDocItem } from '../services/api';
import './Chunk.css';

// 相似度徽章样式
const similarityBadgeClass = (similarity: number): string => {
  if (similarity >= 0.8) return 'badge-success';
  if (similarity >= 0.6) return 'badge-warning';
  return 'badge-danger';
};

// 相似片段模态框
const SimilarModal: React.FC<{
  show: boolean;
  onClose: () => void;
  similarChunks: any[];
  similarityThreshold: number;
  onThresholdChange: (value: number) => void;
  onReSearch: () => void;
}> = ({ show, onClose, similarChunks, similarityThreshold, onThresholdChange, onReSearch }) => {
  if (!show) return null;

  return (
    <div className="modal-mask modal-large" onClick={onClose}>
      <div className="modal-box modal-large-box" onClick={e => e.stopPropagation()}>
        <div className="modal-title">
          <span>
            <i className="fas fa-search"></i> 相似片段
          </span>
          <button className="modal-close" onClick={onClose}>
            <i className="fas fa-times"></i>
          </button>
        </div>
        <div className="modal-body">
          <div className="similar-threshold-control">
            <label>
              <i className="fas fa-chart-line"></i> 相似度阈值:
            </label>
            <input
              type="number"
              className="form-input"
              value={similarityThreshold}
              onChange={e => onThresholdChange(parseFloat(e.target.value))}
              min={0}
              max={1}
              step={0.05}
            />
            <button className="btn btn-primary" onClick={onReSearch}>
              <i className="fas fa-search"></i> 重新搜索
            </button>
          </div>
          {similarChunks.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">
                <i className="fas fa-search-minus"></i>
              </div>
              <h4>未找到相似片段</h4>
              <p>当前片段没有相似度高于阈值的片段</p>
            </div>
          ) : (
            <div className="similar-chunks-list">
              {similarChunks.map((similarChunk) => (
                <div key={similarChunk.chunk_id} className="similar-chunk-card">
                  <div className="similar-chunk-header">
                    <span className="similar-doc-name">
                      <i className="fas fa-file"></i> {similarChunk.document_name}
                    </span>
                    <span className="similar-chunk-num">#{similarChunk.chunk_num}</span>
                    <span className={`similar-badge ${similarityBadgeClass(similarChunk.similarity)}`}>
                      相似度: {(similarChunk.similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="similar-chunk-content">
                    {similarChunk.content}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="modal-footer">
          <button className="btn btn-default" onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  );
};

// 原文预览模态框
const OriginalDocModal: React.FC<{
  show: boolean;
  onClose: () => void;
  content: string;
}> = ({ show, onClose, content }) => {
  if (!show) return null;

  return (
    <div className="modal-mask modal-large" onClick={onClose}>
      <div className="modal-box modal-large-box" onClick={e => e.stopPropagation()}>
        <div className="modal-title">
          <span>
            <i className="fas fa-file-alt"></i> 原文档预览
          </span>
          <button className="modal-close" onClick={onClose}>
            <i className="fas fa-times"></i>
          </button>
        </div>
        <div className="modal-body modal-preview-body">
          <div className="doc-preview">
            <pre>{content}</pre>
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn btn-default" onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  );
};

// 批量切分配置栏
const BatchConfigBar: React.FC<{
  isBatchMode: boolean;
  config: any;
  selectedCount: number;
  batchProcessing: boolean;
  batchProgress: { current: number; total: number } | null;
  onConfigChange: (config: any) => void;
  onBatchChunk: () => void;
}> = ({ isBatchMode, config, selectedCount, batchProcessing, batchProgress, onConfigChange, onBatchChunk }) => {
  if (!isBatchMode) return null;

  return (
    <div className="batch-config-bar">
      <div className="batch-config-left">
        <label>
          <i className="fas fa-th-list"></i> 切分方式:
        </label>
        <select
          className="form-select"
          value={config.type}
          onChange={e => onConfigChange({ ...config, type: e.target.value })}
        >
          <option value="layered">🧠 分层智能切分</option>
          <option value="layered_llm">🤖 分层LLM切分</option>
          <option value="hybrid">🔀 混合切分</option>
          <option value="intelligent">💰 财务报销制度切分</option>
        </select>
      </div>
      <div className="batch-config-right">
        <span>已选 <strong>{selectedCount}</strong> 个文档</span>
        <button
          className="btn btn-primary"
          onClick={onBatchChunk}
          disabled={selectedCount === 0 || batchProcessing}
        >
          {batchProcessing ? (
            <>
              <i className="fas fa-spinner fa-spin"></i>
              {batchProgress && ` 处理中 ${batchProgress.current}/${batchProgress.total}`}
            </>
          ) : (
            <>
              <i className="fas fa-play"></i> 开始批量切分
            </>
          )}
        </button>
      </div>
    </div>
  );
};

// 主内容组件
const ChunkContent: React.FC = () => {
  const navigate = useNavigate();
  const {
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
    similarChunks,
    setConfig,
    setSimilarityThreshold,
    setIsBatchMode,
    setSelectedBatchDocs,
    setShowSimilarModal,
    setShowOriginalDocModal,
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
  } = useChunk();

  // 检查所有文件是否选中
  const allFilesSelected = () => {
    const allFileIds: string[] = [];
    const collect = (items: LocalDocItem[]) => {
      for (const item of items) {
        if (item.type === 'file' && item.id) allFileIds.push(item.id);
        else if (item.children) collect(item.children);
      }
    };
    collect(localDocs);
    return allFileIds.length > 0 && allFileIds.every(id => selectedBatchDocs.has(id));
  };

  // 未选择文档时的界面
  if (!selectedLocalDoc) {
    return (
      <div className="container">
        <div className="page-header">
          <h1 className="page-title">
            <i className="fas fa-scissors"></i> 文档切分与片段可视化
            <small>RAG Chunking 配置 &amp; 编辑</small>
          </h1>
        </div>

        {/* 本地文档列表 */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <i className="fas fa-folder-open"></i> 本地文档库
            </h3>
            <div className="card-header-actions">
              <div className="mode-toggle">
                <button
                  className={`btn btn-sm ${!isBatchMode ? 'btn-primary' : 'btn-outline'}`}
                  onClick={() => setIsBatchMode(false)}
                >
                  <i className="fas fa-file"></i> 单文档
                </button>
                <button
                  className={`btn btn-sm ${isBatchMode ? 'btn-primary' : 'btn-outline'}`}
                  onClick={() => setIsBatchMode(true)}
                >
                  <i className="fas fa-layer-group"></i> 批量切分
                </button>
              </div>
              <span className="tip-text">{loadingStatus}</span>
            </div>
          </div>

          <BatchConfigBar
            isBatchMode={isBatchMode}
            config={config}
            selectedCount={selectedBatchDocs.size}
            batchProcessing={batchProcessing}
            batchProgress={batchProgress}
            onConfigChange={setConfig}
            onBatchChunk={handleBatchChunk}
          />

          <div className="card-body">
            {localDocs.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon">
                  <i className="fas fa-folder-open"></i>
                </div>
                <h4>暂无本地文档</h4>
                <p>backend/data/docs 目录中没有找到文档</p>
              </div>
            ) : (
              <div className="local-docs-tree-container">
                {isBatchMode && (
                  <div className="file-tree-select-all">
                    <label>
                      <input
                        type="checkbox"
                        checked={allFilesSelected()}
                        onChange={e => handleSelectAllBatchDocs(e.target.checked)}
                      />
                      <span>全选所有文件</span>
                    </label>
                  </div>
                )}
                <ChunkFileTree
                  items={localDocs}
                  isBatchMode={isBatchMode}
                  selectedLocalDoc={selectedLocalDoc}
                  selectedBatchDocs={selectedBatchDocs}
                  expandedFolders={expandedFolders}
                  onSelectDoc={handleSelectLocalDoc}
                  onToggleFolder={toggleFolder}
                  onToggleFolderSelection={toggleFolderSelection}
                  onToggleBatchDoc={handleToggleBatchDoc}
                  isFolderFullySelected={isFolderFullySelected}
                  isFolderPartiallySelected={isFolderPartiallySelected}
                  collectFileIds={collectFileIds}
                />
              </div>
            )}
          </div>
        </div>

        {/* 上传提示 */}
        <div className="card upload-hint-card">
          <div className="card-body">
            <div className="upload-hint-icon">
              <i className="fas fa-cloud-upload-alt"></i>
            </div>
            <h3>或上传新文档</h3>
            <p>也可以从【知识文档管理】页面上传新文档</p>
            <button className="btn btn-primary" onClick={() => navigate('/documents')}>
              <i className="fas fa-arrow-left"></i> 前往文档管理
            </button>
          </div>
        </div>
      </div>
    );
  }

  // 已选择文档时的界面
  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-scissors"></i> 文档切分与片段可视化
          <small>RAG Chunking 配置 &amp; 编辑</small>
        </h1>
      </div>

      {/* 配置面板 */}
      <ChunkConfigPanel
        config={config}
        similarityThreshold={similarityThreshold}
        onConfigChange={setConfig}
        onThresholdChange={setSimilarityThreshold}
        onChunk={handleChunk}
        onReset={handleReset}
        onShowOriginal={() => setShowOriginalDocModal(true)}
      />

      {/* 切分结果 */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-list-ul"></i> 切分结果
            {chunks.length > 0 && (
              <span className="badge badge-primary">{chunks.length}</span>
            )}
          </h3>
          <div className="card-header-actions">
            <button
              className="btn btn-sm btn-outline"
              onClick={() => {
                setSelectedBatchDocs(new Set());
                handleSelectLocalDoc('');
              }}
            >
              <i className="fas fa-arrow-left"></i> 返回文档列表
            </button>
          </div>
        </div>
        <div className="card-body">
          <ChunkList
            chunks={chunks}
            selectedChunks={selectedChunks}
            expandedChunks={expandedChunks}
            onSelectAll={handleSelectAll}
            onSelectChunk={handleSelectChunk}
            onToggleExpand={handleToggleExpand}
            onFindSimilar={handleFindSimilar}
            onBatchDelete={handleBatchDelete}
            onBatchMerge={handleBatchMerge}
          />
        </div>
      </div>

      {/* 模态框 */}
      <SimilarModal
        show={showSimilarModal}
        onClose={() => setShowSimilarModal(false)}
        similarChunks={similarChunks}
        similarityThreshold={similarityThreshold}
        onThresholdChange={setSimilarityThreshold}
        onReSearch={handleReSearchSimilar}
      />

      <OriginalDocModal
        show={showOriginalDocModal}
        onClose={() => setShowOriginalDocModal(false)}
        content={docContent}
      />
    </div>
  );
};

// Chunk 主组件
const Chunk: React.FC = () => {
  return (
    <ChunkProvider>
      <ChunkContent />
    </ChunkProvider>
  );
};

export default Chunk;
