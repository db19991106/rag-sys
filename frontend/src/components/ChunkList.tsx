import React from 'react';
import type { Chunk } from '../types';

interface ChunkListProps {
  chunks: Chunk[];
  selectedChunks: Set<string>;
  expandedChunks: Set<string>;
  onSelectAll: (checked: boolean) => void;
  onSelectChunk: (id: string, checked: boolean) => void;
  onToggleExpand: (chunkId: string) => void;
  onFindSimilar: (chunkId: string, chunkContent: string) => void;
  onBatchDelete: () => void;
  onBatchMerge: () => void;
}

export const ChunkList: React.FC<ChunkListProps> = ({
  chunks,
  selectedChunks,
  expandedChunks,
  onSelectAll,
  onSelectChunk,
  onToggleExpand,
  onFindSimilar,
  onBatchDelete,
  onBatchMerge,
}) => {
  if (chunks.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">
          <i className="fas fa-cut"></i>
        </div>
        <h4>暂无切分结果</h4>
        <p>配置切分策略后点击「执行切分」按钮生成片段</p>
      </div>
    );
  }

  return (
    <>
      <div className="toolbar">
        <div className="toolbar-left">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={selectedChunks.size === chunks.length && chunks.length > 0}
              onChange={e => onSelectAll(e.target.checked)}
            />
            <span>全选</span>
          </label>
          <div className="divider"></div>
          <button 
            className="btn btn-sm btn-outline-danger" 
            onClick={onBatchDelete} 
            disabled={selectedChunks.size === 0}
          >
            <i className="fas fa-trash-alt"></i> 批量删除
          </button>
          <button 
            className="btn btn-sm btn-outline-success" 
            onClick={onBatchMerge} 
            disabled={selectedChunks.size < 2}
          >
            <i className="fas fa-object-group"></i> 合并选中
          </button>
        </div>
        <div className="toolbar-right">
          <span className="selected-count">
            已选 <strong>{selectedChunks.size}</strong> 个片段
          </span>
        </div>
      </div>

      <div className="chunk-grid">
        {chunks.map(chunk => (
          <div key={chunk.id} className="chunk-card">
            <div className="chunk-card-header">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedChunks.has(chunk.id)}
                  onChange={e => onSelectChunk(chunk.id, e.target.checked)}
                />
              </label>
              <span className="chunk-number">#{chunk.num}</span>
              <span className="chunk-length">{chunk.length} 字符</span>
              <button 
                className="btn btn-sm btn-outline-info"
                onClick={() => onToggleExpand(chunk.id)}
                title={expandedChunks.has(chunk.id) ? "收起" : "展开"}
              >
                <i className={`fas ${expandedChunks.has(chunk.id) ? 'fa-chevron-up' : 'fa-chevron-down'}`}></i>
              </button>
            </div>
            <div className={`chunk-card-body ${expandedChunks.has(chunk.id) ? 'expanded' : ''}`}>
              <div className="chunk-content">
                {chunk.content}
              </div>
            </div>
            <div className="chunk-card-footer">
              <button 
                className="btn btn-sm btn-outline-primary"
                onClick={() => onFindSimilar(chunk.id, chunk.content)}
                title="查找相似片段"
              >
                <i className="fas fa-search"></i> 相似片段
              </button>
            </div>
          </div>
        ))}
      </div>
    </>
  );
};

export default ChunkList;
