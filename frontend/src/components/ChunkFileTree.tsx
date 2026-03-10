import React from 'react';
import type { LocalDocItem } from '../services/api';

interface ChunkFileTreeProps {
  items: LocalDocItem[];
  isBatchMode: boolean;
  selectedLocalDoc: string;
  selectedBatchDocs: Set<string>;
  expandedFolders: Set<string>;
  onSelectDoc: (id: string) => void;
  onToggleFolder: (path: string) => void;
  onToggleFolderSelection: (item: LocalDocItem) => void;
  onToggleBatchDoc: (docId: string, checked: boolean) => void;
  isFolderFullySelected: (item: LocalDocItem) => boolean;
  isFolderPartiallySelected: (item: LocalDocItem) => boolean;
  collectFileIds: (item: LocalDocItem) => string[];
}

// 获取文件图标
const getFileIcon = (extension?: string): string => {
  if (!extension) return 'fa-file';
  const iconMap: Record<string, string> = {
    '.md': 'fa-file-alt',
    '.txt': 'fa-file-lines',
    '.pdf': 'fa-file-pdf',
    '.docx': 'fa-file-word',
    '.doc': 'fa-file-word',
    '.html': 'fa-file-code',
    '.json': 'fa-file-code',
    '.xlsx': 'fa-file-excel',
    '.xls': 'fa-file-excel',
  };
  return iconMap[extension.toLowerCase()] || 'fa-file';
};

export const ChunkFileTree: React.FC<ChunkFileTreeProps> = ({
  items,
  isBatchMode,
  selectedLocalDoc,
  selectedBatchDocs,
  expandedFolders,
  onSelectDoc,
  onToggleFolder,
  onToggleFolderSelection,
  onToggleBatchDoc,
  isFolderFullySelected,
  isFolderPartiallySelected,
  collectFileIds,
}) => {
  const renderItem = (item: LocalDocItem, depth: number = 0): React.ReactNode => {
    const isExpanded = expandedFolders.has(item.path);
    const paddingLeft = depth * 20;

    if (item.type === 'folder') {
      const allSelected = isFolderFullySelected(item);
      const partiallySelected = isFolderPartiallySelected(item);
      const fileCount = collectFileIds(item).length;

      return (
        <div key={item.path} className="file-tree-folder">
          <div
            className="file-tree-item file-tree-folder-header"
            style={{ paddingLeft: `${paddingLeft + 12}px` }}
          >
            {isBatchMode && (
              <input
                type="checkbox"
                checked={allSelected}
                ref={el => {
                  if (el) el.indeterminate = partiallySelected;
                }}
                onChange={() => onToggleFolderSelection(item)}
                onClick={(e) => e.stopPropagation()}
              />
            )}
            <i
              className={`fas fa-chevron-${isExpanded ? 'down' : 'right'} file-tree-arrow`}
              onClick={() => onToggleFolder(item.path)}
            ></i>
            <i
              className={`fas fa-folder${isExpanded ? '-open' : ''} file-tree-icon folder-icon`}
              onClick={() => onToggleFolder(item.path)}
            ></i>
            <span className="file-tree-name" onClick={() => onToggleFolder(item.path)}>{item.name}</span>
            <span className="file-tree-count">{fileCount} 个文件</span>
          </div>
          {isExpanded && item.children && (
            <div className="file-tree-children">
              {item.children.map(child => renderItem(child, depth + 1))}
            </div>
          )}
        </div>
      );
    } else {
      return (
        <div
          key={item.path}
          className={`file-tree-item file-tree-file ${selectedLocalDoc === item.id ? 'selected' : ''}`}
          style={{ paddingLeft: `${paddingLeft + 32}px` }}
          onClick={() => !isBatchMode && onSelectDoc(item.id!)}
        >
          {isBatchMode && (
            <input
              type="checkbox"
              checked={selectedBatchDocs.has(item.id || '')}
              onChange={(e) => {
                e.stopPropagation();
                if (item.id) {
                  onToggleBatchDoc(item.id, e.target.checked);
                }
              }}
              onClick={(e) => e.stopPropagation()}
            />
          )}
          <i className={`fas ${getFileIcon(item.extension)} file-tree-icon file-icon`}></i>
          <span className="file-tree-name">{item.name}</span>
          <span className="file-tree-size">{item.size}</span>
          {!isBatchMode && (
            <button
              className="file-tree-select-btn"
              onClick={(e) => {
                e.stopPropagation();
                onSelectDoc(item.id!);
              }}
            >
              <i className="fas fa-check"></i> 选择
            </button>
          )}
        </div>
      );
    }
  };

  return (
    <div className="local-docs-tree">
      {items.map(item => renderItem(item))}
    </div>
  );
};

export default ChunkFileTree;
