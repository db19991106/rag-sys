import React, { useState } from 'react';
import type { LocalDocItem } from '../types';
import './FileTree.css';

interface FileTreeProps {
  items: LocalDocItem[];
  selectedId?: string;
  onSelect: (item: LocalDocItem) => void;
  showFiles?: boolean;
  defaultExpanded?: boolean;
  maxHeight?: string;
}

export const FileTree: React.FC<FileTreeProps> = ({
  items,
  selectedId,
  onSelect,
  showFiles = true,
  defaultExpanded = false,
  maxHeight = '400px',
}) => {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    defaultExpanded ? new Set(items.filter(i => i.type === 'folder').map(i => i.path)) : new Set()
  );

  const toggleFolder = (path: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const getFileIcon = (extension?: string): string => {
    if (!extension) return 'fa-file';
    const ext = extension.toLowerCase();
    const iconMap: Record<string, string> = {
      pdf: 'fa-file-pdf',
      doc: 'fa-file-word',
      docx: 'fa-file-word',
      txt: 'fa-file-alt',
      md: 'fa-file-alt',
      xlsx: 'fa-file-excel',
      xls: 'fa-file-excel',
      ppt: 'fa-file-powerpoint',
      pptx: 'fa-file-powerpoint',
      png: 'fa-file-image',
      jpg: 'fa-file-image',
      jpeg: 'fa-file-image',
      gif: 'fa-file-image',
      zip: 'fa-file-archive',
      rar: 'fa-file-archive',
      json: 'fa-file-code',
      js: 'fa-file-code',
      ts: 'fa-file-code',
      py: 'fa-file-code',
    };
    return iconMap[ext] || 'fa-file';
  };

  const renderItem = (item: LocalDocItem, depth: number = 0) => {
    const isFolder = item.type === 'folder';
    const isExpanded = expandedFolders.has(item.path);
    const isSelected = selectedId === item.id || selectedId === item.path;

    if (!showFiles && !isFolder) return null;

    return (
      <div key={item.path} className="file-tree-group">
        <div
          className={`file-tree-item ${isSelected ? 'active' : ''}`}
          style={{ paddingLeft: `calc(var(--space-3) + ${depth * 16}px)` }}
          onClick={() => !isFolder && onSelect(item)}
        >
          {isFolder ? (
            <>
              <span
                className="file-tree-folder-toggle"
                onClick={(e) => toggleFolder(item.path, e)}
              >
                <i className={`fas fa-chevron-${isExpanded ? 'down' : 'right'}`}></i>
              </span>
              <span className="file-tree-icon folder">
                <i className="fas fa-folder"></i>
              </span>
            </>
          ) : (
            <>
              <span className="file-tree-folder-placeholder"></span>
              <span className="file-tree-icon file">
                <i className={`fas ${getFileIcon(item.extension)}`}></i>
              </span>
            </>
          )}
          <span className="file-tree-name">{item.name}</span>
          {item.size && <span className="file-tree-size">{item.size}</span>}
        </div>

        {isFolder && isExpanded && item.children && (
          <div className="file-tree-children">
            {item.children.map(child => renderItem(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="file-tree" style={{ maxHeight, overflowY: 'auto' }}>
      {items.length === 0 ? (
        <div className="file-tree-empty">
          <i className="fas fa-folder-open"></i>
          <span>暂无文件</span>
        </div>
      ) : (
        items.map(item => renderItem(item))
      )}
    </div>
  );
};

export default FileTree;
