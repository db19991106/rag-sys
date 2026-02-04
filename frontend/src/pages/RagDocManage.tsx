import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppData } from '../contexts/AppDataContext';
import { documentApi } from '../services/api';
import { formatFileSize } from '../utils/format';
import type { Document } from '../types';
import './RagDocManage.css';

const RagDocManage: React.FC = () => {
  const navigate = useNavigate();
  const {
    documents,
    addDocuments,
    updateDocument,
    deleteDocument,
    batchDeleteDocuments,
    selectedDocuments,
    setSelectedDocuments,
    setSelectedDocument
  } = useAppData();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const isLoadedRef = useRef(false);

  const [showRenameModal, setShowRenameModal] = useState(false);
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [currentDocId, setCurrentDocId] = useState<string | null>(null);
  const [newName, setNewName] = useState('');
  const [previewDoc, setPreviewDoc] = useState<Document | null>(null);

  const loadDocumentsFromBackend = async () => {
    try {
      const docs = await documentApi.list();
      const convertedDocs: Document[] = docs.map(doc => ({
        id: doc.id as any,
        name: doc.name,
        size: formatFileSize(doc.size),
        time: new Date(doc.upload_time).toLocaleString('zh-CN'),
        status: doc.status as any,
        preview: `这是${doc.name}的文档内容预览...`,
        category: doc.category || '未分类',
        tags: doc.tags || []
      }));

      const currentDocIds = documents.map(d => d.id);
      if (currentDocIds.length > 0) {
        batchDeleteDocuments(currentDocIds as any);
      }
      addDocuments(convertedDocs);
      isLoadedRef.current = true;
    } catch (error) {
      console.error('加载文档列表失败:', error);
    }
  };

  useEffect(() => {
    if (!isLoadedRef.current) {
      loadDocumentsFromBackend();
    }
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const supportExt = ['.txt', '.pdf', '.docx', '.md'];

    for (const file of Array.from(files)) {
      const ext = '.' + file.name.split('.').pop();
      if (!ext) continue;
      const extLower = ext.toLowerCase();
      if (!supportExt.includes(extLower)) {
        alert(`文件${file.name}格式不支持,仅支持TXT/PDF/DOCX/MD`);
        continue;
      }

      try {
        await documentApi.upload(file);
        alert(`文档 ${file.name} 上传成功!`);
      } catch (error) {
        console.error('上传失败:', error);
        alert(`文档 ${file.name} 上传失败: ${error instanceof Error ? error.message : '未知错误'}`);
      }
    }

    if (fileInputRef.current?.value) {
      fileInputRef.current.value = '';
    }

    loadDocumentsFromBackend();
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedDocuments(new Set(documents.map(doc => doc.id)));
    } else {
      setSelectedDocuments(new Set());
    }
  };

  const handleSelectDoc = (id: string, checked: boolean) => {
    const newSelected = new Set(selectedDocuments);
    if (checked) {
      newSelected.add(id);
    } else {
      newSelected.delete(id);
    }
    setSelectedDocuments(newSelected);
  };

  const handleBatchDelete = async () => {
    if (selectedDocuments.size === 0) {
      alert('请先选择要删除的文档');
      return;
    }
    if (confirm(`确定要删除选中的${selectedDocuments.size}个文档吗?删除后不可恢复`)) {
      try {
        const docIds = Array.from(selectedDocuments);
        await documentApi.batchDelete(docIds);
        batchDeleteDocuments(docIds);
        setSelectedDocuments(new Set());
        alert('批量删除成功!');
        loadDocumentsFromBackend();
      } catch (error) {
        console.error('批量删除失败:', error);
        alert(`批量删除失败: ${error instanceof Error ? error.message : '未知错误'}`);
      }
    }
  };

  const handleRename = (id: string) => {
    const doc = documents.find(d => d.id === id);
    if (doc) {
      setCurrentDocId(id);
      setNewName(doc.name);
      setShowRenameModal(true);
    }
  };

  const handlePreview = async (id: string) => {
    const doc = documents.find(d => d.id === id);
    if (doc) {
      try {
        const contentResponse = await documentApi.getContent(doc.id);
        setPreviewDoc({
          ...doc,
          preview: contentResponse.content || doc.preview
        });
        setShowPreviewModal(true);
      } catch (error) {
        console.error('获取文档内容失败:', error);
        setPreviewDoc(doc);
        setShowPreviewModal(true);
      }
    }
  };

  const handleConfirmRename = () => {
    if (currentDocId && newName.trim()) {
      const hasExt = newName.includes('.');
      if (!hasExt) {
        alert('请保留文件后缀(如.txt、.pdf)');
        return;
      }
      updateDocument(currentDocId, { name: newName.trim() });
      setShowRenameModal(false);
      setNewName('');
      setCurrentDocId(null);
    }
  };

  const handleReprocess = (id: string) => {
    updateDocument(id, { status: 'pending' });
    alert(`开始重新处理文档`);

    setTimeout(() => {
      updateDocument(id, { status: 'split' });
      setTimeout(() => {
        updateDocument(id, { status: 'index' });
      }, 1000);
    }, 2000);
  };

  const handleChunk = (id: string) => {
    const doc = documents.find(d => d.id === id);
    if (doc) {
      setSelectedDocument(doc);
      navigate('/chunk');
    }
  };

  const handleDelete = async (id: string) => {
    const doc = documents.find(d => d.id === id);
    if (doc && confirm(`确定要删除文档「${doc.name}」吗?删除后不可恢复`)) {
      try {
        await documentApi.delete(id);
        deleteDocument(id);
        alert('文档删除成功!');
        loadDocumentsFromBackend();
      } catch (error) {
        console.error('删除失败:', error);
        alert(`删除失败: ${error instanceof Error ? error.message : '未知错误'}`);
      }
    }
  };

  const getStatusClass = (status: Document['status']) => {
    const statusMap = {
      pending: 'status-pending',
      split: 'status-split',
      index: 'status-index',
      error: 'status-error'
    };
    return statusMap[status];
  };

  const getStatusText = (status: Document['status']) => {
    const statusMap = {
      pending: '未切分',
      split: '已切分',
      index: '已索引',
      error: '异常'
    };
    return statusMap[status];
  };

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-file-alt"></i> 知识文档管理
          <small>RAG数据来源可视化</small>
        </h1>
        <div className="btn-group">
          <label className="upload-btn">
            <i className="fas fa-upload"></i> 上传文档
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              multiple
              accept=".txt,.pdf,.docx,.md"
            />
          </label>
          {selectedDocuments.size > 0 && (
            <button className="btn btn-danger" onClick={handleBatchDelete}>
              <i className="fas fa-trash"></i> 批量删除 ({selectedDocuments.size})
            </button>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">已上传文档列表</h3>
          <div className="header-actions">
            <p className="file-tip">
              支持格式:<span>TXT、PDF、DOCX、MD</span> | 状态:未切分/已切分/已索引/异常
            </p>
          </div>
        </div>
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: '5%' }}>
                <input
                  type="checkbox"
                  checked={selectedDocuments.size === documents.length && documents.length > 0}
                  onChange={e => handleSelectAll(e.target.checked)}
                />
              </th>
              <th style={{ width: '35%' }}>文档名称</th>
              <th style={{ width: '10%' }}>文件大小</th>
              <th style={{ width: '15%' }}>上传时间</th>
              <th style={{ width: '10%' }}>处理状态</th>
              <th style={{ width: '25%' }}>操作</th>
            </tr>
          </thead>
          <tbody>
            {documents.length === 0 ? (
              <tr>
                <td colSpan={6}>
                  <div className="empty-box">
                    <div className="empty-icon">
                      <i className="fas fa-file-alt"></i>
                    </div>
                    <p>暂无上传文档,点击上方「上传文档」添加RAG数据源</p>
                  </div>
                </td>
              </tr>
            ) : (
              documents.map((doc, index) => (
                <tr key={`${doc.id}-${index}`}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedDocuments.has(doc.id)}
                      onChange={e => handleSelectDoc(doc.id, e.target.checked)}
                    />
                  </td>
                  <td>
                    <div className="doc-name">
                      <div className="doc-icon">
                        <i className="fas fa-file"></i>
                      </div>
                      <span>{doc.name}</span>
                    </div>
                  </td>
                  <td className="doc-size">{doc.size}</td>
                  <td className="doc-time">{doc.time}</td>
                  <td>
                    <span className={`status ${getStatusClass(doc.status)}`}>
                      {getStatusText(doc.status)}
                    </span>
                  </td>
                  <td className="operate">
                    <button className="op-btn op-preview" onClick={() => handlePreview(doc.id)}>
                      <i className="fas fa-eye"></i> 预览
                    </button>
                    <button className="op-btn op-chunk" onClick={() => handleChunk(doc.id)}>
                      <i className="fas fa-cut"></i> 切分
                    </button>
                    <button className="op-btn op-rename" onClick={() => handleRename(doc.id)}>
                      <i className="fas fa-edit"></i> 重命名
                    </button>
                    <button className="op-btn op-reprocess" onClick={() => handleReprocess(doc.id)}>
                      <i className="fas fa-sync-alt"></i> 重新处理
                    </button>
                    <button className="op-btn op-delete" onClick={() => handleDelete(doc.id)}>
                      <i className="fas fa-trash"></i> 删除
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {showRenameModal && (
        <div className="modal-mask" onClick={() => setShowRenameModal(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>重命名文档</span>
              <button className="modal-close" onClick={() => setShowRenameModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <label className="form-label" htmlFor="newName">
                新文档名称
              </label>
              <input
                type="text"
                className="form-input"
                id="newName"
                value={newName}
                onChange={e => setNewName(e.target.value)}
                placeholder="请输入新的文档名称(保留后缀)"
              />
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowRenameModal(false)}>
                取消
              </button>
              <button className="btn btn-primary" onClick={handleConfirmRename}>
                确认重命名
              </button>
            </div>
          </div>
        </div>
      )}

      {showPreviewModal && previewDoc && (
        <div className="modal-mask modal-large" onClick={() => setShowPreviewModal(false)}>
          <div className="modal-box modal-large-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>文档预览 - {previewDoc.name}</span>
              <button className="modal-close" onClick={() => setShowPreviewModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body modal-preview-body">
              <div className="doc-info">
                <div className="info-item">
                  <span className="info-label">文件名:</span>
                  <span className="info-value">{previewDoc.name}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">文件大小:</span>
                  <span className="info-value">{previewDoc.size}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">上传时间:</span>
                  <span className="info-value">{previewDoc.time}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">处理状态:</span>
                  <span className={`info-value status ${getStatusClass(previewDoc.status)}`}>
                    {getStatusText(previewDoc.status)}
                  </span>
                </div>
              </div>
              <div className="doc-content">
                <h4>文档内容预览</h4>
                <div className="content-preview">
                  {previewDoc.preview || '文档内容正在解析中...'}
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowPreviewModal(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RagDocManage;
