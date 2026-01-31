import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppData } from '../contexts/AppDataContext';
import { documentApi } from '../services/api';
import { formatFileSize, formatTime } from '../utils/format';
import type { Document } from '../types';
import './RagDocManage.css';

const RagDocManage: React.FC = () => {
  const navigate = useNavigate();
  const {
    documents,
    addDocument,
    addDocuments,
    updateDocument,
    deleteDocument,
    batchDeleteDocuments,
    selectedDocuments,
    setSelectedDocuments,
    setSelectedDocument
  } = useAppData();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchFileInputRef = useRef<HTMLInputElement>(null);
  const isLoadedRef = useRef(false);

  const [showRenameModal, setShowRenameModal] = useState(false);
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [showBatchImportModal, setShowBatchImportModal] = useState(false);
  const [showCategoryModal, setShowCategoryModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showOCRModal, setShowOCRModal] = useState(false);
  const [showDataSourceModal, setShowDataSourceModal] = useState(false);
  const [showCategoryEditModal, setShowCategoryEditModal] = useState(false);
  const [currentDocId, setCurrentDocId] = useState<number | null>(null);
  const [currentCategoryId, setCurrentCategoryId] = useState<string | null>(null);
  const [newName, setNewName] = useState('');
  const [previewDoc, setPreviewDoc] = useState<Document | null>(null);
  const [editDocContent, setEditDocContent] = useState('');
  const [importProgress, setImportProgress] = useState(0);
  const [importStatus, setImportStatus] = useState<'idle' | 'importing' | 'paused' | 'completed' | 'error'>('idle');
  const [importStats, setImportStats] = useState({ success: 0, failed: 0, skipped: 0 });
  const [importFiles, setImportFiles] = useState<File[]>([]);
  const [importTimer, setImportTimer] = useState<number | null>(null);
  const [categories, setCategories] = useState<Array<{ id: string; name: string; count: number; tags: string[] }>>([
    { id: '1', name: '技术文档', count: 0, tags: ['技术', '文档'] },
    { id: '2', name: '产品资料', count: 0, tags: ['产品', '资料'] },
    { id: '3', name: '用户手册', count: 0, tags: ['用户', '手册'] },
  ]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [newCategoryName, setNewCategoryName] = useState('');
  const [editCategoryName, setEditCategoryName] = useState('');
  const [newTag, setNewTag] = useState('');
  const [ocrConfig, setOCRConfig] = useState({
    engine: 'tesseract',
    language: 'chi_sim+eng',
    precision: 'high',
    tableRecognition: false
  });
  const [showBatchChunkModal, setShowBatchChunkModal] = useState(false);
  const [batchChunkProgress, setBatchChunkProgress] = useState(0);
  const [batchChunkStatus, setBatchChunkStatus] = useState<'idle' | 'chunking' | 'paused' | 'completed' | 'error'>('idle');
  const [batchChunkStats, setBatchChunkStats] = useState({ success: 0, failed: 0, skipped: 0 });
  const [batchChunkTimer, setBatchChunkTimer] = useState<number | null>(null);

  // 从后端加载文档列表
  const loadDocumentsFromBackend = async (forceReload = false) => {
    try {
      const docs = await documentApi.list();
      // 将后端数据转换为前端格式
      const convertedDocs: Document[] = docs.map(doc => ({
        id: doc.id as any, // 保持ID为字符串类型
        name: doc.name,
        size: formatFileSize(doc.size),
        time: new Date(doc.upload_time).toLocaleString('zh-CN'),
        status: doc.status as any,
        preview: `这是${doc.name}的文档内容预览...`,
        category: doc.category || '未分类',
        tags: doc.tags || []
      }));

      // 获取当前所有文档的ID
      const currentDocIds = documents.map(d => d.id);
      // 批量删除当前文档
      if (currentDocIds.length > 0) {
        batchDeleteDocuments(currentDocIds as any);
      }
      // 批量添加新文档
      addDocuments(convertedDocs);
      isLoadedRef.current = true;
    } catch (error) {
      console.error('加载文档列表失败:', error);
    }
  };

  // 组件挂载时自动加载文档列表（只在首次挂载时执行一次）
  useEffect(() => {
    if (!isLoadedRef.current) {
      loadDocumentsFromBackend();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        // 调用后端 API 上传文档
        const response = await documentApi.upload(file);
        alert(`文档 ${file.name} 上传成功!`);
      } catch (error) {
        console.error('上传失败:', error);
        alert(`文档 ${file.name} 上传失败: ${error instanceof Error ? error.message : '未知错误'}`);
      }
    }

    if (fileInputRef.current?.value) {
      fileInputRef.current.value = '';
    }

    // 重新加载文档列表
    loadDocumentsFromBackend();
  };

  const handleBatchImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const supportExt = ['.txt', '.pdf', '.docx', '.md'];
    const validFiles = Array.from(files).filter(file => {
      const ext = '.' + file.name.split('.').pop();
      if (!ext) return false;
      const extLower = ext.toLowerCase();
      return supportExt.includes(extLower);
    });

    if (validFiles.length === 0) {
      alert('没有有效的文件,请选择TXT/PDF/DOCX/MD格式文件');
      return;
    }

    setImportFiles(validFiles);
    setImportStats({ success: 0, failed: 0, skipped: 0 });
    setShowBatchImportModal(true);
    setImportStatus('importing');
    setImportProgress(0);

    let currentIndex = 0;
    const timer = window.setInterval(() => {
      if (importStatus === 'paused') return;

      if (currentIndex >= validFiles.length) {
        clearInterval(timer);
        setImportStatus('completed');
        setImportTimer(null);
        return;
      }

      const file = validFiles[currentIndex];
      const doc: Document = {
        id: Date.now() + Math.floor(Math.random() * 1000),
        name: file.name,
        size: formatFileSize(file.size),
        time: formatTime(new Date()),
        status: 'pending',
        preview: `这是${file.name}的文档内容预览...`,
        category: '未分类'
      };

      addDocument(doc);
      setImportStats(prev => ({ ...prev, success: prev.success + 1 }));
      currentIndex++;
      setImportProgress(Math.floor((currentIndex / validFiles.length) * 100));
    }, 500);

    setImportTimer(timer);

    if (batchFileInputRef.current?.value) {
      batchFileInputRef.current.value = '';
    }
  };

  const handlePauseImport = () => {
    setImportStatus('paused');
  };

  const handleResumeImport = () => {
    setImportStatus('importing');
  };

  const handleCancelImport = () => {
    if (importTimer) {
      clearInterval(importTimer);
      setImportTimer(null);
    }
    setImportStatus('idle');
    setShowBatchImportModal(false);
    setImportProgress(0);
    setImportStats({ success: 0, failed: 0, skipped: 0 });
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedDocuments(new Set(documents.map(doc => doc.id)));
    } else {
      setSelectedDocuments(new Set());
    }
  };

  const handleSelectDoc = (id: number, checked: boolean) => {
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
        const docIds = Array.from(selectedDocuments).map(String);
        await documentApi.batchDelete(docIds);
        batchDeleteDocuments(Array.from(selectedDocuments));
        setSelectedDocuments(new Set());
        alert('批量删除成功!');
        loadDocumentsFromBackend();
      } catch (error) {
        console.error('批量删除失败:', error);
        alert(`批量删除失败: ${error instanceof Error ? error.message : '未知错误'}`);
      }
    }
  };

  const handleBatchChunk = () => {
    if (selectedDocuments.size === 0) {
      alert('请先选择要切分的文档');
      return;
    }
    setShowBatchChunkModal(true);
    setBatchChunkProgress(0);
    setBatchChunkStatus('chunking');
    setBatchChunkStats({ success: 0, failed: 0, skipped: 0 });

    const selectedDocs = Array.from(selectedDocuments);
    let currentIndex = 0;

    const timer = window.setInterval(() => {
      if (batchChunkStatus === 'paused') return;

      if (currentIndex >= selectedDocs.length) {
        clearInterval(timer);
        setBatchChunkStatus('completed');
        setBatchChunkProgress(100);
        return;
      }

      const docId = selectedDocs[currentIndex];
      const doc = documents.find(d => d.id === docId);
      if (doc) {
        // 模拟切分过程
        if (doc.status === 'pending') {
          updateDocument(doc.id, { status: 'split' });
          setBatchChunkStats(prev => ({ ...prev, success: prev.success + 1 }));
        } else if (doc.status === 'split') {
          setBatchChunkStats(prev => ({ ...prev, skipped: prev.skipped + 1 }));
        } else {
          setBatchChunkStats(prev => ({ ...prev, failed: prev.failed + 1 }));
        }
      }

      currentIndex++;
      setBatchChunkProgress(Math.floor((currentIndex / selectedDocs.length) * 100));
    }, 500);

    setBatchChunkTimer(timer);
  };

  const handlePauseBatchChunk = () => {
    setBatchChunkStatus('paused');
  };

  const handleResumeBatchChunk = () => {
    setBatchChunkStatus('chunking');
  };

  const handleCancelBatchChunk = () => {
    if (batchChunkTimer) {
      clearInterval(batchChunkTimer);
      setBatchChunkTimer(null);
    }
    setShowBatchChunkModal(false);
    setBatchChunkStatus('idle');
    setBatchChunkProgress(0);
  };

  const handleRename = (id: number | string) => {
    const doc = documents.find(d => d.id === id);
    if (doc) {
      setCurrentDocId(id as number);
      setNewName(doc.name);
      setShowRenameModal(true);
    }
  };

  const handlePreview = async (id: number | string) => {
    const doc = documents.find(d => d.id === id);
    if (doc) {
      try {
        // 从后端获取真实的文档内容
        const contentResponse = await documentApi.getContent(String(doc.id));
        setPreviewDoc({
          ...doc,
          preview: contentResponse.content || doc.preview
        });
        setShowPreviewModal(true);
      } catch (error) {
        console.error('获取文档内容失败:', error);
        // 如果获取失败，使用预览文本
        setPreviewDoc(doc);
        setShowPreviewModal(true);
      }
    }
  };

  const handleEdit = (id: number | string) => {
    const doc = documents.find(d => d.id === id);
    if (doc) {
      setCurrentDocId(id as number);
      setEditDocContent(doc.preview || '');
      setShowEditModal(true);
    }
  };

  const handleSaveEdit = () => {
    if (currentDocId) {
      updateDocument(currentDocId, { preview: editDocContent });
      setShowEditModal(false);
      alert('文档内容已保存');
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

  const handleReprocess = (id: number | string) => {
    updateDocument(id, { status: 'pending' });
    alert(`开始重新处理文档`);

    setTimeout(() => {
      updateDocument(id, { status: 'split' });
      setTimeout(() => {
        updateDocument(id, { status: 'index' });
      }, 1000);
    }, 2000);
  };

  const handleChunk = (id: number | string) => {
    const doc = documents.find(d => d.id === id);
    if (doc) {
      setSelectedDocument(doc);
      navigate('/chunk');
    }
  };

  const handleDelete = async (id: number) => {
    const doc = documents.find(d => d.id === id);
    if (doc && confirm(`确定要删除文档「${doc.name}」吗?删除后不可恢复`)) {
      try {
        await documentApi.delete(String(id));
        deleteDocument(id);
        alert('文档删除成功!');
        loadDocumentsFromBackend();
      } catch (error) {
        console.error('删除失败:', error);
        alert(`删除失败: ${error instanceof Error ? error.message : '未知错误'}`);
      }
    }
  };

  const handleAddCategory = () => {
    if (newCategoryName.trim()) {
      const newCategory = {
        id: Date.now().toString(),
        name: newCategoryName.trim(),
        count: 0,
        tags: []
      };
      setCategories([...categories, newCategory]);
      setNewCategoryName('');
    }
  };

  const handleEditCategory = (id: string) => {
    const category = categories.find(c => c.id === id);
    if (category) {
      setCurrentCategoryId(id);
      setEditCategoryName(category.name);
      setShowCategoryEditModal(true);
    }
  };

  const handleSaveCategoryEdit = () => {
    if (currentCategoryId && editCategoryName.trim()) {
      setCategories(prev =>
        prev.map(cat =>
          cat.id === currentCategoryId
            ? { ...cat, name: editCategoryName.trim() }
            : cat
        )
      );
      setShowCategoryEditModal(false);
      setEditCategoryName('');
      setCurrentCategoryId(null);
    }
  };

  const handleDeleteCategory = (id: string) => {
    if (confirm('确定要删除此分类吗?')) {
      setCategories(categories.filter(c => c.id !== id));
    }
  };

  const handleAddTag = (categoryId: string) => {
    if (newTag.trim()) {
      setCategories(prev =>
        prev.map(cat =>
          cat.id === categoryId
            ? { ...cat, tags: [...cat.tags, newTag.trim()] }
            : cat
        )
      );
      setNewTag('');
    }
  };

  const handleDeleteTag = (categoryId: string, tagIndex: number) => {
    setCategories(prev =>
      prev.map(cat =>
        cat.id === categoryId
          ? { ...cat, tags: cat.tags.filter((_, i) => i !== tagIndex) }
          : cat
      )
    );
  };

  const handleBatchSetCategory = (category: string) => {
    if (selectedDocuments.size === 0) {
      alert('请先选择要设置分类的文档');
      return;
    }
    Array.from(selectedDocuments).forEach(id => {
      updateDocument(id, { category });
    });
    setSelectedDocuments(new Set());
    alert(`已将${selectedDocuments.size}个文档设置为「${category}」分类`);
  };

  const handleTestOCR = () => {
    alert('OCR测试功能开发中...\n当前配置:\n' +
      `引擎: ${ocrConfig.engine}\n` +
      `语言: ${ocrConfig.language}\n` +
      `精度: ${ocrConfig.precision}\n` +
      `表格识别: ${ocrConfig.tableRecognition ? '开启' : '关闭'}`);
  };

  const handleDatabaseConnect = () => {
    alert('数据库连接测试功能开发中...\n请先配置数据库连接参数');
  };

  const filteredDocuments = selectedCategory === 'all'
    ? documents
    : documents.filter(doc => doc.category === selectedCategory);

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
          <label className="upload-btn batch-import-btn">
            <i className="fas fa-folder-open"></i> 批量导入
            <input
              type="file"
              ref={batchFileInputRef}
              onChange={handleBatchImport}
              multiple
              accept=".txt,.pdf,.docx,.md"
            />
          </label>
          <button className="btn btn-default" onClick={() => setShowCategoryModal(true)}>
            <i className="fas fa-tags"></i> 文档分类
          </button>
          <button className="btn btn-default" onClick={() => setShowOCRModal(true)}>
            <i className="fas fa-image"></i> OCR设置
          </button>
          <button className="btn btn-default" onClick={() => setShowDataSourceModal(true)}>
            <i className="fas fa-database"></i> 数据源导入
          </button>
          {selectedDocuments.size > 0 && (
            <button className="btn btn-primary" onClick={handleBatchChunk}>
              <i className="fas fa-cut"></i> 批量切分 ({selectedDocuments.size})
            </button>
          )}
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
            <div className="filter-bar">
              <label className="filter-label">分类筛选:</label>
              <select
                className="filter-select"
                value={selectedCategory}
                onChange={e => setSelectedCategory(e.target.value)}
              >
                <option value="all">全部分类</option>
                {categories.map(cat => (
                  <option key={cat.id} value={cat.name}>{cat.name}</option>
                ))}
              </select>
              {selectedDocuments.size > 0 && (
                <select
                  className="filter-select"
                  onChange={e => handleBatchSetCategory(e.target.value)}
                  value=""
                >
                  <option value="">批量设置分类</option>
                  {categories.map(cat => (
                    <option key={cat.id} value={cat.name}>{cat.name}</option>
                  ))}
                </select>
              )}
            </div>
          </div>
        </div>
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: '5%' }}>
                <input
                  type="checkbox"
                  checked={selectedDocuments.size === filteredDocuments.length && filteredDocuments.length > 0}
                  onChange={e => handleSelectAll(e.target.checked)}
                />
              </th>
              <th style={{ width: '30%' }}>文档名称</th>
              <th style={{ width: '10%' }}>文件大小</th>
              <th style={{ width: '10%' }}>上传时间</th>
              <th style={{ width: '10%' }}>分类</th>
              <th style={{ width: '10%' }}>处理状态</th>
              <th style={{ width: '25%' }}>操作</th>
            </tr>
          </thead>
          <tbody>
            {filteredDocuments.length === 0 ? (
              <tr>
                <td colSpan={7}>
                  <div className="empty-box">
                    <div className="empty-icon">
                      <i className="fas fa-file-alt"></i>
                    </div>
                    <p>暂无上传文档,点击上方「上传文档」或「批量导入」添加RAG数据源</p>
                  </div>
                </td>
              </tr>
            ) : (
              filteredDocuments.map(doc => (
                <tr key={doc.id}>
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
                    <span className="category-badge">{doc.category || '未分类'}</span>
                  </td>
                  <td>
                    <span className={`status ${getStatusClass(doc.status)}`}>
                      {getStatusText(doc.status)}
                    </span>
                  </td>
                  <td className="operate">
                    <button className="op-btn op-preview" onClick={() => handlePreview(doc.id)}>
                      <i className="fas fa-eye"></i> 预览
                    </button>
                    <button className="op-btn op-edit" onClick={() => handleEdit(doc.id)}>
                      <i className="fas fa-edit"></i> 编辑
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

      {/* 重命名模态框 */}
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

      {/* 文档预览模态框 */}
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
                  <span className="info-label">分类:</span>
                  <span className="info-value">{previewDoc.category || '未分类'}</span>
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
              <button className="btn btn-primary" onClick={() => {
                alert('下载功能开发中...');
                setShowPreviewModal(false);
              }}>
                <i className="fas fa-download"></i> 下载文档
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 文档编辑模态框 */}
      {showEditModal && (
        <div className="modal-mask modal-large" onClick={() => setShowEditModal(false)}>
          <div className="modal-box modal-large-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>编辑文档内容</span>
              <button className="modal-close" onClick={() => setShowEditModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <label className="form-label">文档内容</label>
              <textarea
                className="form-textarea"
                value={editDocContent}
                onChange={e => setEditDocContent(e.target.value)}
                placeholder="请输入文档内容..."
                rows={20}
              />
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowEditModal(false)}>
                取消
              </button>
              <button className="btn btn-primary" onClick={handleSaveEdit}>
                <i className="fas fa-save"></i> 保存
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 批量导入进度模态框 */}
      {showBatchImportModal && (
        <div className="modal-mask" onClick={() => {}}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>批量导入文档</span>
              {importStatus !== 'completed' && (
                <button className="modal-close" onClick={handleCancelImport}>
                  <i className="fas fa-times"></i>
                </button>
              )}
            </div>
            <div className="modal-body">
              <div className="import-progress-container">
                {importStatus === 'completed' ? (
                  <div className="import-success">
                    <i className="fas fa-check-circle"></i>
                    <p>批量导入完成!</p>
                    <div className="import-stats">
                      <div className="stat-item">
                        <span className="stat-label">成功:</span>
                        <span className="stat-value success">{importStats.success}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">失败:</span>
                        <span className="stat-value error">{importStats.failed}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">跳过:</span>
                        <span className="stat-value warning">{importStats.skipped}</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${importProgress}%` }}
                      ></div>
                    </div>
                    <p className="progress-text">
                      {importStatus === 'importing' && `正在导入... ${importProgress}%`}
                      {importStatus === 'paused' && '已暂停'}
                      {importStatus === 'idle' && '准备导入...'}
                    </p>
                    <p className="progress-detail">
                      共 {importFiles.length} 个文件 | 已处理 {Math.floor(importProgress / 100 * importFiles.length)} 个
                    </p>
                    <div className="import-controls">
                      {importStatus === 'importing' && (
                        <button className="btn btn-warning" onClick={handlePauseImport}>
                          <i className="fas fa-pause"></i> 暂停
                        </button>
                      )}
                      {importStatus === 'paused' && (
                        <button className="btn btn-primary" onClick={handleResumeImport}>
                          <i className="fas fa-play"></i> 继续
                        </button>
                      )}
                      <button className="btn btn-danger" onClick={handleCancelImport}>
                        <i className="fas fa-times"></i> 取消
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
            {importStatus === 'completed' && (
              <div className="modal-footer">
                <button className="btn btn-default" onClick={handleCancelImport}>
                  关闭
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 批量切分模态框 */}
      {showBatchChunkModal && (
        <div className="modal-mask" onClick={() => {}}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>批量切分文档</span>
              {batchChunkStatus !== 'completed' && (
                <button className="modal-close" onClick={handleCancelBatchChunk}>
                  <i className="fas fa-times"></i>
                </button>
              )}
            </div>
            <div className="modal-body">
              <div className="import-progress-container">
                {batchChunkStatus === 'completed' ? (
                  <div className="import-success">
                    <i className="fas fa-check-circle"></i>
                    <p>批量切分完成!</p>
                    <div className="import-stats">
                      <div className="stat-item">
                        <span className="stat-label">成功:</span>
                        <span className="stat-value success">{batchChunkStats.success}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">失败:</span>
                        <span className="stat-value error">{batchChunkStats.failed}</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-label">跳过:</span>
                        <span className="stat-value warning">{batchChunkStats.skipped}</span>
                      </div>
                    </div>
                    <div className="chunk-report">
                      <p><strong>处理报告:</strong></p>
                      <ul>
                        <li>共处理 {selectedDocuments.size} 个文档</li>
                        <li>成功切分 {batchChunkStats.success} 个文档</li>
                        <li>跳过已切分 {batchChunkStats.skipped} 个文档</li>
                        {batchChunkStats.failed > 0 && <li>处理失败 {batchChunkStats.failed} 个文档</li>}
                      </ul>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${batchChunkProgress}%` }}
                      ></div>
                    </div>
                    <p className="progress-text">
                      {batchChunkStatus === 'chunking' && `正在切分... ${batchChunkProgress}%`}
                      {batchChunkStatus === 'paused' && '已暂停'}
                      {batchChunkStatus === 'idle' && '准备切分...'}
                    </p>
                    <p className="progress-detail">
                      共 {selectedDocuments.size} 个文档 | 已处理 {Math.floor(batchChunkProgress / 100 * selectedDocuments.size)} 个
                    </p>
                    <div className="import-controls">
                      {batchChunkStatus === 'chunking' && (
                        <button className="btn btn-warning" onClick={handlePauseBatchChunk}>
                          <i className="fas fa-pause"></i> 暂停
                        </button>
                      )}
                      {batchChunkStatus === 'paused' && (
                        <button className="btn btn-primary" onClick={handleResumeBatchChunk}>
                          <i className="fas fa-play"></i> 继续
                        </button>
                      )}
                      <button className="btn btn-danger" onClick={handleCancelBatchChunk}>
                        <i className="fas fa-times"></i> 取消
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
            {batchChunkStatus === 'completed' && (
              <div className="modal-footer">
                <button className="btn btn-default" onClick={handleCancelBatchChunk}>
                  关闭
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 文档分类管理模态框 */}
      {showCategoryModal && (
        <div className="modal-mask modal-large" onClick={() => setShowCategoryModal(false)}>
          <div className="modal-box modal-large-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>文档分类管理</span>
              <button className="modal-close" onClick={() => setShowCategoryModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="category-form">
                <input
                  type="text"
                  className="form-input"
                  value={newCategoryName}
                  onChange={e => setNewCategoryName(e.target.value)}
                  placeholder="输入新分类名称"
                />
                <button className="btn btn-primary" onClick={handleAddCategory}>
                  <i className="fas fa-plus"></i> 添加
                </button>
              </div>
              <div className="category-list">
                {categories.map(category => (
                  <div key={category.id} className="category-item">
                    <div className="category-header">
                      <span className="category-name">{category.name}</span>
                      <div className="category-actions">
                        <button className="op-btn op-edit" onClick={() => handleEditCategory(category.id)}>
                          <i className="fas fa-edit"></i> 编辑
                        </button>
                        <button className="op-btn op-delete" onClick={() => handleDeleteCategory(category.id)}>
                          <i className="fas fa-trash"></i> 删除
                        </button>
                      </div>
                    </div>
                    <div className="category-tags">
                      <span className="category-count">{category.count} 个文档</span>
                      <div className="tags-list">
                        {category.tags.map((tag, index) => (
                          <span key={index} className="tag">
                            {tag}
                            <i className="fas fa-times" onClick={() => handleDeleteTag(category.id, index)}></i>
                          </span>
                        ))}
                      </div>
                      <div className="tag-input-group">
                        <input
                          type="text"
                          className="form-input tag-input"
                          value={newTag}
                          onChange={e => setNewTag(e.target.value)}
                          placeholder="添加标签"
                          onKeyPress={e => e.key === 'Enter' && handleAddTag(category.id)}
                        />
                        <button className="btn btn-sm" onClick={() => handleAddTag(category.id)}>
                          <i className="fas fa-plus"></i>
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowCategoryModal(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 分类编辑模态框 */}
      {showCategoryEditModal && (
        <div className="modal-mask" onClick={() => setShowCategoryEditModal(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>编辑分类</span>
              <button className="modal-close" onClick={() => setShowCategoryEditModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <label className="form-label">分类名称</label>
              <input
                type="text"
                className="form-input"
                value={editCategoryName}
                onChange={e => setEditCategoryName(e.target.value)}
                placeholder="请输入分类名称"
              />
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowCategoryEditModal(false)}>
                取消
              </button>
              <button className="btn btn-primary" onClick={handleSaveCategoryEdit}>
                保存
              </button>
            </div>
          </div>
        </div>
      )}

      {/* OCR配置模态框 */}
      {showOCRModal && (
        <div className="modal-mask modal-large" onClick={() => setShowOCRModal(false)}>
          <div className="modal-box modal-large-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>OCR识别配置</span>
              <button className="modal-close" onClick={() => setShowOCRModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="ocr-config-form">
                <div className="form-item">
                  <label className="form-label">OCR引擎</label>
                  <select
                    className="form-input"
                    value={ocrConfig.engine}
                    onChange={e => setOCRConfig({ ...ocrConfig, engine: e.target.value })}
                  >
                    <option value="tesseract">Tesseract</option>
                    <option value="baidu">百度OCR</option>
                    <option value="aliyun">阿里云OCR</option>
                    <option value="google">Google Vision</option>
                  </select>
                </div>
                <div className="form-item">
                  <label className="form-label">识别语言</label>
                  <select
                    className="form-input"
                    value={ocrConfig.language}
                    onChange={e => setOCRConfig({ ...ocrConfig, language: e.target.value })}
                  >
                    <option value="chi_sim">简体中文</option>
                    <option value="chi_tra">繁体中文</option>
                    <option value="eng">英文</option>
                    <option value="chi_sim+eng">中英文混合</option>
                    <option value="jpn">日文</option>
                    <option value="kor">韩文</option>
                  </select>
                </div>
                <div className="form-item">
                  <label className="form-label">识别精度</label>
                  <select
                    className="form-input"
                    value={ocrConfig.precision}
                    onChange={e => setOCRConfig({ ...ocrConfig, precision: e.target.value })}
                  >
                    <option value="low">低精度(快速)</option>
                    <option value="medium">中等精度</option>
                    <option value="high">高精度(准确)</option>
                  </select>
                </div>
                <div className="form-item">
                  <label className="form-label">表格识别</label>
                  <div className="toggle-switch">
                    <input
                      type="checkbox"
                      id="tableRecognition"
                      checked={ocrConfig.tableRecognition}
                      onChange={e => setOCRConfig({ ...ocrConfig, tableRecognition: e.target.checked })}
                    />
                    <label htmlFor="tableRecognition" className="switch-label">
                      <span className="switch-slider"></span>
                    </label>
                    <span className="switch-text">
                      {ocrConfig.tableRecognition ? '已开启' : '已关闭'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowOCRModal(false)}>
                取消
              </button>
              <button className="btn btn-default" onClick={handleTestOCR}>
                <i className="fas fa-play"></i> 测试OCR
              </button>
              <button className="btn btn-primary" onClick={() => {
                alert('OCR配置已保存!');
                setShowOCRModal(false);
              }}>
                <i className="fas fa-save"></i> 保存配置
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 数据源导入模态框 */}
      {showDataSourceModal && (
        <div className="modal-mask modal-large" onClick={() => setShowDataSourceModal(false)}>
          <div className="modal-box modal-large-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>数据源导入</span>
              <button className="modal-close" onClick={() => setShowDataSourceModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="data-source-tabs">
                <button className="tab active">数据库导入</button>
                <button className="tab">API导入</button>
              </div>
              <div className="data-source-content">
                <div className="form-item">
                  <label className="form-label">数据库类型</label>
                  <select className="form-input">
                    <option>MySQL</option>
                    <option>PostgreSQL</option>
                    <option>MongoDB</option>
                    <option>Elasticsearch</option>
                  </select>
                </div>
                <div className="form-item">
                  <label className="form-label">主机地址</label>
                  <input type="text" className="form-input" placeholder="例如: localhost" />
                </div>
                <div className="form-item">
                  <label className="form-label">端口</label>
                  <input type="text" className="form-input" placeholder="例如: 3306" />
                </div>
                <div className="form-item">
                  <label className="form-label">数据库名称</label>
                  <input type="text" className="form-input" placeholder="输入数据库名称" />
                </div>
                <div className="form-item">
                  <label className="form-label">用户名</label>
                  <input type="text" className="form-input" placeholder="输入用户名" />
                </div>
                <div className="form-item">
                  <label className="form-label">密码</label>
                  <input type="password" className="form-input" placeholder="输入密码" />
                </div>
                <div className="form-item">
                  <label className="form-label">表名</label>
                  <input type="text" className="form-input" placeholder="输入表名" />
                </div>
                <div className="form-item">
                  <label className="form-label">
                    <input type="checkbox" />
                    <span>启用定时导入</span>
                  </label>
                  <input type="text" className="form-input" placeholder="例如: 每天凌晨2点" />
                </div>
                <div className="form-item">
                  <label className="form-label">
                    <input type="checkbox" />
                    <span>启用增量更新</span>
                  </label>
                  <input type="text" className="form-input" placeholder="增量字段名称" />
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowDataSourceModal(false)}>
                取消
              </button>
              <button className="btn btn-default" onClick={handleDatabaseConnect}>
                <i className="fas fa-plug"></i> 测试连接
              </button>
              <button className="btn btn-primary" onClick={() => alert('导入功能开发中...')}>
                <i className="fas fa-download"></i> 导入
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RagDocManage;