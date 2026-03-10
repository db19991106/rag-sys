import React, { useState, useEffect } from 'react';
import { vectorDbApi } from '../services/api';
import type { VectorDocument } from '../services/api';
import './VectorManage.css';

const VectorManage: React.FC = () => {
  const [documents, setDocuments] = useState<VectorDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [totalDocuments, setTotalDocuments] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const [dbType, setDbType] = useState<string>('');
  const [dimension, setDimension] = useState<number>(0);
  const [dbSize, setDbSize] = useState<number>(0);
  const [expandedDocId, setExpandedDocId] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);

  useEffect(() => {
    loadVectorDocuments();
  }, []);

  const loadVectorDocuments = async () => {
    setLoading(true);
    try {
      // 并行获取文档列表和向量库状态
      const [documentsResponse, statusResponse] = await Promise.all([
        vectorDbApi.getDocuments(),
        vectorDbApi.getStatus()
      ]);

      if (documentsResponse.success && documentsResponse.data) {
        setDocuments(documentsResponse.data.documents);
        setTotalDocuments(documentsResponse.data.total_documents);
        setTotalChunks(documentsResponse.data.total_chunks);
      }

      // 设置向量库类型和维度
      setDbType(statusResponse.db_type || '');
      setDimension(statusResponse.dimension || 0);
      setDbSize(statusResponse.db_size || 0);
    } catch (error) {
      console.error('加载向量库文档失败:', error);
      alert(`加载向量库文档失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteDocument = async (documentId: string, documentName: string) => {
    if (!confirm(`确定要从向量库中删除文档「${documentName}」及其所有向量吗？\n\n此操作不可恢复！`)) {
      return;
    }

    setLoading(true);
    try {
      const response = await vectorDbApi.deleteDocument(documentId);
      if (response.success) {
        alert(`成功删除文档「${documentName}」`);
        setShowDeleteConfirm(null);
        loadVectorDocuments();
      } else {
        alert(`删除失败: ${response.message}`);
      }
    } catch (error) {
      console.error('删除文档失败:', error);
      alert(`删除文档失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteChunk = async (vectorId: string, chunkNum: number, _documentId: string) => {
    if (!confirm(`确定要从向量库中删除片段 #${chunkNum} 吗？\n\n此操作不可恢复！`)) {
      return;
    }

    setLoading(true);
    try {
      const response = await vectorDbApi.deleteChunk(vectorId);
      if (response.success) {
        alert(`成功删除片段 #${chunkNum}`);
        loadVectorDocuments();
      } else {
        alert(`删除失败: ${response.message}`);
      }
    } catch (error) {
      console.error('删除片段失败:', error);
      alert(`删除片段失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = (documentId: string) => {
    setExpandedDocId(expandedDocId === documentId ? null : documentId);
  };

  const handleClearVectorDb = async () => {
    if (!confirm('确定要清空向量数据库吗？\n\n此操作将删除所有向量数据和元数据，且不可恢复！')) {
      return;
    }

    if (!confirm('再次确认：此操作不可恢复，确定要继续吗？')) {
      return;
    }

    setLoading(true);
    try {
      const response = await vectorDbApi.clearVectorDb();
      if (response.success) {
        alert('向量数据库已清空');
        setDocuments([]);
        setTotalDocuments(0);
        setTotalChunks(0);
      } else {
        alert(`清空失败: ${response.message || '未知错误'}`);
      }
    } catch (error) {
      console.error('清空向量数据库失败:', error);
      alert(`清空向量数据库失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-database"></i> 向量库管理
          <small>Vector Database Management</small>
        </h1>
        <div className="page-actions">
          <button className="btn btn-outline-primary" onClick={loadVectorDocuments} disabled={loading}>
            <i className={`fas ${loading ? 'fa-spinner fa-spin' : 'fa-sync-alt'}`}></i> 刷新
          </button>
        </div>
      </div>

      <div className="stats-cards">
        <div className="stat-card primary">
          <div className="stat-icon">
            <i className="fas fa-file-alt"></i>
          </div>
          <div className="stat-content">
            <div className="stat-value">{totalDocuments}</div>
            <div className="stat-label">文档数量</div>
          </div>
        </div>
        <div className="stat-card success">
          <div className="stat-icon">
            <i className="fas fa-cubes"></i>
          </div>
          <div className="stat-content">
            <div className="stat-value">{totalChunks}</div>
            <div className="stat-label">向量数量</div>
          </div>
        </div>
        <div className="stat-card info">
          <div className="stat-icon">
            <i className="fas fa-memory"></i>
          </div>
          <div className="stat-content">
            <div className="stat-value">{totalChunks > 0 ? ((totalChunks / totalDocuments).toFixed(1)) : 0}</div>
            <div className="stat-label">平均切分数量</div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-list"></i> 向量库文档列表
          </h3>
          <div className="card-header-actions">
            <span className="tip-text">
              {dbType && `${dbType.toUpperCase()} | `}{totalDocuments} 文档 | {totalChunks} 向量{dimension > 0 && ` | ${dimension} 维`}{dbSize > 0 && ` | ${dbSize >= 1024 * 1024 * 1024 ? (dbSize / (1024 * 1024 * 1024)).toFixed(1) + ' GB' : (dbSize / (1024 * 1024)).toFixed(1) + ' MB'}`}
            </span>
            <button 
              className="btn btn-sm btn-danger" 
              onClick={handleClearVectorDb}
              disabled={loading || (totalChunks === 0 && dbSize === 0)}
              title="清空所有向量数据"
            >
              <i className="fas fa-trash-alt"></i> 清空向量库
            </button>
          </div>
        </div>
        <div className="card-body">
          {loading && documents.length === 0 ? (
            <div className="loading-state">
              <i className="fas fa-spinner fa-spin"></i>
              <span>加载中...</span>
            </div>
          ) : documents.length === 0 ? (
            <div className="empty-state">
              <i className="fas fa-inbox"></i>
              <span>向量库中暂无文档</span>
              <small>请先上传文档并进行切分和向量化</small>
            </div>
          ) : (
            <div className="document-list">
              {documents.map((doc) => (
                <div key={doc.document_id} className="document-item">
                  <div className="document-header">
                    <div className="document-info">
                      <div className="document-name">
                        <i className="fas fa-file-alt"></i>
                        {doc.document_name || '未命名文档'}
                      </div>
                      <div className="document-meta">
                        <span className="meta-item">
                          <i className="fas fa-fingerprint"></i>
                          ID: {doc.document_id.slice(0, 8)}...
                        </span>
                        <span className="meta-item">
                          <i className="fas fa-cubes"></i>
                          {doc.chunk_count} 个向量
                        </span>
                      </div>
                    </div>
                    <div className="document-actions">
                      <button
                        className="btn btn-sm btn-outline-info"
                        onClick={() => toggleExpand(doc.document_id)}
                      >
                        <i className={`fas ${expandedDocId === doc.document_id ? 'fa-chevron-up' : 'fa-chevron-down'}`}></i>
                        {expandedDocId === doc.document_id ? '收起' : '查看片段'}
                      </button>
                      <button
                        className="btn btn-sm btn-outline-danger"
                        onClick={() => handleDeleteDocument(doc.document_id, doc.document_name)}
                      >
                        <i className="fas fa-trash"></i>
                        删除
                      </button>
                    </div>
                  </div>

                  {expandedDocId === doc.document_id && (
                    <div className="document-chunks">
                      <div className="chunks-header">
                        <span>向量片段列表 ({doc.chunks.length})</span>
                      </div>
                      <div className="chunks-list">
                        {doc.chunks.map((chunk) => (
                          <div key={chunk.vector_id} className="chunk-item">
                            <div className="chunk-header">
                              <div className="chunk-num">
                                <i className="fas fa-cube"></i>
                                片段 #{chunk.chunk_num}
                              </div>
                              <button
                                className="btn btn-sm btn-outline-danger"
                                onClick={() => handleDeleteChunk(chunk.vector_id, chunk.chunk_num, doc.document_id)}
                                title="删除此片段"
                              >
                                <i className="fas fa-trash"></i>
                                删除
                              </button>
                            </div>
                            <div className="chunk-content">
                              {chunk.content}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {showDeleteConfirm && (
        <div className="modal-mask" onClick={() => setShowDeleteConfirm(null)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-exclamation-triangle"></i> 确认删除
              </span>
              <button className="modal-close" onClick={() => setShowDeleteConfirm(null)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <p>确定要从向量库中删除该文档及其所有向量吗？</p>
              <p className="warning-text">此操作不可恢复！</p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowDeleteConfirm(null)}>
                取消
              </button>
              <button
                className="btn btn-danger"
                onClick={() => {
                  if (showDeleteConfirm) {
                    const doc = documents.find(d => d.document_id === showDeleteConfirm);
                    if (doc) {
                      handleDeleteDocument(doc.document_id, doc.document_name);
                    }
                  }
                }}
              >
                <i className="fas fa-trash"></i> 确认删除
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VectorManage;