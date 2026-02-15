import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppData } from '../contexts/AppDataContext';
import { chunkingApi, documentApi, retrievalApi } from '../services/api';
import type { Chunk, ChunkConfig } from '../types';
import './Chunk.css';

const Chunk: React.FC = () => {
  const navigate = useNavigate();
  const { selectedDocument, setSelectedDocument, setChunks, chunks, documents, addDocuments, batchDeleteDocuments } = useAppData();

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

  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [docContent, setDocContent] = useState('');
  const [selectedChunks, setSelectedChunks] = useState<Set<string>>(new Set());
  const [showOriginalDocModal, setShowOriginalDocModal] = useState(false);
  const [showSimilarModal, setShowSimilarModal] = useState(false);
  const [similarChunks, setSimilarChunks] = useState<any[]>([]);
  const [currentSimilarChunkId, setCurrentSimilarChunkId] = useState<string>('');
  const [currentSimilarChunkContent, setCurrentSimilarChunkContent] = useState<string>('');
  const [showDocSelector, setShowDocSelector] = useState(false);
  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set());
  
  // 本地文档列表状态
  const [localDocs, setLocalDocs] = useState<{ id: string; name: string; size: string; path: string; content: string }[]>([]);
  const [selectedLocalDoc, setSelectedLocalDoc] = useState<string>('');
  
  // 添加调试状态
  const [loadingStatus, setLoadingStatus] = useState<string>('');

  useEffect(() => {
    const loadDocuments = async () => {
      try {
        const docs = await documentApi.list();
        if (docs) {
          const currentDocIds = documents.map(d => String(d.id));
          if (currentDocIds.length > 0) {
            batchDeleteDocuments(currentDocIds);
          }
          const convertedDocs = docs.map((doc: any) => ({
            id: doc.id as any,
            name: doc.name,
            size: doc.size,
            time: new Date(doc.upload_time).toLocaleString('zh-CN'),
            status: doc.status as any,
            preview: `这是${doc.name}的文档内容预览...`,
            category: doc.category || '未分类',
            tags: doc.tags || []
          }));
          addDocuments(convertedDocs);
        }
      } catch (error) {
        console.error('加载文档列表失败:', error);
      }
    };
    loadDocuments();
  }, [documents, addDocuments, batchDeleteDocuments]);

  // 加载本地文档列表
  useEffect(() => {
    // 直接硬编码一些文档数据用于测试
    const testDocs = [
      {
        id: "01_企业协作平台用户手册",
        name: "01_企业协作平台用户手册.md",
        size: "1.2 KB",
        path: "data/docs/01_企业协作平台用户手册.md",
        content: "# 企业协作平台用户手册 v2.0\n\n## 1. 产品概述\n\n企业协作平台是一款面向中大型组织的综合办公解决方案，支持即时通讯、文档协作、任务管理等功能。"
      },
      {
        id: "02_智能客服系统产品规格",
        name: "02_智能客服系统产品规格.md", 
        size: "1.3 KB",
        path: "data/docs/02_智能客服系统产品规格.md",
        content: "# 智能客服系统产品规格说明书\n\n## 产品名称\n智能客服系统 AI-CS v3.5\n\n## 目标用户\n- 电商企业\n- 金融机构"
      }
    ];
    
    console.log('设置测试文档数据:', testDocs.length, '个文档');
    setLocalDocs(testDocs);
    setLoadingStatus(`已加载${testDocs.length}个测试文档`);
  }, []);

  useEffect(() => {
    if (!selectedDocument) {
      setDocContent('# 请先选择一个文档\n\n请从【知识文档管理】页面选择一个文档,然后才能进行文档切分操作。\n\n文档切分功能需要先选择一个文档作为切分对象。');
      return;
    }

    const loadDocumentContent = async () => {
      try {
        setDocContent(`# ${selectedDocument.name}\n\n加载中...`);
        const contentResponse = await documentApi.getContent(String(selectedDocument.id));
        if (contentResponse) {
          setDocContent(contentResponse.content);
        } else {
          setDocContent(`# ${selectedDocument.name}\n\n## 文档内容预览\n\n这是${selectedDocument.name}的文档内容预览。`);
        }
      } catch (error) {
        console.error('加载文档内容失败:', error);
        setDocContent(`# ${selectedDocument.name}\n\n## 文档内容预览\n\n加载失败，请重试。`);
      }
    };

    loadDocumentContent();
  }, [selectedDocument]);

  const handleChunk = async () => {
    if (!selectedDocument) {
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

      const response = await chunkingApi.split(String(selectedDocument.id), backendConfig as any);

      const newChunks: Chunk[] = response.chunks.map(chunk => ({
        id: chunk.id,
        num: chunk.num,
        content: chunk.content,
        length: chunk.length
      }));

      setChunks(newChunks);
      setSelectedChunks(new Set());
      
      try {
        console.log('开始自动向量化...');
        const embedResponse = await chunkingApi.embed(String(selectedDocument.id));
        console.log('自动向量化完成:', embedResponse);
        alert(`✅ 切分完成！共生成 ${newChunks.length} 个片段，并已自动向量化完成，现在可以用于问答了。`);
      } catch (embedError) {
        console.error('自动向量化失败:', embedError);
        alert(`⚠️ 切分完成！共生成 ${newChunks.length} 个片段，但自动向量化失败。\n\n请手动点击"向量化"按钮完成向量化，否则文档无法用于问答。`);
      }
    } catch (error) {
      console.error('切分失败:', error);
      alert(`❌ 切分失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  };

  const handleFindSimilar = async (chunkId: string, chunkContent: string) => {
    try {
      const response = await retrievalApi.findSimilarChunks(
        chunkId,
        chunkContent,
        similarityThreshold,
        5
      );

      setCurrentSimilarChunkId(chunkId);
      setCurrentSimilarChunkContent(chunkContent);
      setSimilarChunks(response.similar_chunks);
      setShowSimilarModal(true);
    } catch (error) {
      console.error('查找相似片段失败:', error);
      alert(`❌ 查找相似片段失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  };

  const handleReSearchSimilar = async () => {
    if (!currentSimilarChunkId || !currentSimilarChunkContent) {
      return;
    }

    try {
      const response = await retrievalApi.findSimilarChunks(
        currentSimilarChunkId,
        currentSimilarChunkContent,
        similarityThreshold,
        5
      );

      setSimilarChunks(response.similar_chunks);
    } catch (error) {
      console.error('重新查找相似片段失败:', error);
      alert(`重新查找相似片段失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  };

  const handleReset = () => {
    setChunks([]);
    setSelectedChunks(new Set());
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedChunks(new Set(chunks.map(c => c.id)));
    } else {
      setSelectedChunks(new Set());
    }
  };

  const handleSelectChunk = (id: string, checked: boolean) => {
    const newSelected = new Set(selectedChunks);
    if (checked) {
      newSelected.add(id);
    } else {
      newSelected.delete(id);
    }
    setSelectedChunks(newSelected);
  };

  const handleBatchDelete = () => {
    if (selectedChunks.size === 0) {
      alert('请先选择要删除的片段');
      return;
    }
    if (confirm(`确定要删除选中的${selectedChunks.size}个片段吗?`)) {
      setChunks(chunks.filter(c => !selectedChunks.has(c.id)));
      setSelectedChunks(new Set());
      alert('批量删除成功!');
    }
  };

  const handleBatchMerge = () => {
    if (selectedChunks.size < 2) {
      alert('请至少选择2个片段进行合并');
      return;
    }

    const selectedChunksList = chunks.filter(c => selectedChunks.has(c.id));
    const mergedContent = selectedChunksList.map(c => c.content).join('\n\n');

    const mergedChunk: Chunk = {
      id: `merged_${Date.now()}`,
      num: Math.min(...selectedChunksList.map(c => c.num)),
      content: mergedContent,
      length: mergedContent.length
    };

    setChunks(chunks.filter(c => !selectedChunks.has(c.id)));
    setChunks([...chunks.filter(c => !selectedChunks.has(c.id)), mergedChunk]);
    setSelectedChunks(new Set());

    alert('片段合并成功!');
  };

  const handleDocumentSelect = async (docId: string) => {
    try {
      const doc = documents.find(d => d.id === docId);
      if (doc) {
        setSelectedDocument(doc);
        setDocContent(`# ${doc.name}\n\n加载中...`);

        const contentResponse = await documentApi.getContent(docId);
        if (contentResponse) {
          setDocContent(contentResponse.content);
        } else {
          setDocContent(`# ${doc.name}\n\n## 文档内容预览\n\n这是${doc.name}的文档内容预览。`);
        }

        setShowDocSelector(false);
      }
    } catch (error) {
      console.error('加载文档失败:', error);
      alert('加载文档失败');
    }
  };

  const handleToggleExpand = (chunkId: string) => {
    const newExpandedChunks = new Set(expandedChunks);
    if (newExpandedChunks.has(chunkId)) {
      newExpandedChunks.delete(chunkId);
    } else {
      newExpandedChunks.add(chunkId);
    }
    setExpandedChunks(newExpandedChunks);
  };

  // 选择本地文档
  const handleSelectLocalDoc = (docId: string) => {
    const doc = localDocs.find(d => d.id === docId);
    if (doc) {
      setSelectedLocalDoc(docId);
      setDocContent(doc.content);
      // 创建一个模拟的文档对象
      setSelectedDocument({
        id: doc.id,
        name: doc.name,
        size: doc.size,
        time: new Date().toLocaleString('zh-CN'),
        status: 'index',
        preview: '',
        category: '本地文档',
        tags: []
      });
    }
  };

  if (!selectedDocument) {
    return (
      <div className="container">
        <div className="page-header">
          <h1 className="page-title">
            <i className="fas fa-scissors"></i> 文档切分与片段可视化
            <small>RAG Chunking 配置 &amp; 编辑</small>
          </h1>
        </div>
        
        {/* 本地文档列表表格 */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <i className="fas fa-folder-open"></i> 本地文档库
            </h3>
            <span className="tip-text">共 {localDocs.length} 个文档 {loadingStatus && `(${loadingStatus})`}</span>
          </div>
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
              <div className="local-docs-table-container" style={{ overflowX: 'auto' }}>
                <table className="local-docs-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ backgroundColor: 'var(--bg-light)', borderBottom: '2px solid var(--border)' }}>
                      <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '600', color: 'var(--text-secondary)' }}>序号</th>
                      <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '600', color: 'var(--text-secondary)' }}>文档名称</th>
                      <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '600', color: 'var(--text-secondary)' }}>大小</th>
                      <th style={{ padding: '12px 16px', textAlign: 'center', fontWeight: '600', color: 'var(--text-secondary)' }}>操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {localDocs.map((doc, index) => (
                      <tr 
                        key={doc.id} 
                        style={{ 
                          borderBottom: '1px solid var(--border)',
                          backgroundColor: selectedLocalDoc === doc.id ? 'var(--active)' : 'transparent',
                          cursor: 'pointer',
                          transition: 'background-color 0.2s'
                        }}
                        onClick={() => handleSelectLocalDoc(doc.id)}
                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--active)'}
                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = selectedLocalDoc === doc.id ? 'var(--active)' : 'transparent'}
                      >
                        <td style={{ padding: '12px 16px', color: 'var(--text-light)' }}>{index + 1}</td>
                        <td style={{ padding: '12px 16px', color: 'var(--text-main)', fontWeight: '500' }}>
                          <i className="fas fa-file-alt" style={{ marginRight: '8px', color: 'var(--primary)' }}></i>
                          {doc.name}
                        </td>
                        <td style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>{doc.size}</td>
                        <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                          <button 
                            className="btn btn-sm btn-primary"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSelectLocalDoc(doc.id);
                            }}
                          >
                            <i className="fas fa-check"></i> 选择
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        <div className="card" style={{ marginTop: '20px' }}>
          <div className="card-body" style={{ textAlign: 'center', padding: '40px 20px' }}>
            <div style={{ fontSize: '48px', color: 'var(--text-light)', marginBottom: '20px' }}>
              <i className="fas fa-cloud-upload-alt"></i>
            </div>
            <h3 style={{ marginBottom: '12px', color: 'var(--text-main)' }}>或上传新文档</h3>
            <p style={{ marginBottom: '20px', color: 'var(--text-secondary)' }}>
              也可以从【知识文档管理】页面上传新文档
            </p>
            <button className="btn btn-primary" onClick={() => navigate('/documents')}>
              <i className="fas fa-arrow-left"></i> 前往文档管理
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-scissors"></i> 文档切分与片段可视化
          <small>RAG Chunking 配置 &amp; 编辑</small>
        </h1>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-cog"></i> 切分策略配置
          </h3>
          <div className="doc-info-badge">
            <i className="fas fa-file"></i>
            <span>{selectedDocument?.name}</span>
            <span className="separator">|</span>
            <span>{selectedDocument?.size}</span>
            <button
              className="btn btn-sm btn-outline-primary"
              onClick={() => setShowDocSelector(true)}
              style={{ marginLeft: '12px' }}
            >
              <i className="fas fa-exchange-alt"></i> 切换文档
            </button>
          </div>
        </div>
        <div className="card-body">
          <div className="chunk-config-grid">
            <div className="config-card config-card-full">
              <div className="config-card-header">
                <i className="fas fa-th-list"></i>
                <span>切分方式</span>
              </div>
              <select
                className="form-select form-select-lg"
                value={config.type}
                onChange={e => setConfig({ ...config, type: e.target.value as any })}
              >
                <optgroup label="财务制度">
                  <option value="intelligent">💰 财务报销制度切分</option>
                </optgroup>
                <optgroup label="其他文档类型">
                  <option value="product">📄 产品文档切分</option>
                  <option value="technical">⚙️ 技术规范切分</option>
                  <option value="compliance">📋 合规文件切分</option>
                  <option value="hr">👥 HR文档切分</option>
                  <option value="project">📊 项目管理切分</option>
                  <option value="hybrid">🔀 混合切分-标题切分</option>
                </optgroup>
              </select>
            </div>

            <div className="config-card">
              <div className="config-card-header">
                <i className="fas fa-ruler-horizontal"></i>
                <span>Token数量</span>
              </div>
              <input
                type="number"
                className="form-input form-input-lg"
                value={config.chunkTokenSize}
                onChange={e => setConfig({ ...config, chunkTokenSize: parseInt(e.target.value) })}
                min={128}
                max={2048}
                placeholder="128-2048"
              />
              <div className="config-hint">tokens</div>
            </div>

            <div className="config-card">
              <div className="config-card-header">
                <i className="fas fa-layer-group"></i>
                <span>重叠百分比</span>
              </div>
              <input
                type="number"
                className="form-input form-input-lg"
                value={config.overlappedPercent * 100}
                onChange={e => setConfig({ ...config, overlappedPercent: parseFloat(e.target.value) / 100 })}
                min={0}
                max={50}
                placeholder="0-50"
              />
              <div className="config-hint">%</div>
            </div>

            <div className="config-card">
              <div className="config-card-header">
                <i className="fas fa-chart-line"></i>
                <span>相似度阈值</span>
              </div>
              <input
                type="number"
                className="form-input form-input-lg"
                value={similarityThreshold}
                onChange={e => setSimilarityThreshold(parseFloat(e.target.value))}
                min={0}
                max={1}
                step={0.05}
                placeholder="0.0-1.0"
              />
              <div className="config-hint">0-1</div>
            </div>
          </div>

          <div className="action-bar">
            <div className="action-bar-left">
              <button className="btn btn-primary btn-lg" onClick={handleChunk}>
                <i className="fas fa-play"></i> 执行切分
              </button>
              <button className="btn btn-outline" onClick={handleReset}>
                <i className="fas fa-undo"></i> 重置
              </button>
            </div>
            <div className="action-bar-right">
              <button className="btn btn-icon-only" onClick={() => setShowOriginalDocModal(true)} title="查看原文档">
                <i className="fas fa-file-alt"></i>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-list-ul"></i> 切分结果
            {chunks.length > 0 && (
              <span className="badge badge-primary">{chunks.length}</span>
            )}
          </h3>
        </div>
        <div className="card-body">
          <div className="toolbar">
            <div className="toolbar-left">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedChunks.size === chunks.length && chunks.length > 0}
                  onChange={e => handleSelectAll(e.target.checked)}
                />
                <span>全选</span>
              </label>
              <div className="divider"></div>
              <button className="btn btn-sm btn-outline-danger" onClick={handleBatchDelete} disabled={selectedChunks.size === 0}>
                <i className="fas fa-trash-alt"></i> 批量删除
              </button>
              <button className="btn btn-sm btn-outline-success" onClick={handleBatchMerge} disabled={selectedChunks.size < 2}>
                <i className="fas fa-object-group"></i> 合并选中
              </button>
            </div>
            <div className="toolbar-right">
              <span className="selected-count">
                已选 <strong>{selectedChunks.size}</strong> 个片段
              </span>
            </div>
          </div>

          {chunks.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">
                <i className="fas fa-cut"></i>
              </div>
              <h4>暂无切分结果</h4>
              <p>配置切分策略后点击「执行切分」按钮生成片段</p>
            </div>
          ) : (
            <div className="chunk-grid">
              {chunks.map(chunk => (
                <div key={chunk.id} className="chunk-card">
                  <div className="chunk-card-header">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={selectedChunks.has(chunk.id)}
                        onChange={e => handleSelectChunk(chunk.id, e.target.checked)}
                      />
                    </label>
                    <span className="chunk-number">#{chunk.num}</span>
                    <span className="chunk-length">{chunk.length} 字符</span>
                    <button 
                      className="btn btn-sm btn-outline-info"
                      onClick={() => handleToggleExpand(chunk.id)}
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
                      onClick={() => handleFindSimilar(chunk.id, chunk.content)}
                      title="查找相似片段"
                    >
                      <i className="fas fa-search"></i> 相似片段
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {showSimilarModal && (
        <div className="modal-mask modal-large" onClick={() => setShowSimilarModal(false)}>
          <div className="modal-box modal-large-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-search"></i> 相似片段
              </span>
              <button className="modal-close" onClick={() => setShowSimilarModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="similar-threshold-control" style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#f8f9fa', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '15px' }}>
                <label style={{ fontWeight: '500', color: '#333', marginBottom: '0' }}>
                  <i className="fas fa-chart-line" style={{ marginRight: '5px' }}></i>
                  相似度阈值:
                </label>
                <input
                  type="number"
                  className="form-input"
                  style={{ width: '120px', marginBottom: '0' }}
                  value={similarityThreshold}
                  onChange={e => setSimilarityThreshold(parseFloat(e.target.value))}
                  min={0}
                  max={1}
                  step={0.05}
                />
                <button
                  className="btn btn-primary"
                  onClick={handleReSearchSimilar}
                >
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
              <button className="btn btn-default" onClick={() => setShowSimilarModal(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {showOriginalDocModal && (
        <div className="modal-mask modal-large" onClick={() => setShowOriginalDocModal(false)}>
          <div className="modal-box modal-large-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-file-alt"></i> 原文档预览
              </span>
              <button className="modal-close" onClick={() => setShowOriginalDocModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body modal-preview-body">
              <div className="doc-preview">
                <pre>{docContent}</pre>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowOriginalDocModal(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {showDocSelector && (
        <div className="modal-mask" onClick={() => setShowDocSelector(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>选择文档</span>
              <button className="modal-close" onClick={() => setShowDocSelector(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="document-selector-list">
                {documents.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-state-icon">
                      <i className="fas fa-file-alt"></i>
                    </div>
                    <h4>暂无文档</h4>
                    <p>请先在【知识文档管理】页面上传文档</p>
                  </div>
                ) : (
                  documents.map(doc => (
                    <div
                      key={doc.id}
                      className="document-item"
                      onClick={() => handleDocumentSelect(doc.id as string)}
                    >
                      <div className="document-item-icon">
                        <i className="fas fa-file"></i>
                      </div>
                      <div className="document-item-info">
                        <div className="document-item-name">{doc.name}</div>
                        <div className="document-item-meta">
                          <span>{doc.size}</span>
                          <span>{doc.time}</span>
                          <span className={`status status-${doc.status}`}>
                            {doc.status}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

function similarityBadgeClass(similarity: number): string {
  if (similarity >= 0.8) return 'badge-success';
  if (similarity >= 0.6) return 'badge-warning';
  return 'badge-danger';
};

export default Chunk;