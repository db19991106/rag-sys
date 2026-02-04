import React, { useState, useEffect } from 'react';
import { useAppData } from '../contexts/AppDataContext';
import { embeddingApi, vectorDbApi } from '../services/api';
import type { Chunk } from '../types';
import './Embedding.css';

const Embedding: React.FC = () => {
  const { chunks } = useAppData();

  const [chunkVecList, setChunkVecList] = useState<Chunk[]>([]);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [currentModel, setCurrentModel] = useState('BAAI/bge-base-zh-v1.5');
  const [vectorDimension, setVectorDimension] = useState(512);
  const [isGenerating, setIsGenerating] = useState(false);

  const availableModels = [
    { name: 'BAAI/bge-small-zh-v1.5', type: 'bge', dimension: 512, desc: '轻量级中文模型,速度快' },
    { name: 'BAAI/bge-base-zh-v1.5', type: 'bge', dimension: 768, desc: '平衡型中文模型' },
    { name: 'BAAI/bge-large-zh-v1.5', type: 'bge', dimension: 1024, desc: '高精度中文模型' },
    { name: 'text2vec-base-chinese', type: 'text2vec', dimension: 768, desc: '通用中文文本模型' },
    { name: 'ernie-embeddings-v2', type: 'ernie', dimension: 1024, desc: '百度ERNIE模型' }
  ];

  const loadActualChunkVecData = async () => {
    try {
      const response = await vectorDbApi.getDocuments();
      const vectorDbData = response.data || {};
      const documents = vectorDbData.documents || [];

      const vectorStatusMap: Record<string, 'success'> = {};
      documents.forEach(doc => {
        doc.chunks.forEach(chunk => {
          vectorStatusMap[chunk.vector_id] = 'success';
        });
      });

      const actualData: Chunk[] = chunks.map(chunk => ({
        ...chunk,
        vecStatus: vectorStatusMap[chunk.id] || 'pending'
      }));

      setChunkVecList(actualData);
    } catch (error) {
      console.error('加载实际向量化状态失败:', error);
      const defaultData: Chunk[] = chunks.map(chunk => ({
        ...chunk,
        vecStatus: 'pending'
      }));
      setChunkVecList(defaultData);
    }
  };

  const handleSwitchModel = async (model: any) => {
    try {
      const response = await embeddingApi.load({
        model_type: model.type,
        model_name: model.name,
        batch_size: 32,
        device: 'cpu'
      });

      if (response.status && response.status !== 'error') {
        setCurrentModel(model.name);
        setVectorDimension(model.dimension);
        setShowModelSelector(false);
        alert(`已切换模型: ${model.name}`);
      } else {
        alert(`模型加载失败: ${response.message}`);
      }
    } catch (error) {
      console.error('切换模型失败:', error);
      alert(`切换模型失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  };

  useEffect(() => {
    loadActualChunkVecData();
  }, [chunks]);

  useEffect(() => {
    const loadSystemSettings = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/settings`);
        if (response.ok) {
          const settings = await response.json();
          if (settings.data && settings.data.embedding_model_name) {
            setCurrentModel(settings.data.embedding_model_name);
            const model = availableModels.find(m => m.name === settings.data.embedding_model_name);
            if (model) {
              setVectorDimension(model.dimension);
            }
          }
        }
      } catch (error) {
        console.error('加载系统设置失败:', error);
      }
    };

    loadSystemSettings();
  }, []);

  const handleGenerateVec = async () => {
    if (chunks.length === 0) {
      alert('请先在文档切分页面切分文档，然后再进行向量化');
      return;
    }

    try {
      const status = await embeddingApi.getStatus();
      if (!status.success || !status.data.is_loaded) {
        alert('请先加载嵌入模型，然后再进行向量化');
        return;
      }
    } catch (error) {
      console.error('检查模型状态失败:', error);
      alert('无法检查模型状态，请确保已加载嵌入模型');
      return;
    }

    const docId = chunks.length > 0 ? chunks[0].id.split('_')[0] : null;

    if (!docId) {
      alert('无法确定文档ID，请先选择文档');
      return;
    }

    try {
      setIsGenerating(true);
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/chunking/embed?doc_id=${docId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const result = await response.json();
        alert(`向量化完成！\n${result.message}`);
        await loadActualChunkVecData();
      } else {
        const error = await response.json();
        alert(`向量化失败: ${error.detail || error.message || '未知错误'}`);
      }
    } catch (error) {
      console.error('向量化失败:', error);
      alert(`向量化失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const getStatusClass = (status?: string) => {
    const statusMap: Record<string, string> = {
      pending: 'status-pending',
      processing: 'status-processing',
      success: 'status-success',
      error: 'status-error'
    };
    return statusMap[status || 'pending'] || 'status-pending';
  };

  const getStatusText = (status?: string) => {
    const statusMap: Record<string, string> = {
      pending: '待向量化',
      processing: '正在向量化',
      success: '已向量化·已索引',
      error: '向量化失败'
    };
    return statusMap[status || 'pending'] || '待向量化';
  };

  const getStatusIcon = (status?: string) => {
    const statusMap: Record<string, string> = {
      pending: 'fas fa-clock',
      processing: 'fas fa-spinner loading',
      success: 'fas fa-check-circle',
      error: 'fas fa-exclamation-circle'
    };
    return statusMap[status || 'pending'] || 'fas fa-clock';
  };

  const successCount = chunkVecList.filter(c => c.vecStatus === 'success').length;
  const processingCount = chunkVecList.filter(c => c.vecStatus === 'processing').length;
  const pendingCount = chunkVecList.filter(c => c.vecStatus === 'pending').length;
  const errorCount = chunkVecList.filter(c => c.vecStatus === 'error').length;

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-vector-square"></i> 向量表示与索引状态展示
          <small>RAG Embedding 可视化 & 索引管理</small>
        </h1>
      </div>

      <div className="stat-card-group">
        <div className="stat-card">
          <div className="stat-label">已向量化片段</div>
          <div className="stat-value">
            {successCount} <span className="stat-unit">/ {chunkVecList.length} 个</span>
          </div>
        </div>
        <div className="stat-card stat-card-processing">
          <div className="stat-label">正在向量化</div>
          <div className="stat-value">{processingCount} <span className="stat-unit">个</span></div>
        </div>
        <div className="stat-card stat-card-pending">
          <div className="stat-label">待向量化片段</div>
          <div className="stat-value">{pendingCount} <span className="stat-unit">个</span></div>
        </div>
        <div className="stat-card stat-card-error">
          <div className="stat-label">向量化失败</div>
          <div className="stat-value">{errorCount} <span className="stat-unit">个</span></div>
        </div>
      </div>

      <div className="model-switcher">
        <div className="model-info">
          <div className="model-label">当前模型</div>
          <div className="model-name">{currentModel}</div>
          <div className="model-meta">
            <span className="badge">{availableModels.find(m => m.name === currentModel)?.type.toUpperCase()}</span>
            <span className="badge">{vectorDimension}d</span>
          </div>
        </div>
        <button className="btn btn-sm btn-primary" onClick={() => setShowModelSelector(true)}>
          <i className="fas fa-exchange-alt"></i> 切换模型
        </button>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-list-ol"></i> 切分片段向量状态列表
          </h3>
        </div>
        <div className="card-body">
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th style={{ width: '10%' }}>片段编号</th>
                  <th style={{ width: '40%' }}>片段内容</th>
                  <th style={{ width: '25%' }}>向量状态</th>
                  <th style={{ width: '25%' }}>操作</th>
                </tr>
              </thead>
              <tbody>
                {chunkVecList.map(chunk => (
                  <tr key={chunk.id}>
                    <td><span className="chunk-num">{chunk.num}</span></td>
                    <td className="chunk-content">{chunk.content}</td>
                    <td>
                      <span className={`vec-status ${getStatusClass(chunk.vecStatus)}`}>
                        <i className={getStatusIcon(chunk.vecStatus)}></i> {getStatusText(chunk.vecStatus)}
                      </span>
                    </td>
                    <td>
                      <button className="btn btn-sm btn-default" onClick={() => alert(`重新生成片段${chunk.num}的向量`)}>
                        <i className="fas fa-magic"></i> 重新生成
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="action-bar">
        <div className="action-bar-left">
          <button className="btn btn-primary btn-lg" onClick={handleGenerateVec} disabled={isGenerating}>
            <i className={`fas ${isGenerating ? 'fa-spinner fa-spin' : 'fa-magic'}`}></i> 批量生成向量
          </button>
          <button className="btn btn-default" onClick={async () => {
            await loadActualChunkVecData();
            alert('状态刷新完成!');
          }}>
            <i className="fas fa-sync-alt"></i> 刷新状态
          </button>
        </div>
      </div>

      {showModelSelector && (
        <div className="modal-mask" onClick={() => setShowModelSelector(false)}>
          <div className="modal-box modal-large" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-exchange-alt"></i> 切换嵌入模型
              </span>
              <button className="modal-close" onClick={() => setShowModelSelector(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="model-list">
                {availableModels.map((model, index) => (
                  <div
                    key={index}
                    className={`model-option ${currentModel === model.name ? 'selected' : ''}`}
                    onClick={() => handleSwitchModel(model)}
                  >
                    <div className="model-header">
                      <strong>{model.name}</strong>
                      {currentModel === model.name && <span className="current-badge">当前使用</span>}
                    </div>
                    <div className="model-desc">{model.desc}</div>
                    <div className="model-meta">
                      <span className="badge">{model.type.toUpperCase()}</span>
                      <span className="badge">{model.dimension}d</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowModelSelector(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Embedding;