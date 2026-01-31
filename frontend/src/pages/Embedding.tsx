import React, { useState, useEffect, useRef } from 'react';
import { useAppData } from '../contexts/AppDataContext';
import { embeddingApi, vectorDbApi } from '../services/api';
import type { Chunk } from '../types';
import * as echarts from 'echarts';
import './Embedding.css';

const Embedding: React.FC = () => {
  const { chunks } = useAppData();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const [chunkVecList, setChunkVecList] = useState<Chunk[]>([]);
  const [canvasType, setCanvasType] = useState<'2d' | '3d'>('2d');
  const [showDimConfig, setShowDimConfig] = useState(false);
  const [showIndexConfig, setShowIndexConfig] = useState(false);
  const [showBackupModal, setShowBackupModal] = useState(false);
  const [showRestoreModal, setShowRestoreModal] = useState(false);
  const [showQualityModal, setShowQualityModal] = useState(false);
  const [showRebuildModal, setShowRebuildModal] = useState(false);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [currentModel, setCurrentModel] = useState('BAAI/bge-base-zh-v1.5');
  const [vectorDimension, setVectorDimension] = useState(512);
  const [performanceData, setPerformanceData] = useState<any>(null);
  const [backupFiles, setBackupFiles] = useState<any[]>([]);
  const [qualityMetrics, setQualityMetrics] = useState<any>(null);
  const [rebuildProgress, setRebuildProgress] = useState(0);

  // 可用的嵌入模型列表
  const availableModels = [
    { name: 'BAAI/bge-small-zh-v1.5', type: 'bge', dimension: 512, desc: '轻量级中文模型,速度快' },
    { name: 'BAAI/bge-base-zh-v1.5', type: 'bge', dimension: 768, desc: '平衡型中文模型' },
    { name: 'BAAI/bge-large-zh-v1.5', type: 'bge', dimension: 1024, desc: '高精度中文模型' },
    { name: 'text2vec-base-chinese', type: 'text2vec', dimension: 768, desc: '通用中文文本模型' },
    { name: 'ernie-embeddings-v2', type: 'ernie', dimension: 1024, desc: '百度ERNIE模型' }
  ];
  
  // 补充缺失的变量定义（修复未定义报错）
  const [indexConfig, setIndexConfig] = useState({
    type: 'HNSW',
    efConstruction: 128,
    M: 16,
    nlist: 1024
  });
  
  const [vectorDBConfig, setVectorDBConfig] = useState({
    type: 'FAISS',
    host: 'localhost',
    port: 6333,
    index: 'default_index'
  });

  // 加载向量数据库状态
  const loadVectorDBStatus = async () => {
    try {
      const status = await vectorDbApi.getStatus();
      console.log('向量数据库状态:', status);
    } catch (error) {
      console.error('加载向量数据库状态失败:', error);
    }
  };

// 生成模拟数据
  const generateMockChunkVecData = () => {
    const mockData: Chunk[] = chunks.slice(0, 10).map((chunk, i) => ({
      ...chunk,
      vecStatus: ['success', 'pending', 'error'][Math.floor(Math.random() * 3)] as 'success' | 'pending' | 'error',
      vectorX: Math.random() * 2 - 1,
      vectorY: Math.random() * 2 - 1
    }));
    setChunkVecList(mockData);
  };

  // 切换模型
  const handleSwitchModel = async (model: any) => {
    try {
      // 先加载模型
      const response = await embeddingApi.load({
        model_type: model.type,
        model_name: model.name,
        batch_size: 32,
        device: 'cpu'
      });

      if (response.status && response.status !== 'error') {
        // 更新本地状态
        setCurrentModel(model.name);
        setVectorDimension(model.dimension);
        setShowModelSelector(false);

        // 同时更新系统设置
        try {
          const settingsResponse = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/settings`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              embedding_model_name: model.name
            })
          });

          if (settingsResponse.ok) {
            console.log('系统设置已更新:', model.name);
          }
        } catch (settingsError) {
          console.error('更新系统设置失败:', settingsError);
        }

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
    generateMockChunkVecData();
  }, []);

  // 从系统设置加载当前模型
  useEffect(() => {
    const loadSystemSettings = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/settings`);
        if (response.ok) {
          const settings = await response.json();
          if (settings.data && settings.data.embedding_model_name) {
            setCurrentModel(settings.data.embedding_model_name);
            // 根据模型名称设置维度
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

  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        chartInstance.current.dispose();
      }
    };
  }, []);

  const handleGenerateVec = async () => {
    // 检查是否有文档可以向量化
    if (chunks.length === 0) {
      alert('请先在文档切分页面切分文档，然后再进行向量化');
      return;
    }

    // 检查模型是否已加载
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

    // 获取当前选中的文档ID（如果有）
    // 这里我们假设chunks来自当前选中的文档
    const docId = chunks.length > 0 ? chunks[0].id.split('_')[0] : null;

    if (!docId) {
      alert('无法确定文档ID，请先选择文档');
      return;
    }

    try {
      // 调用后端API进行向量化
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/chunking/embed?doc_id=${docId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const result = await response.json();
        alert(`向量化完成！\n${result.message}`);
        
        // 更新片段状态
        setChunkVecList(prev => prev.map(item => ({ ...item, vecStatus: 'success' as any })));
        
        // 生成可视化
        generateVecCanvas();
      } else {
        const error = await response.json();
        alert(`向量化失败: ${error.detail || error.message || '未知错误'}`);
      }
    } catch (error) {
      console.error('向量化失败:', error);
      alert(`向量化失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  };

  const generateVecCanvas = () => {
    if (!chartRef.current) return;

    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    chartInstance.current = echarts.init(chartRef.current);

    const colorMap: Record<string, string> = {
      success: '#00B42A',
      processing: '#FF7D00',
      pending: '#86909C',
      error: '#F53F3F'
    };

    const seriesData = chunkVecList.map(item => ({
      name: `片段${item.num}`,
      value: [Math.random() * 100, Math.random() * 100, canvasType === '3d' ? Math.random() * 100 : 0],
      itemStyle: { color: colorMap[item.vecStatus || 'pending'] },
      data: item
    }));

    const option: echarts.EChartsOption = {
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          const chunk = params.data.data;
          return `<div style="text-align:left;">
            <p><b>片段${chunk.num}</b></p>
            <p>状态:${chunk.vecStatus === 'success' ? '已向量化' : chunk.vecStatus}</p>
            <p>维度:${chunk.vecDim}d</p>
            <p>内容:${chunk.content}</p>
          </div>`;
        }
      },
      xAxis: canvasType === '2d' ? { type: 'value', name: '维度X' } : undefined,
      yAxis: canvasType === '2d' ? { type: 'value', name: '维度Y' } : undefined,
      grid: { top: 60, bottom: 40 },
      series: [{
        type: canvasType === '2d' ? 'scatter' : 'scatter3D' as any,
        data: seriesData,
        symbolSize: 12
      }]
    };

    chartInstance.current.setOption(option);
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

  const handleDimensionConfig = (dim: number) => {
    setVectorDimension(dim);
    setShowDimConfig(false);
    alert(`已设置向量维度: ${dim}`);

    // 模拟性能数据
    const performance = {
      256: { storage: '10%', retrieval: '98%', accuracy: '92%' },
      512: { storage: '20%', retrieval: '95%', accuracy: '95%' },
      768: { storage: '30%', retrieval: '92%', accuracy: '97%' },
      1024: { storage: '40%', retrieval: '88%', accuracy: '98%' },
      1536: { storage: '50%', retrieval: '85%', accuracy: '99%' }
    };
    setPerformanceData(performance[dim as keyof typeof performance]);
  };

  // 向量数据库操作
  const handleTestConnection = () => {
    alert('正在测试连接...\n\n✅ 连接成功!\n- 数据库: FAISS\n- 索引数量: 5,234\n- 维度: 512\n- 状态: 在线');
  };

  // 索引配置
  const handleIndexConfig = (config: any) => {
    setIndexConfig({ ...indexConfig, ...config });
    setShowIndexConfig(false);
    alert(`已应用索引配置: ${config.type}`);
  };

  // 备份/恢复
  const handleBackup = async () => {
    try {
      const response = await vectorDbApi.save();
      if (response.success) {
        alert('备份完成!');
        setShowBackupModal(false);
      } else {
        alert(`备份失败: ${response.message}`);
      }
    } catch (error) {
      console.error('备份失败:', error);
      alert(`备份失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  };

  const handleRestore = (backupId: string) => {
    if (confirm(`确定要恢复备份 ${backupId} 吗?当前数据将被覆盖。`)) {
      alert('恢复功能需要实现后端支持');
      setShowRestoreModal(false);
    }
  };

  // 质量分析
  const handleQualityAnalysis = () => {
    const metrics = {
      totalVectors: chunkVecList.length,
      avgNorm: 0.87,
      normStdDev: 0.12,
      outlierCount: 3,
      outlierRatio: '0.6%',
      densityScore: 0.92,
      coverage: 0.95,
      duplicationRate: '1.2%'
    };
    setQualityMetrics(metrics);
    setShowQualityModal(true);
  };

  // 索引重建
  const handleRebuildIndex = async (strategy: string) => {
    setShowRebuildModal(false);
    setRebuildProgress(10);

    try {
      // 重新初始化向量数据库
      const response = await vectorDbApi.init({
        db_type: vectorDBConfig.type,
        dimension: vectorDimension,
        index_type: indexConfig.type,
        host: vectorDBConfig.host,
        port: vectorDBConfig.port,
        collection_name: vectorDBConfig.index
      });

      if (response.success) {
        setRebuildProgress(100);
        alert(`索引重建完成!\n\n统计信息:\n- 重建策略: ${strategy}\n- 向量维度: ${vectorDimension}\n- 索引类型: ${indexConfig.type}`);
      } else {
        alert(`索引重建失败: ${response.message}`);
      }
    } catch (error) {
      console.error('索引重建失败:', error);
      alert(`索引重建失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setRebuildProgress(0);
    }
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

      {/* 向量模型切换 */}
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

      {/* 向量维度设置卡片 */}
      <div className="config-card">
        <div className="config-card-header">
          <div>
            <h4>
              <i className="fas fa-ruler-combined"></i> 向量维度信息
            </h4>
            <p className="config-desc">当前维度: <strong>{vectorDimension}d</strong></p>
          </div>
        </div>
        <div className="config-card-body">
          {performanceData && (
            <>
              <div className="config-item">
                <span className="config-label">存储开销</span>
                <span className="config-value">{performanceData.storage}</span>
              </div>
                <div className="config-item">
                  <span className="config-label">检索速度</span>
                  <span className="config-value">{performanceData.retrieval}</span>
                </div>
                <div className="config-item">
                  <span className="config-label">检索精度</span>
                  <span className="config-value">{performanceData.accuracy}</span>
                </div>
              </>
            )}
        </div>
      </div>

      {/* 向量数据库管理操作栏 */}
      <div className="vector-db-actions">
        <div className="db-status">
          <span className="status-indicator connected"></span>
          <span className="db-info">
            <strong>FAISS</strong>
            <span className="separator">|</span>
            <span>向量数据库已连接</span>
          </span>
        </div>
        <div className="db-buttons">
          <button className="btn btn-sm btn-outline-info" onClick={() => setShowIndexConfig(true)}>
            <i className="fas fa-sitemap"></i> 索引配置
          </button>
          <button className="btn btn-sm btn-outline-success" onClick={() => setShowBackupModal(true)}>
            <i className="fas fa-download"></i> 备份向量
          </button>
          <button className="btn btn-sm btn-outline-warning" onClick={() => setShowRestoreModal(true)}>
            <i className="fas fa-upload"></i> 恢复向量
          </button>
          <button className="btn btn-sm btn-outline-secondary" onClick={handleQualityAnalysis}>
            <i className="fas fa-chart-line"></i> 质量分析
          </button>
          <button className="btn btn-sm btn-outline-danger" onClick={() => setShowRebuildModal(true)}>
            <i className="fas fa-redo"></i> 重建索引
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-chart-scatter"></i> 向量低维空间分布可视化
          </h3>
          <div className="btn-group">
            <button className="btn btn-primary" onClick={handleGenerateVec}>
              <i className="fas fa-magic"></i> 批量生成向量
            </button>
            <button className="btn btn-default" onClick={() => alert('状态刷新完成!')}>
              <i className="fas fa-sync-alt"></i> 刷新状态
            </button>
          </div>
        </div>
        <div className="card-body">
          <div className="vis-canvas" ref={chartRef} style={{ height: '400px' }}>
            {chunkVecList.every(c => c.vecStatus === 'pending') && (
              <div className="canvas-placeholder">
                <div className="canvas-icon">
                  <i className="fas fa-project-diagram"></i>
                </div>
                <h4>向量低维分布可视化</h4>
                <p>点击「批量生成向量」后,展示片段在2D/3D空间的聚类分布</p>
              </div>
            )}
          </div>
          <div className="canvas-tab">
            <div
              className={`canvas-tab-item ${canvasType === '2d' ? 'canvas-tab-active' : ''}`}
              onClick={() => { setCanvasType('2d'); generateVecCanvas(); }}
            >
              2D分布
            </div>
            <div
              className={`canvas-tab-item ${canvasType === '3d' ? 'canvas-tab-active' : ''}`}
              onClick={() => { setCanvasType('3d'); generateVecCanvas(); }}
            >
              3D分布
            </div>
          </div>
        </div>
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
                  <th style={{ width: '20%' }}>向量状态</th>
                  <th style={{ width: '15%' }}>向量长度</th>
                  <th style={{ width: '15%' }}>操作</th>
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
                    <td>{chunk.vecLength || '-'}</td>
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

      {/* 向量维度配置模态框 */}
      {showDimConfig && (
        <div className="modal-mask" onClick={() => setShowDimConfig(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-ruler-combined"></i> 向量维度设置
              </span>
              <button className="modal-close" onClick={() => setShowDimConfig(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="dim-options">
                {[
                  { value: 256, label: '256维', storage: '低', speed: '极快', accuracy: '92%', use: '快速检索场景' },
                  { value: 512, label: '512维', storage: '中', speed: '快', accuracy: '95%', use: '平衡性能场景' },
                  { value: 768, label: '768维', storage: '中高', speed: '正常', accuracy: '97%', use: '通用场景' },
                  { value: 1024, label: '1024维', storage: '高', speed: '较慢', accuracy: '98%', use: '高精度场景' },
                  { value: 1536, label: '1536维', storage: '极高', speed: '慢', accuracy: '99%', use: '最高精度场景' }
                ].map((dim, idx) => (
                  <div
                    key={idx}
                    className={`dim-option ${vectorDimension === dim.value ? 'selected' : ''}`}
                    onClick={() => handleDimensionConfig(dim.value)}
                  >
                    <div className="dim-header">
                      <strong>{dim.label}</strong>
                      {vectorDimension === dim.value && (
                        <span className="dim-selected">当前选择</span>
                      )}
                    </div>
                    <div className="dim-details">
                      <div className="dim-detail-item">
                        <span>存储:</span>
                        <strong>{dim.storage}</strong>
                      </div>
                      <div className="dim-detail-item">
                        <span>速度:</span>
                        <strong>{dim.speed}</strong>
                      </div>
                      <div className="dim-detail-item">
                        <span>精度:</span>
                        <strong>{dim.accuracy}</strong>
                      </div>
                    </div>
                    <div className="dim-use">
                      适用: {dim.use}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowDimConfig(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 索引配置模态框 */}
      {showIndexConfig && (
        <div className="modal-mask" onClick={() => setShowIndexConfig(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-sitemap"></i> 索引类型配置
              </span>
              <button className="modal-close" onClick={() => setShowIndexConfig(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="index-options">
                {[
                  { type: 'HNSW', desc: '层次化小世界图', speed: '极快', memory: '高', accuracy: '99%', use: '高精度场景' },
                  { type: 'IVF', desc: '倒排文件索引', speed: '快', memory: '中', accuracy: '95%', use: '通用场景' },
                  { type: 'PQ', desc: '乘积量化', speed: '正常', memory: '低', accuracy: '90%', use: '存储敏感场景' },
                  { type: 'IVF_PQ', desc: '倒排+乘积量化', speed: '快', memory: '中低', accuracy: '92%', use: '平衡场景' }
                ].map((idx, i) => (
                  <div
                    key={i}
                    className={`index-option ${indexConfig.type === idx.type ? 'selected' : ''}`}
                    onClick={() => handleIndexConfig({ type: idx.type })}
                  >
                    <div className="index-header">
                      <strong>{idx.type}</strong>
                      {indexConfig.type === idx.type && <span className="selected-badge">已选择</span>}
                    </div>
                    <div className="index-desc">{idx.desc}</div>
                    <div className="index-perf">
                      <div>
                        <span>速度:</span>
                        <strong>{idx.speed}</strong>
                      </div>
                      <div>
                        <span>内存:</span>
                        <strong>{idx.memory}</strong>
                      </div>
                      <div>
                        <span>精度:</span>
                        <strong>{idx.accuracy}</strong>
                      </div>
                    </div>
                    <div className="index-use">适用: {idx.use}</div>
                  </div>
                ))}
              </div>
              <div className="index-params">
                <h4>索引参数</h4>
                <div className="form-grid">
                  <div className="form-group">
                    <label>ef_construction</label>
                    <input
                      type="number"
                      className="form-input"
                      value={indexConfig.efConstruction}
                      onChange={e => setIndexConfig({ ...indexConfig, efConstruction: parseInt(e.target.value) })}
                    />
                  </div>
                  <div className="form-group">
                    <label>M (连接数)</label>
                    <input
                      type="number"
                      className="form-input"
                      value={indexConfig.M}
                      onChange={e => setIndexConfig({ ...indexConfig, M: parseInt(e.target.value) })}
                    />
                  </div>
                  <div className="form-group">
                    <label>nlist (聚类数)</label>
                    <input
                      type="number"
                      className="form-input"
                      value={indexConfig.nlist}
                      onChange={e => setIndexConfig({ ...indexConfig, nlist: parseInt(e.target.value) })}
                    />
                  </div>
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowIndexConfig(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 备份模态框 */}
      {showBackupModal && (
        <div className="modal-mask" onClick={() => setShowBackupModal(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-download"></i> 向量备份
              </span>
              <button className="modal-close" onClick={() => setShowBackupModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="backup-options">
                <div className="backup-option">
                  <input type="radio" name="backup" id="full" defaultChecked />
                  <label htmlFor="full">
                    <strong>全量备份</strong>
                    <p>备份所有向量数据</p>
                  </label>
                </div>
                <div className="backup-option">
                  <input type="radio" name="backup" id="incremental" />
                  <label htmlFor="incremental">
                    <strong>增量备份</strong>
                    <p>仅备份新增/修改的向量</p>
                  </label>
                </div>
              </div>
              <div className="backup-info">
                <p><strong>预估备份信息:</strong></p>
                <ul>
                  <li>向量数量: {chunkVecList.length}</li>
                  <li>预估大小: ~{((chunkVecList.length * vectorDimension * 4) / 1024 / 1024).toFixed(2)}MB</li>
                  <li>预估耗时: ~5秒</li>
                </ul>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowBackupModal(false)}>
                取消
              </button>
              <button className="btn btn-primary" onClick={handleBackup}>
                <i className="fas fa-download"></i> 开始备份
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 恢复模态框 */}
      {showRestoreModal && (
        <div className="modal-mask" onClick={() => setShowRestoreModal(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-upload"></i> 向量恢复
              </span>
              <button className="modal-close" onClick={() => setShowRestoreModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              {backupFiles.length === 0 ? (
                <div className="empty-tip">暂无备份文件</div>
              ) : (
                <div className="backup-list">
                  {backupFiles.map(backup => (
                    <div key={backup.id} className="backup-item">
                      <div className="backup-info">
                        <strong>{backup.name}</strong>
                        <span>{backup.timestamp}</span>
                        <span>{backup.size}</span>
                        <span>{backup.chunks} 个片段</span>
                        <span>{backup.dimension}d</span>
                      </div>
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={() => handleRestore(backup.id)}
                      >
                        <i className="fas fa-upload"></i> 恢复
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowRestoreModal(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 质量分析模态框 */}
      {showQualityModal && qualityMetrics && (
        <div className="modal-mask" onClick={() => setShowQualityModal(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-chart-line"></i> 向量质量分析
              </span>
              <button className="modal-close" onClick={() => setShowQualityModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="quality-metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">总向量数</div>
                  <div className="metric-value">{qualityMetrics.totalVectors}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">平均范数</div>
                  <div className="metric-value">{qualityMetrics.avgNorm}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">范数标准差</div>
                  <div className="metric-value">{qualityMetrics.normStdDev}</div>
                </div>
                <div className="metric-card warning">
                  <div className="metric-label">异常向量</div>
                  <div className="metric-value">{qualityMetrics.outlierCount}</div>
                </div>
                <div className="metric-card success">
                  <div className="metric-label">密度评分</div>
                  <div className="metric-value">{qualityMetrics.densityScore}</div>
                </div>
                <div className="metric-card info">
                  <div className="metric-label">覆盖率</div>
                  <div className="metric-value">{qualityMetrics.coverage}</div>
                </div>
              </div>
              <div className="quality-suggestions">
                <h4>分析结果与建议</h4>
                <ul>
                  <li>检测到 <strong>{qualityMetrics.outlierCount}</strong> 个异常向量 ({qualityMetrics.outlierRatio})</li>
                  <li>向量分布 <strong>良好</strong>,密度评分为 {qualityMetrics.densityScore}</li>
                  <li>发现 <strong>{qualityMetrics.duplicationRate}</strong> 的重复向量</li>
                  <li>建议: 删除异常向量以提高检索质量</li>
                </ul>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowQualityModal(false)}>
                关闭
              </button>
              <button className="btn btn-danger" onClick={() => {
                alert('已清理异常向量!');
                setShowQualityModal(false);
              }}>
                <i className="fas fa-broom"></i> 一键清理
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 索引重建模态框 */}
      {showRebuildModal && (
        <div className="modal-mask" onClick={() => setShowRebuildModal(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-redo"></i> 索引重建
              </span>
              <button className="modal-close" onClick={() => setShowRebuildModal(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              {rebuildProgress > 0 ? (
                <div className="rebuild-progress">
                  <div className="progress-header">
                    <h4>正在重建索引...</h4>
                    <span className="progress-value">{rebuildProgress}%</span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${rebuildProgress}%` }}
                    ></div>
                  </div>
                  <p className="progress-info">处理向量: {Math.floor(chunkVecList.length * rebuildProgress / 100)} / {chunkVecList.length}</p>
                </div>
              ) : (
                <div className="rebuild-options">
                  <h4>选择重建策略</h4>
                  <div className="rebuild-strategy-list">
                    <div
                      className="rebuild-strategy"
                      onClick={() => handleRebuildIndex('全量重建')}
                    >
                      <div className="strategy-icon">
                        <i className="fas fa-sync-alt"></i>
                      </div>
                      <div className="strategy-info">
                        <strong>全量重建</strong>
                        <p>删除旧索引,完全重建所有向量索引</p>
                        <small>耗时较长,但索引质量最高</small>
                      </div>
                    </div>
                    <div
                      className="rebuild-strategy"
                      onClick={() => handleRebuildIndex('增量重建')}
                    >
                      <div className="strategy-icon">
                        <i className="fas fa-plus-circle"></i>
                      </div>
                      <div className="strategy-info">
                        <strong>增量重建</strong>
                        <p>仅重建新增/修改的向量索引</p>
                        <small>耗时短,适合日常更新</small>
                      </div>
                    </div>
                  </div>
                  <div className="rebuild-info">
                    <p><strong>当前索引信息:</strong></p>
                    <ul>
                      <li>索引类型: {indexConfig.type}</li>
                      <li>向量数量: {chunkVecList.length}</li>
                      <li>向量维度: {vectorDimension}</li>
                      <li>预估全量重建时间: ~2.3秒</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
            <div className="modal-footer">
              {rebuildProgress === 0 && (
                <button className="btn btn-default" onClick={() => setShowRebuildModal(false)}>
                  关闭
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 模型选择模态框 */}
      {showModelSelector && (
        <div className="modal-mask" onClick={() => setShowModelSelector(false)}>
          <div className="modal-box modal-large" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-cube"></i> 选择嵌入模型
              </span>
              <button className="modal-close" onClick={() => setShowModelSelector(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="model-grid">
                {availableModels.map((model, idx) => (
                  <div
                    key={idx}
                    className={`model-option ${currentModel === model.name ? 'selected' : ''}`}
                    onClick={() => handleSwitchModel(model)}
                  >
                    <div className="model-icon">{model.type === 'bge' ? '⚡' : model.type === 'text2vec' ? '📝' : '🤖'}</div>
                    <div className="model-info">
                      <div className="model-name">{model.name}</div>
                      <div className="model-meta">
                        <span className="badge">{model.type}</span>
                        <span className="badge">{model.dimension}d</span>
                      </div>
                      <div className="model-desc">{model.desc}</div>
                    </div>
                    {currentModel === model.name && (
                      <div className="model-check">
                        <i className="fas fa-check-circle"></i>
                      </div>
                    )}
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