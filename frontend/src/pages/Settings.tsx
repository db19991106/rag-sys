import React, { useState, useEffect } from 'react';
import { settingsApi } from '../services/api';
import './Settings.css';

interface SystemSettings {
  // 嵌入模型配置
  embedding_model_type: string;
  embedding_model_name: string;
  embedding_device: string;
  embedding_batch_size: number;
  
  // 重排序配置
  enable_rerank: boolean;
  reranker_type: string;
  reranker_model: string;
  reranker_top_k: number;
  reranker_threshold: number;
  
  // 向量数据库配置
  vector_db_type: string;
  vector_db_dimension: number;
  vector_db_index_type: string;
  vector_db_host: string | null;
  vector_db_port: number | null;
  vector_db_collection_name: string | null;
  
  // 智能切分配置
  intelligent_splitting_enabled?: boolean;
  splitting_strategy?: string;
  intelligent_splitting_sensitivity?: number;
  min_chunk_size?: number;
  max_chunk_size?: number;
  special_rules_enabled?: boolean;
}

// 可用的嵌入模型列表
const EMBEDDING_MODELS = [
  { name: 'BAAI/bge-small-zh-v1.5', type: 'bge', dimension: 512, desc: '轻量级中文模型，速度快' },
  { name: 'BAAI/bge-base-zh-v1.5', type: 'bge', dimension: 768, desc: '平衡型中文模型（推荐）' },
  { name: 'BAAI/bge-large-zh-v1.5', type: 'bge', dimension: 1024, desc: '高精度中文模型' },
  { name: 'text2vec-base-chinese', type: 'sentence-transformers', dimension: 768, desc: '通用中文文本模型' },
  { name: 'moka-ai/m3e-base', type: 'bge', dimension: 768, desc: 'M3E模型，性能优异' },
  { name: 'openai/text-embedding-3-small', type: 'openai', dimension: 1536, desc: 'OpenAI官方模型' }
];

// 可用的重排序模型列表
const RERANKER_MODELS = [
  { name: 'BAAI/bge-reranker-base', type: 'cross_encoder', desc: '中文重排序模型（推荐）' },
  { name: 'BAAI/bge-reranker-large', type: 'cross_encoder', desc: '高精度中文重排序模型' },
  { name: 'cross-encoder/ms-marco-multilingual', type: 'cross_encoder', desc: '多语言重排序模型' },
  { name: 'BAAI/bge-reranker-v2-m3', type: 'cross_encoder', desc: 'M3E重排序模型' }
];

const Settings: React.FC = () => {
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [activeTab, setActiveTab] = useState<'embedding' | 'reranker' | 'vector_db' | 'content_organization'>('embedding');

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      console.log('开始加载系统设置...');
      const data = await settingsApi.get();
      console.log('系统设置加载成功:', data);
      setSettings(data);
    } catch (error) {
      console.error('加载设置失败:', error);
      showMessage('error', '加载设置失败: ' + (error instanceof Error ? error.message : '未知错误'));
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async () => {
    if (!settings) return;
    
    try {
      setSaving(true);
      await settingsApi.update(settings);
      showMessage('success', '设置保存成功！');
    } catch (error) {
      showMessage('error', '保存设置失败: ' + (error instanceof Error ? error.message : '未知错误'));
    } finally {
      setSaving(false);
    }
  };

  const resetSettings = async () => {
    if (!confirm('确定要重置所有设置为默认值吗？此操作不可恢复。')) {
      return;
    }
    
    try {
      setLoading(true);
      await settingsApi.reset();
      await loadSettings();
      showMessage('success', '设置已重置为默认值');
    } catch (error) {
      showMessage('error', '重置设置失败: ' + (error instanceof Error ? error.message : '未知错误'));
    } finally {
      setLoading(false);
    }
  };

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 3000);
  };

  if (loading) {
    return (
      <div className="container">
        <div className="page-header">
          <h1 className="page-title">
            <i className="fas fa-cog"></i> 系统设置
          </h1>
        </div>
        <div className="loading-spinner">加载中...</div>
      </div>
    );
  }

  if (!settings) {
    return (
      <div className="container">
        <div className="page-header">
          <h1 className="page-title">
            <i className="fas fa-cog"></i> 系统设置
          </h1>
        </div>
        <div className="error-message">无法加载设置</div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-cog"></i> 系统设置
        </h1>
        <div className="header-actions">
          <button className="btn btn-outline-danger" onClick={resetSettings}>
            <i className="fas fa-undo"></i> 重置默认
          </button>
          <button className="btn btn-primary" onClick={saveSettings} disabled={saving}>
            <i className="fas fa-save"></i> {saving ? '保存中...' : '保存设置'}
          </button>
        </div>
      </div>

      {message && (
        <div className={`alert alert-${message.type}`}>
          <i className={`fas fa-${message.type === 'success' ? 'check-circle' : 'exclamation-circle'}`}></i>
          {message.text}
        </div>
      )}

      <div className="settings-container">
        {/* 标签页导航 */}
        <div className="settings-tabs">
          <button
            className={`tab-btn ${activeTab === 'embedding' ? 'active' : ''}`}
            onClick={() => setActiveTab('embedding')}
          >
            <i className="fas fa-vector-square"></i> 向量索引模型
          </button>
          <button
            className={`tab-btn ${activeTab === 'reranker' ? 'active' : ''}`}
            onClick={() => setActiveTab('reranker')}
          >
            <i className="fas fa-sort-amount-down"></i> 重排序配置
          </button>
          <button
            className={`tab-btn ${activeTab === 'vector_db' ? 'active' : ''}`}
            onClick={() => setActiveTab('vector_db')}
          >
            <i className="fas fa-database"></i> 向量数据库
          </button>
          <button
            className={`tab-btn ${activeTab === 'content_organization' ? 'active' : ''}`}
            onClick={() => setActiveTab('content_organization')}
          >
            <i className="fas fa-align-left"></i> 内容组织
          </button>
        </div>

        {/* 向量索引模型设置 */}
        {activeTab === 'embedding' && (
          <div className="settings-panel">
            <div className="panel-header">
              <h3>向量索引模型配置</h3>
              <p>配置用于文本向量化的嵌入模型</p>
            </div>
            <div className="panel-body">
              <div className="form-section">
                <h4>模型配置</h4>
                <div className="form-item">
                  <label className="form-label">模型类型</label>
                  <select
                    className="form-select"
                    value={settings.embedding_model_type}
                    onChange={(e) => setSettings({ ...settings, embedding_model_type: e.target.value })}
                  >
                    <option value="bge">BGE (Beijing Academy of Artificial Intelligence)</option>
                    <option value="sentence-transformers">Sentence Transformers</option>
                    <option value="openai">OpenAI Embeddings</option>
                  </select>
                </div>
                <div className="form-item">
                  <label className="form-label">模型名称</label>
                  <select
                    className="form-select"
                    value={settings.embedding_model_name}
                    onChange={(e) => {
                      setSettings({ ...settings, embedding_model_name: e.target.value });
                    }}
                  >
                    {EMBEDDING_MODELS.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name} ({model.type}) - {model.desc}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="form-item">
                  <label className="form-label">运行设备</label>
                  <select
                    className="form-select"
                    value={settings.embedding_device}
                    onChange={(e) => setSettings({ ...settings, embedding_device: e.target.value })}
                  >
                    <option value="cpu">CPU</option>
                    <option value="cuda">CUDA (GPU)</option>
                  </select>
                </div>
                <div className="form-item">
                  <label className="form-label">批处理大小</label>
                  <input
                    type="number"
                    className="form-input"
                    value={settings.embedding_batch_size}
                    onChange={(e) => setSettings({ ...settings, embedding_batch_size: parseInt(e.target.value) })}
                    min="1"
                    max="128"
                  />
                  <small className="form-hint">批量处理的文本数量，影响向量生成速度</small>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 重排序配置 */}
        {activeTab === 'reranker' && (
          <div className="settings-panel">
            <div className="panel-header">
              <h3>重排序配置</h3>
              <p>配置检索结果的重排序策略</p>
            </div>
            <div className="panel-body">
              <div className="form-section">
                <h4>基础设置</h4>
                <div className="form-item">
                  <label className="form-label">启用重排序</label>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={settings.enable_rerank}
                      onChange={(e) => setSettings({ ...settings, enable_rerank: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                  <small className="form-hint">启用后会对检索结果进行重新排序，提升检索质量</small>
                </div>
              </div>

              {settings.enable_rerank && (
                <div className="form-section">
                  <h4>重排序器配置</h4>
                  <div className="form-item">
                    <label className="form-label">重排序器类型</label>
                    <select
                      className="form-select"
                      value={settings.reranker_type}
                      onChange={(e) => setSettings({ ...settings, reranker_type: e.target.value })}
                    >
                      <option value="cross_encoder">Cross Encoder (推荐)</option>
                      <option value="colbert">ColBERT</option>
                      <option value="mmr">MMR (最大边际相关性)</option>
                    </select>
                  </div>
                  <div className="form-item">
                                    <label className="form-label">模型名称</label>
                                    <select
                                      className="form-select"
                                      value={settings.reranker_model}
                                      onChange={(e) => setSettings({ ...settings, reranker_model: e.target.value })}
                                    >
                                      {RERANKER_MODELS.map((model) => (
                                        <option key={model.name} value={model.name}>
                                          {model.name} - {model.desc}
                                        </option>
                                      ))}
                                    </select>
                                  </div>                  <div className="form-item">
                    <label className="form-label">保留数量 (Top-K)</label>
                    <input
                      type="number"
                      className="form-input"
                      value={settings.reranker_top_k}
                      onChange={(e) => setSettings({ ...settings, reranker_top_k: parseInt(e.target.value) })}
                      min="1"
                      max="100"
                    />
                  </div>
                  <div className="form-item">
                    <label className="form-label">分数阈值</label>
                    <input
                      type="number"
                      className="form-input"
                      value={settings.reranker_threshold}
                      onChange={(e) => setSettings({ ...settings, reranker_threshold: parseFloat(e.target.value) })}
                      min="0"
                      max="1"
                      step="0.05"
                    />
                    <small className="form-hint">低于此阈值的结果将被过滤</small>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 向量数据库配置 */}
        {activeTab === 'vector_db' && (
          <div className="settings-panel">
            <div className="panel-header">
              <h3>向量数据库配置</h3>
              <p>配置向量数据库连接参数</p>
            </div>
            <div className="panel-body">
              <div className="form-section">
                <h4>数据库类型</h4>
                <div className="form-item">
                  <label className="form-label">数据库类型</label>
                  <select
                    className="form-select"
                    value={settings.vector_db_type}
                    onChange={(e) => setSettings({ ...settings, vector_db_type: e.target.value })}
                  >
                    <option value="faiss">FAISS (本地文件)</option>
                    <option value="milvus">Milvus (分布式)</option>
                    <option value="qdrant">Qdrant (分布式)</option>
                  </select>
                </div>
                <div className="form-item">
                  <label className="form-label">向量维度</label>
                  <input
                    type="number"
                    className="form-input"
                    value={settings.vector_db_dimension}
                    onChange={(e) => setSettings({ ...settings, vector_db_dimension: parseInt(e.target.value) })}
                    min="1"
                    max="4096"
                  />
                  <small className="form-hint">必须与嵌入模型的输出维度一致</small>
                </div>
                <div className="form-item">
                  <label className="form-label">索引类型 (FAISS)</label>
                  <select
                    className="form-select"
                    value={settings.vector_db_index_type}
                    onChange={(e) => setSettings({ ...settings, vector_db_index_type: e.target.value })}
                  >
                    <option value="HNSW">HNSW (高性能)</option>
                    <option value="IVF">IVF (倒排文件)</option>
                    <option value="PQ">PQ (乘积量化)</option>
                    <option value="Flat">Flat (精确搜索)</option>
                  </select>
                </div>
              </div>

              {(settings.vector_db_type === 'milvus' || settings.vector_db_type === 'qdrant') && (
                <div className="form-section">
                  <h4>连接配置</h4>
                  <div className="form-item">
                    <label className="form-label">主机地址</label>
                    <input
                      type="text"
                      className="form-input"
                      value={settings.vector_db_host || ''}
                      onChange={(e) => setSettings({ ...settings, vector_db_host: e.target.value || null })}
                      placeholder="localhost"
                    />
                  </div>
                  <div className="form-item">
                    <label className="form-label">端口</label>
                    <input
                      type="number"
                      className="form-input"
                      value={settings.vector_db_port || ''}
                      onChange={(e) => setSettings({ ...settings, vector_db_port: parseInt(e.target.value) || null })}
                      placeholder="19530"
                      min="1"
                      max="65535"
                    />
                  </div>
                  {settings.vector_db_type === 'milvus' && (
                    <div className="form-item">
                      <label className="form-label">集合名称</label>
                      <input
                        type="text"
                        className="form-input"
                        value={settings.vector_db_collection_name || ''}
                        onChange={(e) => setSettings({ ...settings, vector_db_collection_name: e.target.value || null })}
                        placeholder="rag_vectors"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* 内容组织配置 */}
        {activeTab === 'content_organization' && (
          <div className="settings-panel">
            <div className="panel-header">
              <h3>内容组织配置</h3>
              <p>配置内容切分策略和智能切分参数</p>
            </div>
            <div className="panel-body">
              <div className="form-section">
                <h4>智能切分设置</h4>
                <div className="form-item">
                  <label className="form-label">启用智能切分</label>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={settings.intelligent_splitting_enabled}
                      onChange={(e) => setSettings({ ...settings, intelligent_splitting_enabled: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                  <small className="form-hint">启用后会根据文件类型自动选择切分策略</small>
                </div>
                
                <div className="form-item">
                  <label className="form-label">切分策略</label>
                  <select
                    className="form-select"
                    value={settings.splitting_strategy}
                    onChange={(e) => setSettings({ ...settings, splitting_strategy: e.target.value })}
                  >
                    <option value="intelligent">智能切分 (推荐)</option>
                    <option value="size_based">基于大小</option>
                    <option value="section_based">基于章节</option>
                    <option value="page_based">基于页面</option>
                  </select>
                  <small className="form-hint">选择内容切分的策略</small>
                </div>
              </div>

              {settings.intelligent_splitting_enabled && (
                <div className="form-section">
                  <h4>智能切分参数</h4>
                  <div className="form-item">
                    <label className="form-label">敏感度</label>
                    <input
                      type="number"
                      className="form-input"
                      value={settings.intelligent_splitting_sensitivity}
                      onChange={(e) => setSettings({ ...settings, intelligent_splitting_sensitivity: parseFloat(e.target.value) })}
                      min="0.1"
                      max="1.0"
                      step="0.1"
                    />
                    <small className="form-hint">文件类型识别的敏感度，值越高识别越严格</small>
                  </div>
                  
                  <div className="form-item">
                    <label className="form-label">最小切分单元大小</label>
                    <input
                      type="number"
                      className="form-input"
                      value={settings.min_chunk_size}
                      onChange={(e) => setSettings({ ...settings, min_chunk_size: parseInt(e.target.value) })}
                      min="50"
                      max="500"
                      step="50"
                    />
                    <small className="form-hint">每个切分单元的最小字符数</small>
                  </div>
                  
                  <div className="form-item">
                    <label className="form-label">最大切分单元大小</label>
                    <input
                      type="number"
                      className="form-input"
                      value={settings.max_chunk_size}
                      onChange={(e) => setSettings({ ...settings, max_chunk_size: parseInt(e.target.value) })}
                      min="200"
                      max="2000"
                      step="100"
                    />
                    <small className="form-hint">每个切分单元的最大字符数</small>
                  </div>
                  
                  <div className="form-item">
                    <label className="form-label">启用特殊规则</label>
                    <label className="toggle-switch">
                      <input
                        type="checkbox"
                        checked={settings.special_rules_enabled}
                        onChange={(e) => setSettings({ ...settings, special_rules_enabled: e.target.checked })}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                    <small className="form-hint">启用针对特定文件类型的特殊切分规则</small>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Settings;
