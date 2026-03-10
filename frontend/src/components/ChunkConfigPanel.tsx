import React from 'react';
import type { ChunkConfig } from '../types';

interface ChunkConfigPanelProps {
  config: ChunkConfig;
  similarityThreshold: number;
  onConfigChange: (config: ChunkConfig) => void;
  onThresholdChange: (threshold: number) => void;
  onChunk: () => void;
  onReset: () => void;
  onShowOriginal: () => void;
}

export const ChunkConfigPanel: React.FC<ChunkConfigPanelProps> = ({
  config,
  similarityThreshold,
  onConfigChange,
  onThresholdChange,
  onChunk,
  onReset,
  onShowOriginal,
}) => {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">
          <i className="fas fa-cog"></i> 切分策略配置
        </h3>
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
              onChange={e => onConfigChange({ ...config, type: e.target.value as any })}
            >
              <optgroup label="基础切分">
                <option value="naive">📝 分隔符切分</option>
              </optgroup>
              <optgroup label="智能切分">
                <option value="layered">🧠 分层智能切分</option>
                <option value="layered_llm">🤖 分层LLM切分</option>
                <option value="hybrid">🔀 混合切分-标题切分</option>
              </optgroup>
              <optgroup label="财务制度">
                <option value="intelligent">💰 财务报销制度切分</option>
              </optgroup>
              <optgroup label="其他文档类型">
                <option value="product">📄 产品文档切分</option>
                <option value="technical">⚙️ 技术规范切分</option>
                <option value="compliance">📋 合规文件切分</option>
                <option value="hr">👥 HR文档切分</option>
                <option value="project">📊 项目管理切分</option>
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
              onChange={e => onConfigChange({ ...config, chunkTokenSize: parseInt(e.target.value) })}
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
              onChange={e => onConfigChange({ ...config, overlappedPercent: parseFloat(e.target.value) / 100 })}
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
              onChange={e => onThresholdChange(parseFloat(e.target.value))}
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
            <button className="btn btn-primary btn-lg" onClick={onChunk}>
              <i className="fas fa-play"></i> 执行切分
            </button>
            <button className="btn btn-outline" onClick={onReset}>
              <i className="fas fa-undo"></i> 重置
            </button>
          </div>
          <div className="action-bar-right">
            <button className="btn btn-icon-only" onClick={onShowOriginal} title="查看原文档">
              <i className="fas fa-file-alt"></i>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChunkConfigPanel;
