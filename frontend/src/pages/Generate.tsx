import React from 'react';
import { useAppData } from '../contexts/AppDataContext';
import './Generate.css';

const Generate: React.FC = () => {
  const { chunks } = useAppData();

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-wand-magic-sparkles"></i> 生成上下文展示
          <small>Generation Context Viewer</small>
        </h1>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-list"></i> 可用上下文片段
            {chunks.length > 0 && (
              <span className="badge badge-primary">{chunks.length}</span>
            )}
          </h3>
        </div>
        <div className="card-body">
          {chunks.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">
                <i className="fas fa-layer-group"></i>
              </div>
              <h4>暂无上下文片段</h4>
              <p>请先在文档切分页面切分文档并生成片段</p>
            </div>
          ) : (
            <div className="chunks-list">
              {chunks.map((chunk) => (
                <div key={chunk.id} className="chunk-item">
                  <div className="chunk-header">
                    <span className="chunk-number">#{chunk.num}</span>
                    <span className="chunk-length">{chunk.length} 字符</span>
                  </div>
                  <div className="chunk-content">
                    {chunk.content}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="info-box">
        <div className="info-box-icon">
          <i className="fas fa-info-circle"></i>
        </div>
        <div className="info-box-content">
          <h4>提示</h4>
          <p>上下文片段可用于 RAG 生成。实际生成功能请在对话页面使用。</p>
        </div>
      </div>
    </div>
  );
};

export default Generate;