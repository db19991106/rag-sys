import React, { useState } from 'react';
import { useAppData } from '../contexts/AppDataContext';
import './History.css';

const History: React.FC = () => {
  const { operationLogs, retrievalHistory } = useAppData();
  const [activeTab, setActiveTab] = useState<'operations' | 'retrievals'>('operations');

  const handleExportLog = () => {
    const data = JSON.stringify(operationLogs, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `operation-logs-${new Date().getTime()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClearLogs = () => {
    if (confirm('确定要清空所有操作日志吗?')) {
      alert('清空日志功能开发中...');
    }
  };

  const modules = Array.from(new Set(operationLogs.map(log => log.module)));

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-history"></i> 操作历史记录
          <small>系统操作审计与追溯</small>
        </h1>
        <div className="btn-group">
          <button className="btn btn-primary" onClick={handleExportLog}>
            <i className="fas fa-download"></i> 导出日志
          </button>
          <button className="btn btn-danger" onClick={handleClearLogs}>
            <i className="fas fa-trash"></i> 清空日志
          </button>
        </div>
      </div>

      <div className="history-container">
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'operations' ? 'active' : ''}`}
            onClick={() => setActiveTab('operations')}
          >
            <i className="fas fa-list-alt"></i> 操作日志
          </button>
          <button
            className={`tab ${activeTab === 'retrievals' ? 'active' : ''}`}
            onClick={() => setActiveTab('retrievals')}
          >
            <i className="fas fa-search"></i> 检索历史
          </button>
        </div>

        {activeTab === 'operations' && (
          <div className="tab-content">
            <div className="filter-bar">
              <span className="filter-label">按模块筛选:</span>
              <div className="filter-buttons">
                <button
                  className="filter-btn active"
                  onClick={() => {}}
                >
                  全部
                </button>
                {modules.map(module => (
                  <button
                    key={module}
                    className="filter-btn"
                    onClick={() => {}}
                  >
                    {module}
                  </button>
                ))}
              </div>
            </div>

            <div className="card">
              <table className="table">
                <thead>
                  <tr>
                    <th style={{ width: '15%' }}>时间</th>
                    <th style={{ width: '15%' }}>模块</th>
                    <th style={{ width: '20%' }}>操作</th>
                    <th style={{ width: '20%' }}>用户</th>
                    <th style={{ width: '30%' }}>详情</th>
                  </tr>
                </thead>
                <tbody>
                  {operationLogs.length === 0 ? (
                    <tr>
                      <td colSpan={5}>
                        <div className="empty-box">
                          <div className="empty-icon">
                            <i className="fas fa-history"></i>
                          </div>
                          <p>暂无操作日志</p>
                        </div>
                      </td>
                    </tr>
                  ) : (
                    operationLogs.map(log => (
                      <tr key={log.id}>
                        <td className="log-time">{log.timestamp}</td>
                        <td className="log-module">
                          <span className="module-badge">{log.module}</span>
                        </td>
                        <td className="log-operation">{log.operation}</td>
                        <td className="log-user">{log.user}</td>
                        <td className="log-details">
                          <code className="details-code">
                            {JSON.stringify(log.details, null, 2)}
                          </code>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'retrievals' && (
          <div className="tab-content">
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">检索历史记录</h3>
                <span className="tip-text">共 {retrievalHistory.length} 条检索记录</span>
              </div>
              <table className="table">
                <thead>
                  <tr>
                    <th style={{ width: '15%' }}>时间</th>
                    <th style={{ width: '30%' }}>查询内容</th>
                    <th style={{ width: '15%' }}>Top-K</th>
                    <th style={{ width: '15%' }}>相似度阈值</th>
                    <th style={{ width: '10%' }}>结果数</th>
                    <th style={{ width: '15%' }}>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {retrievalHistory.length === 0 ? (
                    <tr>
                      <td colSpan={6}>
                        <div className="empty-box">
                          <div className="empty-icon">
                            <i className="fas fa-search"></i>
                          </div>
                          <p>暂无检索历史记录</p>
                        </div>
                      </td>
                    </tr>
                  ) : (
                    retrievalHistory.map(history => (
                      <tr key={history.id}>
                        <td className="log-time">{history.timestamp}</td>
                        <td className="query-text">{history.query}</td>
                        <td className="config-value">{history.config.topK}</td>
                        <td className="config-value">{history.config.simThreshold}</td>
                        <td className="result-count">{history.results.length}</td>
                        <td className="operate">
                          <button className="op-btn">
                            <i className="fas fa-redo"></i> 重新检索
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default History;