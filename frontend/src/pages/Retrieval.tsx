import React, { useState } from 'react';
import { retrievalApi } from '../services/api';
import { extractKeywords, highlightContent, getSimilarityScoreClass } from '../utils/format';
import type { RetrievalResult } from '../types';
import './Retrieval.css';

const Retrieval: React.FC = () => {
  const [topK, setTopK] = useState(5);
  const [simThreshold, setSimThreshold] = useState(0.7);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  const executeRetrieval = async () => {
    if (!query.trim()) {
      alert('请输入查询问题!');
      return;
    }

    setIsSearching(true);
    setResults([]);

    try {
      const response = await retrievalApi.search(query, {
        top_k: topK,
        similarity_threshold: simThreshold,
        algorithm: 'cosine'
      });

      const keywords = extractKeywords(query);

      const formattedResults: RetrievalResult[] = response.results.map((result) => ({
        id: result.chunk_id,
        num: result.chunk_num,
        content: result.content,
        sim: result.similarity,
        matchKeywords: keywords.filter(k => result.content.includes(k))
      }));

      setResults(formattedResults);
    } catch (error) {
      console.error('检索失败:', error);
      alert(`检索失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      executeRetrieval();
    }
  };

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-magnifying-glass"></i> 向量检索系统
          <small>Vector Retrieval System</small>
        </h1>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-search"></i> 检索配置
          </h3>
        </div>
        <div className="card-body">
          <div className="retrieval-config">
            <div className="config-item">
              <label className="config-label">返回结果数 (Top-K)</label>
              <input
                type="number"
                className="form-input"
                value={topK}
                onChange={e => setTopK(parseInt(e.target.value))}
                min={1}
                max={20}
              />
            </div>

            <div className="config-item">
              <label className="config-label">相似度阈值</label>
              <input
                type="number"
                className="form-input"
                value={simThreshold}
                onChange={e => setSimThreshold(parseFloat(e.target.value))}
                min={0}
                max={1}
                step={0.1}
              />
            </div>

            <div className="config-item config-item-full">
              <label className="config-label">查询问题</label>
              <input
                type="text"
                className="form-input"
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="请输入您的问题..."
              />
            </div>
          </div>

          <div className="action-bar">
            <button className="btn btn-primary btn-lg" onClick={executeRetrieval} disabled={isSearching}>
              <i className={`fas ${isSearching ? 'fa-spinner fa-spin' : 'fa-search'}`}></i>
              执行检索
            </button>
          </div>
        </div>
      </div>

      {results.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <i className="fas fa-list"></i> 检索结果
              <span className="badge badge-primary">{results.length}</span>
            </h3>
          </div>
          <div className="card-body">
            <div className="results-list">
              {results.map((result, index) => (
                <div key={result.id} className="result-item">
                  <div className="result-header">
                    <div className="result-meta">
                      <span className="result-rank">#{index + 1}</span>
                      <span className="result-chunk">片段 #{result.num}</span>
                      <span className={`similarity-score ${getSimilarityScoreClass(result.sim)}`}>
                        相似度: {(result.sim * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="result-content">
                    <div dangerouslySetInnerHTML={{ __html: highlightContent(result.content, result.matchKeywords || []) }} />
                  </div>
                  {result.matchKeywords && result.matchKeywords.length > 0 && (
                    <div className="result-keywords">
                      <span className="keywords-label">匹配关键词:</span>
                      <div className="keywords-list">
                        {result.matchKeywords.map((keyword, idx) => (
                          <span key={idx} className="keyword-tag">{keyword}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {results.length === 0 && !isSearching && query && (
        <div className="empty-state">
          <div className="empty-state-icon">
            <i className="fas fa-search-minus"></i>
          </div>
          <h4>未找到相关结果</h4>
          <p>请尝试调整相似度阈值或更换查询词</p>
        </div>
      )}
    </div>
  );
};

export default Retrieval;