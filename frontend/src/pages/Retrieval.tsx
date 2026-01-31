import React, { useState, useEffect, useRef } from 'react';
import { useAppData } from '../contexts/AppDataContext';
import { retrievalApi } from '../services/api';
import { extractKeywords, highlightContent, getSimilarityScoreClass } from '../utils/format';
import type { RetrievalResult } from '../types';
import * as echarts from 'echarts';
import './Retrieval.css';

const Retrieval: React.FC = () => {
  const { chunks } = useAppData();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const [topK, setTopK] = useState(5);
  const [simThreshold, setSimThreshold] = useState(0.7);
  const [query, setQuery] = useState('RAG的核心流程是什么?');
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [processSteps, setProcessSteps] = useState<React.ReactNode[]>([]);

  // 检索策略相关状态
  const [showAlgoConfig, setShowAlgoConfig] = useState(false);
  const [showRerankConfig, setShowRerankConfig] = useState(false);
  const [showFilterConfig, setShowFilterConfig] = useState(false);
  const [showPerfMonitor, setShowPerfMonitor] = useState(false);
  const [showABTest, setShowABTest] = useState(false);

  const [algoConfig, setAlgoConfig] = useState({
    type: 'cosine',
    desc: '余弦相似度',
    speed: '快',
    accuracy: '高',
    use: '文本相似度'
  });

  const [rerankConfig, setRerankConfig] = useState({
    enabled: true,
    model: 'bge-reranker-base',
    topK: 10,
    threshold: 0.5
  });

  const [filterConfig, setFilterConfig] = useState({
    simRange: [0.7, 1.0],
    dateRange: 'all',
    categories: [] as string[],
    tags: [] as string[]
  });

  const [perfMetrics, setPerfMetrics] = useState<any>(null);
  const [abTestConfig, setABTestConfig] = useState<any>(null);
  const [abTestResults, setAbTestResults] = useState<any>(null);

  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        chartInstance.current.dispose();
      }
    };
  }, []);

  // 检索策略处理函数
  const handleAlgoConfig = (algo: any) => {
    setAlgoConfig(algo);
    setShowAlgoConfig(false);
    alert(`已切换算法: ${algo.desc}`);
  };

  const handleRerankConfig = (config: any) => {
    setRerankConfig(config);
    setShowRerankConfig(false);
    alert('重排序配置已更新');
  };

  const handleFilterConfig = (config: any) => {
    setFilterConfig(config);
    setShowFilterConfig(false);
    alert('过滤条件已应用');
  };

  const handlePerfAnalysis = () => {
    const metrics = {
      avgLatency: 150,
      p99Latency: 320,
      throughput: 850,
      successRate: 98.5,
      errorRate: 1.5,
      cacheHitRate: 45.2,
      avgSimilarity: 0.85,
      resultRelevance: 0.92
    };
    setPerfMetrics(metrics);
    setShowPerfMonitor(true);
  };

  const handleABTest = (config: any) => {
    setABTestConfig(config);
    setAbTestResults({
      testDuration: '24h',
      trafficA: '50%',
      trafficB: '50%',
      avgLatencyA: 145,
      avgLatencyB: 158,
      relevanceA: 0.91,
      relevanceB: 0.89,
      successRateA: 99.1,
      successRateB: 97.8,
      recommendation: '策略A性能更优'
    });
    setShowABTest(true);
  };

  const executeRetrieval = async () => {
    if (!query.trim()) {
      alert('请输入查询问题!');
      return;
    }

    setResults([]);
    setProcessSteps([
      <div className="process-step">
        <div className="step-icon">1</div>
        <div><b>检索初始化</b>:获取检索参数(Top-K={topK},相似度阈值={simThreshold})</div>
      </div>,
      <div className="process-step">
        <div className="step-icon loading"></div>
        <div>正在将查询问题向量化...</div>
      </div>
    ]);

    try {
      // 调用后端 API 执行检索
      const response = await retrievalApi.search(query, {
        top_k: topK,
        similarity_threshold: simThreshold,
        algorithm: algoConfig.type
      });

      const keywords = extractKeywords(query);

      // 将后端返回的数据转换为前端格式
      const mockResults: RetrievalResult[] = response.results.map((result, index) => ({
        id: result.chunk_id,
        num: result.chunk_num,
        content: result.content,
        sim: result.similarity,
        matchKeywords: keywords.filter(k => result.content.includes(k)),
        vecStatus: 'success'
      }));

      setResults(mockResults);

      setProcessSteps([
        <div className="process-step">
          <div className="step-icon step-success">2</div>
          <div><b>查询向量化完成</b>:基于嵌入模型将查询转为向量</div>
        </div>,
        <div className="process-step">
          <div className="step-icon step-success">3</div>
          <div><b>相似度计算完成</b>:完成{algoConfig.desc}计算,提取匹配关键词「{keywords.join('、')}」</div>
        </div>,
        <div className="process-step">
          <div className="step-icon step-success">4</div>
          <div><b>结果筛选完成</b>:筛选出{mockResults.length}个符合条件的片段</div>
        </div>,
        <div className="process-step">
          <div className="step-icon step-success">5</div>
          <div className="process-tip">检索完成!共耗时{response.latency_ms.toFixed(0)}ms,展示Top-{mockResults.length}匹配结果</div>
        </div>
      ]);

      renderSimChart(mockResults);
    } catch (error) {
      console.error('检索失败:', error);
      setProcessSteps([
        ...setProcessSteps,
        <div className="process-step">
          <div className="step-icon step-error">✗</div>
          <div><b>检索失败</b>: {error instanceof Error ? error.message : '未知错误'}</div>
        </div>
      ]);
      alert(`检索失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  };

  const renderSimChart = (data: RetrievalResult[]) => {
    if (!chartRef.current) return;

    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    chartInstance.current = echarts.init(chartRef.current);

    const xData = data.map(item => `片段${item.num}`);
    const yData = data.map(item => item.sim);
    const colorData = yData.map(sim => {
      if (sim >= 0.8) return '#00B42A';
      else if (sim >= 0.7) return '#165DFF';
      else if (sim >= 0.6) return '#FF7D00';
      else return '#86909C';
    });

    const option: echarts.EChartsOption = {
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const data = params[0];
          return `片段${data.axisValueLabel}: 相似度 ${data.value.toFixed(4)}`;
        }
      },
      xAxis: {
        type: 'category',
        data: xData,
        axisLabel: { interval: 0 }
      },
      yAxis: {
        type: 'value',
        min: simThreshold - 0.1,
        max: 1.0,
        name: '相似度'
      },
      grid: { top: 30, bottom: 40, left: 60, right: 20 },
      series: [{
        type: 'bar',
        data: yData.map((val, i) => ({ value: val, itemStyle: { color: colorData[i] } })),
        barWidth: '60%',
        label: {
          show: true,
          position: 'top',
          formatter: '{c}',
          fontSize: 12
        }
      }]
    };

    chartInstance.current.setOption(option);
  };

  const clearResults = () => {
    setResults([]);
    setProcessSteps([
      <div className="process-tip">🔍 检索过程将在点击「执行检索」后实时展示...</div>
    ]);
    if (chartInstance.current) {
      chartInstance.current.dispose();
    }
  };

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-search"></i> 查询与检索过程可视化
          <small>RAG Retrieval 匹配 & 结果展示</small>
        </h1>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-cog"></i> 检索配置与查询输入
          </h3>
        </div>
        <div className="card-body">
          <div className="retrieval-config">
            <div className="form-item">
              <label className="form-label">Top-K 数量</label>
              <input
                type="number"
                className="form-input"
                value={topK}
                onChange={e => setTopK(parseInt(e.target.value))}
                min="1"
                max="20"
              />
            </div>
            <div className="form-item">
              <label className="form-label">相似度阈值</label>
              <input
                type="number"
                className="form-input"
                value={simThreshold}
                onChange={e => setSimThreshold(parseFloat(e.target.value))}
                step="0.05"
                min="0"
                max="1"
              />
            </div>
          </div>

          <label className="form-label">用户查询问题</label>
          <textarea
            className="form-textarea"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="请输入您的查询问题"
            rows={3}
          />

          <div className="btn-group">
            <button className="btn btn-primary" onClick={executeRetrieval}>
              <i className="fas fa-search"></i> 执行检索
            </button>
            <button className="btn btn-default" onClick={clearResults}>
              <i className="fas fa-trash"></i> 清空结果
            </button>
          </div>

          {/* 检索策略按钮组 */}
          <div className="strategy-buttons">
            <button className="btn btn-sm btn-outline-primary" onClick={() => setShowAlgoConfig(true)}>
              <i className="fas fa-calculator"></i> 算法配置
            </button>
            <button className="btn btn-sm btn-outline-info" onClick={() => setShowRerankConfig(true)}>
              <i className="fas fa-sort-amount-down"></i> 重排序设置
            </button>
            <button className="btn btn-sm btn-outline-success" onClick={() => setShowFilterConfig(true)}>
              <i className="fas fa-filter"></i> 结果过滤
            </button>
            <button className="btn btn-sm btn-outline-warning" onClick={handlePerfAnalysis}>
              <i className="fas fa-tachometer-alt"></i> 性能监控
            </button>
            <button className="btn btn-sm btn-outline-secondary" onClick={() => setShowABTest(true)}>
              <i className="fas fa-vial"></i> A/B测试
            </button>
          </div>

          <div className="retrieval-process">{processSteps}</div>
        </div>
      </div>

      {results.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">
              <i className="fas fa-clipboard-list"></i> Top-K 检索结果
            </h3>
            <span className="tip-text">共{results.length}个匹配结果 | 余弦相似度 ≥ {simThreshold}</span>
          </div>
          <div className="card-body">
            <div ref={chartRef} style={{ width: '100%', height: '200px', marginBottom: '30px' }}></div>

            <table className="table">
              <thead>
                <tr>
                  <th style={{ width: '10%' }}>片段编号</th>
                  <th style={{ width: '45%' }}>匹配片段内容</th>
                  <th style={{ width: '15%' }}>相似度分数</th>
                  <th style={{ width: '15%' }}>向量状态</th>
                  <th style={{ width: '15%' }}>操作</th>
                </tr>
              </thead>
              <tbody>
                {results.map(result => (
                  <tr key={result.id}>
                    <td><span className="chunk-num">{result.num}</span></td>
                    <td className="chunk-content">
                      <span dangerouslySetInnerHTML={{
                        __html: highlightContent(result.content, result.matchKeywords)
                      }} />
                    </td>
                    <td>
                      <span className={`similarity-score ${getSimilarityScoreClass(result.sim)}`}>
                        {result.sim.toFixed(4)}
                      </span>
                    </td>
                    <td>
                      <span className="similarity-score score-high">
                        <i className="fas fa-check-circle"></i> 已索引
                      </span>
                    </td>
                    <td>
                      <button className="btn btn-sm btn-default" onClick={() => alert(`查看片段${result.num}详情`)}>
                        <i className="fas fa-eye"></i> 查看详情
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 算法配置模态框 */}
      {showAlgoConfig && (
        <div className="modal-mask" onClick={() => setShowAlgoConfig(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-calculator"></i> 相似度算法配置
              </span>
              <button className="modal-close" onClick={() => setShowAlgoConfig(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="algo-grid">
                {[
                  { type: 'cosine', desc: '余弦相似度', formula: 'cos(θ) = (A·B) / (||A|| × ||B||)', speed: '快', accuracy: '高', use: '文本相似度', icon: '📐' },
                  { type: 'euclidean', desc: '欧氏距离', formula: 'd = √∑(Ai-Bi)²', speed: '极快', accuracy: '中', use: '空间距离', icon: '📏' },
                  { type: 'dot', desc: '点积', formula: 'A·B = ∑Ai×Bi', speed: '极快', accuracy: '中', use: '高维向量', icon: '•' },
                  { type: 'manhattan', desc: '曼哈顿距离', formula: 'd = ∑|Ai-Bi|', speed: '快', accuracy: '中低', use: '网格数据', icon: '🏙️' }
                ].map((algo, idx) => (
                  <div
                    key={idx}
                    className={`algo-option ${algoConfig.type === algo.type ? 'selected' : ''}`}
                    onClick={() => handleAlgoConfig(algo)}
                  >
                    <div className="algo-icon">{algo.icon}</div>
                    <div className="algo-info">
                      <div className="algo-name">{algo.desc}</div>
                      <div className="algo-formula">{algo.formula}</div>
                      <div className="algo-meta">
                        <span className="meta-item">速度: {algo.speed}</span>
                        <span className="meta-item">精度: {algo.accuracy}</span>
                        <span className="meta-item">适用: {algo.use}</span>
                      </div>
                    </div>
                    {algoConfig.type === algo.type && (
                      <div className="algo-check">
                        <i className="fas fa-check-circle"></i>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowAlgoConfig(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 重排序配置模态框 */}
      {showRerankConfig && (
        <div className="modal-mask" onClick={() => setShowRerankConfig(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-sort-amount-down"></i> 重排序配置
              </span>
              <button className="modal-close" onClick={() => setShowRerankConfig(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="rerank-toggle">
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={rerankConfig.enabled}
                    onChange={e => handleRerankConfig({ ...rerankConfig, enabled: e.target.checked })}
                  />
                  <span className="toggle-text">启用重排序</span>
                </label>
              </div>

              {rerankConfig.enabled && (
                <div className="rerank-options">
                  <div className="form-item">
                    <label className="form-label">重排序模型</label>
                    <select
                      className="form-select"
                      value={rerankConfig.model}
                      onChange={e => handleRerankConfig({ ...rerankConfig, model: e.target.value })}
                    >
                      <option value="bge-reranker-base">BGE Reranker Base</option>
                      <option value="bge-reranker-large">BGE Reranker Large</option>
                      <option value="cohere-rerank">Cohere Rerank</option>
                      <option value="cross-encoder">Cross-Encoder</option>
                    </select>
                  </div>

                  <div className="form-item">
                    <label className="form-label">重排序Top-K</label>
                    <input
                      type="number"
                      className="form-input"
                      value={rerankConfig.topK}
                      onChange={e => handleRerankConfig({ ...rerankConfig, topK: parseInt(e.target.value) })}
                      min="5"
                      max="50"
                    />
                  </div>

                  <div className="form-item">
                    <label className="form-label">重排序阈值</label>
                    <input
                      type="range"
                      className="form-range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={rerankConfig.threshold}
                      onChange={e => handleRerankConfig({ ...rerankConfig, threshold: parseFloat(e.target.value) })}
                    />
                    <div className="range-value">{rerankConfig.threshold}</div>
                  </div>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-outline-primary" onClick={() => alert('正在测试重排序效果...\n\n✅ 测试完成!\n- 平均相关度提升: +15%\n- 排序准确率: 92%')}>
                <i className="fas fa-vial"></i> 测试效果
              </button>
              <button className="btn btn-default" onClick={() => setShowRerankConfig(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 结果过滤配置模态框 */}
      {showFilterConfig && (
        <div className="modal-mask" onClick={() => setShowFilterConfig(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-filter"></i> 检索结果过滤
              </span>
              <button className="modal-close" onClick={() => setShowFilterConfig(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="filter-section">
                <h4>相似度范围</h4>
                <div className="range-filter">
                  <label>最低: {filterConfig.simRange[0]}</label>
                  <input
                    type="range"
                    className="form-range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={filterConfig.simRange[0]}
                    onChange={e => setFilterConfig({ ...filterConfig, simRange: [parseFloat(e.target.value), filterConfig.simRange[1]] })}
                  />
                </div>
                <div className="range-filter">
                  <label>最高: {filterConfig.simRange[1]}</label>
                  <input
                    type="range"
                    className="form-range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={filterConfig.simRange[1]}
                    onChange={e => setFilterConfig({ ...filterConfig, simRange: [filterConfig.simRange[0], parseFloat(e.target.value)] })}
                  />
                </div>
              </div>

              <div className="filter-section">
                <h4>日期范围</h4>
                <select
                  className="form-select"
                  value={filterConfig.dateRange}
                  onChange={e => setFilterConfig({ ...filterConfig, dateRange: e.target.value })}
                >
                  <option value="all">全部时间</option>
                  <option value="today">今天</option>
                  <option value="week">最近一周</option>
                  <option value="month">最近一月</option>
                  <option value="year">最近一年</option>
                </select>
              </div>

              <div className="filter-section">
                <h4>文档分类</h4>
                <div className="checkbox-group">
                  {['技术文档', '用户手册', 'API文档', 'FAQ', '案例'].map((cat, idx) => (
                    <label key={idx} className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={filterConfig.categories.includes(cat)}
                        onChange={e => {
                          const newCats = e.target.checked
                            ? [...filterConfig.categories, cat]
                            : filterConfig.categories.filter(c => c !== cat);
                          setFilterConfig({ ...filterConfig, categories: newCats });
                        }}
                      />
                      <span>{cat}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setFilterConfig({
                simRange: [0.6, 1.0],
                dateRange: 'all',
                categories: [],
                tags: []
              })}>
                <i className="fas fa-undo"></i> 重置过滤
              </button>
              <button className="btn btn-primary" onClick={() => handleFilterConfig(filterConfig)}>
                <i className="fas fa-check"></i> 应用过滤
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 性能监控模态框 */}
      {showPerfMonitor && perfMetrics && (
        <div className="modal-mask" onClick={() => setShowPerfMonitor(false)}>
          <div className="modal-box modal-large" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-tachometer-alt"></i> 检索性能监控
              </span>
              <button className="modal-close" onClick={() => setShowPerfMonitor(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="perf-metrics-grid">
                <div className="perf-card primary">
                  <div className="perf-icon">⚡</div>
                  <div className="perf-value">{perfMetrics.avgLatency}ms</div>
                  <div className="perf-label">平均延迟</div>
                </div>
                <div className="perf-card success">
                  <div className="perf-icon">📊</div>
                  <div className="perf-value">{perfMetrics.throughput}</div>
                  <div className="perf-label">吞吐量(ops/s)</div>
                </div>
                <div className="perf-card info">
                  <div className="perf-icon">✅</div>
                  <div className="perf-value">{perfMetrics.successRate}%</div>
                  <div className="perf-label">成功率</div>
                </div>
                <div className="perf-card warning">
                  <div className="perf-icon">🎯</div>
                  <div className="perf-value">{perfMetrics.resultRelevance}</div>
                  <div className="perf-label">结果相关度</div>
                </div>
                <div className="perf-card secondary">
                  <div className="perf-icon">💾</div>
                  <div className="perf-value">{perfMetrics.cacheHitRate}%</div>
                  <div className="perf-label">缓存命中率</div>
                </div>
                <div className="perf-card error">
                  <div className="perf-icon">⚠️</div>
                  <div className="perf-value">{perfMetrics.errorRate}%</div>
                  <div className="perf-label">错误率</div>
                </div>
              </div>

              <div className="perf-details">
                <h4>性能详情</h4>
                <div className="perf-list">
                  <div className="perf-item">
                    <span>P99延迟</span>
                    <strong>{perfMetrics.p99Latency}ms</strong>
                  </div>
                  <div className="perf-item">
                    <span>平均相似度</span>
                    <strong>{perfMetrics.avgSimilarity}</strong>
                  </div>
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-outline-info" onClick={() => alert('性能报告已导出\n\n- 导出时间: ' + new Date().toLocaleString() + '\n- 导出格式: CSV\n- 文件大小: 125KB')}>
                <i className="fas fa-download"></i> 导出报告
              </button>
              <button className="btn btn-default" onClick={() => setShowPerfMonitor(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* A/B测试模态框 */}
      {showABTest && (
        <div className="modal-mask" onClick={() => setShowABTest(false)}>
          <div className="modal-box modal-large" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span>
                <i className="fas fa-vial"></i> 检索A/B测试
              </span>
              <button className="modal-close" onClick={() => setShowABTest(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              {!abTestConfig ? (
                <div className="ab-test-setup">
                  <h4>创建测试计划</h4>
                  <div className="form-item">
                    <label className="form-label">测试名称</label>
                    <input type="text" className="form-input" placeholder="输入测试名称" />
                  </div>
                  <div className="ab-strategies">
                    <div className="strategy-config">
                      <h5>策略A (对照组)</h5>
                      <select className="form-select">
                        <option>余弦相似度 + Top-5</option>
                        <option>余弦相似度 + Top-10</option>
                      </select>
                    </div>
                    <div className="strategy-config">
                      <h5>策略B (实验组)</h5>
                      <select className="form-select">
                        <option>重排序 + Top-5</option>
                        <option>重排序 + Top-10</option>
                      </select>
                    </div>
                  </div>
                  <div className="form-item">
                    <label className="form-label">流量分配</label>
                    <div className="traffic-slider">
                      <span>策略A: 50%</span>
                      <input type="range" className="form-range" min="0" max="100" value="50" />
                      <span>策略B: 50%</span>
                    </div>
                  </div>
                  <button
                    className="btn btn-primary"
                    onClick={() => handleABTest({ name: '测试1', duration: '24h' })}
                  >
                    <i className="fas fa-play"></i> 开始测试
                  </button>
                </div>
              ) : (
                <div className="ab-test-results">
                  <h4>测试结果</h4>
                  <div className="test-summary">
                    <div className="summary-item">
                      <span>测试时长</span>
                      <strong>{abTestResults.testDuration}</strong>
                    </div>
                    <div className="summary-item">
                      <span>流量分配</span>
                      <strong>A: {abTestResults.trafficA} | B: {abTestResults.trafficB}</strong>
                    </div>
                  </div>

                  <div className="result-comparison">
                    <h5>性能对比</h5>
                    <div className="comparison-grid">
                      <div className="compare-item">
                        <div className="compare-label">平均延迟</div>
                        <div className="compare-values">
                          <div className="value-a">{abTestResults.avgLatencyA}ms</div>
                          <div className="vs">VS</div>
                          <div className="value-b">{abTestResults.avgLatencyB}ms</div>
                        </div>
                        <div className="compare-winner">策略A 胜出</div>
                      </div>
                      <div className="compare-item">
                        <div className="compare-label">结果相关度</div>
                        <div className="compare-values">
                          <div className="value-a">{abTestResults.relevanceA}</div>
                          <div className="vs">VS</div>
                          <div className="value-b">{abTestResults.relevanceB}</div>
                        </div>
                        <div className="compare-winner">策略A 胜出</div>
                      </div>
                      <div className="compare-item">
                        <div className="compare-label">成功率</div>
                        <div className="compare-values">
                          <div className="value-a">{abTestResults.successRateA}%</div>
                          <div className="vs">VS</div>
                          <div className="value-b">{abTestResults.successRateB}%</div>
                        </div>
                        <div className="compare-winner">策略A 胜出</div>
                      </div>
                    </div>
                  </div>

                  <div className="recommendation">
                    <h5>测试结论</h5>
                    <div className="recommendation-box">
                      <i className="fas fa-lightbulb"></i>
                      <span>{abTestResults.recommendation}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowABTest(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Retrieval;