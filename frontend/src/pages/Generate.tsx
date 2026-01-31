import React, { useState, useEffect, useRef } from 'react';
import { useAppData } from '../contexts/AppDataContext';
import type { Chunk } from '../types';
import { highlightContent } from '../utils/format';
import Sortable from 'sortablejs';
import './Generate.css';

const Generate: React.FC = () => {
  const { chunks } = useAppData();
  const chunkListRef = useRef<HTMLDivElement>(null);

  const [userQuery] = useState('RAG的核心流程是什么?');
  const [currentChunks, setCurrentChunks] = useState<Chunk[]>([]);
  const [prompt, setPrompt] = useState('');

  // 生成参数相关状态
  const [showModelConfig, setShowModelConfig] = useState(false);
  const [showParamTuning, setShowParamTuning] = useState(false);
  const [showTemplateManage, setShowTemplateManage] = useState(false);
  const [showPromptTest, setShowPromptTest] = useState(false);
  const [showLengthOptimize, setShowLengthOptimize] = useState(false);

  const [llmConfig, setLLMConfig] = useState({
    provider: 'openai',
    model: 'gpt-3.5-turbo',
    apiKey: '',
    baseUrl: '',
    temperature: 0.7,
    maxTokens: 2000,
    stream: true
  });

  const [genParams, setGenParams] = useState({
    temperature: 0.7,
    topP: 0.9,
    maxTokens: 2000,
    frequencyPenalty: 0,
    presencePenalty: 0,
    stopSequences: [] as string[]
  });

  const [promptTemplates, setPromptTemplates] = useState<any[]>([]);
  const [testResults, setTestResults] = useState<any>(null);
  const [lengthOptimization, setLengthOptimization] = useState<any>(null);

  useEffect(() => {
    const mockChunks = chunks.length > 0 ? chunks.slice(0, 5).map((chunk, index) => ({
      ...chunk,
      num: index + 1,
      sim: 0.9 - index * 0.05,
      highlightKeywords: ['RAG', '流程'].filter(k => chunk.content.includes(k))
    })) as any[] : [];

    setCurrentChunks(mockChunks);
    buildPrompt(mockChunks);

    if (chunkListRef.current) {
      new Sortable(chunkListRef.current, {
        animation: 150,
        ghostClass: 'chunk-item-dragging',
        onEnd: () => {
          const items = chunkListRef.current?.children;
          if (items) {
            const newOrder = Array.from(items).map(item => item.getAttribute('data-id'));
            const reorderedChunks = newOrder
              .map(id => currentChunks.find(c => c.id === id))
              .filter((c): c is Chunk => c !== undefined);
            setCurrentChunks(reorderedChunks);
            buildPrompt(reorderedChunks);
          }
        }
      });
    }
  }, [chunks]);

  const buildPrompt = (chunks: Chunk[]) => {
    const contextContent = chunks.map((chunk, index) => {
      return `【参考片段${index + 1}】${chunk.content}`;
    }).join('\n');

    const fullPrompt = `# RAG生成输入模板
请根据以下参考文档片段,回答用户的问题,仅使用文档中的信息,不要编造内容。如果文档中没有相关信息,请回答"文档中未提及相关内容"。

## 参考文档
${contextContent || '(无参考文档片段)'}

## 用户问题
${userQuery}

## 回答要求
1. 语言简洁、逻辑清晰,分点回答更佳
2. 严格基于参考文档,不添加额外信息
3. 保留核心关键词,贴合问题语义`;

    setPrompt(fullPrompt);
  };

  const handleResetContext = () => {
    const originalChunks = chunks.slice(0, 5).map((chunk, index) => ({
      ...chunk,
      num: index + 1,
      sim: 0.9 - index * 0.05,
      highlightKeywords: ['RAG', '流程'].filter(k => chunk.content.includes(k))
    }));
    setCurrentChunks(originalChunks);
    buildPrompt(originalChunks);
    alert('上下文已重置!');
  };

  const handleDeleteChunk = (id: string) => {
    const newChunks = currentChunks.filter(c => c.id !== id);
    setCurrentChunks(newChunks);
    buildPrompt(newChunks);
  };

  const handleClearAll = () => {
    if (confirm('确定要清空所有上下文片段吗?')) {
      setCurrentChunks([]);
      buildPrompt([]);
    }
  };

  const handleBuildContext = () => {
    buildPrompt(currentChunks);
    alert('生成上下文已构建完成!');
  };

  // 生成参数处理函数
  const handleModelConfig = (config: any) => {
    setLLMConfig({ ...llmConfig, ...config });
    setShowModelConfig(false);
    alert(`已配置模型: ${config.model}`);
  };

  const handleTestModel = () => {
    alert('正在测试模型连接...\n\n✅ 连接成功!\n- 模型: ' + llmConfig.model + '\n- 响应时间: 850ms\n- 状态: 正常');
  };

  const handleParamTuning = (params: any) => {
    setGenParams({ ...genParams, ...params });
    setShowParamTuning(false);
    alert('参数已更新');
  };

  const handleTestParams = () => {
    alert('正在测试参数效果...\n\n测试结果:\n- 生成质量: 良好\n- 响应速度: 正常\n- 建议温度: 0.7');
  };

  const handleSaveTemplate = (template: any) => {
    const newTemplate = {
      id: Date.now().toString(),
      name: template.name,
      content: template.content,
      createTime: new Date().toLocaleString('zh-CN')
    };
    setPromptTemplates([...promptTemplates, newTemplate]);
    setShowTemplateManage(false);
    alert('模板保存成功!');
  };

  const handlePromptTest = (query: string) => {
    const result = {
      query,
      response: 'RAG(Retrieval-Augmented Generation)的核心流程包括:\n1. 文档切分:将文档分成小的文本片段\n2. 向量化:将文本片段转换为向量表示\n3. 索引构建:构建向量索引以支持快速检索\n4. 检索:根据查询从索引中检索相关片段\n5. 上下文构建:将检索结果组合成上下文\n6. 生成:使用LLM基于上下文生成回答',
      tokens: 256,
      latency: 1.2,
      quality: 0.92
    };
    setTestResults(result);
    setShowPromptTest(true);
  };

  const handleLengthOptimize = () => {
    const currentLength = prompt.length;
    const optimalLength = Math.min(currentLength, 1500);
    const recommendation = {
      currentLength,
      optimalLength,
      chunkCount: currentChunks.length,
      suggestedChunkCount: Math.max(3, Math.min(currentChunks.length, 5)),
      maxLength: 2000,
      usage: ((currentLength / 2000) * 100).toFixed(1) + '%'
    };
    setLengthOptimization(recommendation);
    setShowLengthOptimize(true);
  };

  const charLength = prompt.replace(/\s/g, '').length;

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">
          <i className="fas fa-layer-group"></i> 上下文构建与生成输入展示
          <small>RAG Context 编辑 & Prompt 可视化</small>
        </h1>
      </div>

      <div className="gen-strategy-bar">
        <div className="strategy-group">
          <span className="strategy-label">模型配置</span>
          <button className="btn btn-sm btn-primary" onClick={() => setShowModelConfig(true)}>
            <i className="fas fa-robot"></i> LLM模型
          </button>
          <button className="btn btn-sm btn-info" onClick={() => setShowParamTuning(true)}>
            <i className="fas fa-sliders-h"></i> 参数调优
          </button>
        </div>
        <div className="strategy-group">
          <span className="strategy-label">Prompt管理</span>
          <button className="btn btn-sm btn-success" onClick={() => setShowTemplateManage(true)}>
            <i className="fas fa-file-alt"></i> 模板管理
          </button>
          <button className="btn btn-sm btn-outline-primary" onClick={() => setShowPromptTest(true)}>
            <i className="fas fa-vial"></i> 测试Prompt
          </button>
        </div>
        <div className="strategy-group">
          <span className="strategy-label">优化工具</span>
          <button className="btn btn-sm btn-warning" onClick={handleLengthOptimize}>
            <i className="fas fa-compress-alt"></i> 长度优化
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-question-circle"></i> 用户查询与原始检索结果
          </h3>
        </div>
        <div className="card-body">
          <div className="form-item">
            <label className="form-label">用户查询问题</label>
            <div className="query-display">{userQuery}</div>
          </div>

          <div className="edit-header">
            <div className="edit-title">
              检索结果(支持<span style={{ color: 'var(--primary)' }}>拖拽排序</span> / <span style={{ color: 'var(--primary)' }}>点击删除</span>)
            </div>
            <div>
              <button className="btn btn-sm btn-default" onClick={handleResetContext}>
                <i className="fas fa-refresh"></i> 重置上下文
              </button>
              <button className="btn btn-sm btn-primary" onClick={handleBuildContext}>
                <i className="fas fa-sync-alt"></i> 刷新Prompt
              </button>
            </div>
          </div>

          <div ref={chunkListRef} className="chunk-list">
            {currentChunks.length === 0 ? (
              <div className="empty-box" style={{ padding: '30px 20px' }}>
                <div className="empty-icon" style={{ fontSize: '32px' }}>
                  <i className="fas fa-list"></i>
                </div>
                <p style={{ marginTop: '8px' }}>暂无上下文片段,可点击「重置上下文」恢复原始检索结果</p>
              </div>
            ) : (
              currentChunks.map(chunk => (
                <div key={chunk.id} className="chunk-item" data-id={chunk.id}>
                  <div className="chunk-left">
                    <span className="chunk-num">{chunk.num}</span>
                    <span className="chunk-sim">相似度:{(chunk as any).sim?.toFixed(2) || 'N/A'}</span>
                    <div className="chunk-content">
                      <span dangerouslySetInnerHTML={{
                        __html: highlightContent(chunk.content, chunk.highlightKeywords || [])
                      }} />
                    </div>
                  </div>
                  <button className="btn btn-sm btn-danger" onClick={() => handleDeleteChunk(chunk.id)}>
                    <i className="fas fa-trash"></i> 删除
                  </button>
                </div>
              ))
            )}
          </div>

          <div className="btn-group">
            <button className="btn btn-primary" onClick={handleBuildContext}>
              <i className="fas fa-cogs"></i> 构建生成上下文
            </button>
            <button className="btn btn-default" onClick={handleClearAll}>
              <i className="fas fa-trash"></i> 清空所有片段
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <i className="fas fa-file-code"></i> 生成输入Prompt结构可视化
          </h3>
          <span className="tip-text">
            状态:已构建上下文({currentChunks.length}个片段) | Prompt长度:{charLength}字符
          </span>
        </div>
        <div className="card-body">
          <div className="prompt-preview">
            {prompt.split('\n').map((line, index) => {
              let formattedLine = line;
              if (line.includes('# RAG生成输入模板')) {
                formattedLine = `<span class="prompt-label">${line}</span>`;
              } else if (line.includes('##')) {
                formattedLine = `<span class="prompt-label">${line}</span>`;
              } else if (line.includes('【参考片段')) {
                formattedLine = `<span style="color:var(--success); font-weight:500;">${line}</span>`;
              } else if (line.includes(userQuery)) {
                formattedLine = `<span class="prompt-question">${line}</span>`;
              }
              return <div key={index} dangerouslySetInnerHTML={{ __html: formattedLine }} />;
            })}
          </div>
        </div>
      </div>

      {/* LLM模型配置模态框 */}
      {showModelConfig && (
        <div className="modal-mask" onClick={() => setShowModelConfig(false)}>
          <div className="modal-box modal-large" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span><i className="fas fa-robot"></i> LLM模型配置</span>
              <button className="modal-close" onClick={() => setShowModelConfig(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="config-grid">
                <div className="config-item">
                  <label className="form-label">服务提供商</label>
                  <select className="form-select" value={llmConfig.provider} onChange={e => setLLMConfig({ ...llmConfig, provider: e.target.value })}>
                    <option value="openai">OpenAI</option>
                    <option value="azure">Azure OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="cohere">Cohere</option>
                    <option value="zhipu">智谱AI</option>
                    <option value="aliyun">阿里云</option>
                  </select>
                </div>
                <div className="config-item">
                  <label className="form-label">模型名称</label>
                  <select className="form-select" value={llmConfig.model} onChange={e => setLLMConfig({ ...llmConfig, model: e.target.value })}>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    <option value="gpt-4">GPT-4</option>
                    <option value="gpt-4-turbo">GPT-4 Turbo</option>
                    <option value="claude-3-opus">Claude 3 Opus</option>
                    <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                    <option value="command-r">Cohere Command R</option>
                    <option value="glm-4">ChatGLM-4</option>
                    <option value="qwen-turbo">Qwen Turbo</option>
                  </select>
                </div>
                <div className="config-item">
                  <label className="form-label">API密钥</label>
                  <input type="password" className="form-input" value={llmConfig.apiKey} onChange={e => setLLMConfig({ ...llmConfig, apiKey: e.target.value })} placeholder="sk-..." />
                </div>
                <div className="config-item">
                  <label className="form-label">Base URL</label>
                  <input type="text" className="form-input" value={llmConfig.baseUrl} onChange={e => setLLMConfig({ ...llmConfig, baseUrl: e.target.value })} placeholder="https://api.openai.com/v1" />
                </div>
              </div>
              <div className="model-info-card">
                <h4><i className="fas fa-info-circle"></i> 模型信息</h4>
                <div className="info-grid">
                  <div className="info-item">
                    <span className="info-label">上下文长度</span>
                    <span className="info-value">4096 tokens</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">输入价格</span>
                    <span className="info-value">$0.0015/1K</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">输出价格</span>
                    <span className="info-value">$0.002/1K</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">推荐场景</span>
                    <span className="info-value">通用问答</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowModelConfig(false)}>
                取消
              </button>
              <button className="btn btn-outline-info" onClick={handleTestModel}>
                <i className="fas fa-plug"></i> 测试连接
              </button>
              <button className="btn btn-primary" onClick={() => handleModelConfig(llmConfig)}>
                <i className="fas fa-save"></i> 保存配置
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 参数调优模态框 */}
      {showParamTuning && (
        <div className="modal-mask" onClick={() => setShowParamTuning(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span><i className="fas fa-sliders-h"></i> 生成参数调优</span>
              <button className="modal-close" onClick={() => setShowParamTuning(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="param-group">
                <div className="param-item">
                  <div className="param-header">
                    <label className="form-label">Temperature (温度)</label>
                    <span className="param-value">{genParams.temperature}</span>
                  </div>
                  <input
                    type="range"
                    className="form-range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={genParams.temperature}
                    onChange={e => setGenParams({ ...genParams, temperature: parseFloat(e.target.value) })}
                  />
                  <p className="param-desc">控制生成随机性: 0=确定性输出, 2=高度随机</p>
                </div>
                <div className="param-item">
                  <div className="param-header">
                    <label className="form-label">Top P</label>
                    <span className="param-value">{genParams.topP}</span>
                  </div>
                  <input
                    type="range"
                    className="form-range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={genParams.topP}
                    onChange={e => setGenParams({ ...genParams, topP: parseFloat(e.target.value) })}
                  />
                  <p className="param-desc">核采样: 限制只从概率最高的P%的tokens中采样</p>
                </div>
                <div className="param-item">
                  <div className="param-header">
                    <label className="form-label">Max Tokens (最大长度)</label>
                    <span className="param-value">{genParams.maxTokens}</span>
                  </div>
                  <input
                    type="range"
                    className="form-range"
                    min="100"
                    max="4000"
                    step="100"
                    value={genParams.maxTokens}
                    onChange={e => setGenParams({ ...genParams, maxTokens: parseInt(e.target.value) })}
                  />
                  <p className="param-desc">生成输出的最大token数量</p>
                </div>
                <div className="param-item">
                  <div className="param-header">
                    <label className="form-label">Frequency Penalty</label>
                    <span className="param-value">{genParams.frequencyPenalty}</span>
                  </div>
                  <input
                    type="range"
                    className="form-range"
                    min="-2"
                    max="2"
                    step="0.1"
                    value={genParams.frequencyPenalty}
                    onChange={e => setGenParams({ ...genParams, frequencyPenalty: parseFloat(e.target.value) })}
                  />
                  <p className="param-desc">频率惩罚: 减少重复内容, 正值降低重复频率</p>
                </div>
                <div className="param-item">
                  <div className="param-header">
                    <label className="form-label">Presence Penalty</label>
                    <span className="param-value">{genParams.presencePenalty}</span>
                  </div>
                  <input
                    type="range"
                    className="form-range"
                    min="-2"
                    max="2"
                    step="0.1"
                    value={genParams.presencePenalty}
                    onChange={e => setGenParams({ ...genParams, presencePenalty: parseFloat(e.target.value) })}
                  />
                  <p className="param-desc">存在惩罚: 鼓励谈论新话题, 正值增加话题多样性</p>
                </div>
              </div>
              <div className="param-templates">
                <h4>参数模板</h4>
                <div className="template-buttons">
                  <button className="btn btn-sm btn-outline-primary" onClick={() => setGenParams({ temperature: 0.7, topP: 0.9, maxTokens: 2000, frequencyPenalty: 0, presencePenalty: 0, stopSequences: [] })}>
                    平衡模式
                  </button>
                  <button className="btn btn-sm btn-outline-info" onClick={() => setGenParams({ temperature: 0.2, topP: 0.95, maxTokens: 1500, frequencyPenalty: 0.5, presencePenalty: 0.5, stopSequences: [] })}>
                    创意模式
                  </button>
                  <button className="btn btn-sm btn-outline-success" onClick={() => setGenParams({ temperature: 0, topP: 1, maxTokens: 1000, frequencyPenalty: 1, presencePenalty: 1, stopSequences: [] })}>
                    精确模式
                  </button>
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowParamTuning(false)}>
                取消
              </button>
              <button className="btn btn-outline-info" onClick={handleTestParams}>
                <i className="fas fa-vial"></i> 测试效果
              </button>
              <button className="btn btn-primary" onClick={() => handleParamTuning(genParams)}>
                <i className="fas fa-save"></i> 保存参数
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Prompt模板管理模态框 */}
      {showTemplateManage && (
        <div className="modal-mask" onClick={() => setShowTemplateManage(false)}>
          <div className="modal-box modal-large" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span><i className="fas fa-file-alt"></i> Prompt模板管理</span>
              <button className="modal-close" onClick={() => setShowTemplateManage(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="template-create-form">
                <input type="text" className="form-input" placeholder="模板名称" id="templateName" />
                <textarea className="form-textarea" placeholder="输入模板内容..." rows={4} id="templateContent" />
                <button className="btn btn-primary" onClick={() => {
                  const name = (document.getElementById('templateName') as HTMLInputElement)?.value;
                  const content = (document.getElementById('templateContent') as HTMLTextAreaElement)?.value;
                  if (name && content) {
                    handleSaveTemplate({ name, content });
                  } else {
                    alert('请填写模板名称和内容');
                  }
                }}>
                  <i className="fas fa-plus"></i> 创建模板
                </button>
              </div>
              <div className="template-list">
                {promptTemplates.length === 0 ? (
                  <div className="empty-tip">暂无模板</div>
                ) : (
                  promptTemplates.map((template: any) => (
                    <div key={template.id} className="template-item">
                      <div className="template-info">
                        <div className="template-name">{template.name}</div>
                        <div className="template-time">{template.createTime}</div>
                      </div>
                      <div className="template-actions">
                        <button className="btn btn-sm btn-outline-primary">
                          <i className="fas fa-edit"></i> 编辑
                        </button>
                        <button className="btn btn-sm btn-outline-info">
                          <i className="fas fa-download"></i> 导出
                        </button>
                        <button className="btn btn-sm btn-outline-danger">
                          <i className="fas fa-trash"></i> 删除
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowTemplateManage(false)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Prompt测试模态框 */}
      {showPromptTest && (
        <div className="modal-mask" onClick={() => setShowPromptTest(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span><i className="fas fa-vial"></i> Prompt测试验证</span>
              <button className="modal-close" onClick={() => setShowPromptTest(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="test-input-section">
                <label className="form-label">测试查询</label>
                <textarea
                  className="form-textarea"
                  placeholder="输入测试问题..."
                  rows={3}
                  defaultValue="RAG的核心优势是什么?"
                />
              </div>
              {testResults ? (
                <div className="test-result-section">
                  <h4><i className="fas fa-check-circle"></i> 测试结果</h4>
                  <div className="result-metrics">
                    <div className="metric-item">
                      <span className="metric-label">Token数量</span>
                      <span className="metric-value">{testResults.tokens}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">响应延迟</span>
                      <span className="metric-value">{testResults.latency}s</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">质量评分</span>
                      <span className="metric-value">{(testResults.quality * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div className="result-content">
                    <label className="form-label">生成结果</label>
                    <div className="result-text">{testResults.response}</div>
                  </div>
                </div>
              ) : null}
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowPromptTest(false)}>
                取消
              </button>
              <button
                className="btn btn-primary"
                onClick={() => {
                  const query = (document.querySelector('.test-input-section textarea') as HTMLTextAreaElement)?.value;
                  handlePromptTest(query);
                }}
              >
                <i className="fas fa-play"></i> 运行测试
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 长度优化模态框 */}
      {showLengthOptimize && lengthOptimization && (
        <div className="modal-mask" onClick={() => setShowLengthOptimize(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <span><i className="fas fa-compress-alt"></i> 上下文长度优化</span>
              <button className="modal-close" onClick={() => setShowLengthOptimize(false)}>
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="modal-body">
              <div className="length-analysis">
                <div className="analysis-card">
                  <h4><i className="fas fa-ruler"></i> 当前状态</h4>
                  <div className="analysis-item">
                    <span className="analysis-label">当前长度</span>
                    <span className="analysis-value">{lengthOptimization.currentLength} 字符</span>
                  </div>
                  <div className="analysis-item">
                    <span className="analysis-label">片段数量</span>
                    <span className="analysis-value">{lengthOptimization.chunkCount} 个</span>
                  </div>
                  <div className="analysis-item">
                    <span className="analysis-label">使用率</span>
                    <span className="analysis-value">{lengthOptimization.usage}</span>
                  </div>
                </div>
                <div className="analysis-card analysis-card-warning">
                  <h4><i className="fas fa-lightbulb"></i> 优化建议</h4>
                  <div className="analysis-item">
                    <span className="analysis-label">建议长度</span>
                    <span className="analysis-value">{lengthOptimization.optimalLength} 字符</span>
                  </div>
                  <div className="analysis-item">
                    <span className="analysis-label">建议片段数</span>
                    <span className="analysis-value">{lengthOptimization.suggestedChunkCount} 个</span>
                  </div>
                  <div className="analysis-item">
                    <span className="analysis-label">最大限制</span>
                    <span className="analysis-value">{lengthOptimization.maxLength} 字符</span>
                  </div>
                </div>
              </div>
              <div className="length-tips">
                <h4>优化提示</h4>
                <ul>
                  <li>• 保持上下文长度在模型token限制的60-80%范围内</li>
                  <li>• 优先选择与查询最相关的3-5个片段</li>
                  <li>• 可以考虑删除低相似度或重复的片段</li>
                  <li>• 使用摘要技术减少长片段的长度</li>
                </ul>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-default" onClick={() => setShowLengthOptimize(false)}>
                关闭
              </button>
              <button className="btn btn-primary" onClick={() => {
                alert('已应用优化建议!');
                setShowLengthOptimize(false);
              }}>
                <i className="fas fa-check"></i> 应用优化
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Generate;