// 格式化文件大小
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
  else return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

/**
 * XSS防护 - 转义HTML特殊字符
 * 防止XSS攻击，当需要渲染用户输入内容时必须使用此函数
 */
export function escapeHtml(unsafe: string): string {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * 安全地渲染Markdown内容
 * 防止XSS攻击同时保留Markdown格式
 */
export function safeRenderMarkdown(content: string): string {
  // 先转义HTML特殊字符
  const escaped = escapeHtml(content);
  
  // 然后允许安全的Markdown语法（代码块、粗体、斜体等）
  // 注意：这里只实现基本的Markdown语法，不直接渲染HTML标签
  return escaped
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/\n/g, '<br>');
}

// 提取关键词
export function extractKeywords(query: string): string[] {
  const stopWords = ['的', '是', '什么', '哪些', '有', '如何', '怎么', '为', '与', '和', '及', '在', '于'];
  const punctuation = /[，。！？？；：""''()（）\[\]【】、]/g;
  return query
    .replace(punctuation, '')
    .split(' ')
    .filter(word => word.trim() && !stopWords.includes(word) && word.length > 1)
    .slice(0, 5);
}

/**
 * 安全地高亮匹配关键词
 * 防止XSS攻击
 */
export function highlightContent(content: string, keywords: string[]): string {
  if (keywords.length === 0) return escapeHtml(content);
  
  // 先转义内容防止XSS
  const escapedContent = escapeHtml(content);
  
  // 转义关键词中的特殊字符
  const escapedKeywords = keywords.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  
  // 高亮匹配的关键词
  const reg = new RegExp(`(${escapedKeywords.join('|')})`, 'g');
  return escapedContent.replace(reg, '<span class="highlight">$1</span>');
}

// 获取相似度分数样式类
export function getSimilarityScoreClass(sim: number): string {
  if (sim >= 0.8) return 'score-high';
  else if (sim >= 0.7) return 'similarity-score';
  else if (sim >= 0.6) return 'score-middle';
  else return 'score-low';
}