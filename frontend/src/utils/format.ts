// 格式化文件大小
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
  else return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// 格式化时间
export function formatTime(date: Date): string {
  const y = date.getFullYear();
  const m = (date.getMonth() + 1).toString().padStart(2, '0');
  const d = date.getDate().toString().padStart(2, '0');
  const h = date.getHours().toString().padStart(2, '0');
  const min = date.getMinutes().toString().padStart(2, '0');
  const s = date.getSeconds().toString().padStart(2, '0');
  return `${y}-${m}-${d} ${h}:${min}:${s}`;
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

// 高亮匹配关键词
export function highlightContent(content: string, keywords: string[]): string {
  if (keywords.length === 0) return content;
  const reg = new RegExp(`(${keywords.join('|')})`, 'g');
  return content.replace(reg, '<span class="highlight">$1</span>');
}

// 获取相似度分数样式类
export function getSimilarityScoreClass(sim: number): string {
  if (sim >= 0.8) return 'score-high';
  else if (sim >= 0.7) return 'similarity-score';
  else if (sim >= 0.6) return 'score-middle';
  else return 'score-low';
}