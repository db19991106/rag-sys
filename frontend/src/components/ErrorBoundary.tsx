import React, { Component, ErrorInfo, ReactNode } from 'react';
import './ErrorBoundary.css';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  private handleReload = () => {
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="error-boundary">
          <div className="error-boundary-content">
            <div className="error-boundary-icon">
              <i className="fas fa-exclamation-triangle"></i>
            </div>
            <h2 className="error-boundary-title">出现了一些问题</h2>
            <p className="error-boundary-message">
              很抱歉，应用程序遇到了一个错误。请尝试刷新页面或联系技术支持。
            </p>
            {import.meta.env.DEV && this.state.error && (
              <div className="error-boundary-details">
                <details>
                  <summary>错误详情</summary>
                  <pre>{this.state.error.toString()}</pre>
                  {this.state.errorInfo && (
                    <pre>{this.state.errorInfo.componentStack}</pre>
                  )}
                </details>
              </div>
            )}
            <div className="error-boundary-actions">
              <button className="btn btn-secondary" onClick={this.handleRetry}>
                <i className="fas fa-redo"></i>
                重试
              </button>
              <button className="btn btn-primary" onClick={this.handleReload}>
                <i className="fas fa-sync"></i>
                刷新页面
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// 页面加载失败的回退组件
export const PageLoadError: React.FC<{ onRetry?: () => void }> = ({ onRetry }) => (
  <div className="error-boundary">
    <div className="error-boundary-content">
      <div className="error-boundary-icon">
        <i className="fas fa-file-exclamation"></i>
      </div>
      <h2 className="error-boundary-title">页面加载失败</h2>
      <p className="error-boundary-message">
        无法加载此页面，请检查网络连接后重试。
      </p>
      <div className="error-boundary-actions">
        <button className="btn btn-primary" onClick={onRetry || (() => window.location.reload())}>
          <i className="fas fa-redo"></i>
          重新加载
        </button>
      </div>
    </div>
  </div>
);

export default ErrorBoundary;
