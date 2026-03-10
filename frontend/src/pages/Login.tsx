import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import './Login.css';

const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showSuccess, setShowSuccess] = useState(false);
  const { login, isLoading, error } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const success = await login(username, password);
    if (success) {
      setShowSuccess(true);
      setTimeout(() => {
        navigate('/chat');
      }, 1500);
    }
  };

  const handleReset = () => {
    setUsername('');
    setPassword('');
  };

  if (showSuccess) {
    return (
      <div className="login-container">
        <div className="success-page">
          <div className="success-icon">
            <i className="fas fa-check-circle"></i>
          </div>
          <h2>登录成功</h2>
          <p>欢迎进入系统，正在为您跳转...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="login-container">
      <div className="login-box">
        <div className="login-header">
          <div className="login-logo">
            <i className="fas fa-brain"></i>
          </div>
          <h2 className="login-title">RAG 助手</h2>
          <p className="login-subtitle">智能文档检索与问答系统</p>
        </div>
        
        <form onSubmit={handleLogin}>
          <div className="form-item">
            <div className="input-icon">
              <i className="fas fa-user"></i>
            </div>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="请输入用户名"
              required
              disabled={isLoading}
            />
          </div>
          
          <div className="form-item">
            <div className="input-icon">
              <i className="fas fa-lock"></i>
            </div>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="请输入密码"
              required
              disabled={isLoading}
            />
          </div>
          
          {error && (
            <div className="login-error">
              <i className="fas fa-exclamation-circle"></i>
              <span>{error}</span>
            </div>
          )}
          
          <div className="btn-group">
            <button type="submit" className="btn-login" disabled={isLoading}>
              {isLoading ? (
                <>
                  <i className="fas fa-spinner fa-spin"></i>
                  <span>登录中...</span>
                </>
              ) : (
                <span>登录</span>
              )}
            </button>
            <button type="button" className="btn-reset" onClick={handleReset} disabled={isLoading}>
              重置
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Login;
