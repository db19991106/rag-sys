import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import './Login.css';

const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showSuccess, setShowSuccess] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    const success = await login(username, password);
    if (success) {
      setShowSuccess(true);
      setTimeout(() => {
        navigate('/chat');
      }, 2000);
    } else {
      alert('账号或密码错误,请重试!');
    }
  };

  const handleReset = () => {
    setUsername('');
    setPassword('');
  };

  const handleBack = () => {
    setShowSuccess(false);
    handleReset();
  };

  if (showSuccess) {
    return (
      <div className="login-container">
        <div className="success-page">
          <h2>登录成功!</h2>
          <p>欢迎进入系统,正在为您跳转...</p>
          <button className="back-btn" onClick={handleBack}>
            返回登录页
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="login-container">
      <div className="login-box">
        <h2 className="login-title">用户登录</h2>
        <form onSubmit={handleLogin}>
          <div className="form-item">
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            <label htmlFor="username">账号/手机号</label>
          </div>
          <div className="form-item">
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <label htmlFor="password">密码</label>
          </div>
          <div className="btn-group">
            <button type="submit" className="btn-login">
              登录
            </button>
            <button type="button" className="btn-reset" onClick={handleReset}>
              重置
            </button>
          </div>
          <p className="tip">测试账号:admin | 测试密码:123456</p>
        </form>
      </div>
    </div>
  );
};

export default Login;