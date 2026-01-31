import React, { useState, useEffect } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import Chat from '../pages/Chat';
import './Layout.css';

const Layout: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const menuItems = [

      { path: '/chat', label: '智能对话', icon: 'fa-comments', desc: 'RAG 智能问答' },

      { path: '/documents', label: '知识文档管理', icon: 'fa-file-alt', desc: '上传与管理文档' },

      { path: '/chunk', label: '文档切分', icon: 'fa-scissors', desc: '智能切分与编辑' },

      { path: '/embedding', label: '向量入库', icon: 'fa-vector-square', desc: '向量化与索引' },

      { path: '/vector', label: '向量库管理', icon: 'fa-database', desc: '向量库管理' },

      { path: '/retrieval', label: '查询检索', icon: 'fa-search', desc: '智能检索与策略' },

      { path: '/generate', label: '上下文构建', icon: 'fa-layer-group', desc: '上下文与生成' },

      { path: '/history', label: '历史记录', icon: 'fa-history', desc: '操作历史追踪' },

    ];

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  return (
    <div className="layout">
      {/* 顶部导航栏 */}
      <header className="top-nav">
        <div className="nav-container">
          {/* Logo区域 */}
          <div className="nav-logo">
            <Link to="/chat" className="logo-link">
              <div className="logo-icon-wrapper">
                <i className="fas fa-brain"></i>
              </div>
              <div className="logo-text">
                <h2>RAG助手</h2>
                <span>Retrieval-Augmented Generation</span>
              </div>
            </Link>
          </div>

          {/* 导航菜单 */}
          <nav className={`nav-menu ${mobileMenuOpen ? 'mobile-open' : ''}`}>
            <ul>
              {menuItems.map((item, index) => (
                <li key={item.path} style={{ animationDelay: `${index * 0.1}s` }}>
                  <Link
                    to={item.path}
                    className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
                  >
                    <span className="nav-icon">
                      <i className={`fas ${item.icon}`}></i>
                    </span>
                    <span className="nav-text">{item.label}</span>
                    {location.pathname === item.path && <span className="nav-indicator"></span>}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          {/* 用户区域 */}
          <div className="nav-user">
            <div className="user-dropdown">
              <button
                className="user-dropdown-btn"
                onClick={() => setDropdownOpen(!dropdownOpen)}
              >
                <div className="user-avatar-wrapper">
                  <i className="fas fa-user"></i>
                </div>
                <div className="user-info-text">
                  <span className="user-name">{user?.username || '用户'}</span>
                  <span className="user-role">管理员</span>
                </div>
                <i className={`fas fa-chevron-down dropdown-arrow ${dropdownOpen ? 'open' : ''}`}></i>
              </button>
              <div className={`dropdown-menu ${dropdownOpen ? 'show' : ''}`}>
                <Link to="/profile" className="dropdown-item">
                  <i className="fas fa-user-cog"></i>
                  个人设置
                </Link>
                <Link to="/settings" className="dropdown-item">
                  <i className="fas fa-cog"></i>
                  系统设置
                </Link>
                <div className="dropdown-divider"></div>
                <button className="dropdown-item dropdown-item-danger" onClick={handleLogout}>
                  <i className="fas fa-sign-out-alt"></i>
                  退出登录
                </button>
              </div>
            </div>

            {/* 移动端菜单按钮 */}
            <button
              className="mobile-menu-btn"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <i className={`fas ${mobileMenuOpen ? 'fa-times' : 'fa-bars'}`}></i>
            </button>
          </div>
        </div>
      </header>

      {/* 主内容区 */}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;