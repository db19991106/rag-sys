import React, { useState, useEffect } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import './Layout.css';

const Layout: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    { path: '/chat', label: '对话', icon: 'fa-message' },
    { path: '/documents', label: '文档', icon: 'fa-file-lines' },
    { path: '/chunk', label: '切分', icon: 'fa-scissors' },
    { path: '/embedding', label: '向量', icon: 'fa-vector-square' },
    { path: '/vector', label: '库管理', icon: 'fa-database' },
    { path: '/retrieval', label: '检索', icon: 'fa-magnifying-glass' },
    { path: '/generate', label: '生成', icon: 'fa-wand-magic-sparkles' },
    { path: '/settings', label: '设置', icon: 'fa-gear' },
  ];

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('.user-menu')) {
        setUserMenuOpen(false);
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="layout">
      {/* Header */}
      <header className="header">
        <div className="header-container">
          {/* Logo */}
          <Link to="/chat" className="logo-link">
            <div className="logo-icon">
              <i className="fa-solid fa-brain" />
            </div>
            <span className="logo-text">RAG助手</span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="nav">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-link ${isActive(item.path) ? 'active' : ''}`}
              >
                <i className={`fa-solid ${item.icon} nav-icon`} />
                <span>{item.label}</span>
              </Link>
            ))}
          </nav>

          {/* User Menu */}
          <div className="user-menu">
            <button
              className="user-menu-button"
              onClick={() => setUserMenuOpen(!userMenuOpen)}
            >
              <div className="user-avatar">
                {user?.username?.charAt(0).toUpperCase() || 'U'}
              </div>
              <i
                className={`fa-solid fa-chevron-down user-dropdown-arrow ${userMenuOpen ? 'open' : ''}`}
              />
            </button>

            {/* Dropdown Menu */}
            {userMenuOpen && (
              <div className="dropdown-menu">
                <div className="dropdown-user-info">
                  <div className="dropdown-user-name">
                    {user?.username || '用户'}
                  </div>
                  <div className="dropdown-user-email">
                    {user?.email || 'admin@example.com'}
                  </div>
                </div>
                <div className="dropdown-divider" />
                <button
                  className="dropdown-item dropdown-item-danger"
                  onClick={handleLogout}
                >
                  <i className="fa-solid fa-arrow-right-from-bracket" />
                  <span>退出登录</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
