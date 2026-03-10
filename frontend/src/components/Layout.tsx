import React, { useState, useEffect } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import './Layout.css';

const Layout: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  // Chat 页面专用：可折叠侧边栏状态
  const [chatSidebarVisible, setChatSidebarVisible] = useState(false);
  const [chatSidebarPinned, setChatSidebarPinned] = useState(false);

  // Chat 页面使用全屏布局，但有可折叠侧边栏
  const isChatPage = location.pathname === '/chat';

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    { path: '/documents', label: '文档', icon: 'fa-file-lines' },
    { path: '/chunk', label: '切分', icon: 'fa-scissors' },
    { path: '/embedding', label: '向量', icon: 'fa-vector-square' },
    { path: '/vector', label: '库管理', icon: 'fa-database' },
    { path: '/retrieval', label: '检索', icon: 'fa-magnifying-glass' },
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

  // Chat 页面专用：导航项
  const chatNavItems = [
    { path: '/documents', label: '文档管理', icon: 'fa-file-lines' },
    { path: '/chunk', label: '文档切分', icon: 'fa-scissors' },
    { path: '/embedding', label: '嵌入管理', icon: 'fa-vector-square' },
    { path: '/vector', label: '向量数据库', icon: 'fa-database' },
    { path: '/retrieval', label: '检索测试', icon: 'fa-magnifying-glass' },
    { path: '/settings', label: '系统设置', icon: 'fa-gear' },
  ];

  // Chat 页面：悬停显示侧边栏
  const handleChatSidebarEnter = () => {
    if (!chatSidebarPinned) {
      setChatSidebarVisible(true);
    }
  };

  const handleChatSidebarLeave = () => {
    if (!chatSidebarPinned) {
      setChatSidebarVisible(false);
    }
  };

  const toggleChatSidebarPin = () => {
    setChatSidebarPinned(!chatSidebarPinned);
    setChatSidebarVisible(!chatSidebarPinned);
  };

  // Chat 页面使用全屏布局 + 可折叠侧边栏
  if (isChatPage) {
    return (
      <div className="layout layout-fullscreen chat-layout">
        {/* 悬停触发区域 */}
        <div 
          className="chat-sidebar-trigger"
          onMouseEnter={handleChatSidebarEnter}
        />
        
        {/* 可折叠侧边栏 */}
        <div 
          className={`chat-sidebar ${chatSidebarVisible || chatSidebarPinned ? 'visible' : ''}`}
          onMouseLeave={handleChatSidebarLeave}
        >
          {/* Logo 区域 */}
          <div className="chat-sidebar-header">
            <Link to="/chat" className="chat-logo-link">
              <div className="chat-logo-icon">
                <i className="fa-solid fa-brain" />
              </div>
              <span className="chat-logo-text">RAG助手</span>
            </Link>
            <button
              className={`chat-sidebar-pin ${chatSidebarPinned ? 'pinned' : ''}`}
              onClick={toggleChatSidebarPin}
              title={chatSidebarPinned ? '取消固定' : '固定侧边栏'}
            >
              <i className={`fa-solid ${chatSidebarPinned ? 'fa-thumbtack' : 'fa-thumbtack fa-rotate-90'}`} />
            </button>
          </div>

          {/* 导航菜单 */}
          <nav className="chat-sidebar-nav">
            <div className="chat-nav-section">
              <div className="chat-nav-section-title">导航</div>
              {chatNavItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className="chat-nav-item"
                  title={item.label}
                >
                  <div className="chat-nav-item-icon">
                    <i className={`fa-solid ${item.icon}`} />
                  </div>
                  <span className="chat-nav-item-label">{item.label}</span>
                </Link>
              ))}
            </div>
          </nav>

          {/* 底部用户区域 */}
          <div className="chat-sidebar-footer">
            <div className="chat-user-info">
              <div className="chat-user-avatar">
                {user?.username?.charAt(0).toUpperCase() || 'U'}
              </div>
              <div className="chat-user-details">
                <span className="chat-user-name">{user?.username || '用户'}</span>
                <span className="chat-user-email">{user?.email || 'admin@example.com'}</span>
              </div>
              <button
                className="chat-logout-btn"
                onClick={handleLogout}
                title="退出登录"
              >
                <i className="fa-solid fa-arrow-right-from-bracket" />
              </button>
            </div>
          </div>
        </div>

        {/* 主内容区域 */}
        <main className="layout-content layout-content-fullscreen">
          <Outlet />
        </main>
      </div>
    );
  }

  return (
    <div className={`layout ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      {/* 左侧导航栏 */}
      <aside className="sidebar">
        {/* Logo 区域 */}
        <div className="sidebar-header">
          <Link to="/chat" className="logo-link">
            <div className="logo-icon">
              <i className="fa-solid fa-brain" />
            </div>
            {!sidebarCollapsed && <span className="logo-text">RAG助手</span>}
          </Link>
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            title={sidebarCollapsed ? '展开侧边栏' : '收起侧边栏'}
          >
            <i className={`fa-solid ${sidebarCollapsed ? 'fa-chevron-right' : 'fa-chevron-left'}`} />
          </button>
        </div>

        {/* 快捷入口 - 对话 */}
        <div className="sidebar-quick-actions">
          <Link to="/chat" className="quick-action-btn">
            <div className="quick-action-icon">
              <i className="fa-solid fa-message" />
            </div>
            {!sidebarCollapsed && <span className="quick-action-label">新对话</span>}
          </Link>
        </div>

        {/* 导航菜单 */}
        <nav className="sidebar-nav">
          <div className="nav-section">
            {!sidebarCollapsed && <div className="nav-section-title">功能</div>}
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-item ${isActive(item.path) ? 'active' : ''}`}
                title={item.label}
              >
                <div className="nav-item-icon">
                  <i className={`fa-solid ${item.icon}`} />
                </div>
                {!sidebarCollapsed && <span className="nav-item-label">{item.label}</span>}
                {isActive(item.path) && <div className="nav-item-indicator" />}
              </Link>
            ))}
          </div>
        </nav>

        {/* 底部用户区域 */}
        <div className="sidebar-footer">
          <div className="user-menu">
            <button
              className="user-menu-button"
              onClick={() => setUserMenuOpen(!userMenuOpen)}
            >
              <div className="user-avatar">
                {user?.username?.charAt(0).toUpperCase() || 'U'}
              </div>
              {!sidebarCollapsed && (
                <>
                  <div className="user-info">
                    <span className="user-name">{user?.username || '用户'}</span>
                    <span className="user-email">{user?.email || 'admin@example.com'}</span>
                  </div>
                  <i className={`fa-solid fa-chevron-down user-dropdown-arrow ${userMenuOpen ? 'open' : ''}`} />
                </>
              )}
            </button>

            {/* 用户下拉菜单 */}
            {userMenuOpen && (
              <div className="dropdown-menu">
                <div className="dropdown-header">
                  <div className="dropdown-avatar">
                    {user?.username?.charAt(0).toUpperCase() || 'U'}
                  </div>
                  <div className="dropdown-user-info">
                    <div className="dropdown-user-name">{user?.username || '用户'}</div>
                    <div className="dropdown-user-email">{user?.email || 'admin@example.com'}</div>
                  </div>
                </div>
                <div className="dropdown-divider" />
                <button className="dropdown-item dropdown-item-danger" onClick={handleLogout}>
                  <i className="fa-solid fa-arrow-right-from-bracket" />
                  <span>退出登录</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* 主内容区域 */}
      <div className="layout-main">
        {/* 顶部标题栏 */}
        <header className="layout-header">
          <div className="header-left">
            <h1 className="page-title">
              {navItems.find(item => isActive(item.path))?.label || 'RAG助手'}
            </h1>
          </div>
          <div className="header-right">
            {/* 可以添加页面级别的操作按钮 */}
          </div>
        </header>

        {/* 页面内容 */}
        <main className="layout-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;
