import React, { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import type { UserInfo } from '../types';

interface AuthContextType {
  user: UserInfo | null;
  token: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  hasPermission: (permission: string) => boolean;
  getPermissions: () => string[];
  isLoading: boolean;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// 角色权限映射
const rolePermissions: Record<string, string[]> = {
  admin: [
    'manage_users',
    'manage_documents',
    'manage_settings',
    'view_audit_logs',
    'manage_vector_db',
    'manage_embedding_models',
    'run_evaluations',
    'export_data'
  ],
  editor: [
    'manage_documents',
    'manage_vector_db',
    'run_evaluations'
  ],
  viewer: [
    'view_documents',
    'run_retrieval',
    'run_generation'
  ],
  user: [
    'run_retrieval',
    'run_generation'
  ]
};

// API 基础 URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserInfo | null>(() => {
    const saved = sessionStorage.getItem('user');
    return saved ? JSON.parse(saved) : null;
  });
  
  const [token, setToken] = useState<string | null>(() => {
    return sessionStorage.getItem('token');
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 登录函数 - 调用后端 API
  const login = useCallback(async (username: string, password: string): Promise<boolean> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login?username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: '登录失败' }));
        throw new Error(errorData.detail || '用户名或密码错误');
      }

      const data = await response.json();
      const { access_token } = data;

      // 存储 token
      sessionStorage.setItem('token', access_token);
      setToken(access_token);

      // 根据用户名确定角色和权限
      let permissions: string[] = rolePermissions.user;
      let email = `${username}@example.com`;

      if (username === 'admin') {
        permissions = rolePermissions.admin;
        email = 'admin@example.com';
      } else if (username === 'editor') {
        permissions = rolePermissions.editor;
        email = 'editor@example.com';
      } else if (username === 'viewer') {
        permissions = rolePermissions.viewer;
        email = 'viewer@example.com';
      }

      const userInfo: UserInfo = {
        id: username === 'admin' ? '1' : username === 'editor' ? '2' : '3',
        username,
        email,
        permissions,
        isAuthenticated: true,
        lastLogin: new Date().toISOString(),
        created_at: new Date().toISOString()
      };

      setUser(userInfo);
      sessionStorage.setItem('user', JSON.stringify(userInfo));
      
      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '登录失败，请检查网络连接';
      setError(errorMessage);
      console.error('登录失败:', err);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    setToken(null);
    sessionStorage.removeItem('user');
    sessionStorage.removeItem('token');
  }, []);

  const hasPermission = useCallback((permission: string): boolean => {
    if (!user) return false;
    return user.permissions.includes(permission);
  }, [user]);

  const getPermissions = useCallback((): string[] => {
    return user?.permissions || [];
  }, [user]);

  return (
    <AuthContext.Provider value={{ 
      user, 
      token,
      login, 
      logout, 
      hasPermission, 
      getPermissions,
      isLoading,
      error 
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// 获取 Authorization header 的辅助函数
export const getAuthHeaders = (): Record<string, string> => {
  const token = sessionStorage.getItem('token');
  if (token) {
    return { Authorization: `Bearer ${token}` };
  }
  return {};
};
