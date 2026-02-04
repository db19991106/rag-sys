import React, { createContext, useContext, useState } from 'react';
import type { ReactNode } from 'react';
import type { UserInfo } from '../types';

interface AuthContextType {
  user: UserInfo | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  hasPermission: (permission: string) => boolean;
  getPermissions: () => string[];
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

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserInfo | null>(() => {
    const saved = localStorage.getItem('user');
    return saved ? JSON.parse(saved) : null;
  });

  const login = async (username: string, password: string): Promise<boolean> => {
    // 模拟登录验证
    if (username === 'admin' && password === '123456') {
      const userInfo: UserInfo = {
        id: '1',
        username,
        email: 'admin@example.com',
        permissions: rolePermissions.admin,
        isAuthenticated: true,
        lastLogin: new Date().toISOString(),
        created_at: new Date().toISOString()
      };
      setUser(userInfo);
      localStorage.setItem('user', JSON.stringify(userInfo));
      return true;
    } else if (username === 'editor' && password === '123456') {
      const userInfo: UserInfo = {
        id: '2',
        username,
        email: 'editor@example.com',
        permissions: rolePermissions.editor,
        isAuthenticated: true,
        lastLogin: new Date().toISOString(),
        created_at: new Date().toISOString()
      };
      setUser(userInfo);
      localStorage.setItem('user', JSON.stringify(userInfo));
      return true;
    } else if (username === 'viewer' && password === '123456') {
      const userInfo: UserInfo = {
        id: '3',
        username,
        email: 'viewer@example.com',
        permissions: rolePermissions.viewer,
        isAuthenticated: true,
        lastLogin: new Date().toISOString(),
        created_at: new Date().toISOString()
      };
      setUser(userInfo);
      localStorage.setItem('user', JSON.stringify(userInfo));
      return true;
    }
    return false;
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('user');
  };

  const hasPermission = (permission: string): boolean => {
    if (!user) return false;
    return user.permissions.includes(permission);
  };

  const getPermissions = (): string[] => {
    return user?.permissions || [];
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, hasPermission, getPermissions }}>
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