import React, { createContext, useContext, useState } from 'react';
import type { ReactNode } from 'react';
import type { UserInfo } from '../types';

interface AuthContextType {
  user: UserInfo | null;
  login: (username: string, password: string) => boolean;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserInfo | null>(() => {
    const saved = localStorage.getItem('user');
    return saved ? JSON.parse(saved) : null;
  });

  const login = (username: string, password: string): boolean => {
    // 模拟登录验证
    if (username === 'admin' && password === '123456') {
      const userInfo: UserInfo = {
        username,
        isAuthenticated: true
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

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
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