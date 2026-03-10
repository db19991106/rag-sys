import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { AppDataProvider } from './contexts/AppDataContext';
import { ErrorBoundary, PageLoadError } from './components/ErrorBoundary';
import Layout from './components/Layout';
import Login from './pages/Login';
import './index.css';

// 懒加载页面组件
const Chat = lazy(() => import('./pages/Chat'));
const RagDocManage = lazy(() => import('./pages/RagDocManage'));
const ChunkPage = lazy(() => import('./pages/Chunk'));
const Embedding = lazy(() => import('./pages/Embedding'));
const Retrieval = lazy(() => import('./pages/Retrieval'));
const Generate = lazy(() => import('./pages/Generate'));
const VectorManage = lazy(() => import('./pages/VectorManage'));
const Settings = lazy(() => import('./pages/Settings'));

// 页面加载组件
const PageLoader: React.FC = () => (
  <div className="page-loader">
    <div className="page-loader-content">
      <div className="spinner spinner-lg"></div>
      <span className="page-loader-text">加载中...</span>
    </div>
  </div>
);

// 懒加载包装器
const LazyPage: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <Suspense fallback={<PageLoader />}>
    <ErrorBoundary fallback={<PageLoadError />}>
      {children}
    </ErrorBoundary>
  </Suspense>
);

// 路由保护
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isAuthenticated = localStorage.getItem('user') !== null;
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
};

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <AppDataProvider>
          <BrowserRouter>
            <div className="app-wrapper">
              <Routes>
                <Route path="/login" element={<Login />} />
                <Route
                  path="/"
                  element={
                    <ProtectedRoute>
                      <Layout />
                    </ProtectedRoute>
                  }
                >
                  <Route index element={<Navigate to="/chat" replace />} />
                  <Route
                    path="chat"
                    element={
                      <LazyPage>
                        <Chat />
                      </LazyPage>
                    }
                  />
                  <Route
                    path="documents"
                    element={
                      <LazyPage>
                        <RagDocManage />
                      </LazyPage>
                    }
                  />
                  <Route
                    path="chunk"
                    element={
                      <LazyPage>
                        <ChunkPage />
                      </LazyPage>
                    }
                  />
                  <Route
                    path="embedding"
                    element={
                      <LazyPage>
                        <Embedding />
                      </LazyPage>
                    }
                  />
                  <Route
                    path="retrieval"
                    element={
                      <LazyPage>
                        <Retrieval />
                      </LazyPage>
                    }
                  />
                  <Route
                    path="generate"
                    element={
                      <LazyPage>
                        <Generate />
                      </LazyPage>
                    }
                  />
                  <Route
                    path="vector"
                    element={
                      <LazyPage>
                        <VectorManage />
                      </LazyPage>
                    }
                  />
                  <Route
                    path="settings"
                    element={
                      <LazyPage>
                        <Settings />
                      </LazyPage>
                    }
                  />
                </Route>
                {/* 404 处理 */}
                <Route path="*" element={<Navigate to="/chat" replace />} />
              </Routes>
            </div>
          </BrowserRouter>
        </AppDataProvider>
      </AuthProvider>
    </ErrorBoundary>
  );
};

export default App;