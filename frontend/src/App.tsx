import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { AppDataProvider } from './contexts/AppDataContext';
import Login from './pages/Login';
import Layout from './components/Layout';
import Chat from './pages/Chat';
import RagDocManage from './pages/RagDocManage';
import ChunkPage from './pages/Chunk';
import Embedding from './pages/Embedding';
import Retrieval from './pages/Retrieval';
import Generate from './pages/Generate';
import History from './pages/History';
import VectorManage from './pages/VectorManage';
import Settings from './pages/Settings';
import './styles/variables.css';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isAuthenticated = localStorage.getItem('user') !== null;
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
};

const App: React.FC = () => {
  return (
    <AuthProvider>
      <AppDataProvider>
        <BrowserRouter>
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
              <Route path="chat" element={<Chat />} />
              <Route path="documents" element={<RagDocManage />} />
              <Route path="chunk" element={<ChunkPage />} />
              <Route path="embedding" element={<Embedding />} />
              <Route path="retrieval" element={<Retrieval />} />
              <Route path="generate" element={<Generate />} />
              <Route path="history" element={<History />} />
              <Route path="vector" element={<VectorManage />} />
              <Route path="settings" element={<Settings />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AppDataProvider>
    </AuthProvider>
  );
};

export default App;