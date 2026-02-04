import React, { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import type {
  Document,
  Chunk,
} from '../types';

interface AppDataContextType {
  documents: Document[];
  addDocument: (doc: Document) => void;
  addDocuments: (docs: Document[]) => void;
  updateDocument: (id: string, updates: Partial<Document>) => void;
  deleteDocument: (id: string) => void;
  batchDeleteDocuments: (ids: string[]) => void;
  chunks: Chunk[];
  setChunks: (chunks: Chunk[]) => void;
  selectedDocument: Document | null;
  setSelectedDocument: (doc: Document | null) => void;
  selectedDocuments: Set<string>;
  setSelectedDocuments: (ids: Set<string>) => void;
}

const AppDataContext = createContext<AppDataContextType | undefined>(undefined);

export const AppDataProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());

  const addDocument = useCallback((doc: Document) => {
    setDocuments(prev => [...prev, doc]);
  }, []);

  const addDocuments = useCallback((docs: Document[]) => {
    setDocuments(prev => [...prev, ...docs]);
  }, []);

  const updateDocument = useCallback((id: string, updates: Partial<Document>) => {
    setDocuments(prev =>
      prev.map(doc => (doc.id === id ? { ...doc, ...updates } : doc))
    );
  }, []);

  const deleteDocument = useCallback((id: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== id));
  }, []);

  const batchDeleteDocuments = useCallback((ids: string[]) => {
    setDocuments(prev => prev.filter(doc => !ids.includes(doc.id)));
  }, []);

  const value: AppDataContextType = {
    documents,
    addDocument,
    addDocuments,
    updateDocument,
    deleteDocument,
    batchDeleteDocuments,
    chunks,
    setChunks,
    selectedDocument,
    setSelectedDocument,
    selectedDocuments,
    setSelectedDocuments,
  };

  return (
    <AppDataContext.Provider value={value}>
      {children}
    </AppDataContext.Provider>
  );
};

export const useAppData = () => {
  const context = useContext(AppDataContext);
  if (context === undefined) {
    throw new Error('useAppData must be used within an AppDataProvider');
  }
  return context;
};