import React from 'react';
import { Modal } from './Modal';
import './Dialog.css';

interface DialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  type?: 'danger' | 'warning' | 'info';
  loading?: boolean;
}

export const Dialog: React.FC<DialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = '确认',
  cancelText = '取消',
  type = 'danger',
  loading = false,
}) => {
  const iconMap = {
    danger: 'fa-exclamation-triangle',
    warning: 'fa-exclamation-circle',
    info: 'fa-info-circle',
  };

  const footer = (
    <div className="dialog-actions">
      <button className="btn btn-secondary" onClick={onClose} disabled={loading}>
        {cancelText}
      </button>
      <button
        className={`btn btn-${type === 'danger' ? 'danger' : 'primary'}`}
        onClick={onConfirm}
        disabled={loading}
      >
        {loading && <span className="spinner spinner-sm"></span>}
        {confirmText}
      </button>
    </div>
  );

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="sm" footer={footer}>
      <div className={`dialog-content dialog-${type}`}>
        <div className="dialog-icon">
          <i className={`fas ${iconMap[type]}`}></i>
        </div>
        <div className="dialog-info">
          <h4 className="dialog-title">{title}</h4>
          <p className="dialog-message">{message}</p>
        </div>
      </div>
    </Modal>
  );
};

// 确认对话框的 Promise 版本（待完善实现）
// 实际使用时应该结合状态管理
export const confirmDialog = async (
  _options: Omit<DialogProps, 'isOpen' | 'onClose' | 'onConfirm'>
): Promise<boolean> => {
  // 这是一个概念示例，需要更完整的实现
  // 目前建议直接使用 Dialog 组件
  console.warn('confirmDialog is not fully implemented, please use Dialog component directly');
  return false;
};

export default Dialog;
