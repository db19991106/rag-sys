import React from 'react';
import './Loading.css';

interface LoadingProps {
  text?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const Loading: React.FC<LoadingProps> = ({
  text = '加载中...',
  size = 'md',
  className = ''
}) => {
  const sizeClasses = {
    sm: 'loading-sm',
    md: 'loading-md',
    lg: 'loading-lg'
  };

  return (
    <div className={`loading-container ${sizeClasses[size]} ${className}`}>
      <div className="loading-spinner">
        <i className="fas fa-circle-notch fa-spin"></i>
      </div>
      {text && <div className="loading-text">{text}</div>}
    </div>
  );
};

export default Loading;