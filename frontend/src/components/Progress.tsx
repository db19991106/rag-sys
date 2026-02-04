import React from 'react';
import './Progress.css';

interface ProgressProps {
  value: number;
  max?: number;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
  color?: 'primary' | 'success' | 'warning' | 'error';
  className?: string;
}

export const Progress: React.FC<ProgressProps> = ({
  value,
  max = 100,
  showLabel = false,
  size = 'md',
  color = 'primary',
  className = ''
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  const sizeClasses = {
    sm: 'progress-sm',
    md: 'progress-md',
    lg: 'progress-lg'
  };

  const colorClasses = {
    primary: 'progress-primary',
    success: 'progress-success',
    warning: 'progress-warning',
    error: 'progress-error'
  };

  return (
    <div className={`progress-container ${sizeClasses[size]} ${className}`}>
      {showLabel && (
        <span className="progress-label">{Math.round(percentage)}%</span>
      )}
      <div className={`progress-bar ${colorClasses[color]}`}>
        <div
          className="progress-fill"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

export default Progress;