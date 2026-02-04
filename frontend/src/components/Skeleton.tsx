import React from 'react';
import './Skeleton.css';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  lines?: number;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'text',
  width,
  height,
  lines = 1
}) => {
  const style: React.CSSProperties = {
    width: width || '100%',
    height: height || 'auto'
  };

  if (variant === 'text') {
    return (
      <div className={`skeleton skeleton-text ${className}`} style={style}>
        {Array.from({ length: lines }).map((_, index) => (
          <div key={index} className="skeleton-line" style={{ width: index === lines - 1 ? '60%' : '100%' }} />
        ))}
      </div>
    );
  }

  return (
    <div className={`skeleton skeleton-${variant} ${className}`} style={style} />
  );
};

export default Skeleton;