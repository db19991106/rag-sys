import React from 'react';
import './Empty.css';

interface EmptyProps {
  icon?: string;
  title?: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export const Empty: React.FC<EmptyProps> = ({
  icon = 'fas fa-inbox',
  title = '暂无数据',
  description = '还没有相关内容',
  action,
  className = ''
}) => {
  return (
    <div className={`empty-state ${className}`}>
      <div className="empty-icon">
        <i className={icon}></i>
      </div>
      <h4 className="empty-title">{title}</h4>
      <p className="empty-description">{description}</p>
      {action && <div className="empty-action">{action}</div>}
    </div>
  );
};

export default Empty;