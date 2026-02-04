import React from 'react';
import './Card.css';

interface CardProps {
  children: React.ReactNode;
  title?: string;
  icon?: string;
  actions?: React.ReactNode;
  className?: string;
  padding?: 'sm' | 'md' | 'lg';
}

export const Card: React.FC<CardProps> = ({
  children,
  title,
  icon,
  actions,
  className = '',
  padding = 'md'
}) => {
  const paddingClasses = {
    sm: 'card-padding-sm',
    md: 'card-padding-md',
    lg: 'card-padding-lg'
  };

  return (
    <div className={`card ${paddingClasses[padding]} ${className}`}>
      {(title || icon || actions) && (
        <div className="card-header">
          <div className="card-title">
            {icon && <i className={icon}></i>}
            {title && <h3>{title}</h3>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </div>
      )}
      <div className="card-body">
        {children}
      </div>
    </div>
  );
};

export default Card;