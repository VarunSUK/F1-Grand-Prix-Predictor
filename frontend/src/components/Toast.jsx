import React, { useEffect } from 'react';

const Toast = ({ message, type = 'error', onClose, duration = 4000 }) => {
  useEffect(() => {
    if (!duration) return;
    const timer = setTimeout(() => {
      onClose && onClose();
    }, duration);
    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const typeStyles = {
    error: 'bg-red-600 text-white',
    success: 'bg-green-600 text-white',
    info: 'bg-blue-600 text-white',
    warning: 'bg-yellow-500 text-white',
  };

  return (
    <div className={`fixed top-6 right-6 z-50 shadow-lg rounded-lg px-6 py-4 flex items-center space-x-3 ${typeStyles[type] || typeStyles.error}`}
      role="alert"
    >
      <span className="font-semibold">
        {type === 'error' && '❌'}
        {type === 'success' && '✅'}
        {type === 'info' && 'ℹ️'}
        {type === 'warning' && '⚠️'}
      </span>
      <span>{message}</span>
      <button
        onClick={onClose}
        className="ml-4 text-white hover:text-gray-200 focus:outline-none"
        aria-label="Close notification"
      >
        ×
      </button>
    </div>
  );
};

export default Toast; 