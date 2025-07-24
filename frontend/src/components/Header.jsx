import React from 'react';

const Header = () => {
  return (
    <header className="bg-gradient-to-r from-red-600 to-red-800 text-white shadow-lg">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center">
              <span className="text-red-600 font-bold text-xl">🏎️</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold">F1 Predictor</h1>
              <p className="text-red-100 text-sm">AI-Powered Grand Prix Predictions</p>
            </div>
          </div>
          <nav className="hidden md:flex space-x-6">
            <a href="#" className="text-white hover:text-red-200 transition-colors">
              Predictions
            </a>
            <a href="#" className="text-white hover:text-red-200 transition-colors">
              Features
            </a>
            <a href="#" className="text-white hover:text-red-200 transition-colors">
              About
            </a>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header; 