/* eslint-env jest */
import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Header from './Header';

describe('Header', () => {
  it('renders the app title and branding', () => {
    render(<Header />);
    expect(screen.getByText('F1 Predictor')).toBeInTheDocument();
    expect(screen.getByText('AI-Powered Grand Prix Predictions')).toBeInTheDocument();
    expect(screen.getByText('🏎️')).toBeInTheDocument();
  });
}); 