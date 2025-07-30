/* eslint-env jest */
import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import PredictionCard from './PredictionCard';

describe('PredictionCard', () => {
  it('renders loading state when no prediction data', () => {
    render(<PredictionCard prediction={null} />);
    expect(screen.getByText('No prediction available')).toBeInTheDocument();
  });

  it('renders prediction data when available', () => {
    const mockPrediction = {
      predicted_winner: 'Max Verstappen',
      confidence: 0.85,
      podium_predictions: [
        { position: 1, driver: 'Max Verstappen', probability: 0.85 },
        { position: 2, driver: 'Lewis Hamilton', probability: 0.10 },
        { position: 3, driver: 'Charles Leclerc', probability: 0.05 }
      ]
    };

    render(<PredictionCard prediction={mockPrediction} />);
    expect(screen.getAllByText('Max Verstappen')).toHaveLength(2);
    expect(screen.getAllByText('85.0%')).toHaveLength(2);
    expect(screen.getByText('Lewis Hamilton')).toBeInTheDocument();
    expect(screen.getByText('Charles Leclerc')).toBeInTheDocument();
  });
}); 