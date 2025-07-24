import axios from 'axios';

// Utility to handle and format API errors
function handleApiError(error, context = '') {
  let message = 'An unknown error occurred.';
  let status = null;
  let details = null;
  let isNetwork = false;
  let isTimeout = false;
  let isCORS = false;

  if (axios.isAxiosError(error)) {
    if (error.response) {
      // Server responded with a status code outside 2xx
      status = error.response.status;
      message = error.response.data?.message || error.response.statusText || 'Server error';
      details = error.response.data;
    } else if (error.request) {
      // No response received
      message = 'No response from server. Please check your network connection or backend status.';
      isNetwork = true;
      if (error.code === 'ECONNABORTED') {
        isTimeout = true;
        message = 'Request timed out. Please try again.';
      }
      // CORS errors are hard to detect, but if request exists and no response, likely CORS or network
      if (error.message && error.message.includes('Network Error')) {
        isCORS = true;
        message = 'Network or CORS error. Please check backend and browser settings.';
      }
    } else {
      // Something else happened
      message = error.message;
    }
  } else if (error instanceof Error) {
    message = error.message;
  }

  if (import.meta.env.MODE === 'development') {
    // eslint-disable-next-line no-console
    console.error(`API Error${context ? ' in ' + context : ''}:`, error);
  }

  return { error: true, message, status, details, isNetwork, isTimeout, isCORS };
}

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  timeout: 30000, // Increased timeout to 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

console.log('🔌 API client initialized with baseURL:', api.defaults.baseURL);

// API functions
export const f1Api = {
  // Get prediction for a specific race
  async getPrediction(season, round) {
    console.log('📡 Making prediction API call:', { season, round });
    try {
      const response = await api.get(`/predict/${season}/${round}`);
      console.log('✅ Prediction API response:', response.data);
      if (!response.data || response.data.error) {
        throw new Error('Prediction data is empty or invalid.');
      }
      return response.data;
    } catch (error) {
      throw handleApiError(error, 'getPrediction');
    }
  },

  // Get features for a specific race
  async getFeatures(season, round) {
    console.log('📡 Making features API call:', { season, round });
    try {
      const response = await api.get(`/features/${season}/${round}`);
      console.log('✅ Features API response:', response.data);
      if (!response.data || response.data.error) {
        throw new Error('Features data is empty or invalid.');
      }
      return response.data;
    } catch (error) {
      throw handleApiError(error, 'getFeatures');
    }
  },

  // Get available races for a season
  async getSeasonRaces(season) {
    console.log('📡 Making season races API call:', { season });
    try {
      const response = await api.get(`/seasons/${season}/races`);
      console.log('✅ Season races API response:', response.data);
      if (!response.data || response.data.error) {
        throw new Error('Season races data is empty or invalid.');
      }
      // Return only the races array
      return response.data.races || [];
    } catch (error) {
      throw handleApiError(error, 'getSeasonRaces');
    }
  },

  // Get the most recent round for a season
  async getMostRecentRound(season) {
    const races = await this.getSeasonRaces(season);
    if (Array.isArray(races) && races.length > 0) {
      // Find the highest round number
      return Math.max(...races.map(r => Number(r.round)));
    }
    return 1; // fallback
  },

  // Test prediction endpoint
  async testPrediction() {
    console.log('📡 Making test prediction API call');
    try {
      const response = await api.get('/test-predict');
      console.log('✅ Test prediction API response:', response.data);
      if (!response.data || response.data.error) {
        throw new Error('Test prediction data is empty or invalid.');
      }
      return response.data;
    } catch (error) {
      throw handleApiError(error, 'testPrediction');
    }
  },

  // Health check
  async healthCheck() {
    console.log('📡 Making health check API call');
    try {
      const response = await axios.get('http://localhost:8000/health');
      console.log('✅ Health check API response:', response.data);
      if (!response.data || response.data.status !== 'healthy') {
        throw new Error('Backend health check failed.');
      }
      return response.data;
    } catch (error) {
      throw handleApiError(error, 'healthCheck');
    }
  }
};

export default f1Api; 