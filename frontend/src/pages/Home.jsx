import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import RacePicker from '../components/RacePicker';
import PredictionCard from '../components/PredictionCard';
import FeaturePanel from '../components/FeaturePanel';
import { f1Api } from '../api/f1Api';
import Toast from '../components/Toast';

const Home = () => {
  const [selectedSeason, setSelectedSeason] = useState(2024);
  const [selectedRound, setSelectedRound] = useState(1);
  const [prediction, setPrediction] = useState(null);
  const [features, setFeatures] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('unknown');
  const [races, setRaces] = useState([]);
  const [seasons, setSeasons] = useState([2024]);
  const [predictionRequested, setPredictionRequested] = useState(false);
  const [unavailableRounds, setUnavailableRounds] = useState({});
  const [toast, setToast] = useState(null);
  const [predictHover, setPredictHover] = useState(false);

  useEffect(() => {
    checkBackendStatus();
  }, []);

  useEffect(() => {
    async function setInitialMostRecentRound() {
      const mostRecent = await f1Api.getMostRecentRound(selectedSeason);
      setSelectedRound(mostRecent);
    }
    setInitialMostRecentRound();
    // eslint-disable-next-line
  }, []);

  useEffect(() => {
    async function fetchSeasonsAndRaces() {
      const years = [2021, 2022, 2023, 2024, 2025];
      const availableSeasons = [];
      for (const year of years) {
        try {
          const raceList = await f1Api.getSeasonRaces(year);
          if (raceList && raceList.length > 0) {
            availableSeasons.push(year);
          }
        } catch {
          // Ignore errors for missing years
        }
      }
      setSeasons(availableSeasons.reverse());
      if (availableSeasons.length > 0) {
        setSelectedSeason(availableSeasons[0]);
      }
    }
    fetchSeasonsAndRaces();
  }, []);

  useEffect(() => {
    async function fetchRaces() {
      const raceList = await f1Api.getSeasonRaces(selectedSeason);
      setRaces(raceList);
    }
    if (selectedSeason) fetchRaces();
  }, [selectedSeason]);

  const checkBackendStatus = async () => {
    try {
      const status = await f1Api.healthCheck();
      setBackendStatus(status.status === 'healthy' ? 'connected' : 'error');
    } catch (error) {
      setBackendStatus('error');
      setToast({ message: error.message || 'Backend connection failed.', type: 'error' });
      if (import.meta.env.MODE === 'development') {
        console.error('Backend connection failed:', error);
      }
    }
  };

  const loadPrediction = async () => {
    setIsLoading(true);
    setError(null);
    setToast(null);
    let predictionData = null;
    let featuresData = null;
    let hasError = false;
    try {
      const [predictionRes, featuresRes] = await Promise.allSettled([
        f1Api.getPrediction(selectedSeason, selectedRound),
        f1Api.getFeatures(selectedSeason, selectedRound)
      ]);
      if (predictionRes.status === 'fulfilled') {
        predictionData = predictionRes.value;
      } else {
        hasError = true;
      }
      if (featuresRes.status === 'fulfilled') {
        featuresData = featuresRes.value;
      }
      if (predictionData) {
        setPrediction(predictionData);
      }
      if (featuresData && featuresData.features) {
        setFeatures(featuresData.features);
      }
      if (hasError || !predictionData) {
        setError('Prediction unavailable for this race. Please try another.');
        setToast({ message: 'No prediction data available for this race.', type: 'warning' });
        setUnavailableRounds(prev => {
          const updated = { ...prev };
          if (!updated[selectedSeason]) updated[selectedSeason] = new Set();
          updated[selectedSeason].add(selectedRound);
          return { ...updated };
        });
      }
    } catch (error) {
      setError('Prediction unavailable for this race. Please try another.');
      setToast({ message: error.message || 'Failed to load prediction data.', type: 'error' });
      setUnavailableRounds(prev => {
        const updated = { ...prev };
        if (!updated[selectedSeason]) updated[selectedSeason] = new Set();
        updated[selectedSeason].add(selectedRound);
        return { ...updated };
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSeasonChange = async (season) => {
    setSelectedSeason(season);
    setPredictionRequested(false); // Reset prediction request state
    setError(null); // Reset error state
    const mostRecent = await f1Api.getMostRecentRound(season);
    setSelectedRound(mostRecent);
  };

  const handleRoundChange = (round) => {
    setSelectedRound(round);
    setPredictionRequested(false); // Reset prediction request state
    setError(null); // Reset error state
  };

  const handlePredict = async () => {
    setPredictionRequested(true);
    await loadPrediction();
  };

  const isUnavailable = unavailableRounds[selectedSeason] && unavailableRounds[selectedSeason].has(selectedRound);

  const getMissingFields = (note) => {
    if (!note || !note.includes('missing fields:')) return [];
    const match = note.match(/missing fields: ([\w, ]+)/);
    if (match && match[1]) {
      return match[1].split(',').map(f => f.trim());
    }
    return [];
  };
  const missingFields = predictionRequested && prediction?.note ? getMissingFields(prediction.note) : [];

  const predictionUnavailable = isUnavailable || error === 'Prediction unavailable for this race. Please try another.' || (prediction && prediction.status === 'unavailable');
  const predictionError = predictionRequested && predictionUnavailable;

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)' }}>
      <div className="max-w-4xl mx-auto px-4 py-8 space-y-6">
        <header className="flex items-center justify-center gap-4 mb-6">
          <h1
            className="text-4xl font-extrabold flex items-center gap-2 tracking-widest drop-shadow-lg font-[Orbitron,sans-serif] neon-text neon-animate"
            style={{ fontFamily: 'Orbitron, Montserrat, Arial, sans-serif', color: '#e0e7ef', textShadow: '0 0 12px #00eaff, 0 2px 16px #00eaff99' }}
          >
            🏎️ F1 Predictor
          </h1>
          <a href="https://github.com/VarunSUK" target="_blank" rel="noopener noreferrer" aria-label="GitHub Profile">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-8 h-8 text-gray-200 hover:text-blue-400 transition-colors duration-200">
              <path d="M12 2C6.477 2 2 6.484 2 12.021c0 4.428 2.865 8.184 6.839 9.504.5.092.682-.217.682-.482 0-.237-.009-.868-.014-1.703-2.782.605-3.369-1.342-3.369-1.342-.454-1.155-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.004.07 1.532 1.032 1.532 1.032.892 1.53 2.341 1.088 2.91.832.091-.647.35-1.088.636-1.339-2.221-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.987 1.029-2.686-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.025A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.295 2.748-1.025 2.748-1.025.546 1.378.202 2.397.1 2.65.64.699 1.028 1.593 1.028 2.686 0 3.847-2.337 4.695-4.566 4.944.359.309.678.919.678 1.852 0 1.336-.012 2.417-.012 2.747 0 .267.18.577.688.479C19.138 20.2 22 16.447 22 12.021 22 6.484 17.523 2 12 2z"/>
            </svg>
          </a>
        </header>
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
        <p className="mb-4 text-center text-lg neon-text" style={{ color: '#e0e7ef', fontFamily: 'Montserrat, Arial, sans-serif', textShadow: '0 0 8px #00eaff, 0 2px 8px #00eaff99' }}>AI-powered Grand Prix Predictions</p>
        <section className="card glass p-6 w-full max-w-2xl mx-auto flex flex-col items-center gap-4 transition-all duration-200 hover:scale-[1.02] hover:shadow-2xl focus-within:scale-[1.02] focus-within:shadow-2xl overflow-visible">
          <h2
            className="text-2xl font-bold tracking-widest font-[Orbitron,sans-serif] flex items-center gap-2 mb-2 drop-shadow neon-text neon-animate"
            style={{ fontFamily: 'Orbitron, Montserrat, Arial, sans-serif', color: '#e0e7ef', textShadow: '0 0 8px #00eaff, 0 2px 8px #00eaff99' }}
          >
            <span role="img" aria-label="calendar">📅</span> Select Race
          </h2>
          <p className="text-sm mb-2 text-center w-full font-[Montserrat,sans-serif] tracking-wide" style={{ color: '#a5f3fc', textShadow: '0 0 6px #00eaff88' }}>Choose season and round to get predictions.</p>
          <RacePicker
            selectedSeason={selectedSeason}
            selectedRound={selectedRound}
            onSeasonChange={handleSeasonChange}
            onRoundChange={handleRoundChange}
            isLoading={isLoading}
            error={error}
            onRetry={loadPrediction}
            races={races}
            seasons={seasons}
            unavailableRounds={unavailableRounds[selectedSeason] ? Array.from(unavailableRounds[selectedSeason]) : []}
          />
          <button
            onClick={handlePredict}
            aria-label="Predict race outcome"
            role="button"
            disabled={isLoading || predictionError}
            onMouseEnter={() => setPredictHover(true)}
            onMouseLeave={() => setPredictHover(false)}
            style={{
              background: predictHover
                ? 'linear-gradient(90deg, #3b82f6 0%, #00eaff 100%)'
                : 'linear-gradient(90deg, #00eaff 0%, #3b82f6 100%)',
              color: '#fff',
              border: 'none',
              borderRadius: '0.5rem',
              boxShadow: predictHover
                ? '0 0 24px #00eaffcc, 0 2px 8px #1e293b'
                : '0 0 8px #00eaff99, 0 1.5px 1.5px #0006',
              fontFamily: "'Orbitron', 'Montserrat', Arial, sans-serif",
              fontWeight: 700,
              letterSpacing: '0.05em',
              padding: '0.5rem 2rem',
              minWidth: 120,
              maxWidth: 220,
              fontSize: '1.1rem',
              cursor: 'pointer',
              display: 'inline-block',
              marginTop: '0.5rem',
              transform: predictHover ? 'scale(1.07)' : 'scale(1.0)',
              transition: 'all 0.22s cubic-bezier(0.4,0,0.2,1)',
              outline: 'none',
            }}
          >
            {isLoading ? (
              <span className="flex items-center justify-center"><span className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></span>Loading...</span>
            ) : predictionError ? 'Unavailable' : 'Predict'}
          </button>
          {predictionError && (
            <div className="bg-yellow-100 text-yellow-800 border border-yellow-300 px-4 py-2 rounded-md text-sm font-medium shadow-sm my-4 flex items-center gap-2 justify-center" aria-live="assertive" role="alert">
              <span role="img" aria-label="Warning">⚠️</span> Prediction unavailable for this race. Please try another.
              <span className="ml-2 text-xs text-yellow-700">Try a different round — round 1 is usually available.</span>
            </div>
          )}
        </section>
        <div className="my-6">
          <div className={`inline-flex items-center px-3 py-2 rounded-full text-sm font-medium ${
            backendStatus === 'connected' 
              ? 'bg-green-900 text-green-200' 
              : backendStatus === 'error'
              ? 'bg-red-900 text-red-200'
              : 'bg-yellow-900 text-yellow-200'
          }`}>
            <div className={`w-2 h-2 rounded-full mr-2 ${
              backendStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'
            }`}></div>
            Backend: {backendStatus === 'connected' ? 'Connected' : backendStatus === 'error' ? 'Disconnected' : 'Checking...'}
          </div>
        </div>
        {predictionRequested && prediction?.note && !predictionError && (
          <div className="mb-6 bg-blue-900/80 border border-blue-700 rounded-lg p-4">
            <div className="flex items-center">
              <div className="text-blue-300 mr-2">ℹ️</div>
              <p className="text-blue-200">{prediction.note}</p>
            </div>
          </div>
        )}
        {isLoading && (
          <div className="flex justify-center items-center my-6" aria-live="polite" role="status">
            <span className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400 mr-2"></span>
            <span className="text-gray-300">Fetching prediction...</span>
          </div>
        )}
        {predictionRequested && !predictionError && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-fadeIn" aria-live="polite">
            <div>
              <PredictionCard prediction={prediction} error={error} missingFields={missingFields} />
            </div>
            <div>
              <FeaturePanel 
                features={features} 
                driverName={prediction?.predicted_winner}
                error={error}
                missingFields={missingFields}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Home; 