import React from 'react';
import { useReducedMotion } from 'framer-motion';
import { teamInfo } from '../data/teamInfo';
import { teamNameAliases } from '../data/teamNameAliases';

const getTeamInfo = (team) => teamInfo[teamNameAliases[team] || team] || {};

const PredictionCard = ({ prediction, error }) => {
  const shouldReduceMotion = useReducedMotion();
  if (error) {
    return (
      <div className="bg-red-100 border border-red-300 text-red-800 rounded-lg p-4 text-center my-4" role="alert">
        <span role="img" aria-label="Error">❌</span> {error}
      </div>
    );
  }
  if (!prediction) {
    return (
      <div className="bg-gray-100 border border-gray-300 text-gray-700 rounded-lg p-4 text-center my-4" role="status">
        No prediction available
      </div>
    );
  }
  const { predicted_winner, confidence, key_features, podium_predictions = [], all_driver_predictions = [] } = prediction;
  const winnerTeam = key_features?.team || '';
  const winnerInfo = getTeamInfo(winnerTeam);
  const isP1 = podium_predictions[0]?.driver === predicted_winner;

  // Use all_driver_predictions if available, else fallback to podium_predictions
  const driverList = all_driver_predictions && all_driver_predictions.length > 0 ? all_driver_predictions : podium_predictions;
  // Sort by probability descending
  const sortedDrivers = driverList.slice().sort((a, b) => (b.probability || 0) - (a.probability || 0));

  return (
    <>
      <div
        className="w-full max-w-2xl mx-auto shadow-lg rounded-2xl p-8 flex flex-col items-center text-center gap-2 transition-all duration-300 ease-in-out hover:scale-105 neon-card glass"
        style={{
          background: winnerInfo.color
            ? `linear-gradient(135deg, ${winnerInfo.color} 0%, #1e293b 100%)`
            : 'linear-gradient(135deg, #3b82f6 0%, #1e293b 100%)',
          border: '2px solid #00eaff',
          boxShadow: '0 0 32px #00eaff88, 0 2px 8px #1e293b',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
        }}
        initial={shouldReduceMotion ? false : { opacity: 0, y: 40 }}
        animate={shouldReduceMotion ? {} : { opacity: 1, y: 0 }}
        exit={shouldReduceMotion ? {} : { opacity: 0, y: 40 }}
        tabIndex={0}
      >
        <div className="flex flex-col items-center w-full px-4 py-6 sm:px-6">
          {winnerInfo.logo && (
            <img src={winnerInfo.logo} className="w-12 h-12 mb-1 object-contain" alt="team logo" style={{ maxWidth: 48, maxHeight: 48 }} />
          )}
          <div className="flex items-center justify-center gap-2">
            {isP1 && <span className="text-2xl drop-shadow">🥇</span>}
            <span className="text-2xl sm:text-3xl font-extrabold tracking-widest neon-text neon-animate" style={{ fontFamily: 'Orbitron, Montserrat, Arial, sans-serif', color: '#e0e7ef', textShadow: '0 0 12px #00eaff, 0 2px 16px #00eaff99' }}>{predicted_winner || 'N/A'}</span>
          </div>
          <div className="text-sm sm:text-base neon-text neon-animate uppercase mt-1" style={{ color: '#00eaff', fontFamily: 'Orbitron, Montserrat, Arial, sans-serif', textShadow: '0 0 8px #00eaff' }}>{teamNameAliases[winnerTeam] || winnerTeam}</div>
          {/* Confidence Bar */}
          <div className="w-full mt-3 bg-white/30 h-2 rounded">
            <div
              style={{ width: `${(confidence * 100) || 0}%` }}
              className="h-full rounded bg-white transition-all duration-500"
            />
          </div>
          <p className="neon-text text-lg mt-2" style={{ color: '#00eaff', fontFamily: 'Orbitron, Montserrat, Arial, sans-serif', textShadow: '0 0 8px #00eaff' }}>Confidence: {(confidence * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Full Driver Prediction Table */}
      <div className="w-full max-w-7xl mt-8 bg-white shadow-md rounded-lg overflow-x-auto mx-auto" style={{ minWidth: 700 }}>
        <div className="p-4 sm:p-6">
          {sortedDrivers.length === 0 ? (
            <div className="text-center text-gray-500 py-8 text-lg">🏁 No data</div>
          ) : (
            <table
              className="w-full text-left border-collapse text-sm sm:text-base neon-table"
              style={{
                background: '#1e293b',
                color: '#e0e7ef',
                borderRadius: '1rem',
                boxShadow: '0 0 16px #00eaff44',
                overflow: 'hidden',
              }}
            >
              <thead
                className="bg-[#102030] text-[#00eaff] text-xs uppercase tracking-widest font-[Orbitron,sans-serif]"
                style={{
                  background: 'linear-gradient(90deg, #102030 60%, #00eaff 100%)',
                  color: '#00eaff',
                  textShadow: '0 0 8px #00eaff',
                  fontFamily: 'Orbitron, Montserrat, Arial, sans-serif',
                }}
              >
                <tr style={{ borderBottom: '2px solid #00eaff' }}>
                  <th className="py-2 px-3">Pos</th>
                  <th className="py-2 px-3">Driver</th>
                  <th className="py-2 px-3">Team</th>
                  <th className="py-2 px-3">Logo</th>
                  <th className="py-2 px-3">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {sortedDrivers.map((driver, i) => {
                  const resolvedTeam = teamNameAliases[driver.team] || driver.team;
                  const info = teamInfo[resolvedTeam] || {};
                  return (
                    <tr
                      key={driver.name || driver.driver || i}
                      className={`hover:bg-[#102030] transition`}
                      style={{ borderLeft: `6px solid ${info.color || '#00eaff'}` }}
                    >
                      <td className={`py-2 px-3 font-medium ${i === 0 ? 'bg-yellow-100' : i === 1 ? 'bg-gray-100' : i === 2 ? 'bg-orange-100' : ''}`}>{i + 1}</td>
                      <td className={`py-2 px-3 ${i === 0 ? 'bg-yellow-100' : i === 1 ? 'bg-gray-100' : i === 2 ? 'bg-orange-100' : ''}`}>{driver.name || driver.driver}</td>
                      <td className={`py-2 px-3 ${i === 0 ? 'bg-yellow-100' : i === 1 ? 'bg-gray-100' : i === 2 ? 'bg-orange-100' : ''}`}>{resolvedTeam}</td>
                      <td className={`py-2 px-3 ${i === 0 ? 'bg-yellow-100' : i === 1 ? 'bg-gray-100' : i === 2 ? 'bg-orange-100' : ''}`}>{info.logo ? (<img src={info.logo} alt="logo" className="w-8 h-8 object-contain inline-block align-middle" style={{ maxWidth: 32, maxHeight: 32 }} />) : (<span className="text-gray-400">—</span>)}</td>
                      <td className={`py-2 px-3 ${i === 0 ? 'bg-yellow-100' : i === 1 ? 'bg-gray-100' : i === 2 ? 'bg-orange-100' : ''}`}>{(driver.probability * 100).toFixed(1)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </>
  );
};

export default PredictionCard; 