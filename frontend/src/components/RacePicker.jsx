import React from 'react';
import { motion, useReducedMotion } from 'framer-motion';

const RacePicker = ({ selectedSeason, selectedRound, onSeasonChange, onRoundChange, isLoading, error, races, seasons = [], unavailableRounds = [] }) => {
  const shouldReduceMotion = useReducedMotion();
  const availableRaces = races || [];
  const unavailableSet = new Set(unavailableRounds);

  if (isLoading && availableRaces.length === 0) {
    return (
      <motion.div
        className="flex items-center justify-center py-6"
        initial={shouldReduceMotion ? false : { opacity: 0, scale: 0.8 }}
        animate={shouldReduceMotion ? {} : { opacity: 1, scale: 1 }}
        exit={shouldReduceMotion ? {} : { opacity: 0, scale: 0.8 }}
        transition={{ type: 'spring', stiffness: 120, damping: 18 }}
      >
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-2"></div>
        <span className="text-gray-600">Loading races...</span>
      </motion.div>
    );
  }

  if (availableRaces.length === 0) {
    return (
      <div className="text-center text-gray-500 py-6">
        <div>No races available for this season.</div>
      </div>
    );
  }

  return (
    <motion.div
      className="bg-white/70 backdrop-blur-md shadow-xl rounded-lg p-4 transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl focus-within:scale-[1.02] focus-within:shadow-2xl"
      initial={shouldReduceMotion ? false : { opacity: 0, y: 40 }}
      animate={shouldReduceMotion ? {} : { opacity: 1, y: 0 }}
      exit={shouldReduceMotion ? {} : { opacity: 0, y: 40 }}
      transition={{ type: 'spring', stiffness: 80, damping: 18 }}
      tabIndex={0}
      style={{ overflow: 'visible' }}
    >
      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-3">
          <div className="flex items-center">
            <div className="text-red-600 mr-2">⚠️</div>
            <p className="text-red-800">{error}</p>
          </div>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label htmlFor="season" className="block text-sm font-medium text-blue-700 mb-2">
            <span className="font-[Orbitron,sans-serif] neon-text neon-animate" style={{ color: '#00eaff', fontFamily: 'Orbitron, Montserrat, Arial, sans-serif', textShadow: '0 0 8px #00eaff' }}>Season</span>
          </label>
          <select
            id="season"
            aria-label="Select season"
            role="listbox"
            value={selectedSeason}
            onChange={(e) => onSeasonChange(parseInt(e.target.value))}
            disabled={isLoading}
            className="rounded-lg px-3 py-2 w-full text-base font-bold font-[Orbitron,sans-serif] bg-[#1e293b] border-2 border-[#00eaff] text-[#e0e7ef] shadow-lg focus:outline-none focus:ring-2 focus:ring-[#00eaff] focus:border-[#00eaff] transition-all duration-200 hover:scale-105 hover:shadow-xl neon-text"
            style={{ transition: 'all 0.2s cubic-bezier(0.4,0,0.2,1)', boxShadow: '0 0 8px #00eaff99' }}
          >
            {seasons.map((season) => (
              <option key={season} value={season} aria-selected={selectedSeason === season}>
                {season}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="round" className="block text-sm font-medium text-red-700 mb-2">
            <span className="font-[Orbitron,sans-serif] neon-text neon-animate" style={{ color: '#00eaff', fontFamily: 'Orbitron, Montserrat, Arial, sans-serif', textShadow: '0 0 8px #00eaff' }}>Race</span>
          </label>
          <select
            id="round"
            aria-label="Select race round"
            role="listbox"
            value={selectedRound}
            onChange={(e) => onRoundChange(parseInt(e.target.value))}
            disabled={isLoading}
            className="rounded-lg px-3 py-2 w-full text-base font-bold font-[Orbitron,sans-serif] bg-[#1e293b] border-2 border-[#00eaff] text-[#e0e7ef] shadow-lg focus:outline-none focus:ring-2 focus:ring-[#00eaff] focus:border-[#00eaff] transition-all duration-200 hover:scale-105 hover:shadow-xl neon-text"
            style={{ transition: 'all 0.2s cubic-bezier(0.4,0,0.2,1)', boxShadow: '0 0 8px #00eaff99' }}
          >
            {availableRaces.map((race) => (
              <option key={race.round} value={race.round} aria-selected={selectedRound === race.round}>
                Round {race.round}: {race.name}
                {unavailableSet.has(race.round) ? ' (Unavailable)' : ''}
              </option>
            ))}
          </select>
        </div>
      </div>
    </motion.div>
  );
};

export default RacePicker; 