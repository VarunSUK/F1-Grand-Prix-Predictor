import React from 'react';
import { motion, useReducedMotion } from 'framer-motion';

const FeaturePanel = ({ features, driverName, error, missingFields = [] }) => {
  const shouldReduceMotion = useReducedMotion();
  if (error) {
    return (
      <div className="bg-white/70 backdrop-blur-md rounded-lg shadow-xl p-6 transition-all">
        <div className="text-center text-red-600 font-semibold">⚠️ {error}</div>
      </div>
    );
  }
  if (!features || features.length === 0) {
    return (
      <motion.div
        className="bg-blue-50/70 backdrop-blur-md border border-blue-300 p-4 rounded-lg shadow-xl transition-all"
        aria-live="polite"
        initial={shouldReduceMotion ? false : { opacity: 0, y: 40 }}
        animate={shouldReduceMotion ? {} : { opacity: 1, y: 0 }}
        exit={shouldReduceMotion ? {} : { opacity: 0, y: 40 }}
        transition={{ type: 'spring', stiffness: 80, damping: 18 }}
        tabIndex={0}
      >
        <p className="text-center text-sm italic text-gray-500 py-4">No feature data available for this race.</p>
      </motion.div>
    );
  }
  // Find the driver's features
  const driverFeatures = features.find(f => f.driver === driverName) || features[0];

  const formatValue = (value, type = 'text') => {
    if (value === null || value === undefined) return 'N/A';
    switch (type) {
      case 'time':
        return typeof value === 'number' ? `${value.toFixed(3)}s` : value;
      case 'percentage':
        return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : value;
      case 'position':
        return typeof value === 'number' ? `${value}${value === 1 ? 'st' : value === 2 ? 'nd' : value === 3 ? 'rd' : 'th'}` : value;
      case 'number':
        return typeof value === 'number' ? value.toFixed(2) : value;
      default:
        return value;
    }
  };

  const featureCategories = [
    {
      title: 'Qualifying Performance',
      icon: '🏁',
      features: [
        { key: 'grid_position', label: 'Grid Position', type: 'position' },
        { key: 'qualifying_time', label: 'Qualifying Time', type: 'time' },
        { key: 'qualifying_performance', label: 'Qualifying Performance', type: 'percentage' },
        { key: 'grid_position_score', label: 'Grid Position Score', type: 'percentage' }
      ]
    },
    {
      title: 'Team & Strategy',
      icon: '🏎️',
      features: [
        { key: 'team', label: 'Team', type: 'text' },
        { key: 'team_consistency', label: 'Team Consistency', type: 'percentage' },
        { key: 'avg_stint_length', label: 'Avg Stint Length', type: 'number' },
        { key: 'total_pit_stops', label: 'Total Pit Stops', type: 'number' }
      ]
    },
    {
      title: 'Race Data',
      icon: '📊',
      features: [
        { key: 'total_laps', label: 'Total Laps', type: 'number' },
        { key: 'compound_usage', label: 'Compound Usage', type: 'text' },
        { key: 'final_position', label: 'Final Position', type: 'position' },
        { key: 'points_scored', label: 'Points Scored', type: 'number' }
      ]
    }
  ];

  return (
    <motion.div
      className="bg-blue-50/70 backdrop-blur-md border border-blue-300 p-4 rounded-lg shadow-xl transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl focus-within:scale-[1.02] focus-within:shadow-2xl"
      aria-live="polite"
      initial={shouldReduceMotion ? false : { opacity: 0, y: 40 }}
      animate={shouldReduceMotion ? {} : { opacity: 1, y: 0 }}
      exit={shouldReduceMotion ? {} : { opacity: 0, y: 40 }}
      transition={{ type: 'spring', stiffness: 80, damping: 18 }}
      tabIndex={0}
    >
      <h2 className="text-xl font-bold text-blue-900 mb-4 flex items-center">📊 Key Features</h2>
      {driverName && (
        <div className="mb-6 p-4 bg-blue-100 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-800 mb-2">Selected Driver</h3>
          <p className="text-blue-600 font-medium">{driverName}</p>
        </div>
      )}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {featureCategories.map((category, categoryIndex) => (
          <div key={categoryIndex} className="border border-gray-200 rounded-lg p-4 bg-white transition hover:bg-gray-50">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <span className="mr-2">{category.icon}</span>
              {category.title}
            </h3>
            <div className="grid grid-cols-1 gap-4">
              {category.features.map((feature, featureIndex) => (
                <div key={featureIndex} className={`bg-gray-50 p-3 rounded-lg${missingFields.includes(feature.key) ? ' border-2 border-yellow-400' : ''}`} aria-label={feature.label + (missingFields.includes(feature.key) ? ' (missing)' : '')}>
                  <div className="text-sm text-gray-700 font-medium mb-1">
                    {feature.label} {missingFields.includes(feature.key) && <span className="text-yellow-600" title="Missing or default value">⚠️</span>}
                  </div>
                  <div className={`text-lg font-semibold ${missingFields.includes(feature.key) ? 'text-yellow-700' : 'text-black'}`}> 
                    {formatValue(driverFeatures[feature.key], feature.type)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      {/* Additional Features */}
      <div className="mt-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">🔧 Additional Features</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.entries(driverFeatures)
            .filter(([key]) => !['driver', 'team'].includes(key))
            .filter(([key]) => !featureCategories.some(cat => cat.features.some(f => f.key === key)))
            .slice(0, 8)
            .map(([key, value], index) => (
              <div key={index} className={`bg-gray-50 p-2 rounded text-center${missingFields.includes(key) ? ' border-2 border-yellow-400' : ''}`} aria-label={key.replace(/_/g, ' ') + (missingFields.includes(key) ? ' (missing)' : '')}>
                <div className="text-xs text-gray-700 font-medium mb-1">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} {missingFields.includes(key) && <span className="text-yellow-600" title="Missing or default value">⚠️</span>}
                </div>
                <div className={`text-sm font-semibold ${missingFields.includes(key) ? 'text-yellow-700' : 'text-black'}`}> 
                  {formatValue(value)}
                </div>
              </div>
            ))}
        </div>
      </div>
    </motion.div>
  );
};

export default FeaturePanel; 