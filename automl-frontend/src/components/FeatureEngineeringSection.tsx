// FeatureEngineeringSection.tsx - Feature Engineering UI Component

import React from 'react';
import {
  getColumnTypeInfo,
  getImputationOptions,
  getConfigurationWarnings,
  getSampleValues,
  includeAllFeatures,
  resetAllToAuto
} from '../utils/featureEngineering';
import type { FeatureConfig } from '../utils/featureEngineering';

interface DatasetPreview {
  columns: string[];
  rows: Record<string, any>[];
}

interface AdvancedSettings {
  cvFolds: number;
  trainSplit: number;
  maxRuntime: number;
}

interface FeatureEngineeringSectionProps {
  datasetPreview: DatasetPreview;
  selectedTargetColumn: string;
  featureConfig: FeatureConfig;
  setFeatureConfig: React.Dispatch<React.SetStateAction<FeatureConfig>>;
  advancedSettings: AdvancedSettings;
  handleAdvancedSettingsChange: (field: string, value: number | string) => void;
  showFeatureEngineering: boolean;
  setShowFeatureEngineering: React.Dispatch<React.SetStateAction<boolean>>;
}

const FeatureEngineeringSection: React.FC<FeatureEngineeringSectionProps> = ({
  datasetPreview,
  selectedTargetColumn,
  featureConfig,
  setFeatureConfig,
  advancedSettings,
  handleAdvancedSettingsChange,
  showFeatureEngineering,
  setShowFeatureEngineering
}) => {
  // Feature engineering handlers
  const handleFeatureConfigChange = (column: string, field: string, value: string | boolean) => {
    setFeatureConfig(prev => ({
      ...prev,
      [column]: {
        ...prev[column],
        [field]: value
      }
    }));
  };

  // Master checkboxes handlers
  const handleIncludeAll = () => {
    setFeatureConfig(prev => includeAllFeatures(prev, selectedTargetColumn));
  };

  const handleAutoAll = () => {
    setFeatureConfig(prev => resetAllToAuto(prev, selectedTargetColumn));
  };

  return (
    <section className="feature-engineering">
      <div className="feature-header">
        <div className="feature-header-content">
          <h2> Feature Engineering</h2>
          <p>Configure feature inclusion and missing value handling. H2O AutoML will automatically handle encoding and transformations.</p>
        </div>
        <div className="feature-toggle">
          <button 
            className={`btn ${showFeatureEngineering ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => setShowFeatureEngineering(!showFeatureEngineering)}
          >
            {showFeatureEngineering ? ' Hide Configuration' : ' Show Configuration'}
          </button>
        </div>
      </div>
      
      {showFeatureEngineering && (
        <div className="feature-content">
          {/* Middle Row - Advanced Settings and Quick Controls Side by Side */}
          <div className="feature-middle-row">
             <div className="feature-controls-section">
              <h4> Configuration Summary</h4>
              
              {/* Feature Configuration Summary */}
              <div className="feature-summary">
                <h5> Current Configuration</h5>
                <div className="summary-stats">
                  <div className="stat-item">
                    <span className="stat-value">{Object.values(featureConfig).filter(config => config.include).length}</span>
                    <span className="stat-label">Features Included</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{Object.values(featureConfig).filter(config => config.impute !== 'auto').length}</span>
                    <span className="stat-label">Custom Imputation</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{datasetPreview.columns.length - Object.values(featureConfig).filter(config => config.include).length}</span>
                    <span className="stat-label">Features Excluded</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{datasetPreview.columns.length}</span>
                    <span className="stat-label">Total Columns</span>
                  </div>
                </div>
              </div>

              {/* Configuration Warnings */}
              {(() => {
                const warnings = getConfigurationWarnings(featureConfig, datasetPreview, selectedTargetColumn);
                return warnings.length > 0 && (
                  <div className="config-warnings">
                    <h5> Configuration Warnings</h5>
                    <div className="warnings-list">
                      {warnings.map((warning, index) => (
                        <div key={index} className="warning-item">
                          {warning}
                        </div>
                      ))}
                    </div>
                    <p className="warning-note">
                      <strong>Note:</strong> These warnings help prevent H2O training errors. You can proceed, but consider adjusting the configuration for better results.
                    </p>
                  </div>
                );
              })()}
            </div>
            <div className="advanced-settings">
              <h4> Advanced Training Settings</h4>
              <div className="settings-grid">
                <div className="setting-item">
                  <label>Cross-Validation Folds:</label>
                  <input 
                    type="number" 
                    min="3" 
                    max="10" 
                    value={advancedSettings.cvFolds}
                    onChange={(e) => handleAdvancedSettingsChange('cvFolds', parseInt(e.target.value))}
                  />
                </div>
                <div className="setting-item">
                  <label>Train Split Ratio:</label>
                  <input 
                    type="range" 
                    min="0.6" 
                    max="0.9" 
                    step="0.05"
                    value={advancedSettings.trainSplit}
                    onChange={(e) => handleAdvancedSettingsChange('trainSplit', parseFloat(e.target.value))}
                  />
                  <span>{Math.round(advancedSettings.trainSplit * 100)}%</span>
                </div>
                <div className="setting-item">
                  <label>Max Runtime (seconds):</label>
                  <input 
                    type="number" 
                    min="60" 
                    max="3600" 
                    step="60"
                    value={advancedSettings.maxRuntime}
                    onChange={(e) => handleAdvancedSettingsChange('maxRuntime', parseInt(e.target.value))}
                  />
                </div>
              </div>
            </div>
          
          </div>

          {/* Feature Configuration Table - Full Width at Bottom */}
          <div className="feature-table-section">
            <div className="feature-table-container">
              <h4> Feature Selection & Missing Value Configuration</h4>
              <p className="table-description">Select features to include and configure missing value handling. H2O AutoML will automatically determine optimal encoding and transformations.</p>
              
              <div className="table-container">
                <table className="feature-table">
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Type</th>
                      <th>Missing</th>
                      <th>
                        <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                          Include
                          <input 
                            type="checkbox" 
                            title="Include all features"
                            onChange={handleIncludeAll}
                            style={{cursor: 'pointer'}}
                          />
                        </div>
                      </th>
                      <th>
                        <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                          Missing Values
                          <input 
                            type="checkbox" 
                            title="Set all to auto"
                            onChange={handleAutoAll}
                            style={{cursor: 'pointer'}}
                          />
                        </div>
                      </th>
                      <th>Sample Values</th>
                    </tr>
                  </thead>
                  <tbody>
                    {datasetPreview.columns.map((col) => {
                      const isTarget = col === selectedTargetColumn;
                      const colTypeInfo = getColumnTypeInfo(col, datasetPreview);
                      const imputationOptions = getImputationOptions(col, datasetPreview);
                      const sampleValues = getSampleValues(col, datasetPreview);
                      
                      return (
                        <tr key={col} className={isTarget ? 'target-column' : ''}>
                          <td>
                            <div className="column-info">
                              <strong>{col}</strong>
                              {isTarget && <span className="target-badge">TARGET</span>}
                            </div>
                          </td>
                          <td>
                            <div className="type-info">
                              <span 
                                className="type-badge" 
                                style={{backgroundColor: colTypeInfo.color, color: 'white'}}
                                title={colTypeInfo.description}
                              >
                                {colTypeInfo.icon} {colTypeInfo.label}
                              </span>
                            </div>
                          </td>
                          <td>
                            {colTypeInfo.missingCount > 0 ? (
                              <span className="missing-count" style={{color: '#dc3545'}}>
                                {colTypeInfo.missingCount} ({colTypeInfo.missingPercent}%)
                              </span>
                            ) : (
                              <span className="no-missing" style={{color: '#28a745'}}>None</span>
                            )}
                          </td>
                          <td>
                            <input 
                              type="checkbox" 
                              checked={isTarget || featureConfig[col]?.include || false}
                              disabled={isTarget}
                              onChange={(e) => handleFeatureConfigChange(col, 'include', e.target.checked)}
                              className="feature-checkbox"
                            />
                          </td>
                          <td>
                            <select 
                              value={featureConfig[col]?.impute || 'auto'}
                              disabled={isTarget || !featureConfig[col]?.include}
                              onChange={(e) => handleFeatureConfigChange(col, 'impute', e.target.value)}
                              className="feature-select"
                              title={`Options for ${colTypeInfo.label} data type`}
                            >
                              {imputationOptions.map(option => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                          </td>
                          <td className="sample-values">
                            {sampleValues}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};

export default FeatureEngineeringSection;
