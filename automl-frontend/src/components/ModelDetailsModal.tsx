import React, { useState, useEffect } from 'react';

interface Model {
  model_id: string;
  auc?: number;
  logloss?: number;
  rmse?: number;
  mean_residual_deviance?: number;
  mlflow_run_id?: string;
  model_type?: string;
  feature_importance?: Array<{ feature: string; importance: number }>;
  training_time_ms?: number;
  [key: string]: any;
}

interface AdvancedSettings {
  cvFolds: number;
  trainSplit: number;
  maxRuntime: number;
}

interface ModelDetailsModalProps {
  modalModel: Model | null;
  showModal: boolean;
  onClose: () => void;
  filteredLeaderboard: Model[];
  advancedSettings: AdvancedSettings;
  formatValue: (value: any) => string;
  detectedProblemType: string;
  onGoToLiveModel?: () => void;
}

const ModelDetailsModal: React.FC<ModelDetailsModalProps> = ({
  modalModel,
  showModal,
  onClose,
  filteredLeaderboard,
  advancedSettings,
  formatValue,
  detectedProblemType,
  onGoToLiveModel
}) => {
  const [mojoDownloadLoading, setMojoDownloadLoading] = useState(false);
  const [mojoDownloadError, setMojoDownloadError] = useState<string | null>(null);

  useEffect(() => {
    // Reset error when modal model changes
    if (modalModel) {
      setMojoDownloadError(null);
    }
  }, [modalModel]);

  const handleDownloadMojo = async () => {
    if (!modalModel) return;
    setMojoDownloadLoading(true);
    setMojoDownloadError(null);
    console.log(`Downloading MOJO for model ID: ${modalModel.model_id}`);
    try {
      const res = await fetch(`http://localhost:8000/download-mojo/${modalModel.model_id}`);
      if (!res.ok) {
        throw new Error('MOJO not found for this model.');
      }
      const blob = await res.blob();
      // Download as file
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${modalModel.model_id}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      
      // Clear any previous errors on successful download
      setMojoDownloadError(null);
    } catch (err: any) {
      setMojoDownloadError(err.message || 'Failed to download MOJO.');
    } finally {
      setMojoDownloadLoading(false);
    }
  };

  const handleModalContentClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleClose = (e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    setMojoDownloadError(null);
    onClose();
  };

  if (!showModal || !modalModel) return null;

  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div className="modal-content modern-redesign" onClick={handleModalContentClick}>
        {/* Header */}
        <div className="modal-header-modern">
          <div className="header-left">
            <div className="model-rank-badge-modern">
              <span className="rank-text">#{filteredLeaderboard.findIndex(m => m.model_id === modalModel.model_id) + 1}</span>
            </div>
            <div className="header-info">
              <h2 className="modal-title-modern">Model Analysis</h2>
              <p className="modal-subtitle">{modalModel.model_id}</p>
            </div>
          </div>
          <button className="modal-close-modern" onClick={handleClose}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
        
        <div className="modal-body-modern">
          {/* Top Section: Performance Metrics */}
          <div className="metrics-section-modern">
            <h3 className="section-title-modern">
            
              Performance Metrics
            </h3>
            <div className="metrics-grid-modern">
              {['auc', 'logloss', 'aucpr', 'rmse', 'mse', 'mean_per_class_error']
                .filter(k => modalModel[k] !== undefined && modalModel[k] !== null)
                .map((k) => {
                  const isHigherBetter = ['auc', 'aucpr'].includes(k);
                  const value = modalModel[k];
                  const bestInCategory = filteredLeaderboard.reduce((best, model) => {
                    const modelValue = model[k];
                    if (modelValue === undefined || modelValue === null) return best;
                    if (best === null) return modelValue;
                    return isHigherBetter ? Math.max(best, modelValue) : Math.min(best, modelValue);
                  }, null);
                  
                  const isBest = bestInCategory === value;
                  
                  return (
                    <div key={k} className={`metric-card-modern ${isBest ? 'best-metric-modern' : ''}`}>
                      <div className="metric-header-modern">
                        <span className="metric-name-modern">{k.replace(/_/g, ' ').toUpperCase()}</span>
                        {isBest && <span className="best-badge-modern">🏆</span>}
                      </div>
                      <div className="metric-value-modern">{formatValue(value)}</div>
                      <div className="metric-progress-modern">
                        <div 
                          className="metric-progress-bar-modern"
                          style={{ 
                            width: bestInCategory ? 
                              (isHigherBetter ? 
                                `${(value / bestInCategory) * 100}%` : 
                                `${((bestInCategory / value) * 100).toFixed(0)}%`
                              ) : '0%'
                          }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>

          {/* Middle Section: Two Columns */}
          <div className="middle-section-modern">
            {/* Left Column: Model Info & Config */}
            <div className="left-column-modern">
              {/* Model Information */}
              <div className="info-card-modern">
                <h4 className="card-title-modern">
               
                  Model Information
                </h4>
                <div className="info-grid-modern">
                  <div className="info-item-modern">
                    <span className="info-label-modern">Algorithm</span>
                    <span className="info-value-modern">{modalModel.model_type || 'H2O AutoML'}</span>
                  </div>
                  {detectedProblemType && (
                    <div className="info-item-modern">
                      <span className="info-label-modern">Problem Type</span>
                      <span className="info-value-modern problem-type">
                        {detectedProblemType === 'classification' ? ' Classification' : 'Regression'}
                        <span className="problem-type-badge">{detectedProblemType}</span>
                        <span className="problem-type-help" title="Automatically detected by H2O based on your target column data">
                          ℹ️
                        </span>
                      </span>
                    </div>
                  )}
                  {modalModel.training_time_ms && (
                    <div className="info-item-modern">
                      <span className="info-label-modern">Training Time</span>
                      <span className="info-value-modern">{(modalModel.training_time_ms / 1000).toFixed(2)}s</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Training Configuration */}
              <div className="config-card-modern">
                <h4 className="card-title-modern">
                  
                  Configuration
                </h4>
                <div className="config-grid-modern">
                  <div className="config-item-modern">
                    <span className="config-label-modern">Strategy</span>
                    <span className="config-value-modern">H2O AutoML</span>
                  </div>
                  <div className="config-item-modern">
                    <span className="config-label-modern">Validation</span>
                    <span className="config-value-modern">{advancedSettings.cvFolds}-Fold CV</span>
                  </div>
                  <div className="config-item-modern">
                    <span className="config-label-modern">Split</span>
                    <span className="config-value-modern">{(advancedSettings.trainSplit * 100).toFixed(0)}% Train</span>
                  </div>
                  <div className="config-item-modern">
                    <span className="config-label-modern">Runtime</span>
                    <span className="config-value-modern">{advancedSettings.maxRuntime}s</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Feature Importance */}
            <div className="right-column-modern">
              {modalModel.feature_importance && modalModel.feature_importance.length > 0 && (
                <div className="features-card-modern">
                  <h4 className="card-title-modern">
                    
                    Feature Importance
                  </h4>
                  <div className="features-list-modern">
                    {modalModel.feature_importance.slice(0, 8).map((f: any, idx: number) => {
                      const maxImportance = modalModel.feature_importance?.[0]?.importance || 1;
                      const percentage = ((f.importance / maxImportance) * 100).toFixed(1);
                      
                      return (
                        <div key={idx} className="feature-item-modern">
                          <div className="feature-left-modern">
                            <span className="feature-rank-modern">#{idx + 1}</span>
                            <span className="feature-name-modern">{f.feature}</span>
                          </div>
                          <div className="feature-right-modern">
                            <div className="feature-bar-container-modern">
                              <div 
                                className="feature-bar-modern"
                                style={{ width: `${percentage}%` }}
                              ></div>
                            </div>
                            <span className="feature-percentage-modern">{percentage}%</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  {modalModel.feature_importance.length > 8 && (
                    <div className="features-more-modern">
                      + {modalModel.feature_importance.length - 8} more features
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

        </div>

        {/* Footer Actions */}
        <div className="modal-footer-modern">
          <div className="actions-container-modern">
            <button 
              className="action-button-modern primary" 
              onClick={handleDownloadMojo} 
              disabled={mojoDownloadLoading}
            >
              <div className="button-content-modern">
                
                <span className="button-text-modern">
                  {mojoDownloadLoading ? 'Downloading...' : 'Download MOJO'}
                </span>
              </div>
              {mojoDownloadLoading && <div className="button-spinner-modern"></div>}
            </button>
            
            {modalModel.mlflow_run_id && (
              <a
                href={`http://localhost:5000/#/experiments/1/runs/${modalModel.mlflow_run_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="action-button-modern secondary"
                onClick={(e) => {
                
                  const currentUrl = e.currentTarget.href;
                  if (currentUrl.includes('/experiments/1/')) {
                    setTimeout(() => {
                      window.open(`http://localhost:5000/#/experiments/0/runs/${modalModel.mlflow_run_id}`, '_blank');
                    }, 1000);
                  }
                }}
              >
                <div className="button-content-modern">
                  <span className="button-icon-modern">🔬</span>
                  <span className="button-text-modern">View in MLflow</span>
                </div>
                <span className="external-icon-modern">↗</span>
              </a>
            )}
          </div>
          
          {mojoDownloadError && (
            <div className="error-message-modern">
              
              <span>{mojoDownloadError}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelDetailsModal;
