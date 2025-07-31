import React, { useState, useEffect } from 'react';
import { FaTrophy, FaMedal, FaAward, FaRobot, FaCheckCircle, FaMapPin, FaChartLine, FaBullseye, FaRocket, FaEye } from 'react-icons/fa';
import Navbar from './Navbar';
import ModelDetailsModal from './ModelDetailsModal';

interface Model {
  model_id: string;
  mlflow_run_id?: string;
  feature_importance?: any;
  frontend_id?: string;
  [key: string]: any;
}

interface ModelOption {
  id: string;
  name: string;
  shortName: string;
}

interface AdvancedSettings {
  cvFolds: number;
  trainSplit: number;
  maxRuntime: number;
}

interface DeployScreenProps {
  leaderboard: Model[];
  selectedModels: string[];
  modelOptions: ModelOption[];
  selectedModel: string;
  message: string;
  advancedSettings: AdvancedSettings;
  detectedProblemType: string;
  onModelSelect: (modelId: string) => void;
  onBackToHome: () => void;
  onGoToLiveModel?: () => void;
  onNavigate?: (route: string) => void;
}

const getModelIcon = (modelId: string): React.ReactElement => {
  const icons: Record<string, React.ReactElement> = {
    'xgboost': <FaRocket />,
    'randomforest': <FaTrophy />,
    'gbm': <FaMedal />,
    'glm': <FaChartLine />,
    'neural': <FaAward />,
    'ensemble': <FaBullseye />
  };
  return icons[modelId] || <FaRobot />;
};

const DeployScreen: React.FC<DeployScreenProps> = ({
  leaderboard,
  selectedModels,
  modelOptions,
  selectedModel,
  message,
  advancedSettings,
  detectedProblemType,
  onModelSelect,
  onBackToHome,
  onGoToLiveModel,
  onNavigate
}) => {
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [modalModel, setModalModel] = useState<Model | null>(null);
  const [showDemoModal, setShowDemoModal] = useState(false);

  // Filter leaderboard to only show results from selected models
  const filteredLeaderboard = leaderboard.filter(model => {
    // First check if the model has a frontend_id that matches selected models
    if (model.frontend_id && selectedModels.includes(model.frontend_id)) {
      return true;
    }
    
    // Fallback to name matching for backward compatibility
    return selectedModels.some(selectedModelId => {
      const modelOption = modelOptions.find(m => m.id === selectedModelId);
      if (!modelOption) return false;
      
      const modelIdLower = model.model_id.toLowerCase();
      const modelNameLower = modelOption.name.toLowerCase();
      const modelOptionIdLower = modelOption.id.toLowerCase();
      
      return modelIdLower.includes(modelNameLower) ||
             modelIdLower.includes(modelOptionIdLower) ||
             modelNameLower.includes(modelIdLower);
    });
  });

  const formatValue = (value: any): string => {
    if (typeof value === 'number') {
      return value.toFixed(4);
    }
    return String(value);
  };

  // Handler to open modal with model details
  const handleViewDetails = (e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (!selectedModel) return;
    // Find the selected model in leaderboard
    const model = leaderboard.find(m => m.model_id === selectedModel);
    setModalModel(model || null);
    setShowDetailsModal(true);
    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden';
  };

  // Handler to close modal
  const handleCloseModal = (e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    setShowDetailsModal(false);
    // Restore body scroll
    document.body.style.overflow = 'unset';
  };

  // Close modal when clicking outside or pressing Escape
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (showDetailsModal) {
          handleCloseModal();
        }
        if (showDemoModal) {
          setShowDemoModal(false);
        }
      }
    };

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element;
      if (target?.classList.contains('modal-overlay')) {
        if (showDetailsModal) {
          handleCloseModal();
        }
        if (showDemoModal) {
          setShowDemoModal(false);
        }
      }
    };

    if (showDetailsModal || showDemoModal) {
      document.addEventListener('keydown', handleKeyDown);
      document.addEventListener('click', handleClickOutside);
      return () => {
        document.removeEventListener('keydown', handleKeyDown);
        document.removeEventListener('click', handleClickOutside);
        document.body.style.overflow = 'unset';
      };
    }
  }, [showDetailsModal, showDemoModal]);

  return (
    <>
      {/* Fixed Navigation Bar */}
      <Navbar onNavigate={(route) => {
        if (route === '/') {
          onBackToHome();
        } else if (onNavigate) {
          onNavigate(route);
        } else {
          // Fallback navigation using window.location if no onNavigate prop
          window.location.href = route;
        }
      }} currentPage="/deploy" />
      
      <div className="App" style={{ paddingTop: '80px' }}>
        <section className="hero-section">
          <h1 className="hero-title"><FaTrophy /> Training Results</h1>
          <h1 className="hero-subtitle">Model performance leaderboard and deployment options</h1>
        </section>

        {/* Status Messages */}
        {message && (
          <div className={`message ${leaderboard.length > 0 ? 'message-success' : 'message-info'}`}>
            {message}
          </div>
        )}

        {/* Leaderboard */}
        {leaderboard.length > 0 && (
          <section className="leaderboard-section">
            <h2><FaChartLine /> Model Performance Leaderboard</h2>
            <div className="leaderboard-header">
              <p>Results from your selected models ({filteredLeaderboard.length} models trained)</p>
              <div className="trained-models-chips">
                {filteredLeaderboard.map((model, index) => {
                  const modelInfo = modelOptions.find(m => m.name.toLowerCase().includes(model.model_id.toLowerCase()) || 
                    m.id.toLowerCase().includes(model.model_id.toLowerCase()));
                  return (
                    <div key={model.model_id} className={`model-result-chip rank-${index + 1}`}>
                      <span className="chip-icon">{modelInfo ? getModelIcon(modelInfo.id) : <FaRobot />}</span>
                      <span className="chip-name">{model.model_id}</span>
                      <span className="chip-rank">#{index + 1}</span>
                    </div>
                  );
                })}
              </div>
            </div>
            
            <div className="table-container">
              <table className="leaderboard-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    {Object.keys(filteredLeaderboard[0])
                      .filter(col => col !== 'model_id' && col !== 'mlflow_run_id' && col !== 'feature_importance' && col !== 'frontend_id')
                      .map((col) => (
                        <th key={col}>{col.replace(/_/g, ' ').toUpperCase()}</th>
                    ))}
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredLeaderboard.map((model, index) => {
                    const modelInfo = modelOptions.find(m => m.name.toLowerCase().includes(model.model_id.toLowerCase()) || 
                      m.id.toLowerCase().includes(model.model_id.toLowerCase()));
                    return (
                      <tr
                        key={index}
                        className={`${index === 0 ? 'best-model' : ''} ${selectedModel === model.model_id ? 'selected-model' : ''}`}
                      >
                        <td>
                          <div className="rank-cell">
                            {index === 0 && <span className="trophy"><FaTrophy style={{color: '#FFD700'}} /></span>}
                            {index === 1 && <span className="trophy"><FaMedal style={{color: '#C0C0C0'}} /></span>}
                            {index === 2 && <span className="trophy"><FaAward style={{color: '#CD7F32'}} /></span>}
                            <span className="rank-number">#{index + 1}</span>
                          </div>
                        </td>
                        <td>
                          <div className="model-cell">
                            <span className="model-icon">{modelInfo ? getModelIcon(modelInfo.id) : <FaRobot />}</span>
                            <div className="model-info">
                              <div className="model-name">{model.model_id}</div>
                            </div>
                          </div>
                        </td>
                        {Object.entries(model)
                          .filter(([key]) => key !== 'model_id' && key !== 'mlflow_run_id' && key !== 'feature_importance' && key !== 'frontend_id')
                          .map(([key, value], i) => (
                            <td key={i} className="metric-cell">
                              <span className="metric-value">{formatValue(value)}</span>
                              <span className="metric-label">{key.replace(/_/g, ' ')}</span>
                            </td>
                          ))}
                        <td>
                          <button
                            onClick={() => onModelSelect(model.model_id)}
                            className={`btn btn-sm ${selectedModel === model.model_id ? 'btn-success' : 'btn-outline'}`}
                          >
                            {selectedModel === model.model_id ? <><FaCheckCircle /> Selected</> : <><FaMapPin /> Select</>}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Next Steps */}
        {leaderboard.length > 0 && !selectedModel && (
         <section className="hero-section">
            <h4  className="hero-title"><FaMapPin /> Select a Model</h4>
            <h5 className='hero-subtitle'>Click the <strong> Select</strong> button next to a model above to proceed with deployment!</h5>
            <div className="selection-hint">
              <span>Tip: The top-ranked model usually performs best on your dataset.</span>
            </div>
          </section>
        )}

        {selectedModel && (
          <section className="hero-section">
            <h5  className="hero-title"><FaBullseye /> Ready for Deployment</h5>
            <h5  className="hero-subtitle">Model {selectedModel}is selected and ready for the next phase!</h5>
            <div className="deployment-actions">
              <button
                className="btn btn-primary"
                onClick={() => {
                  // Find the selected model details
                  const model = leaderboard.find(m => m.model_id === selectedModel);
                  if (model) {
                    localStorage.setItem('deployedModel', JSON.stringify(model));
                    if (onGoToLiveModel) {
                      onGoToLiveModel();
                    }
                  }
                }}
              >
                <FaRocket /> Deploy Model
              </button>
              <button 
                className="btn btn-outline" 
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  handleViewDetails(e);
                }}
              >
                <FaEye /> View Details
              </button>
            </div>
          </section>
        )}

        {/* Model Details Modal */}
        <ModelDetailsModal
          modalModel={modalModel}
          showModal={showDetailsModal}
          onClose={handleCloseModal}
          filteredLeaderboard={filteredLeaderboard}
          advancedSettings={advancedSettings}
          formatValue={formatValue}
          detectedProblemType={detectedProblemType}
          onGoToLiveModel={onGoToLiveModel}
        />
      </div>
    </>
  );
};

export default DeployScreen;
