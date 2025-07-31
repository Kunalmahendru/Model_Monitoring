import React from 'react';
import { FaRobot, FaSync, FaSpinner, FaRocket, FaTree, FaBolt, FaChartBar, FaBrain, FaBullseye } from 'react-icons/fa';

interface ModelOption {
  id: string;
  name: string;
  shortName: string;
}

interface TrainingProgressModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedModels: string[];
  modelOptions: ModelOption[];
  message: string;
}

const getModelIcon = (modelId: string): React.ReactElement => {
  const icons: Record<string, React.ReactElement> = {
    'xgboost': <FaRocket />,
    'randomforest': <FaTree />,
    'gbm': <FaBolt />,
    'glm': <FaChartBar />,
    'neural': <FaBrain />,
    'ensemble': <FaBullseye />
  };
  return icons[modelId] || <FaRobot />;
};

const TrainingProgressModal: React.FC<TrainingProgressModalProps> = ({
  isOpen,
  onClose,
  selectedModels,
  modelOptions,
  message
}) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content training-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2><FaRobot /> Training Models</h2>
          <p>Please wait while we train your selected models</p>
        </div>

        <div className="modal-body">
          {/* Status Messages */}
          {message && (
            <div className="message message-info">
              {message}
            </div>
          )}

          {/* Training Progress */}
          <div className="training-progress">
            <h3><FaSync className="fa-spin" /> Training in Progress</h3>
            <div className="progress-info">
              <p>Training {selectedModels.length} selected models on your dataset...</p>
              <div className="training-models-list">
                {selectedModels.map(modelId => {
                  const model = modelOptions.find(m => m.id === modelId);
                  return (
                    <div key={modelId} className="training-model-item">
                      <span className="model-icon">{getModelIcon(modelId)}</span>
                      <span className="model-name">{model?.name || modelId}</span>
                      
                    </div>
                  );
                })}
              </div>
              <div className="training-spinner">
                <FaSpinner className="fa-spin" style={{fontSize: '2rem'}} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingProgressModal;
