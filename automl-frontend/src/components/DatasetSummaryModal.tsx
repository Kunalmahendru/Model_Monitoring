import React from 'react';
import '../modal.css';

interface DatasetSummary {
  rows: number;
  columns: number;
  target?: string;
  problem_type?: string;
  problem_confidence?: string;
  target_analysis?: {
    unique_values: number;
    sample_values?: string[];
  };
  top_missing?: { [key: string]: number };
  warnings?: string[];
}

interface DatasetSummaryModalProps {
  datasetSummary: DatasetSummary | null;
  showModal: boolean;
  onClose: () => void;
  selectedTargetColumn: string;
  selectedProblemType: string;
}

const DatasetSummaryModal: React.FC<DatasetSummaryModalProps> = ({
  datasetSummary,
  showModal,
  onClose,
  selectedTargetColumn,
  selectedProblemType
}) => {
  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleContentClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleClose = () => {
    // Ensure body scroll is restored when modal closes
    document.body.style.overflow = 'unset';
    onClose();
  };

  if (!showModal || !datasetSummary) return null;

  // Add effect to handle body scroll when modal is open
  React.useEffect(() => {
    if (showModal) {
      document.body.style.overflow = 'hidden';
      
      // Add escape key handler
      const handleEscape = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
          handleClose();
        }
      };
      
      document.addEventListener('keydown', handleEscape);
      
      return () => {
        document.removeEventListener('keydown', handleEscape);
        document.body.style.overflow = 'unset';
      };
    }
    
    // Cleanup function to restore scroll when component unmounts or modal closes
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [showModal]);

  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div className="modal-content" onClick={handleContentClick}>
        <div className="modal-header">
          <h2>📊 Dataset Summary</h2>
          <button className="modal-close" onClick={handleClose}>✕</button>
        </div>
        <div className="modal-body">
          <div className="summary-stats">
            <div className="stat-item">
              <span className="stat-label">Rows:</span>
              <span className="stat-value">{datasetSummary.rows?.toLocaleString() || 'N/A'}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Columns:</span>
              <span className="stat-value">{datasetSummary.columns || 'N/A'}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Target Column:</span>
              <span className="stat-value">{selectedTargetColumn || datasetSummary.target || 'Not selected'}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Problem Type:</span>
              <span className="stat-value">
                {selectedProblemType ? 
                  (selectedProblemType === 'classification' ? 'Classification' : 'Regression') : 
                  (datasetSummary.problem_type || 'Not selected')
                }
              </span>
            </div>
          </div>

          {datasetSummary.top_missing && Object.keys(datasetSummary.top_missing).length > 0 && (
            <div className="missing-values-section">
              <h4>Missing Values:</h4>
              <ul className="missing-list">
                {Object.entries(datasetSummary.top_missing).slice(0, 5).map(([column, count]) => (
                  <li key={column}>
                    <strong>{column}:</strong> {count} missing
                  </li>
                ))}
              </ul>
            </div>
          )}

          {datasetSummary.warnings && datasetSummary.warnings.length > 0 && (
            <div className="warnings-section">
              <h4>⚠️ Warnings:</h4>
              <ul className="warnings-list">
                {datasetSummary.warnings.map((warning: string, index: number) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="modal-footer">
            <p>Dataset ready for training. You can proceed to configure feature engineering and model selection.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetSummaryModal;
