import React from 'react';

const DownloadPanel: React.FC = () => {
  return (
    <div className="download-section">
      <h2>Download Results</h2>
      <button onClick={() => window.open('http://localhost:8000/download/best_model')}>Download Best Model</button>
      <button onClick={() => window.open('http://localhost:8000/download/predictions')}>Download Predictions</button>
      <button onClick={() => window.open('http://localhost:8000/download/mlflow_logs')}>Download MLflow Logs</button>
    </div>
  );
};

export default DownloadPanel;
