import React, { useState } from 'react';

interface Model {
  model_id: string;
  auc?: number;
  logloss?: number;
  rmse?: number;
  mean_residual_deviance?: number;
  [key: string]: any;
}

interface DatasetPreview {
  columns: string[];
  rows: any[];
}

const modelOptions = [
  { id: 'xgboost', name: 'XGBoost', description: 'H2O XGBoost implementation with gradient boosting' },
  { id: 'randomforest', name: 'Random Forest (DRF)', description: 'H2O Distributed Random Forest algorithm' },
  { id: 'gbm', name: 'Gradient Boosting Machine', description: 'H2O native gradient boosting algorithm' },
  { id: 'glm', name: 'Generalized Linear Model', description: 'H2O GLM for linear and logistic regression' },
  { id: 'neural', name: 'Deep Learning', description: 'H2O deep neural network implementation' },
  { id: 'ensemble', name: 'Stacked Ensemble', description: 'H2O ensemble method combining multiple models' }
];

const getModelIcon = (modelId: string): string => {
  const icons: { [key: string]: string } = {
    'xgboost': '🚀',
    'randomforest': '🌲',
    'gbm': '⚡',
    'glm': '📊',
    'neural': '🧠',
    'ensemble': '🔗'
  };
  return icons[modelId] || '🤖';
};

const getModelTags = (modelId: string): string[] => {
  const tags: { [key: string]: string[] } = {
    'xgboost': ['Boosting', 'Fast', 'H2O Native'],
    'randomforest': ['Ensemble', 'Robust', 'H2O DRF'],
    'gbm': ['Boosting', 'H2O Native', 'Scalable'],
    'glm': ['Linear', 'Interpretable', 'H2O GLM'],
    'neural': ['Deep Learning', 'H2O Native', 'Neural Network'],
    'ensemble': ['Meta-learning', 'H2O Stacked', 'Best Performance']
  };
  return tags[modelId] || ['H2O Algorithm'];
};

const App: React.FC = () => {
  const [currentScreen, setCurrentScreen] = useState<'upload' | 'leaderboard'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [training, setTraining] = useState(false);
  const [message, setMessage] = useState('');
  const [leaderboard, setLeaderboard] = useState<Model[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedModels, setSelectedModels] = useState<string[]>(['xgboost', 'randomforest', 'gbm']);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setUploadSuccess(false);
      setDatasetPreview(null);
      
      // Auto-preview the dataset
      await previewDataset(selectedFile);
    }
  };

  const previewDataset = async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const res = await fetch('http://localhost:8000/preview', {
        method: 'POST',
        body: formData,
      });
      
      const data = await res.json();
      setDatasetPreview(data);
    } catch (error) {
      console.error('Preview failed:', error);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setMessage('');
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('http://localhost:8000/upload-dataset/', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      setMessage(data.message || 'File uploaded successfully.');
      setUploadSuccess(true);
    } catch (error) {
      console.error(error);
      setMessage('Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  const handleTrain = async () => {
    if (!file) {
      setMessage('No file uploaded.');
      return;
    }

    if (selectedModels.length === 0) {
      setMessage('Please select at least one model to train.');
      return;
    }

    setTraining(true);
    setMessage(`Training ${selectedModels.length} selected models... This may take several minutes.`);
    
    // Navigate to leaderboard screen
    setCurrentScreen('leaderboard');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('models', JSON.stringify(selectedModels));

      const res = await fetch('http://localhost:8000/train-model/', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      setMessage(data.message || 'Model training complete.');
      setLeaderboard(data.leaderboard || []);
      
      // Log training info for debugging
      console.log('Training completed:', {
        totalModels: data.leaderboard?.length || 0,
        selectedModels: selectedModels,
        trainingInfo: data.training_info
      });
      
    } catch (error) {
      console.error('Training error:', error);
      setMessage(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setTraining(false);
    }
  };

  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
    setMessage(`Selected model: ${modelId}`);
  };

  const handleModelToggle = (modelId: string) => {
    setSelectedModels(prev => 
      prev.includes(modelId) 
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const formatValue = (value: any): string => {
    if (typeof value === 'number') {
      return value.toFixed(4);
    }
    return String(value);
  };

  const goBackToUpload = () => {
    setCurrentScreen('upload');
    setTraining(false);
    setLeaderboard([]);
    setSelectedModel('');
    setMessage('');
  };

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

  // Upload Screen
  if (currentScreen === 'upload') {
    return (
      <div className="App">
        <header>
          <h1>🤖 AutoML Training Platform</h1>
          <p>Upload your dataset, select models, and train automatically</p>
        </header>

        {/* File Upload Section */}
        <section className="upload-section">
          <h2>📁 Dataset Upload</h2>
          <div className="upload-controls">
            <input 
              type="file" 
              accept=".csv" 
              onChange={handleFileChange}
              className="file-input"
            />
            <button 
              onClick={handleUpload} 
              disabled={uploading || !file}
              className={`btn ${uploadSuccess ? 'btn-success' : 'btn-primary'}`}
            >
              {uploading ? '⏳ Uploading...' : uploadSuccess ? '✅ Uploaded' : '📤 Upload Dataset'}
            </button>
          </div>
        </section>

        {/* Dataset Preview */}
        {datasetPreview && (
          <section className="preview-section">
            <h3>👀 Dataset Preview</h3>
            <p><strong>Columns:</strong> {datasetPreview.columns.length} | <strong>Target:</strong> {datasetPreview.columns[datasetPreview.columns.length - 1]}</p>
            <div className="table-container">
              <table className="preview-table">
                <thead>
                  <tr>
                    {datasetPreview.columns.map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {datasetPreview.rows.map((row, index) => (
                    <tr key={index}>
                      {datasetPreview.columns.map((col) => (
                        <td key={col}>{row[col]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Model Selection */}
        {uploadSuccess && (
          <section className="model-selection">
            <h2>🎯 Model Selection</h2>
            <p>Choose the machine learning models you want to train and compare on your dataset</p>
            
            <div className="model-selection-header">
              <div className="selection-actions">
                <button 
                  className="btn btn-outline btn-sm"
                  onClick={() => setSelectedModels(modelOptions.map(m => m.id))}
                >
                  Select All
                </button>
                <button 
                  className="btn btn-outline btn-sm"
                  onClick={() => setSelectedModels([])}
                >
                  Clear All
                </button>
              </div>
              <div className="selected-indicator">
                <span className="count-badge">{selectedModels.length}</span>
                <span>models selected</span>
              </div>
            </div>

            <div className="modern-model-grid">
              {modelOptions.map((model) => (
                <div 
                  key={model.id}
                  className={`modern-model-card ${selectedModels.includes(model.id) ? 'selected' : ''}`}
                  onClick={() => handleModelToggle(model.id)}
                >
                  <div className="model-status">
                    <div className={`selection-indicator ${selectedModels.includes(model.id) ? 'active' : ''}`}>
                      <div className="checkmark">
                        <svg viewBox="0 0 24 24" width="16" height="16">
                          <path fill="currentColor" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
                        </svg>
                      </div>
                    </div>
                  </div>
                  
                  <div className="model-content">
                    <div className="model-icon">
                      {getModelIcon(model.id)}
                    </div>
                    <h4 className="model-name">{model.name}</h4>
                    <p className="model-description">{model.description}</p>
                    
                    <div className="model-tags">
                      {getModelTags(model.id).map(tag => (
                        <span key={tag} className="model-tag">{tag}</span>
                      ))}
                    </div>
                  </div>
                  
                  <div className="selection-overlay">
                    <span className="selection-text">
                      {selectedModels.includes(model.id) ? 'Selected ✓' : 'Click to select'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
            
            {selectedModels.length > 0 && (
              <div className="selection-summary">
                <div className="summary-content">
                  <h4>Selected Models for Training:</h4>
                  <div className="selected-models-list">
                    {selectedModels.map(modelId => {
                      const model = modelOptions.find(m => m.id === modelId);
                      return (
                        <div key={modelId} className="selected-model-chip">
                          <span className="chip-icon">{getModelIcon(modelId)}</span>
                          <span className="chip-name">{model?.name}</span>
                          <button 
                            className="chip-remove"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleModelToggle(modelId);
                            }}
                          >
                            ×
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </section>
        )}

      {/* Training Section */}
        {uploadSuccess && (
          <section className="training-section">
            <h2>🚀 Start Training</h2>
            <div className="training-controls">
              <button
                onClick={handleTrain}
                disabled={training || selectedModels.length === 0}
                className={`btn btn-primary ${training ? 'btn-loading' : ''}`}
              >
                {training ? '🔄 Training Models...' : `🎯 Train ${selectedModels.length} Selected Models`}
              </button>
              
              {selectedModels.length === 0 && (
                <p className="training-warning">⚠️ Please select at least one model to train</p>
              )}
            </div>
          </section>
        )}

        {/* Status Messages */}
        {message && (
          <div className={`message ${uploadSuccess ? 'message-success' : 'message-info'}`}>
            {message}
          </div>
        )}
      </div>
    );
  }

  // Leaderboard Screen
  return (
    <div className="App">
      <header>
        <h1>🏆 Training Results</h1>
        <p>Model performance leaderboard</p>
        <button onClick={goBackToUpload} className="btn btn-outline back-btn">
          ← Back to Upload
        </button>
      </header>

      {/* Status Messages */}
      {message && (
        <div className={`message ${leaderboard.length > 0 ? 'message-success' : 'message-info'}`}>
          {message}
        </div>
      )}

      {/* Training Progress */}
      {training && (
        <section className="training-progress">
          <h2>🔄 Training in Progress</h2>
          <div className="progress-info">
            <p>Training {selectedModels.length} selected models on your dataset...</p>
            <div className="training-models-list">
              {selectedModels.map(modelId => {
                const model = modelOptions.find(m => m.id === modelId);
                return (
                  <div key={modelId} className="training-model-item">
                    <span className="training-icon">{getModelIcon(modelId)}</span>
                    <span className="training-name">{model?.name}</span>
                    <div className="training-spinner-small">
                      <div className="spinner-small"></div>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="training-spinner">
              <div className="spinner"></div>
            </div>
          </div>
        </section>
      )}

      {/* Leaderboard */}
      {leaderboard.length > 0 && (
        <section className="leaderboard-section">
          <h2>📊 Model Performance Leaderboard</h2>
          <div className="leaderboard-header">
            <p>Results from your selected models ({filteredLeaderboard.length} models trained)</p>
            <div className="trained-models-chips">
              {filteredLeaderboard.map((model, index) => {
                const modelInfo = modelOptions.find(m => m.name.toLowerCase().includes(model.model_id.toLowerCase()) || 
                  m.id.toLowerCase().includes(model.model_id.toLowerCase()));
                return (
                  <div key={model.model_id} className={`model-result-chip rank-${index + 1}`}>
                    <span className="chip-icon">{modelInfo ? getModelIcon(modelInfo.id) : '🤖'}</span>
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
                  {Object.keys(filteredLeaderboard[0]).filter(col => col !== 'model_id').map((col) => (
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
                          {index === 0 && <span className="trophy">🥇</span>}
                          {index === 1 && <span className="trophy">🥈</span>}
                          {index === 2 && <span className="trophy">🥉</span>}
                          <span className="rank-number">#{index + 1}</span>
                        </div>
                      </td>
                      <td>
                        <div className="model-cell">
                          <span className="model-icon">{modelInfo ? getModelIcon(modelInfo.id) : '🤖'}</span>
                          <div className="model-info">
                            <div className="model-name">{model.model_id}</div>
                            <div className="model-type">{modelInfo?.name || 'Unknown Model'}</div>
                          </div>
                        </div>
                      </td>
                      {Object.entries(model).filter(([key]) => key !== 'model_id').map(([key, value], i) => (
                        <td key={i} className="metric-cell">
                          <span className="metric-value">{formatValue(value)}</span>
                          <span className="metric-label">{key.replace(/_/g, ' ')}</span>
                        </td>
                      ))}
                      <td>
                        <button
                          onClick={() => handleModelSelect(model.model_id)}
                          className={`btn btn-sm ${selectedModel === model.model_id ? 'btn-success' : 'btn-outline'}`}
                        >
                          {selectedModel === model.model_id ? '✅ Selected' : '📌 Select'}
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
      {selectedModel && (
        <section className="next-steps">
          <h3>🎯 Ready for Deployment</h3>
          <p>Model <strong>{selectedModel}</strong> is selected and ready for the next phase!</p>
          <div className="deployment-actions">
            <button className="btn btn-primary">🚀 Deploy Model</button>
            <button className="btn btn-outline">📊 View Details</button>
          </div>
        </section>
      )}
    </div>
  );
};

export default App;

// import React, { useState } from 'react';
// import './App.css';

// interface Model {
//   model_id: string;
//   auc?: number;
//   logloss?: number;
//   rmse?: number;
//   mean_residual_deviance?: number;
//   [key: string]: any;
// }

// interface DatasetPreview {
//   columns: string[];
//   rows: any[];
// }

// interface PredictionInput {
//   [key: string]: string | number;
// }

// interface PredictionResult {
//   prediction: number | string;
//   confidence?: number;
//   model_used: string;
// }

// const App: React.FC = () => {
//   const [file, setFile] = useState<File | null>(null);
//   const [training, setTraining] = useState(false);
//   const [message, setMessage] = useState('');
//   const [leaderboard, setLeaderboard] = useState<Model[]>([]);
//   const [uploading, setUploading] = useState(false);
//   const [uploadSuccess, setUploadSuccess] = useState(false);
//   const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null);
//   const [selectedModel, setSelectedModel] = useState<string>('');
  
//   // Prediction states
//   const [predictionInputs, setPredictionInputs] = useState<PredictionInput>({});
//   const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
//   const [predicting, setPredicting] = useState(false);
//   const [predictionFile, setPredictionFile] = useState<File | null>(null);
//   const [batchPredicting, setBatchPredicting] = useState(false);
//   const [batchResults, setBatchResults] = useState<any[]>([]);

//   const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
//     if (e.target.files && e.target.files[0]) {
//       const selectedFile = e.target.files[0];
//       setFile(selectedFile);
//       setUploadSuccess(false);
//       setDatasetPreview(null);
      
//       // Auto-preview the dataset
//       await previewDataset(selectedFile);
//     }
//   };

//   const previewDataset = async (file: File) => {
//     try {
//       const formData = new FormData();
//       formData.append('file', file);
      
//       const res = await fetch('http://localhost:8000/preview', {
//         method: 'POST',
//         body: formData,
//       });
      
//       const data = await res.json();
//       setDatasetPreview(data);
//     } catch (error) {
//       console.error('Preview failed:', error);
//     }
//   };

//   const handleUpload = async () => {
//     if (!file) return;
//     setUploading(true);
//     setMessage('');
//     try {
//       const formData = new FormData();
//       formData.append('file', file);

//       const res = await fetch('http://localhost:8000/upload-dataset/', {
//         method: 'POST',
//         body: formData,
//       });

//       const data = await res.json();
//       setMessage(data.message || 'File uploaded successfully.');
//       setUploadSuccess(true);
//     } catch (error) {
//       console.error(error);
//       setMessage('Upload failed.');
//     } finally {
//       setUploading(false);
//     }
//   };

//   const handleTrain = async () => {
//     if (!file) {
//       setMessage('No file uploaded.');
//       return;
//     }

//     setTraining(true);
//     setMessage('Training in progress... This may take several minutes.');

//     try {
//       const formData = new FormData();
//       formData.append('file', file);

//       const res = await fetch('http://localhost:8000/train-model/', {
//         method: 'POST',
//         body: formData,
//       });

//       const data = await res.json();
//       setMessage(data.message || 'Model training complete.');
//       setLeaderboard(data.leaderboard || []);
      
//       // Auto-select the best model
//       if (data.leaderboard && data.leaderboard.length > 0) {
//         setSelectedModel(data.leaderboard[0].model_id);
//       }
//     } catch (error) {
//       console.error(error);
//       setMessage('Training failed.');
//     } finally {
//       setTraining(false);
//     }
//   };

//   const handleModelSelect = (modelId: string) => {
//     setSelectedModel(modelId);
//     setMessage(`Selected model: ${modelId}`);
//   };

//   const handlePredictionInputChange = (column: string, value: string) => {
//     setPredictionInputs(prev => ({
//       ...prev,
//       [column]: value
//     }));
//   };

//   const handleSinglePrediction = async () => {
//     if (!selectedModel || !datasetPreview) {
//       setMessage('Please select a model and ensure dataset is loaded.');
//       return;
//     }

//     setPredicting(true);
//     try {
//       const response = await fetch('http://localhost:8000/predict/', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//           model_id: selectedModel,
//           inputs: predictionInputs
//         }),
//       });

//       const data = await response.json();
//       setPredictionResult(data);
//       setMessage('Prediction completed successfully!');
//     } catch (error) {
//       console.error('Prediction failed:', error);
//       setMessage('Prediction failed.');
//     } finally {
//       setPredicting(false);
//     }
//   };

//   const handlePredictionFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
//     if (e.target.files && e.target.files[0]) {
//       setPredictionFile(e.target.files[0]);
//     }
//   };

//   const handleBatchPrediction = async () => {
//     if (!predictionFile || !selectedModel) {
//       setMessage('Please select a model and upload a prediction file.');
//       return;
//     }

//     setBatchPredicting(true);
//     try {
//       const formData = new FormData();
//       formData.append('file', predictionFile);
//       formData.append('model_id', selectedModel);

//       const response = await fetch('http://localhost:8000/batch-predict/', {
//         method: 'POST',
//         body: formData,
//       });

//       const data = await response.json();
//       setBatchResults(data.predictions || []);
//       setMessage('Batch prediction completed successfully!');
//     } catch (error) {
//       console.error('Batch prediction failed:', error);
//       setMessage('Batch prediction failed.');
//     } finally {
//       setBatchPredicting(false);
//     }
//   };

//   const downloadBatchResults = () => {
//     if (batchResults.length === 0) return;
    
//     const csv = [
//       Object.keys(batchResults[0]).join(','),
//       ...batchResults.map(row => Object.values(row).join(','))
//     ].join('\n');
    
//     const blob = new Blob([csv], { type: 'text/csv' });
//     const url = window.URL.createObjectURL(blob);
//     const a = document.createElement('a');
//     a.href = url;
//     a.download = 'predictions.csv';
//     a.click();
//     window.URL.revokeObjectURL(url);
//   };

//   const formatValue = (value: any): string => {
//     if (typeof value === 'number') {
//       return value.toFixed(4);
//     }
//     return String(value);
//   };

//   // Get feature columns (excluding target column)
//   const getFeatureColumns = () => {
//     if (!datasetPreview) return [];
//     return datasetPreview.columns.slice(0, -1); // Exclude last column (target)
//   };

//   const getBestModel = () => {
//     return leaderboard.length > 0 ? leaderboard[0] : null;
//   };

//   return (
//     <div className="App">
//       <header>
//         <h1>🤖 AutoML Training Platform</h1>
//         <p>Upload your dataset, train models automatically, and deploy the best performer</p>
//       </header>

//       {/* File Upload Section */}
//       <section className="upload-section">
//         <h2>📁 Dataset Upload</h2>
//         <div className="upload-controls">
//           <input 
//             type="file" 
//             accept=".csv" 
//             onChange={handleFileChange}
//             className="file-input"
//           />
//           <button 
//             onClick={handleUpload} 
//             disabled={uploading || !file}
//             className={`btn ${uploadSuccess ? 'btn-success' : 'btn-primary'}`}
//           >
//             {uploading ? '⏳ Uploading...' : uploadSuccess ? '✅ Uploaded' : '📤 Upload Dataset'}
//           </button>
//         </div>
//       </section>

//       {/* Dataset Preview */}
//       {datasetPreview && (
//         <section className="preview-section">
//           <h3>👀 Dataset Preview</h3>
//           <p><strong>Columns:</strong> {datasetPreview.columns.length} | <strong>Target:</strong> {datasetPreview.columns[datasetPreview.columns.length - 1]}</p>
//           <div className="table-container">
//             <table className="preview-table">
//               <thead>
//                 <tr>
//                   {datasetPreview.columns.map((col) => (
//                     <th key={col}>{col}</th>
//                   ))}
//                 </tr>
//               </thead>
//               <tbody>
//                 {datasetPreview.rows.map((row, index) => (
//                   <tr key={index}>
//                     {datasetPreview.columns.map((col) => (
//                       <td key={col}>{row[col]}</td>
//                     ))}
//                   </tr>
//                 ))}
//               </tbody>
//             </table>
//           </div>
//         </section>
//       )}

//       {/* Training Section */}
//       <section className="training-section">
//         <h2>🚀 Model Training</h2>
//         <button
//           onClick={handleTrain}
//           disabled={training || !uploadSuccess}
//           className={`btn btn-primary ${training ? 'btn-loading' : ''}`}
//         >
//           {training ? '🔄 Training Models...' : '🎯 Train AutoML Models'}
//         </button>
//       </section>

//       {/* Status Messages */}
//       {message && (
//         <div className={`message ${uploadSuccess || leaderboard.length > 0 ? 'message-success' : 'message-info'}`}>
//           {message}
//         </div>
//       )}

//       {/* Leaderboard */}
//       {leaderboard.length > 0 && (
//         <section className="leaderboard-section">
//           <h2>🏆 Model Leaderboard</h2>
//           <p>Click on a model to select it for deployment</p>
//           <div className="table-container">
//             <table className="leaderboard-table">
//               <thead>
//                 <tr>
//                   <th>Rank</th>
//                   {Object.keys(leaderboard[0]).map((col) => (
//                     <th key={col}>{col.replace(/_/g, ' ').toUpperCase()}</th>
//                   ))}
//                   <th>Action</th>
//                 </tr>
//               </thead>
//               <tbody>
//                 {leaderboard.map((model, index) => (
//                   <tr
//                     key={index}
//                     className={`${index === 0 ? 'best-model' : ''} ${selectedModel === model.model_id ? 'selected-model' : ''}`}
//                   >
//                     <td>
//                       {index === 0 && <span className="trophy">🥇</span>}
//                       {index === 1 && <span className="trophy">🥈</span>}
//                       {index === 2 && <span className="trophy">🥉</span>}
//                       #{index + 1}
//                     </td>
//                     {Object.values(model).map((value, i) => (
//                       <td key={i}>{formatValue(value)}</td>
//                     ))}
//                     <td>
//                       <button
//                         onClick={() => handleModelSelect(model.model_id)}
//                         className={`btn btn-sm ${selectedModel === model.model_id ? 'btn-success' : 'btn-outline'}`}
//                       >
//                         {selectedModel === model.model_id ? '✅ Selected' : '📌 Select'}
//                       </button>
//                     </td>
//                   </tr>
//                 ))}
//               </tbody>
//             </table>
//           </div>
//         </section>
//       )}

//       {/* Prediction Section */}
//       {selectedModel && datasetPreview && (
//         <section className="prediction-section">
//           <h2>🔮 Make Predictions</h2>
//           <div className="prediction-container">
            
//             {/* Single Prediction */}
//             <div className="prediction-card">
//               <h3>📊 Single Prediction</h3>
//               <p>Enter values for each feature to get a prediction from the selected model: <strong>{selectedModel}</strong></p>
              
//               <div className="prediction-inputs">
//                 {getFeatureColumns().map((column) => (
//                   <div key={column} className="input-group">
//                     <label>{column}:</label>
//                     <input
//                       type="number"
//                       step="any"
//                       value={predictionInputs[column] || ''}
//                       onChange={(e) => handlePredictionInputChange(column, e.target.value)}
//                       placeholder={`Enter ${column}`}
//                     />
//                   </div>
//                 ))}
//               </div>
              
//               <button
//                 onClick={handleSinglePrediction}
//                 disabled={predicting || getFeatureColumns().some(col => !predictionInputs[col])}
//                 className={`btn btn-primary ${predicting ? 'btn-loading' : ''}`}
//               >
//                 {predicting ? '🔄 Predicting...' : '🎯 Predict'}
//               </button>
              
//               {predictionResult && (
//                 <div className="prediction-result">
//                   <h4>🎯 Prediction Result</h4>
//                   <div className="result-card">
//                     <p><strong>Prediction:</strong> {formatValue(predictionResult.prediction)}</p>
//                     {predictionResult.confidence && (
//                       <p><strong>Confidence:</strong> {(predictionResult.confidence * 100).toFixed(2)}%</p>
//                     )}
//                     <p><strong>Model Used:</strong> {predictionResult.model_used}</p>
//                   </div>
//                 </div>
//               )}
//             </div>

//             {/* Batch Prediction */}
//             <div className="prediction-card">
//               <h3>📈 Batch Prediction</h3>
//               <p>Upload a CSV file with the same feature columns to get predictions for multiple rows</p>
              
//               <div className="upload-controls">
//                 <input 
//                   type="file" 
//                   accept=".csv" 
//                   onChange={handlePredictionFileChange}
//                   className="file-input"
//                 />
//                 <button 
//                   onClick={handleBatchPrediction} 
//                   disabled={batchPredicting || !predictionFile}
//                   className={`btn btn-primary ${batchPredicting ? 'btn-loading' : ''}`}
//                 >
//                   {batchPredicting ? '🔄 Predicting...' : '📊 Batch Predict'}
//                 </button>
//               </div>
              
//               {batchResults.length > 0 && (
//                 <div className="batch-results">
//                   <div className="results-header">
//                     <h4>📋 Batch Results ({batchResults.length} predictions)</h4>
//                     <button onClick={downloadBatchResults} className="btn btn-sm btn-outline">
//                       📥 Download CSV
//                     </button>
//                   </div>
                  
//                   <div className="table-container">
//                     <table className="results-table">
//                       <thead>
//                         <tr>
//                           {Object.keys(batchResults[0]).map((col) => (
//                             <th key={col}>{col}</th>
//                           ))}
//                         </tr>
//                       </thead>
//                       <tbody>
//                         {batchResults.slice(0, 10).map((row, index) => (
//                           <tr key={index}>
//                             {Object.values(row).map((value, i) => (
//                               <td key={i}>{formatValue(value)}</td>
//                             ))}
//                           </tr>
//                         ))}
//                       </tbody>
//                     </table>
//                     {batchResults.length > 10 && (
//                       <p className="results-footer">
//                         Showing first 10 results. Download CSV to see all {batchResults.length} predictions.
//                       </p>
//                     )}
//                   </div>
//                 </div>
//               )}
//             </div>
//           </div>
//         </section>
//       )}

//       {/* Model Info */}
//       {getBestModel() && (
//         <section className="model-info">
//           <h3>🎯 Best Model Information</h3>
//           <div className="model-card">
//             <p><strong>Model ID:</strong> {getBestModel()?.model_id}</p>
//             <p><strong>Performance:</strong> {Object.entries(getBestModel() || {})
//               .filter(([key]) => key !== 'model_id')
//               .map(([key, value]) => `${key}: ${formatValue(value)}`)
//               .join(', ')}</p>
//             <p><strong>Status:</strong> {selectedModel ? '✅ Ready for predictions' : '⏳ Not selected'}</p>
//           </div>
//         </section>
//       )}
//     </div>
//   );
// };

// export default App;