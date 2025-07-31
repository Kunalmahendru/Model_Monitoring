import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import Navbar from './components/Navbar';

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

interface LiveModelProps {
  onBackToHome?: () => void;
  onNavigate?: (route: string) => void;
}

const LiveModel: React.FC<LiveModelProps> = ({ onBackToHome, onNavigate }) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [predictionUrl, setPredictionUrl] = useState<string | null>(null);
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null);
  const [predictionPreview, setPredictionPreview] = useState<DatasetPreview | null>(null);

  // Auto-scroll to bottom when predictions are ready
  useEffect(() => {
    if (predictionUrl) {
      setTimeout(() => {
        window.scrollTo({
          top: document.body.scrollHeight,
          behavior: 'smooth'
        });
      }, 100);
    }
  }, [predictionUrl, predictionPreview]);

  // Get model details from localStorage (set by previous page)
  const model = JSON.parse(localStorage.getItem('deployedModel') || '{}') as Model;

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setDatasetPreview(null);
      setPredictionUrl(null);
      // Optionally preview dataset
      await previewDataset(selectedFile);
    }
  };

  const previewDataset = async (file: File) => {
    try {
      // First try to parse locally with PapaParse for better accuracy
      const text = await file.text();
      const result = Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        transformHeader: (header: string) => header.trim(),
      });

      if (result.errors.length > 0) {
        console.warn('Local CSV parsing warnings:', result.errors);
      }

      if (result.data && result.meta.fields) {
        setDatasetPreview({
          columns: result.meta.fields,
          rows: result.data.slice(0, 5) as any[], // Show first 5 rows
        });
        return;
      }

      // Fallback to backend preview if local parsing fails
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
      setMessage('Preview failed.');
    }
  };

  const previewPredictions = async (blob: Blob) => {
    try {
      const text = await blob.text();
      const result = Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        transformHeader: (header: string) => header.trim(),
      });

      if (result.errors.length > 0) {
        console.warn('CSV parsing warnings:', result.errors);
      }

      setPredictionPreview({
        columns: result.meta.fields || [],
        rows: result.data.slice(0, 5) as any[], // Show first 5 rows for preview
      });
    } catch (error) {
      console.error('Failed to preview predictions:', error);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setUploading(true);
    setMessage('');
    setPredictionUrl(null);
    setPredictionPreview(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model_id', model.model_id); // Send the model ID
      
      console.log('Sending prediction request with model_id:', model.model_id);
      
      const res = await fetch('http://localhost:8000/predict-csv/', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Prediction failed: ${res.status} - ${errorText}`);
      }
      
      const blob = await res.blob();
      // Create download link for predictions
      const url = window.URL.createObjectURL(blob);
      setPredictionUrl(url);
      
      // Preview the predictions
      await previewPredictions(blob);
      
      setMessage('Prediction complete! Download the results below.');
    } catch (err: any) {
      console.error('Prediction error:', err);
      setMessage(`Prediction failed: ${err.message || 'Unknown error'}`);
    } finally {
      setUploading(false);
    }
  };

  if (!model.model_id) {
    return (
      <div className="App">
        <header>
          <h1> No Model Deployed</h1>
          <p>Please go back to the training screen and deploy a model first.</p>
        </header>
      </div>
    );
  }

  return (
    <>
      {/* Fixed Navigation Bar */}
      <Navbar 
        currentPage="/live"
        onNavigate={(route) => {
          if (route === '/') {
            onBackToHome && onBackToHome();
          } else if (onNavigate) {
            onNavigate(route);
          } else {
            // Fallback navigation using window.location if no onNavigate prop
            window.location.href = route;
          }
        }} />
      
      <div className="App" style={{ paddingTop: '80px' }}>
        <section className="hero-section">
          
            <h1 className="hero-title"> Live Model Prediction</h1>
         <p className="hero-subtitle">Model: <strong>{model.model_id}</strong></p>
          <p className="hero-subtitle">Upload a CSV file with feature columns (no target column) to get predictions.</p>
        </section>
      <section className="upload-section">
        <h2> Upload Dataset for Prediction</h2>
        <div className="upload-controls">
          <input type="file" accept=".csv" onChange={handleFileChange} className="file-input" />
          <button onClick={handlePredict} disabled={uploading || !file} className="btn btn-primary">
            {uploading ? ' Predicting...' : ' Generate Predictions'}
          </button>
        </div>
        {file && (
          <p style={{marginTop: '0.5em', fontSize: '14px', color: '#666'}}>
            Selected: <strong>{file.name}</strong> ({(file.size / 1024).toFixed(1)} KB)
          </p>
        )}
      </section>
      {datasetPreview && (
        <section className="preview-section">
          <h3> Dataset Preview</h3>
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
                      <td key={col}>
                        {row[col] === null || row[col] === undefined ? (
                          <span style={{color: '#999', fontStyle: 'italic'}}>null</span>
                        ) : (
                          String(row[col])
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      {predictionUrl && (
        <section className="prediction-section">
          <h3>Predictions Ready</h3>
          <p>Your predictions have been generated and are ready for download!</p>
          <div style={{ marginTop: '1.5rem' }}>
            <a href={predictionUrl} download={`predictions_${file?.name || 'dataset'}.csv`} className="btn btn-success">
               Download Predictions CSV
            </a>
          </div>
        </section>
      )}
      {predictionPreview && (
        <section className="preview-section">
          <h3> Prediction Results Preview</h3>
          <p style={{ color: '#64748b', marginBottom: '1rem' }}>
            Showing first 5 rows of predictions (download CSV for complete results)
          </p>
          <div className="table-container">
            <table className="preview-table">
              <thead>
                <tr>
                  {predictionPreview.columns.map((col) => (
                    <th key={col} style={{ 
                      backgroundColor: col.toLowerCase().includes('predict') ? '#e0f2fe' : 'inherit',
                      color: col.toLowerCase().includes('predict') ? '#0277bd' : 'inherit',
                      fontWeight: col.toLowerCase().includes('predict') ? 'bold' : 'normal'
                    }}>
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {predictionPreview.rows.map((row, index) => (
                  <tr key={index}>
                    {predictionPreview.columns.map((col) => (
                      <td key={col} style={{
                        backgroundColor: col.toLowerCase().includes('predict') ? '#f1f8ff' : 'inherit',
                        fontWeight: col.toLowerCase().includes('predict') ? '600' : 'normal',
                        color: col.toLowerCase().includes('predict') ? '#1565c0' : 'inherit'
                      }}>
                        {row[col] === null || row[col] === undefined ? (
                          <span style={{color: '#999', fontStyle: 'italic'}}>null</span>
                        ) : (
                          String(row[col])
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      {message && (
        <div className={`message ${message.includes('complete') ? 'message-success' : message.includes('failed') ? 'message-error' : 'message-info'}`}>
          {message}
        </div>
      )}
      </div>
    </>
  );
};

export default LiveModel;
