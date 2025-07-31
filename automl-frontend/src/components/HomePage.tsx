import React, { useState } from 'react';
import DatasetSummaryModal from './DatasetSummaryModal';
import Navbar from './Navbar';
import '../modal.css';

interface DatasetPreview {
  columns: string[];
  rows: Record<string, any>[];
}

interface HomePageProps {
  onDatasetUploaded: (data: {
    file: File;
    datasetPreview: DatasetPreview;
    selectedTargetColumn: string;
    selectedProblemType: string; // Add problem type
    datasetSummary: any;
  }) => void;
}

const HomePage: React.FC<HomePageProps> = ({ onDatasetUploaded }) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null);
  const [selectedTargetColumn, setSelectedTargetColumn] = useState<string>('');
  const [selectedProblemType, setSelectedProblemType] = useState<string>(''); // Add problem type state
  const [message, setMessage] = useState('');
  
  // Dataset summary modal state
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [datasetSummary, setDatasetSummary] = useState<any>(null);

  // Ref for the Configure button
  const configureBtnRef = React.useRef<HTMLButtonElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setUploadSuccess(false);
      setDatasetPreview(null);
      setSelectedTargetColumn(''); // Reset target column selection
      setSelectedProblemType(''); // Reset problem type selection
      
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
    if (!file || !selectedTargetColumn || !selectedProblemType) return;
    setUploading(true);
    setMessage('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('target_column', selectedTargetColumn); // Send target column to backend

      const res = await fetch('http://localhost:8000/upload-dataset/', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      setMessage(data.message || 'File uploaded successfully.');
      setUploadSuccess(true);
      setDatasetSummary(data); // Save summary for modal
      setShowSummaryModal(true); // Open modal after upload
    } catch (error) {
      console.error(error);
      setMessage('Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  const handleProceedToTraining = () => {
    if (file && datasetPreview && selectedTargetColumn && selectedProblemType) {
      onDatasetUploaded({
        file,
        datasetPreview,
        selectedTargetColumn,
        selectedProblemType, // Include problem type
        datasetSummary
      });
    }
  };

  return (
    <>
      {/* Fixed Navigation Bar */}
      <Navbar currentPage="/" />

      <div className="App">
        {/* Hero Section */}
        <section className="hero-section">
          <h1 className="hero-title">Transform Data Into Intelligence</h1>
          <p className="hero-subtitle">
            Advanced AutoML platform that automatically builds, optimizes, and deploys 
            machine learning models from your CSV data with enterprise-grade accuracy and performance.
          </p>
        </section>

        {/* File Upload Section */}
        <section className="upload-section">
          <h2>Dataset Upload</h2>
          <div className="upload-controls" style={{display: 'flex', alignItems: 'center', gap: '0.5em'}}>
            <input 
              type="file" 
              accept=".csv" 
              onChange={handleFileChange}
              className="file-input"
            />
            <button 
            onClick={handleUpload} 
            disabled={uploading || !file || !selectedTargetColumn || !selectedProblemType}
            className={`btn ${uploadSuccess ? 'btn-success' : 'btn-primary'}`}
          >
            {uploading ? 'Uploading...' : uploadSuccess ? 'Uploaded' : 'Upload Dataset'}
          </button>
          {/* Info icon with yellow circle */}
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              marginLeft: '0.5em',
              cursor: 'pointer',
              borderRadius: '50%',
              background: '#ffe066',
              width: 28,
              height: 28,
              justifyContent: 'center',
              fontWeight: 'bold',
              fontSize: 18,
              color: '#b8860b',
              boxShadow: '0 0 0 2px #ffe066',
            }}
            title="Show last dataset summary"
            onClick={() => datasetSummary && setShowSummaryModal(true)}
          >
            i
          </span>
        </div>
      </section>

      {/* Dataset Preview */}
      {datasetPreview && (
        <section className="preview-section">
          <h3> Dataset Preview &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          <strong>Columns:</strong> {datasetPreview.columns.length}
          </h3>
          {/* Target Column Selection */}
          <div style={{marginBottom: '1em', padding: '1em', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9'}}>
            <h4 style={{margin: '0 0 0.5em 0', color: '#333'}}> Select Target Column</h4>
            <p style={{margin: '0 0 0.5em 0', fontSize: '14px', color: '#666'}}>
              Choose the column you want to predict (target variable)
            </p>
            <select 
              value={selectedTargetColumn} 
              onChange={(e) => setSelectedTargetColumn(e.target.value)}
              style={{
                width: '100%',
                maxWidth: '300px',
                padding: '8px 12px',
                border: '1px solid #ccc',
                borderRadius: '4px',
                fontSize: '14px'
              }}
            >
              <option value="">-- Select Target Column --</option>
              {datasetPreview.columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
            {selectedTargetColumn && (
              <div style={{margin: '0.5em 0 0 0', fontSize: '12px'}}>
                <p style={{margin: '0', color: '#007bff'}}>
                   Selected: <strong>{selectedTargetColumn}</strong>
                </p>
              </div>
            )}
            {!selectedTargetColumn && (
              <p style={{margin: '0.5em 0 0 0', fontSize: '12px', color: '#dc3545'}}>
                Please select a target column to enable upload
              </p>
            )}
          </div>

          {/* Problem Type Selection - Only show when target column is selected */}
          {selectedTargetColumn && (
            <div style={{marginBottom: '1em', padding: '1em', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f0f8ff'}}>
              <h4 style={{margin: '0 0 0.5em 0', color: '#333'}}> Select Problem Type</h4>
              <p style={{margin: '0 0 0.5em 0', fontSize: '14px', color: '#666'}}>
                What type of machine learning problem is this?
              </p>
              <div style={{display: 'flex', gap: '1em', flexWrap: 'wrap'}}>
                <label style={{display: 'flex', alignItems: 'center', cursor: 'pointer', padding: '8px 12px', border: '1px solid #ccc', borderRadius: '4px', backgroundColor: selectedProblemType === 'classification' ? '#e3f2fd' : 'white'}}>
                  <input 
                    type="radio" 
                    value="classification" 
                    checked={selectedProblemType === 'classification'}
                    onChange={(e) => setSelectedProblemType(e.target.value)}
                    style={{marginRight: '8px'}}
                  />
                  <span> Classification</span>
                </label>
                <label style={{display: 'flex', alignItems: 'center', cursor: 'pointer', padding: '8px 12px', border: '1px solid #ccc', borderRadius: '4px', backgroundColor: selectedProblemType === 'regression' ? '#e3f2fd' : 'white'}}>
                  <input 
                    type="radio" 
                    value="regression" 
                    checked={selectedProblemType === 'regression'}
                    onChange={(e) => setSelectedProblemType(e.target.value)}
                    style={{marginRight: '8px'}}
                  />
                  <span> Regression</span>
                </label>
              </div>
              {selectedProblemType && (
                <div style={{margin: '0.5em 0 0 0', fontSize: '12px'}}>
                  <p style={{margin: '0', color: '#007bff'}}>
                     Selected: <strong>{selectedProblemType === 'classification' ? 'Classification (predicting categories)' : 'Regression (predicting numbers)'}</strong>
                  </p>
                </div>
              )}
              {!selectedProblemType && (
                <p style={{margin: '0.5em 0 0 0', fontSize: '12px', color: '#dc3545'}}>
                   Please select a problem type to enable upload
                </p>
              )}
            </div>
          )}

          <div className="table-container">
            <table className="preview-table">
              <thead>
                <tr>
                  {datasetPreview.columns.map((col) => (
                    <th key={col} style={{backgroundColor: selectedTargetColumn === col ? '#e3f2fd' : 'inherit'}}>
                      {col} {selectedTargetColumn === col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {datasetPreview.rows.map((row, index) => (
                  <tr key={index}>
                    {datasetPreview.columns.map((col) => (
                      <td key={col} style={{backgroundColor: selectedTargetColumn === col ? '#f3f9ff' : 'inherit'}}>
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

      {/* Proceed to Training */}
      {uploadSuccess && (
        <section className="proceed-section">
          <h2>Ready for Training</h2>
          <p>Your dataset has been uploaded successfully. Proceed to configure training settings.</p>
          <button
            ref={configureBtnRef}
            onClick={handleProceedToTraining}
            className="btn btn-primary btn-large"
          >
            Configure Training & Models
          </button>
        </section>
      )}

      {/* Status Messages */}
      {message && (
        <div className={`message ${uploadSuccess ? 'message-success' : 'message-info'}`}>
          {message}
        </div>
      )}

      {/* How It Works Section */}
      <section className="how-it-works-section">
        <h2 className="section-title">How It Works</h2>
        <div className="steps-container">
          <div className="step-item">
            <div className="step-number">1</div>
            <h3 className="step-title">Upload Dataset</h3>
            <p className="step-description">
              Upload your CSV file and get instant data preview with statistical insights
            </p>
          </div>
          <div className="step-item">
            <div className="step-number">2</div>
            <h3 className="step-title">Configure Target</h3>
            <p className="step-description">
              Select your target column and choose between classification or regression
            </p>
          </div>
          <div className="step-item">
            <div className="step-number">3</div>
            <h3 className="step-title">AutoML Training</h3>
            <p className="step-description">
              Our AI automatically builds and optimizes multiple models for best performance
            </p>
          </div>
          <div className="step-item">
            <div className="step-number">4</div>
            <h3 className="step-title">View Leaderboard</h3>
            <p className="step-description">
              Compare all trained models in an interactive leaderboard with performance rankings
            </p>
          </div>
          <div className="step-item">
            <div className="step-number">5</div>
            <h3 className="step-title">MLflow Integration</h3>
            <p className="step-description">
              Track experiments and model versions with integrated MLflow UI for complete lifecycle management
            </p>
          </div>
          <div className="step-item">
            <div className="step-number">6</div>
            <h3 className="step-title">Get Results</h3>
            <p className="step-description">
              Download your trained model with detailed performance metrics and insights
            </p>
          </div>
        </div>
      </section>

      {/* Intelligent Features Section */}
      <section className="features-section">
        <h2 className="section-title">Powerful Features</h2>
        <p className="section-subtitle">
          Everything you need to build world-class machine learning models without the complexity
        </p>
        <div className="features-grid">
          <div className="feature-card">
          
            <h3 className="feature-title">Intelligent AutoML</h3>
            <p className="feature-description">
              Advanced algorithms automatically select, train, and tune multiple machine learning 
              models including XGBoost, Random Forest, Neural Networks, and more to find the 
              optimal solution for your data.
            </p>
          </div>
          <div className="feature-card">
          
            <h3 className="feature-title">Robust & User-Friendly</h3>
            <p className="feature-description">
              Built with simplicity in mind, our platform provides a robust and intuitive interface 
              that makes machine learning accessible to everyone, regardless of technical expertise.
            </p>
          </div>
          <div className="feature-card">
            <h3 className="feature-title">Comprehensive Analytics</h3>
            <p className="feature-description">
              Get detailed data profiling, feature importance analysis, model performance metrics, and 
              interactive visualizations to understand your results.
            </p>
          </div>
          <div className="feature-card">
           
            <h3 className="feature-title">Export & Deploy</h3>
            <p className="feature-description">
              Download trained models in multiple formats (MOJO, POJO, Python) for easy integration 
              into your existing applications and production systems.
            </p>
          </div>
        </div>
      </section>

      {/* Dataset Summary Modal */}
      <DatasetSummaryModal
        datasetSummary={datasetSummary}
        showModal={showSummaryModal}
        onClose={() => {
          setShowSummaryModal(false);
          // Ensure body scroll is restored
          document.body.style.overflow = 'unset';
          // Scroll to Configure button after closing modal
          setTimeout(() => {
            if (configureBtnRef.current) {
              configureBtnRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
          }, 100);
        }}
        selectedTargetColumn={selectedTargetColumn}
        selectedProblemType={selectedProblemType}
      />
      </div>
    </>
  );
};

export default HomePage;
