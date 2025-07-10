// UploadScreen.tsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface DatasetPreview {
  columns: string[];
  rows: any[];
}

const UploadScreen: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null);
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setUploadSuccess(false);
      setDatasetPreview(null);
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
    if (!file) return;

    try {
      const formData = new FormData();
      formData.append('file', file);

      await fetch('http://localhost:8000/train-model/', {
        method: 'POST',
        body: formData,
      });

      // Navigate to leaderboard screen
      navigate('/leaderboard');
    } catch (error) {
      console.error('Training error:', error);
      setMessage('Training failed.');
    }
  };

  return (
    <div className="App">
      <h1>AutoML Training Platform</h1>

      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={uploading || !file}>
        {uploading ? 'Uploading...' : 'Upload Dataset'}
      </button>

      {datasetPreview && (
        <div>
          <h3>Dataset Preview</h3>
          <table>
            <thead>
              <tr>
                {datasetPreview.columns.map(col => <th key={col}>{col}</th>)}
              </tr>
            </thead>
            <tbody>
              {datasetPreview.rows.map((row, i) => (
                <tr key={i}>
                  {datasetPreview.columns.map(col => (
                    <td key={col}>{row[col]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <button onClick={handleTrain} disabled={!uploadSuccess}>
        Train Model
      </button>

      {message && <p>{message}</p>}
    </div>
  );
};

export default UploadScreen;
