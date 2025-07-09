import React, { useState } from 'react';

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [preview, setPreview] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setMessage('');
    setError('');
    setPreview(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      // 1. Upload file (you can skip this if preview directly works)
      const uploadResponse = await fetch('http://localhost:8000/upload-dataset', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        const data = await uploadResponse.json();
        setError(data.detail || 'Upload failed');
        return;
      }

      setMessage('Upload successful! Generating preview...');
      setError('');

      // 2. Generate preview
      const previewForm = new FormData();
      previewForm.append('file', file);

      const previewResponse = await fetch('http://localhost:8000/preview', {
        method: 'POST',
        body: previewForm,
      });

      const previewData = await previewResponse.json();

      if (previewResponse.ok) {
        setPreview(previewData);
      } else {
        setError(previewData.detail || 'Failed to get preview');
      }

    } catch (err) {
      setError('Error connecting to server');
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Upload Dataset (CSV)</h2>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <br /><br />
      <button onClick={handleUpload}>Upload and Preview</button>
      <br /><br />
      {message && <div style={{ color: 'green' }}>{message}</div>}
      {error && <div style={{ color: 'red' }}>{error}</div>}

      {preview && (
        <div style={{ marginTop: 20 }}>
          <h3>Dataset Preview</h3>
          <table border="1" cellPadding="8">
            <thead>
              <tr>
                {preview.columns.map((col, index) => (
                  <th key={index}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {preview.columns.map((col, colIndex) => (
                    <td key={colIndex}>{row[col]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default UploadForm;
