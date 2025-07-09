import React, { useState, useEffect } from 'react';

interface ModelInfo {
  name: string;
  accuracy: number;
  training_time: number;
}

const ModelTable: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    const fetchModels = async () => {
      const res = await fetch('http://localhost:8000/models');
      const data = await res.json();
      setModels(data.models);
    };

    fetchModels();
  }, []);

  return (
    <div className="table-section">
      <h2>Model Leaderboard</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Training Time (s)</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model, index) => (
            <tr key={index}>
              <td>{model.name}</td>
              <td>{model.accuracy}</td>
              <td>{model.training_time}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ModelTable;
