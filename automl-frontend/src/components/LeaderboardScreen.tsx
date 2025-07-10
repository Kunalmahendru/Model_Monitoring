// LeaderboardScreen.tsx
import React, { useEffect, useState } from 'react';

interface Model {
  model_id: string;
  [key: string]: any;
}

const LeaderboardScreen: React.FC = () => {
  const [leaderboard, setLeaderboard] = useState<Model[]>([]);
  const [message, setMessage] = useState('');

  useEffect(() => {
    const fetchLeaderboard = async () => {
      try {
        const res = await fetch('http://localhost:8000/get-leaderboard');
        const data = await res.json();
        setLeaderboard(data.leaderboard || []);
        setMessage(data.message || 'Model training complete.');
      } catch (err) {
        console.error(err);
        setMessage('Failed to load leaderboard.');
      }
    };

    fetchLeaderboard();
  }, []);

  const formatValue = (value: any): string =>
    typeof value === 'number' ? value.toFixed(4) : String(value);

  return (
    <div className="App">
      <h2>🏆 Model Leaderboard</h2>
      <p>{message}</p>
      {leaderboard.length > 0 ? (
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              {Object.keys(leaderboard[0]).map(key => (
                <th key={key}>{key.toUpperCase()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {leaderboard.map((model, index) => (
              <tr key={model.model_id}>
                <td>#{index + 1}</td>
                {Object.values(model).map((value, i) => (
                  <td key={i}>{formatValue(value)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>No models found.</p>
      )}
    </div>
  );
};

export default LeaderboardScreen;
