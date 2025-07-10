import React from 'react';

type Props = {
  selectedAlgos: string[];
  setSelectedAlgos: (algos: string[]) => void;
  onTrain: () => void;
};

const allAlgorithms = ['XGBoost', 'Random Forest', 'GBM', 'Stacked Ensemble', 'GLM'];

const ModelSelectionScreen: React.FC<Props> = ({ selectedAlgos, setSelectedAlgos, onTrain }) => {
  const toggleAlgo = (algo: string) => {
    if (selectedAlgos.includes(algo)) {
      setSelectedAlgos(selectedAlgos.filter(a => a !== algo));
    } else {
      setSelectedAlgos([...selectedAlgos, algo]);
    }
  };

  const selectAll = () => setSelectedAlgos(allAlgorithms);
  const deselectAll = () => setSelectedAlgos([]);

  return (
    <div>
      <h2>Select Models to Train</h2>
      <button onClick={selectAll}>Select All</button>
      <button onClick={deselectAll}>Deselect All</button>
      <ul>
        {allAlgorithms.map(algo => (
          <li key={algo}>
            <label>
              <input
                type="checkbox"
                checked={selectedAlgos.includes(algo)}
                onChange={() => toggleAlgo(algo)}
              />
              {algo}
            </label>
          </li>
        ))}
      </ul>
      <button onClick={onTrain} disabled={selectedAlgos.length === 0}>Train Selected Models</button>
    </div>
  );
};

export default ModelSelectionScreen;
