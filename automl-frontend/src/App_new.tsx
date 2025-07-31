import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import HomePage from './components/HomePage';
import TrainingPageNew from './components/TrainingPageNew';
import DeployScreen from './components/DeployScreen';
import LiveModel from './LiveModel';

interface DatasetPreview {
  columns: string[];
  rows: Record<string, any>[];
}

interface DatasetData {
  file: File;
  datasetPreview: DatasetPreview;
  selectedTargetColumn: string;
  selectedProblemType: string; // Add problem type
  datasetSummary: any;
}

interface TrainingData {
  leaderboard: any[];
  selectedModels: string[];
  selectedModel: string;
  message: string;
  advancedSettings: any;
  detectedProblemType: string;
}

// Context for sharing dataset data across routes
const DatasetContext = React.createContext<{
  datasetData: DatasetData | null;
  setDatasetData: (data: DatasetData | null) => void;
  trainingData: TrainingData | null;
  setTrainingData: (data: TrainingData | null) => void;
}>({
  datasetData: null,
  setDatasetData: () => {},
  trainingData: null,
  setTrainingData: () => {}
});

// Home Page Wrapper Component
const HomePageWrapper: React.FC = () => {
  const navigate = useNavigate();
  const { setDatasetData } = React.useContext(DatasetContext);

  const handleDatasetUploaded = (data: DatasetData) => {
    setDatasetData(data);
    navigate('/training');
  };

  return <HomePage onDatasetUploaded={handleDatasetUploaded} />;
};

// Training Page Wrapper Component
const TrainingPageWrapper: React.FC = () => {
  const navigate = useNavigate();
  const { datasetData, setDatasetData, setTrainingData } = React.useContext(DatasetContext);

  const handleBackToHome = () => {
    setDatasetData(null);
    navigate('/');
  };

  const handleGoToLiveModel = () => {
    navigate('/livemodel');
  };

  const handleShowDeployScreen = (trainingData: TrainingData) => {
    setTrainingData(trainingData);
    navigate('/deploy');
  };

  // Redirect to home if no dataset data
  if (!datasetData) {
    return <Navigate to="/" replace />;
  }

  return (
    <TrainingPageNew
      file={datasetData.file}
      datasetPreview={datasetData.datasetPreview}
      selectedTargetColumn={datasetData.selectedTargetColumn}
      selectedProblemType={datasetData.selectedProblemType} // Add problem type
      onBackToHome={handleBackToHome}
      onGoToLiveModel={handleGoToLiveModel}
      onShowDeployScreen={handleShowDeployScreen}
    />
  );
};

// Deploy Screen Wrapper Component
const DeployScreenWrapper: React.FC = () => {
  const navigate = useNavigate();
  const { trainingData, setTrainingData } = React.useContext(DatasetContext);

  const handleBackToHome = () => {
    setTrainingData(null);
    navigate('/');
  };

  const handleGoToLiveModel = () => {
    navigate('/livemodel');
  };

  const handleNavigate = (route: string) => {
    // Handle navigation to any route
    navigate(route);
  };

  const handleModelSelect = (modelId: string) => {
    if (trainingData) {
      setTrainingData({
        ...trainingData,
        selectedModel: modelId
      });
    }
  };

  if (!trainingData) {
    return <Navigate to="/" replace />;
  }

  return (
    <DeployScreen
      leaderboard={trainingData.leaderboard}
      selectedModels={trainingData.selectedModels}
      modelOptions={[]} // You'll need to pass this from training page
      selectedModel={trainingData.selectedModel}
      message={trainingData.message}
      advancedSettings={trainingData.advancedSettings}
      detectedProblemType={trainingData.detectedProblemType}
      onModelSelect={handleModelSelect}
      onBackToHome={handleBackToHome}
      onGoToLiveModel={handleGoToLiveModel}
      onNavigate={handleNavigate}
    />
  );
};

// Live Model Wrapper Component
const LiveModelWrapper: React.FC = () => {
  const navigate = useNavigate();
  const { setDatasetData, setTrainingData } = React.useContext(DatasetContext);

  const handleBackToHome = () => {
    setDatasetData(null);
    setTrainingData(null);
    navigate('/');
  };

  const handleNavigate = (route: string) => {
    // Handle navigation to any route
    navigate(route);
  };

  return <LiveModel onBackToHome={handleBackToHome} onNavigate={handleNavigate} />;
};

// App Content Component (contains the routes)
const AppContent: React.FC = () => {
  return (
    <Routes>
      {/* Home route - default */}
      <Route path="/" element={<HomePageWrapper />} />
      <Route path="/homepage" element={<Navigate to="/" replace />} />
      
      {/* Training route */}
      <Route path="/training" element={<TrainingPageWrapper />} />
      
      {/* Deploy route */}
      <Route path="/deploy" element={<DeployScreenWrapper />} />
      
      {/* Live Model route */}
      <Route path="/livemodel" element={<LiveModelWrapper />} />
      
      {/* Catch all route - redirect to home */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

// Main App Component with Router
const MainApp: React.FC = () => {
  console.log('MainApp component rendering with React Router...');
  const [datasetData, setDatasetData] = useState<DatasetData | null>(null);
  const [trainingData, setTrainingData] = useState<TrainingData | null>(null);

  return (
    <Router>
      <DatasetContext.Provider value={{ 
        datasetData, 
        setDatasetData, 
        trainingData, 
        setTrainingData 
      }}>
        <AppContent />
      </DatasetContext.Provider>
    </Router>
  );
};

export default MainApp;
