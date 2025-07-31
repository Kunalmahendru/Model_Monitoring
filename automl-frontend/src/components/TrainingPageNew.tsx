import React, { useState, useEffect } from 'react';
import { 
  FaRocket, 
  FaTree, 
  FaBolt, 
  FaChartBar, 
  FaBrain, 
  FaBullseye, 
  FaRobot,
  FaCog,
  FaSpinner,
  FaPlay,
  FaMapPin,
  FaChevronDown,
  FaChevronUp
} from 'react-icons/fa';
import Navbar from './Navbar';
import FeatureEngineeringSection from './FeatureEngineeringSection';
import TrainingProgressModal from './TrainingProgressModal';
import { createInitialFeatureConfig } from '../utils/featureEngineering';
import type { FeatureConfig } from '../utils/featureEngineering';

interface DatasetPreview {
  columns: string[];
  rows: Record<string, any>[];
}

interface AdvancedSettings {
  cvFolds: number;
  trainSplit: number;
  maxRuntime: number;
}

interface Model {
  model_id: string;
  mlflow_run_id?: string;
  feature_importance?: any;
  frontend_id?: string;
  [key: string]: any;
}

interface TrainingPageProps {
  file: File;
  datasetPreview: DatasetPreview;
  selectedTargetColumn: string;
  selectedProblemType: string;
  onBackToHome: () => void;
  onGoToLiveModel?: () => void;
  onNavigate?: (route: string) => void;
  onShowDeployScreen?: (trainingData: {
    leaderboard: Model[];
    selectedModels: string[];
    selectedModel: string;
    message: string;
    advancedSettings: AdvancedSettings;
    detectedProblemType: string;
  }) => void;
}

const modelOptions = [
  { 
    id: 'xgboost', 
    name: 'XGBoost', 
    shortName: 'XGBoost',
    description: 'H2O XGBoost implementation with gradient boosting',
    detailedDescription: 'Extreme Gradient Boosting - one of the most powerful and popular ML algorithms',
    strengths: [
      'Excellent performance on structured data',
      'Built-in regularization prevents overfitting',
      'Handles missing values automatically',
      'Fast training and prediction'
    ],
    bestFor: [
      'Tabular/structured data',
      'Competition-winning results',
      'Mixed data types',
      'Medium to large datasets'
    ],
    complexity: 'Medium',
    speed: 'Fast',
    interpretability: 'Medium',
    memoryUsage: 'Medium',
    hyperparams: 'Many tunable parameters',
    pros: ['High accuracy', 'Robust', 'Industry standard'],
    cons: ['Can overfit', 'Many hyperparameters', 'Less interpretable'],
    useCase: 'Classification & Regression',
    requirements: 'Requires XGBoost backend (may not be available on all systems)',
    tipForBeginner: 'Great choice for most problems - often wins competitions!'
  },
  { 
    id: 'randomforest', 
    name: 'Random Forest (DRF)', 
    shortName: 'Random Forest',
    description: 'H2O Distributed Random Forest algorithm',
    detailedDescription: 'Ensemble of decision trees with built-in feature randomness and bagging',
    strengths: [
      'Very robust and stable',
      'Handles overfitting well',
      'Works with missing data',
      'Provides feature importance'
    ],
    bestFor: [
      'Beginners to ML',
      'Baseline models',
      'Mixed data types',
      'Feature selection'
    ],
    complexity: 'Low',
    speed: 'Fast',
    interpretability: 'High',
    memoryUsage: 'Medium',
    hyperparams: 'Few, easy to tune',
    pros: ['Easy to use', 'Stable', 'Good interpretability'],
    cons: ['Can be biased', 'Less accurate than boosting', 'Memory intensive'],
    useCase: 'Classification & Regression',
    requirements: 'Always available in H2O',
    tipForBeginner: 'Perfect starting point - reliable and easy to understand!'
  },
  { 
    id: 'gbm', 
    name: 'Gradient Boosting Machine', 
    shortName: 'GBM',
    description: 'H2O native gradient boosting algorithm',
    detailedDescription: 'Sequential ensemble method that builds models iteratively to correct errors',
    strengths: [
      'High predictive accuracy',
      'Handles different data types well',
      'Built-in cross-validation',
      'Excellent feature selection'
    ],
    bestFor: [
      'High accuracy requirements',
      'Structured data problems',
      'Feature engineering',
      'Production models'
    ],
    complexity: 'Medium',
    speed: 'Medium',
    interpretability: 'Medium',
    memoryUsage: 'Medium',
    hyperparams: 'Several important ones',
    pros: ['High accuracy', 'Robust', 'Good default settings'],
    cons: ['Can overfit', 'Slower than Random Forest', 'Sensitive to noise'],
    useCase: 'Classification & Regression',
    requirements: 'Native H2O algorithm - always available',
    tipForBeginner: 'H2O\'s flagship algorithm - reliable choice for production!'
  },
  { 
    id: 'glm', 
    name: 'Generalized Linear Model', 
    shortName: 'GLM',
    description: 'H2O GLM for linear and logistic regression',
    detailedDescription: 'Linear models with regularization - fast, interpretable, and statistically sound',
    strengths: [
      'Highly interpretable',
      'Very fast training',
      'Statistical significance tests',
      'Handles large datasets well'
    ],
    bestFor: [
      'Linear relationships',
      'Interpretability requirements',
      'Large datasets',
      'Baseline models'
    ],
    complexity: 'Low',
    speed: 'Very Fast',
    interpretability: 'Very High',
    memoryUsage: 'Low',
    hyperparams: 'Few, well-understood',
    pros: ['Fastest training', 'Highly interpretable', 'Statistically sound'],
    cons: ['Linear assumptions', 'Lower accuracy on complex data', 'Limited flexibility'],
    useCase: 'Classification & Regression',
    requirements: 'Native H2O algorithm - always available',
    tipForBeginner: 'Best for understanding relationships in your data!'
  },
  { 
    id: 'neural', 
    name: 'Deep Learning', 
    shortName: 'Neural Network',
    description: 'H2O Deep Neural Network Implementation and Result Interpretation',
    detailedDescription: 'Multi-layer neural networks capable of learning complex non-linear patterns',
    strengths: [
      'Learns complex patterns',
      'Automatic feature engineering',
      'Handles large datasets',
      'Flexible architecture'
    ],
    bestFor: [
      'Large datasets',
      'Complex non-linear patterns',
      'Image/text data',
      'Deep feature learning'
    ],
    complexity: 'High',
    speed: 'Slow',
    interpretability: 'Low',
    memoryUsage: 'High',
    hyperparams: 'Many architecture choices',
    pros: ['Learns complex patterns', 'Flexible', 'Scales well'],
    cons: ['Black box', 'Requires large data', 'Slow training', 'Many hyperparameters'],
    useCase: 'Classification & Regression',
    requirements: 'May require more memory and computation time',
    tipForBeginner: 'Advanced option - try simpler models first!'
  },
  { 
    id: 'ensemble', 
    name: 'Stacked Ensemble', 
    shortName: 'Ensemble',
    description: 'H2O ensemble method combining multiple models',
    detailedDescription: 'Meta-learning approach that combines predictions from multiple base models',
    strengths: [
      'Often highest accuracy',
      'Combines model strengths',
      'Reduces overfitting',
      'Automatic model selection'
    ],
    bestFor: [
      'Maximum accuracy',
      'Competition settings',
      'Production systems',
      'Combining diverse models'
    ],
    complexity: 'High',
    speed: 'Slow',
    interpretability: 'Low',
    memoryUsage: 'High',
    hyperparams: 'Meta-learner options',
    pros: ['Highest accuracy', 'Combines best models', 'Robust'],
    cons: ['Complex', 'Slow prediction', 'Hard to interpret', 'Resource intensive'],
    useCase: 'Classification & Regression',
    requirements: 'Requires other base models to be trained first',
    tipForBeginner: 'Let AutoML choose the best combination for you!'
  }
];

const getModelIcon = (modelId: string): React.ReactElement => {
  const icons: Record<string, React.ReactElement> = {
    'xgboost': <FaRocket />,
    'randomforest': <FaTree />,
    'gbm': <FaBolt />,
    'glm': <FaChartBar />,
    'neural': <FaBrain />,
    'ensemble': <FaBullseye />
  };
  return icons[modelId] || <FaRobot />;
};

const getModelTags = (modelId: string): string[] => {
  const tags: Record<string, string[]> = {
    'xgboost': ['Boosting', 'High Performance', 'Competition Winner'],
    'randomforest': ['Ensemble', 'Interpretable', 'Robust'],
    'gbm': ['Boosting', 'Native H2O', 'Production Ready'],
    'glm': ['Linear', 'Fast', 'Interpretable'],
    'neural': ['Deep Learning', 'Complex Patterns', 'Flexible'],
    'ensemble': ['Meta Learning', 'Best Accuracy', 'Advanced']
  };
  return tags[modelId] || ['Machine Learning'];
};

const getComplexityColor = (complexity: string) => {
  switch (complexity) {
    case 'Low': return '#28a745';
    case 'Medium': return '#ffc107';
    case 'High': return '#dc3545';
    default: return '#6c757d';
  }
};

const getSpeedColor = (speed: string) => {
  switch (speed) {
    case 'Very Fast': return '#28a745';
    case 'Fast': return '#28a745';
    case 'Medium': return '#ffc107';
    case 'Slow': return '#dc3545';
    default: return '#6c757d';
  }
};

const TrainingPage: React.FC<TrainingPageProps> = ({ 
  file, 
  datasetPreview, 
  selectedTargetColumn,
  selectedProblemType,
  onBackToHome,
  onGoToLiveModel,
  onShowDeployScreen
}) => {
  const [training, setTraining] = useState(false);
  const [message, setMessage] = useState('');
  const [selectedModels, setSelectedModels] = useState<string[]>(['xgboost', 'randomforest', 'gbm']);
  const [expandedModelCard, setExpandedModelCard] = useState<string | null>(null);
  
  // Feature engineering state
  const [showFeatureEngineering, setShowFeatureEngineering] = useState(false);
  const [featureConfig, setFeatureConfig] = useState<FeatureConfig>({});
  const [advancedSettings, setAdvancedSettings] = useState<AdvancedSettings>({
    cvFolds: 5,
    trainSplit: 0.8,
    maxRuntime: 300
  });

  // Initialize feature config when component mounts
  useEffect(() => {
    if (datasetPreview) {
      setFeatureConfig(createInitialFeatureConfig(datasetPreview));
    }
  }, [datasetPreview]);

  // Scroll to top when component mounts
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const handleAdvancedSettingsChange = (field: string, value: number | string) => {
    setAdvancedSettings(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleTrain = async () => {
    if (!file) {
      setMessage('No file uploaded.');
      return;
    }

    if (!selectedTargetColumn) {
      setMessage('Please select a target column.');
      return;
    }

    if (selectedModels.length === 0) {
      setMessage('Please select at least one model to train.');
      return;
    }

    setTraining(true);
    setMessage(`Training ${selectedModels.length} selected models... This may take several minutes.`);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('models', JSON.stringify(selectedModels));
      formData.append('target_column', selectedTargetColumn);
      formData.append('problem_type', selectedProblemType);
      formData.append('feature_config', JSON.stringify(featureConfig));
      formData.append('advanced_settings', JSON.stringify(advancedSettings));

      const res = await fetch('http://localhost:8000/train-model/', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error('Training error response:', errorText);
        throw new Error(`HTTP error! status: ${res.status}, details: ${errorText}`);
      }

      const data = await res.json();
      
      // Navigate to Deploy Screen with training results
      if (onShowDeployScreen) {
        onShowDeployScreen({
          leaderboard: data.leaderboard || [],
          selectedModels,
          selectedModel: '',
          message: data.message || 'Model training complete.',
          advancedSettings,
          detectedProblemType: data.problem_type || ''
        });
      }
      
    } catch (error) {
      console.error('Training error:', error);
      setMessage(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setTraining(false);
    }
  };

  const handleModelToggle = (modelId: string) => {
    setSelectedModels(prev =>
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const handleExpandModelCard = (modelId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    setExpandedModelCard(prev => prev === modelId ? null : modelId);
  };

  return (
    <>
      {/* Fixed Navigation Bar */}
      <Navbar onNavigate={(route) => {
        if (route === '/') {
          onBackToHome();
        }
      }} currentPage="/training" />
      
      <div className="App" style={{ paddingTop: '80px' }}>
        <section className="hero-section">
          <h1 className="hero-title"><FaCog /> Configure Training</h1>
          <h1 className="hero-subtitle">Set up feature engineering, model selection, and training parameters</h1>
        </section>

        {/* Training Progress Modal */}
        <TrainingProgressModal
          isOpen={training}
          onClose={() => {}} // Don't allow closing during training
          selectedModels={selectedModels}
          modelOptions={modelOptions}
          message={message}
        />

        {/* Feature Engineering Section */}
        <FeatureEngineeringSection
          datasetPreview={datasetPreview}
          selectedTargetColumn={selectedTargetColumn}
          featureConfig={featureConfig}
          setFeatureConfig={setFeatureConfig}
          advancedSettings={advancedSettings}
          handleAdvancedSettingsChange={handleAdvancedSettingsChange}
          showFeatureEngineering={showFeatureEngineering}
          setShowFeatureEngineering={setShowFeatureEngineering}
        />

        {/* Model Selection */}
        <section className="model-selection">
          <h2><FaCog /> Model Selection</h2>
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
                className={`modern-model-card enhanced ${selectedModels.includes(model.id) ? 'selected' : ''} ${expandedModelCard === model.id ? 'expanded' : ''}`}
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
                
                <div className="model-header">
                  <div className="model-icon-large">
                    {getModelIcon(model.id)}
                  </div>
                  <div className="model-title">
                    <h4 className="model-name">{model.name}</h4>
                    <span className="model-short-name">{model.shortName}</span>
                  </div>
                </div>

                {/* Compact view - always visible */}
                <div className="model-compact-info">
                  <div className="model-description-brief">
                    <p>{model.description}</p>
                  </div>

                  <div className="model-specs-compact">
                    <div className="spec-row">
                      <span className="spec-label">Complexity:</span>
                      <span className="spec-value" style={{color: getComplexityColor(model.complexity)}}>
                        {model.complexity}
                      </span>
                    </div>
                    <div className="spec-row">
                      <span className="spec-label">Speed:</span>
                      <span className="spec-value" style={{color: getSpeedColor(model.speed)}}>
                        {model.speed}
                      </span>
                    </div>
                  </div>

                  <div className="model-tags">
                    {getModelTags(model.id).slice(0, 2).map((tag: string) => (
                      <span key={tag} className="model-tag enhanced">{tag}</span>
                    ))}
                  </div>
                </div>

                {/* Expand/Collapse Button - Bottom Left */}
                <button 
                  className={`model-expand-btn ${expandedModelCard === model.id ? 'expanded' : ''}`}
                  onClick={(e) => handleExpandModelCard(model.id, e)}
                  title={expandedModelCard === model.id ? 'Collapse Details' : 'Expand Details'}
                >
                  {expandedModelCard === model.id ? <FaChevronUp /> : <FaChevronDown />}
                </button>

                {/* Expanded view - only visible when expanded */}
                {expandedModelCard === model.id && (
                  <div className="model-expanded-details">
                    <div className="model-description-full">
                      <p><strong>Details:</strong> {model.detailedDescription}</p>
                    </div>

                    <div className="model-specs-full">
                      <div className="spec-row">
                        <span className="spec-label">Interpretability:</span>
                        <span className="spec-value">{model.interpretability}</span>
                      </div>
                      <div className="spec-row">
                        <span className="spec-label">Memory Usage:</span>
                        <span className="spec-value">{model.memoryUsage}</span>
                      </div>
                      <div className="spec-row">
                        <span className="spec-label">Hyperparameters:</span>
                        <span className="spec-value">{model.hyperparams}</span>
                      </div>
                    </div>

                    <div className="model-strengths">
                      <h5><FaBolt /> Key Strengths:</h5>
                      <ul>
                        {model.strengths.map((strength, idx) => (
                          <li key={idx}>{strength}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="model-best-for">
                      <h5><FaBullseye /> Best For:</h5>
                      <ul>
                        {model.bestFor.map((useCase, idx) => (
                          <li key={idx}>{useCase}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="model-pros-cons">
                      <div className="pros-section">
                        <h6>Pros:</h6>
                        <ul>
                          {model.pros.map((pro, idx) => (
                            <li key={idx}>{pro}</li>
                          ))}
                        </ul>
                      </div>
                      <div className="cons-section">
                        <h6>Cons:</h6>
                        <ul>
                          {model.cons.map((con, idx) => (
                            <li key={idx}>{con}</li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div className="model-tip">
                      <div className="tip-box">
                        <span className="tip-icon">💡</span>
                        <span className="tip-text">{model.tipForBeginner}</span>
                      </div>
                    </div>

                    <div className="model-all-tags">
                      {getModelTags(model.id).map((tag: string) => (
                        <span key={tag} className="model-tag enhanced">{tag}</span>
                      ))}
                    </div>
                  </div>
                )}
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

        {/* Training Section */}
        <section className="training-section">
          <h2><FaPlay /> Start Training</h2>
          <div className="training-controls">
            <button
              onClick={handleTrain}
              disabled={training || selectedModels.length === 0}
              className={`btn btn-primary ${training ? 'btn-loading' : ''}`}
            >
              {training ? <><FaSpinner className="fa-spin" /> Training Models...</> : <><FaPlay /> Train {selectedModels.length} Selected Models</>}
            </button>
            
            {selectedModels.length === 0 && (
              <p className="training-warning"><FaMapPin /> Please select at least one model to train</p>
            )}
          </div>
        </section>

        {/* Status Messages */}
        {message && (
          <div className="message message-info">
            {message}
          </div>
        )}
      </div>
    </>
  );
};

export default TrainingPage;
