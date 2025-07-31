// featureEngineering.ts - Feature Engineering Utilities

export interface DatasetPreview {
  columns: string[];
  rows: Record<string, any>[];
}

export interface FeatureConfig {
  [columnName: string]: {
    include: boolean;
    impute: string;
    dataType: string;
  };
}

export interface ColumnTypeInfo {
  icon: string;
  label: string;
  color: string;
  description: string;
  missingCount: number;
  missingPercent: string;
  type: string;
}

export interface ImputationOption {
  value: string;
  label: string;
}

export interface EncodingOption {
  value: string;
  label: string;
}

export interface TransformOption {
  value: string;
  label: string;
}

/**
 * Detects the data type of a column based on its values
 */
export const detectColumnType = (column: string, datasetPreview: DatasetPreview): string => {
  if (!datasetPreview) return 'unknown';
  
  const columnData = datasetPreview.rows
    .map(row => row[column])
    .filter(val => val !== null && val !== undefined && val !== '');
  
  if (columnData.length === 0) return 'empty';
  
  // Check if all values can be converted to numbers
  const numericValues = columnData.filter(val => !isNaN(Number(val)) && val !== '');
  const isNumeric = numericValues.length === columnData.length;
  
  if (isNumeric) {
    const uniqueCount = new Set(columnData).size;
    const uniqueRatio = uniqueCount / columnData.length;
    
    // If low unique count and looks like categories (even if numeric)
    if (uniqueCount <= 10 && uniqueRatio < 0.1) {
      return 'categorical_numeric'; // Like 0,1,2 for classes
    }
    
    // Check if all are integers vs floats
    const isInteger = columnData.every(val => Number.isInteger(Number(val)));
    return isInteger ? 'integer' : 'numeric';
  } else {
    // String/text data
    const uniqueCount = new Set(columnData).size;
    const uniqueRatio = uniqueCount / columnData.length;
    
    // If few unique values, it's categorical
    if (uniqueCount <= 50 && uniqueRatio < 0.5) {
      return 'categorical';
    }
    
    return 'text';
  }
};

/**
 * Gets the count of missing values in a column
 */
export const getMissingCount = (column: string, datasetPreview: DatasetPreview): number => {
  if (!datasetPreview) return 0;
  return datasetPreview.rows.filter(row => 
    row[column] === null || row[column] === undefined || row[column] === ''
  ).length;
};

/**
 * Gets comprehensive column type information including missing data stats
 */
export const getColumnTypeInfo = (column: string, datasetPreview: DatasetPreview): ColumnTypeInfo => {
  const colType = detectColumnType(column, datasetPreview);
  const missingCount = getMissingCount(column, datasetPreview);
  const totalRows = datasetPreview?.rows.length || 0;
  const missingPercent = totalRows > 0 ? (missingCount / totalRows * 100).toFixed(1) : '0';
  
  const typeInfo = {
    'numeric': { icon: '🔢', label: 'Numeric', color: '#28a745', description: 'Continuous numerical data' },
    'integer': { icon: '🔢', label: 'Integer', color: '#17a2b8', description: 'Whole numbers' },
    'categorical': { icon: '🏷️', label: 'Categorical', color: '#ffc107', description: 'Categories/classes' },
    'categorical_numeric': { icon: '🏷️', label: 'Categorical (Numeric)', color: '#fd7e14', description: 'Numeric categories (0,1,2...)' },
    'text': { icon: '📝', label: 'Text', color: '#6f42c1', description: 'Text/string data' },
    'empty': { icon: '❌', label: 'Empty', color: '#dc3545', description: 'No valid data' },
    'unknown': { icon: '❓', label: 'Unknown', color: '#6c757d', description: 'Cannot determine type' }
  };
  
  return {
    ...(typeInfo[colType as keyof typeof typeInfo] || typeInfo['unknown']),
    missingCount,
    missingPercent,
    type: colType
  };
};

/**
 * Gets valid imputation options based on column data type
 */
export const getImputationOptions = (column: string, datasetPreview: DatasetPreview): ImputationOption[] => {
  const colType = detectColumnType(column, datasetPreview);
  
  switch (colType) {
    case 'numeric':
    case 'integer':
      return [
        { value: 'mean', label: 'Mean (average)' },
        { value: 'median', label: 'Median (middle value)' },
        { value: 'mode', label: 'Mode (most frequent)' },
        { value: 'forward_fill', label: 'Forward Fill' },
        { value: 'backward_fill', label: 'Backward Fill' },
        { value: 'zero', label: 'Fill with 0' },
        { value: 'auto', label: 'Auto (H2O decides)' }
      ];
    
    case 'categorical':
    case 'categorical_numeric':
      return [
        { value: 'mode', label: 'Mode (most frequent)' },
        { value: 'forward_fill', label: 'Forward Fill' },
        { value: 'backward_fill', label: 'Backward Fill' },
        { value: 'unknown', label: 'Fill with "Unknown"' },
        { value: 'auto', label: 'Auto (H2O decides)' }
      ];
    
    case 'text':
      return [
        { value: 'mode', label: 'Mode (most frequent)' },
        { value: 'unknown', label: 'Fill with "Unknown"' },
        { value: 'empty_string', label: 'Fill with empty string' },
        { value: 'auto', label: 'Auto (H2O decides)' }
      ];
    
    default:
      return [
        { value: 'auto', label: 'Auto (H2O decides)' },
        { value: 'mode', label: 'Mode (most frequent)' }
      ];
  }
};

/**
 * Gets valid encoding options based on column data type
 */
export const getEncodingOptions = (column: string, datasetPreview: DatasetPreview): EncodingOption[] => {
  const colType = detectColumnType(column, datasetPreview);
  
  switch (colType) {
    case 'categorical':
    case 'categorical_numeric':
      return [
        { value: 'auto', label: 'Auto (H2O decides)' },
        { value: 'one_hot', label: 'One-Hot Encoding' },
        { value: 'label', label: 'Label Encoding' },
        { value: 'target', label: 'Target Encoding' },
        { value: 'binary', label: 'Binary Encoding' }
      ];
    
    case 'text':
      return [
        { value: 'auto', label: 'Auto (H2O decides)' },
        { value: 'bag_of_words', label: 'Bag of Words' },
        { value: 'tfidf', label: 'TF-IDF' },
        { value: 'word2vec', label: 'Word2Vec' }
      ];
    
    case 'numeric':
    case 'integer':
    default:
      return [
        { value: 'none', label: 'None (keep as numeric)' },
        { value: 'auto', label: 'Auto (H2O decides)' }
      ];
  }
};

/**
 * Gets valid transform options based on column data type
 */
export const getTransformOptions = (column: string, datasetPreview: DatasetPreview): TransformOption[] => {
  const colType = detectColumnType(column, datasetPreview);
  
  switch (colType) {
    case 'numeric':
    case 'integer':
      return [
        { value: 'none', label: 'None (original values)' },
        { value: 'normalize', label: 'Normalize (0-1 scale)' },
        { value: 'standardize', label: 'Standardize (z-score)' },
        { value: 'log', label: 'Log Transform' },
        { value: 'sqrt', label: 'Square Root' },
        { value: 'box_cox', label: 'Box-Cox Transform' },
        { value: 'auto', label: 'Auto (H2O decides)' }
      ];
    
    case 'categorical':
    case 'categorical_numeric':
    case 'text':
      return [
        { value: 'none', label: 'None (keep original)' },
        { value: 'auto', label: 'Auto (H2O decides)' }
      ];
    
    default:
      return [
        { value: 'none', label: 'None' },
        { value: 'auto', label: 'Auto (H2O decides)' }
      ];
  }
};

/**
 * Validates feature configuration and returns warnings for potential issues
 */
export const getConfigurationWarnings = (
  featureConfig: FeatureConfig, 
  datasetPreview: DatasetPreview, 
  selectedTargetColumn: string
): string[] => {
  const warnings: string[] = [];
  
  Object.entries(featureConfig).forEach(([column, config]) => {
    if (!config.include || column === selectedTargetColumn) return;
    
    const colTypeInfo = getColumnTypeInfo(column, datasetPreview);
    
    // Check for inappropriate imputation
    if (colTypeInfo.type === 'categorical' && ['mean', 'median'].includes(config.impute)) {
      warnings.push(`⚠️ Column "${column}" is categorical but using numerical imputation (${config.impute}). Use "mode" instead.`);
    }
    
    if (colTypeInfo.type === 'text' && ['mean', 'median'].includes(config.impute)) {
      warnings.push(`⚠️ Column "${column}" is text but using numerical imputation (${config.impute}). Use "mode" or "unknown" instead.`);
    }
    
    // Check for inappropriate encoding and transforms will be handled by H2O AutoML automatically
    // Removed encoding and transform validation since we let H2O handle these automatically
    
    // Check for high missing values with drop strategy
    if (config.impute === 'drop' && colTypeInfo.missingCount > datasetPreview.rows.length * 0.5) {
      warnings.push(`⚠️ Column "${column}" has ${colTypeInfo.missingPercent}% missing values. Dropping rows may remove too much data.`);
    }
  });
  
  return warnings;
};

/**
 * Creates initial feature configuration for all columns
 */
export const createInitialFeatureConfig = (datasetPreview: DatasetPreview): FeatureConfig => {
  const initialConfig: FeatureConfig = {};
  
  if (datasetPreview) {
    datasetPreview.columns.forEach((col: string) => {
      initialConfig[col] = {
        include: true,
        impute: 'auto',
        dataType: 'auto'
      };
    });
  }
  
  return initialConfig;
};

/**
 * Updates a specific field in the feature configuration for a column
 */
export const updateFeatureConfig = (
  featureConfig: FeatureConfig, 
  column: string, 
  field: string, 
  value: string | boolean
): FeatureConfig => {
  return {
    ...featureConfig,
    [column]: {
      ...featureConfig[column],
      [field]: value
    }
  };
};

/**
 * Sets all features to be included
 */
export const includeAllFeatures = (
  featureConfig: FeatureConfig, 
  selectedTargetColumn: string
): FeatureConfig => {
  const newConfig = { ...featureConfig };
  Object.keys(newConfig).forEach(col => {
    if (col !== selectedTargetColumn) {
      newConfig[col].include = true;
    }
  });
  return newConfig;
};

/**
 * Resets all feature configurations to auto/default values
 */
export const resetAllToAuto = (
  featureConfig: FeatureConfig, 
  selectedTargetColumn: string
): FeatureConfig => {
  const newConfig = { ...featureConfig };
  Object.keys(newConfig).forEach(col => {
    if (col !== selectedTargetColumn) {
      newConfig[col] = {
        ...newConfig[col],
        impute: 'auto'
      };
    }
  });
  return newConfig;
};

/**
 * Gets sample values for a column (for display purposes)
 */
export const getSampleValues = (column: string, datasetPreview: DatasetPreview, maxLength: number = 50): string => {
  const sampleValues = datasetPreview.rows
    .slice(0, 3)
    .map(row => row[column])
    .filter(val => val !== null && val !== undefined && val !== '')
    .join(', ');
  
  return sampleValues.length > maxLength ? sampleValues.substring(0, maxLength) + '...' : sampleValues;
};
