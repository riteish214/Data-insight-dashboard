import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

class DataProcessor:
    """Handles data processing, analysis, and statistical operations."""
    
    def get_column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get comprehensive column information."""
        info_data = []
        for col in df.columns:
            info_data.append({
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Non-Null Count': df[col].count(),
                'Null Count': df[col].isnull().sum(),
                'Unique Values': df[col].nunique(),
                'Memory Usage (bytes)': df[col].memory_usage()
            })
        return pd.DataFrame(info_data)
    
    def get_statistical_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistical summary for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            summary = numeric_df.describe()
            # Add additional statistics
            summary.loc['median'] = numeric_df.median()
            summary.loc['mode'] = numeric_df.mode().iloc[0] if len(numeric_df.mode()) > 0 else np.nan
            summary.loc['skewness'] = numeric_df.skew()
            summary.loc['kurtosis'] = numeric_df.kurtosis()
            return summary.round(3)
        else:
            return pd.DataFrame()
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the dataset."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Missing Percentage': missing_percentages.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        return {
            'total_missing': missing_counts.sum(),
            'by_column': missing_df,
            'missing_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100
        }
    
    def analyze_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Detailed analysis of a specific column."""
        col_data = df[column]
        
        analysis = {
            'column_name': column,
            'data_type': str(col_data.dtype),
            'total_values': len(col_data),
            'non_null_values': col_data.count(),
            'null_values': col_data.isnull().sum(),
            'unique_values': col_data.nunique(),
            'duplicate_values': col_data.duplicated().sum()
        }
        
        if col_data.dtype in ['int64', 'float64']:
            analysis.update({
                'mean': float(col_data.mean()) if not col_data.empty else None,
                'median': float(col_data.median()) if not col_data.empty else None,
                'std': float(col_data.std()) if not col_data.empty else None,
                'min': float(col_data.min()) if not col_data.empty else None,
                'max': float(col_data.max()) if not col_data.empty else None,
                'q25': float(col_data.quantile(0.25)) if not col_data.empty else None,
                'q75': float(col_data.quantile(0.75)) if not col_data.empty else None,
                'skewness': float(col_data.skew()) if not col_data.empty else None,
                'kurtosis': float(col_data.kurtosis()) if not col_data.empty else None
            })
        
        elif col_data.dtype == 'object':
            value_counts = col_data.value_counts().head(10)
            analysis.update({
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                'average_length': float(col_data.astype(str).str.len().mean()) if not col_data.empty else None,
                'top_values': value_counts.to_dict()
            })
        
        return analysis
    
    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr') -> Tuple[pd.Series, Dict]:
        """Detect outliers in a numeric column."""
        if df[column].dtype not in ['int64', 'float64']:
            return pd.Series(dtype=bool), {}
        
        col_data = df[column].dropna()
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            info = {
                'method': 'IQR',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': outliers.sum()
            }
        
        elif method == 'zscore':
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outliers = z_scores > 3
            
            info = {
                'method': 'Z-Score',
                'threshold': 3,
                'outlier_count': outliers.sum()
            }
        
        return outliers, info
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive data quality report."""
        report = {
            'dataset_shape': df.shape,
            'memory_usage_mb': df.memory_usage().sum() / (1024**2),
            'missing_values': self.analyze_missing_values(df),
            'column_types': df.dtypes.value_counts().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Outlier analysis for numeric columns
        outliers_summary = {}
        for col in report['numeric_columns']:
            outliers, info = self.detect_outliers(df, col)
            outliers_summary[col] = info
        
        report['outliers'] = outliers_summary
        
        return report
