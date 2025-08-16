# Overview

Data Analyzer Dashboard is a Streamlit-based web application for interactive CSV data exploration and analysis. The application provides comprehensive data analytics capabilities including statistical summaries, visualizations, missing value analysis, and export functionality. It's designed to help users quickly understand and analyze datasets through an intuitive web interface.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Framework
- **Streamlit**: Web framework for the entire user interface
- **Plotly**: Interactive visualization library for charts and graphs
- **Wide layout configuration**: Optimized for dashboard-style data presentation

## Modular Component Design
The application follows a modular architecture with separate classes for distinct responsibilities:

- **DataProcessor**: Handles statistical analysis, column information, and missing value detection
- **VisualizationEngine**: Manages all chart creation and visual representations
- **ExportManager**: Handles data export to CSV and Excel formats with embedded charts
- **SampleDataManager**: Provides pre-loaded datasets for demonstration purposes

## Data Processing Architecture
- **Pandas-based**: Core data manipulation using pandas DataFrames
- **NumPy integration**: Statistical calculations and array operations
- **Real-time filtering**: Dynamic data filtering stored in session state
- **Memory-efficient processing**: Column-wise analysis to handle large datasets

## Session Management
- **Streamlit session state**: Persistent storage for DataFrames and user selections
- **State variables**: df, filtered_df, and dataset_name maintained across interactions

## Visualization Strategy
- **Interactive charts**: Plotly-based visualizations with hover effects and zoom capabilities
- **Multiple chart types**: Correlation heatmaps, distribution plots, scatter plots, and statistical summaries
- **Responsive design**: Charts adapt to different screen sizes and data types

## Export Capabilities
- **Multi-format support**: CSV and Excel export options
- **Enhanced Excel export**: Includes multiple worksheets with data, statistics, and embedded charts
- **Chart embedding**: Automatic chart generation within Excel files using xlsxwriter

# External Dependencies

## Core Data Science Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations and array operations
- **scikit-learn**: Sample datasets (Iris, Wine) for demonstration

## Visualization Libraries
- **plotly**: Interactive web-based visualizations
- **seaborn**: Statistical data visualization (imported but not actively used)
- **matplotlib**: Base plotting library (imported but not actively used)

## Web Framework
- **streamlit**: Web application framework and UI components

## Data Export Libraries
- **xlsxwriter**: Excel file creation with chart embedding
- **openpyxl**: Excel file manipulation and chart generation
- **io/base64**: File handling and encoding for downloads

## Utility Libraries
- **datetime**: Timestamp generation for exports
- **warnings**: Error message filtering
- **typing**: Type hints for better code documentation

## Sample Data Sources
- **sklearn.datasets**: Pre-loaded scientific datasets for testing and demonstration
- **Synthetic data generation**: Custom algorithms for creating realistic sample datasets