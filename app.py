import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from visualization import VisualizationEngine
from export_utils import ExportManager
from sample_data import SampleDataManager

# Configure page
st.set_page_config(
    page_title="Data Analyzer Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = ""

# Initialize components
data_processor = DataProcessor()
viz_engine = VisualizationEngine()
export_manager = ExportManager()
sample_data_manager = SampleDataManager()

def main():
    st.title("📊 Data Analyzer Dashboard")
    st.markdown("**Interactive CSV Explorer with Advanced Analytics**")
    
    # Sidebar for data loading
    with st.sidebar:
        st.header("📂 Data Loading")
        
        # Sample datasets
        st.subheader("Sample Datasets")
        sample_options = ["Select a sample dataset", "Iris", "Titanic", "House Prices"]
        selected_sample = st.selectbox("Choose sample dataset:", sample_options)
        
        if selected_sample != "Select a sample dataset":
            if st.button(f"Load {selected_sample}"):
                df, name = sample_data_manager.load_sample_dataset(selected_sample.lower().replace(" ", "_"))
                if df is not None:
                    st.session_state.df = df
                    st.session_state.filtered_df = df.copy()
                    st.session_state.dataset_name = name
                    st.success(f"Loaded {name} dataset!")
                    st.rerun()
        
        st.divider()
        
        # File upload
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your CSV file for analysis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.filtered_df = df.copy()
                st.session_state.dataset_name = uploaded_file.name
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Main content area
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🔍 Data Explorer", 
            "📈 Visualizations", 
            "🔎 Filtering", 
            "💾 Export"
        ])
        
        with tab1:
            show_data_overview(df)
        
        with tab2:
            show_data_explorer(df)
        
        with tab3:
            show_visualizations(df)
        
        with tab4:
            show_filtering_interface(df)
        
        with tab5:
            show_export_options()
    
    else:
        st.info("👆 Please upload a CSV file or select a sample dataset from the sidebar to begin analysis.")
        
        # Show sample of what the dashboard can do
        st.markdown("### 🚀 Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Data Analysis**
            - Comprehensive data summary
            - Statistical insights
            - Missing value analysis
            - Data type detection
            """)
        
        with col2:
            st.markdown("""
            **📈 Visualizations**
            - Interactive charts
            - Correlation heatmaps
            - Distribution plots
            - Custom filtering
            """)
        
        with col3:
            st.markdown("""
            **💾 Export Options**
            - Filtered CSV export
            - Excel reports with charts
            - Statistical summaries
            - Custom reports
            """)

def show_data_overview(df):
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.1f} MB")
    with col4:
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.metric("Missing Data", f"{missing_percent:.1f}%")
    
    # Data types and basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Column Information")
        column_info = data_processor.get_column_info(df)
        st.dataframe(column_info, use_container_width=True)
    
    with col2:
        st.subheader("📈 Statistical Summary")
        summary_stats = data_processor.get_statistical_summary(df)
        st.dataframe(summary_stats, use_container_width=True)
    
    # Missing values analysis
    st.subheader("❌ Missing Values Analysis")
    missing_analysis = data_processor.analyze_missing_values(df)
    if missing_analysis['total_missing'] > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(missing_analysis['by_column'], use_container_width=True)
        with col2:
            fig = viz_engine.create_missing_values_chart(missing_analysis['by_column'])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 No missing values found in the dataset!")

def show_data_explorer(df):
    st.header("🔍 Data Explorer")
    
    # Data preview
    st.subheader("📄 Data Preview")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        show_rows = st.slider("Number of rows to display", 5, min(100, len(df)), 10)
    with col2:
        show_head = st.radio("Show", ["Head", "Tail", "Random"])
    
    if show_head == "Head":
        preview_df = df.head(show_rows)
    elif show_head == "Tail":
        preview_df = df.tail(show_rows)
    else:
        preview_df = df.sample(min(show_rows, len(df)))
    
    st.dataframe(preview_df, use_container_width=True)
    
    # Column analysis
    st.subheader("📊 Column Analysis")
    selected_column = st.selectbox("Select column for detailed analysis:", df.columns)
    
    if selected_column:
        col1, col2 = st.columns(2)
        
        with col1:
            column_stats = data_processor.analyze_column(df, selected_column)
            st.json(column_stats)
        
        with col2:
            if df[selected_column].dtype in ['int64', 'float64']:
                fig = viz_engine.create_distribution_plot(df, selected_column)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = viz_engine.create_value_counts_plot(df, selected_column)
                st.plotly_chart(fig, use_container_width=True)

def show_visualizations(df):
    st.header("📈 Interactive Visualizations")
    
    viz_type = st.selectbox(
        "Choose visualization type:",
        ["Correlation Heatmap", "Distribution Plot", "Scatter Plot", "Bar Chart", "Line Chart", "Box Plot"]
    )
    
    if viz_type == "Correlation Heatmap":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig = viz_engine.create_correlation_heatmap(df[numeric_cols])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
    
    elif viz_type == "Distribution Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Select column:", numeric_cols)
            fig = viz_engine.create_distribution_plot(df, col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns available for distribution plot.")
    
    elif viz_type == "Scatter Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols)
            with col2:
                y_col = st.selectbox("Y-axis:", numeric_cols)
            
            color_col = st.selectbox("Color by (optional):", ["None"] + list(df.columns))
            color_col = None if color_col == "None" else color_col
            
            fig = viz_engine.create_scatter_plot(df, x_col, y_col, color_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for scatter plot.")
    
    elif viz_type == "Bar Chart":
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            col = st.selectbox("Select categorical column:", categorical_cols)
            fig = viz_engine.create_value_counts_plot(df, col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns available for bar chart.")
    
    elif viz_type == "Line Chart":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Select column:", numeric_cols)
            fig = viz_engine.create_line_chart(df, col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns available for line chart.")
    
    elif viz_type == "Box Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Select column:", numeric_cols)
            group_col = st.selectbox("Group by (optional):", ["None"] + list(df.columns))
            group_col = None if group_col == "None" else group_col
            
            fig = viz_engine.create_box_plot(df, col, group_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns available for box plot.")

def show_filtering_interface(df):
    st.header("🔎 Data Filtering")
    
    # Initialize filtered dataframe
    filtered_df = df.copy()
    
    st.subheader("Filter Options")
    
    # Column-based filtering
    for col in df.columns:
        with st.expander(f"Filter by {col}"):
            if df[col].dtype in ['int64', 'float64']:
                # Numeric filtering
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected_range = st.slider(
                    f"Range for {col}",
                    min_val, max_val, (min_val, max_val),
                    key=f"slider_{col}"
                )
                filtered_df = filtered_df[
                    (filtered_df[col] >= selected_range[0]) & 
                    (filtered_df[col] <= selected_range[1])
                ]
            
            elif df[col].dtype == 'object':
                # Categorical filtering
                unique_values = df[col].dropna().unique()
                if len(unique_values) <= 50:  # Only show multiselect for reasonable number of values
                    selected_values = st.multiselect(
                        f"Select values for {col}",
                        unique_values,
                        default=list(unique_values),
                        key=f"multiselect_{col}"
                    )
                    if selected_values:
                        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                else:
                    search_term = st.text_input(f"Search in {col}:", key=f"search_{col}")
                    if search_term:
                        filtered_df = filtered_df[
                            filtered_df[col].str.contains(search_term, case=False, na=False)
                        ]
    
    # Update session state
    st.session_state.filtered_df = filtered_df
    
    # Show filtered results
    st.subheader("📊 Filtered Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Rows", f"{len(df):,}")
    with col2:
        st.metric("Filtered Rows", f"{len(filtered_df):,}")
    
    if len(filtered_df) > 0:
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("No data matches the current filters.")

def show_export_options():
    st.header("💾 Export Options")
    
    if st.session_state.filtered_df is not None:
        filtered_df = st.session_state.filtered_df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 CSV Export")
            csv_data = export_manager.to_csv(filtered_df)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv_data,
                file_name=f"filtered_{st.session_state.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.subheader("📊 Excel Report")
            if st.button("Generate Excel Report with Charts"):
                with st.spinner("Generating Excel report..."):
                    excel_data = export_manager.to_excel_with_charts(
                        filtered_df, 
                        st.session_state.dataset_name
                    )
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name=f"report_{st.session_state.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        # Summary statistics export
        st.subheader("📈 Summary Statistics")
        summary_stats = data_processor.get_statistical_summary(filtered_df)
        csv_summary = export_manager.to_csv(summary_stats)
        st.download_button(
            label="Download Summary Statistics",
            data=csv_summary,
            file_name=f"summary_{st.session_state.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        st.info("No data available for export. Please load a dataset first.")

if __name__ == "__main__":
    main()
