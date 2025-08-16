import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

class VisualizationEngine:
    """Handles all visualization and chart generation."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create an interactive correlation heatmap."""
        correlation_matrix = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Heatmap',
            xaxis_title='Features',
            yaxis_title='Features',
            height=600,
            width=800
        )
        
        return fig
    
    def create_distribution_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create a distribution plot with histogram and box plot."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'Distribution of {column}', f'Box Plot of {column}'],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=df[column].dropna(),
                nbinsx=30,
                name='Distribution',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                x=df[column].dropna(),
                name='Box Plot',
                marker_color='lightcoral'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text=f'Distribution Analysis: {column}'
        )
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                          color_col: Optional[str] = None) -> go.Figure:
        """Create an interactive scatter plot."""
        if color_col:
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col,
                title=f'Scatter Plot: {x_col} vs {y_col}',
                hover_data=[col for col in df.columns if col not in [x_col, y_col, color_col]][:5]
            )
        else:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=f'Scatter Plot: {x_col} vs {y_col}',
                hover_data=[col for col in df.columns if col not in [x_col, y_col]][:5]
            )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(height=600)
        
        return fig
    
    def create_value_counts_plot(self, df: pd.DataFrame, column: str, top_n: int = 20) -> go.Figure:
        """Create a bar chart of value counts."""
        value_counts = df[column].value_counts().head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                marker_color='lightseagreen',
                text=value_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f'Value Counts: {column} (Top {min(top_n, len(value_counts))})',
            xaxis_title=column,
            yaxis_title='Count',
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_line_chart(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create a line chart (useful for time series or sequential data)."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines+markers',
            name=column,
            line=dict(color='royalblue', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=f'Line Chart: {column}',
            xaxis_title='Index',
            yaxis_title=column,
            height=500
        )
        
        return fig
    
    def create_box_plot(self, df: pd.DataFrame, column: str, 
                       group_col: Optional[str] = None) -> go.Figure:
        """Create a box plot, optionally grouped by another column."""
        if group_col:
            fig = px.box(
                df, x=group_col, y=column,
                title=f'Box Plot: {column} by {group_col}'
            )
        else:
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df[column],
                name=column,
                marker_color='lightviolet'
            ))
            fig.update_layout(
                title=f'Box Plot: {column}',
                yaxis_title=column
            )
        
        fig.update_layout(height=500)
        return fig
    
    def create_missing_values_chart(self, missing_df: pd.DataFrame) -> go.Figure:
        """Create a chart showing missing values by column."""
        if len(missing_df) == 0:
            # Return empty chart if no missing values
            fig = go.Figure()
            fig.add_annotation(
                text="No missing values found!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        fig = go.Figure(data=[
            go.Bar(
                x=missing_df['Missing Percentage'],
                y=missing_df['Column'],
                orientation='h',
                marker_color='salmon',
                text=[f"{count} ({pct:.1f}%)" for count, pct in 
                      zip(missing_df['Missing Count'], missing_df['Missing Percentage'])],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Missing Values by Column',
            xaxis_title='Missing Percentage (%)',
            yaxis_title='Columns',
            height=max(400, len(missing_df) * 30),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_pie_chart(self, df: pd.DataFrame, column: str, top_n: int = 10) -> go.Figure:
        """Create a pie chart for categorical data."""
        value_counts = df[column].value_counts().head(top_n)
        
        fig = go.Figure(data=[go.Pie(
            labels=value_counts.index,
            values=value_counts.values,
            hole=.3
        )])
        
        fig.update_layout(
            title=f'Distribution: {column}',
            height=500
        )
        
        return fig
    
    def create_multi_histogram(self, df: pd.DataFrame, columns: list) -> go.Figure:
        """Create overlaid histograms for multiple numeric columns."""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, col in enumerate(columns):
            fig.add_trace(go.Histogram(
                x=df[col].dropna(),
                name=col,
                opacity=0.7,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title='Distribution Comparison',
            xaxis_title='Value',
            yaxis_title='Frequency',
            barmode='overlay',
            height=500
        )
        
        return fig
    
    def create_pairplot_subplot(self, df: pd.DataFrame, columns: list) -> go.Figure:
        """Create a simplified pairplot using subplots."""
        n_cols = len(columns)
        fig = make_subplots(
            rows=n_cols, cols=n_cols,
            subplot_titles=[f"{col1} vs {col2}" for col1 in columns for col2 in columns]
        )
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    # Diagonal: histogram
                    fig.add_trace(
                        go.Histogram(x=df[col1], name=f"{col1} dist", showlegend=False),
                        row=i+1, col=j+1
                    )
                else:
                    # Off-diagonal: scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=df[col2], y=df[col1], mode='markers',
                            name=f"{col1} vs {col2}", showlegend=False,
                            marker=dict(size=3, opacity=0.6)
                        ),
                        row=i+1, col=j+1
                    )
        
        fig.update_layout(
            title='Pairwise Relationships',
            height=200 * n_cols,
            showlegend=False
        )
        
        return fig
