# 📊 Data Insight Dashboard

An interactive **data exploration and visualization dashboard** built with **Streamlit** and **Python**.  
This tool helps users quickly analyze datasets, generate insights, and create compelling visualizations without writing code.

---

## 🚀 Features

- **CSV Upload**: Upload your own datasets for instant analysis  
- **Data Overview**: Automatic summary including  
  - Dataset shape  
  - Column data types  
  - Missing value counts  
  - Descriptive statistics  
- **Interactive Visualizations**:  
  - Line charts  
  - Bar charts  
  - Pie charts  
  - Histograms  
  - Correlation heatmaps  
- **Data Filtering**: Apply column-based filters and conditions  
- **Export Options**:  
  - Download filtered datasets as CSV  
  - Export full reports as Excel with charts  
- **Sample Datasets**: Preloaded Iris, Titanic, and House Prices datasets for demo usage  

---

## 🛠️ Tech Stack

- **Backend & UI**: [Streamlit](https://streamlit.io/)  
- **Data Processing**: Pandas, NumPy  
- **Visualization**: Plotly, Matplotlib, Seaborn  
- **Export**: Openpyxl, XlsxWriter  
- **ML/Analysis (optional)**: Scikit-learn  

---

## 📂 Project Structure

data-insight-dashboard/
├── app.py # Main Streamlit app
├── data_processor.py # Data loading and processing logic
├── visualizations.py # Chart/plot functions
├── utils.py # Filtering, export, helper functions
├── config.py # Settings and constants
├── requirements.txt # Dependencies
└── sample_data/ # Example datasets (Iris, Titanic, House Prices)

yaml
Copy
Edit

---

## ⚙️ Installation

Clone the repo:

```bash
git clone https://github.com/yourusername/data-insight-dashboard.git
cd data-insight-dashboard
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py

📖 Usage
Open the dashboard in your browser (http://localhost:8501)

Upload your CSV file from the sidebar

Explore the Data Overview tab

Create visualizations in the Visualizations tab

Apply filters and download filtered datasets

Export results to CSV or Excel reports

📊 Sample Datasets
Iris Dataset – Flower measurements for classification tasks

Titanic Dataset – Passenger survival dataset

House Prices Dataset – Housing features and sale prices



📜 License
This project is open-source under the MIT License.

pgsql
Copy
Edit

---

