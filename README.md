# 🏎️ Formula 1 Data Analysis and Visualization Project

This project provides a comprehensive pipeline for analyzing and visualizing Formula 1 racing data. It includes scripts and Jupyter Notebooks for cleaning raw datasets, creating master tables, performing Exploratory Data Analysis (EDA), engineering features, and launching an interactive Streamlit UI for data exploration.

> **Goal:** Uncover insights into driver performance, team reliability, track complexity, and race outcomes.

## 📑 Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage Instructions](#usage-instructions)
    - [Data Cleaning](#data-cleaning)
    - [Creating Master Tables](#creating-master-tables)
    - [Exploratory Data Analysis & Feature Engineering](#exploratory-data-analysis--feature-engineering)
    - [Running the Streamlit UI](#running-the-streamlit-ui)
- [Problem Statements](#problem-statements)
- [Outputs](#outputs)

## 📂 Project Structure

```text
📁 Cleaned_Dataset/            # Contains cleaned datasets and master tables
📁 DPL_Datasets/               # Contains raw Formula 1 datasets
📄 analyze_csvs.py             # Python script for generating scatter plots from CSV files
📓 clean_datasets.ipynb        # Notebook for cleaning datasets
📓 create_master_table.ipynb   # Notebook for creating master tables
📓 driver_analysis.ipynb       # Notebook for analyzing driver performance
📓 EDA_script.ipynb            # Notebook for exploratory data analysis
📓 Feature_Engineering.ipynb   # Notebook for advanced feature engineering
📓 Modeling.ipynb              # Notebook for modeling tasks
📄 modules.py                  # Core functions and utility module
📓 Presentation.ipynb          # Notebook for project findings and presentations
📄 requirements.txt            # File specifying Python dependencies
📄 ui.py                       # Streamlit UI script
```

## 🛠️ Prerequisites

- **Python 3.8** or higher
- **Jupyter Notebook**  
  (Install via: `pip install notebook`)
- **Git**
- **Docker** (optional)

## 🚀 Setup Instructions

1. **Clone the Repository**

   Open your terminal and run:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**

   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Docker Setup**

   Build and run the Docker container:
   ```bash
   docker build -t f1-analysis .
   docker run -p 8501:8501 f1-analysis
   ```

## 📊 Usage Instructions

### Data Cleaning

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open and execute `clean_datasets.ipynb`.

### Creating Master Tables

Run the `create_master_table.ipynb` notebook to generate the following CSV files:
- `master_race_results.csv`
- `master_qualifying.csv`
- `master_sprint_results.csv`
- `master_pit_stops.csv`
- `master_lap_times.csv`

### Exploratory Data Analysis & Feature Engineering

1. Execute `EDA_script.ipynb` for data visualization and analysis.
2. Run `Feature_Engineering.ipynb` to generate advanced features.

### Running the Streamlit UI

Launch the interactive UI with:
```bash
streamlit run ui.py
```
## ✅ Problem Statements Solved

This project successfully explored several key areas of Formula 1 racing analytics:

- **Driver Consistency:** ✅ Analyzed performance patterns across seasons
- **Team Reliability:** ✅ Assessed mechanical failures and strategy success
- **Track Complexity:** ✅ Determined impact of circuit design on race outcomes
- **Race Trends:** ✅ Identified patterns in overtaking, accidents, and scoring

The following challenges have been solved using data analysis, predictive modeling, and visualization techniques:

### Driver & Constructor Performance ✅
- ✅ Identified dominant drivers and constructors by analyzing win ratios and podium finishes
- ✅ Assessed the relationship between career longevity and success metrics (wins, podiums, points)

### Qualifying vs. Race Performance ✅
- ✅ Determined how starting grid position impacts final race results
- ✅ Identified drivers who excel at making up positions

### Pit Stop Strategies ✅
- ✅ Evaluated optimal pit stop frequency and timing for race success
- ✅ Analyzed pit stop efficiency and its influence on race outcomes

### Head-to-Head Driver Analysis ✅
- ✅ Discovered which rivalries have been the most competitive
- ✅ Compiled head-to-head stats based on race finishes

### Hypothetical Driver Swaps ✅
- ✅ Simulated driver swaps between different teams and predicted the impact on standings

### Driver Movements & Team Networks ✅
- ✅ Mapped driver transitions across teams using network graph visualizations

### Team Performance Comparison ✅
- ✅ Compared team success rates against different opponents with and without circuit factors

### Driver Consistency in Race Performance ✅
- ✅ Identified drivers with consistent top finishes and those with fluctuating results

### Lap Time Efficiency ✅
- ✅ Compared lap times across different circuits and identified teams maximizing efficiency

### Best Team Lineup ✅
- ✅ Built optimal team lineups based on driver performance trends

### Predictions for 2025 Season ✅
- ✅ Projected Drivers' and Constructors' Championship winners based on historical and current data

### Struggling Teams Analysis ✅
- ✅ Predicted which teams are likely to underperform in upcoming seasons

### Driver-Specific Track Struggles ✅
- ✅ Identified circuits where specific drivers consistently struggle or excel

### Championship Retention Probability ✅
- ✅ Analyzed the probability that a season's winner will retain the title in the next season
- ✅ Studied historical trends of back-to-back champions

### Champion Age Trends ✅
- ✅ Identified the age ranges where drivers consistently win championships across different decades

### Bonus Challenge ✅
- ✅ Successfully predicted future team movements based on past team transitions and transfer trends

## 📈 Outputs


- **Cleaned Datasets:** Processed data files in CSV format
- **Master Tables:** Integrated records with normalized references
- **Visualizations:** Statistical plots and interactive charts
- **Feature Sets:** Engineered variables for predictive modeling
- **Interactive Streamlit UI:** User-friendly web application providing:
    - Real-time data exploration and filtering capabilities
    - Interactive visualizations of driver and constructor performance
    - Comparison tools for head-to-head driver and team analysis
    - Custom reporting and insight generation
    - Responsive design for desktop and mobile access