# EduScore Predictor

> **Advanced ML-powered Student Performance Analysis & Math Score Prediction Dashboard**

## Overview

EduScore Predictor is a production-ready machine learning project that predicts student math scores and surfaces deep insights from a 1,000-student dataset. The Streamlit dashboard features 5 interactive tabs, Plotly visualizations, and an 8-model leaderboard.

## Project Structure

```
eduscore-predictor/
├── app.py                        ← Streamlit entry point
├── setup.py                      ← Package configuration
├── requirements.txt              ← Python dependencies
├── pyrightconfig.json            ← Pyright type-checker config
│
├── data/
│   └── students.csv              ← Source dataset (1,000 records, 8 features)
│
├── notebooks/
│   ├── 01_eda.ipynb              ← Exploratory data analysis
│   └── 02_model_training.ipynb  ← Model experimentation
│
├── artifacts/                    ← Generated at runtime (git-ignored)
│   ├── raw.csv / train.csv / test.csv
│   ├── Model.pkl                 ← Best model (serialised)
│   ├── transform.pkl             ← Fitted preprocessor
│   └── model_scores.json        ← All 8 model R² scores
│
├── logs/                         ← Runtime logs (git-ignored)
│
└── src/
    ├── core/                     ← Shared infrastructure
    │   ├── logger.py             ← Logging configuration
    │   ├── exceptions.py         ← Centralised exception handler
    │   └── utils.py              ← save_object, load_object, evaluate_models
    │
    ├── pipeline/
    │   ├── train_pipeline.py     ← Orchestrates stages 1-3
    │   ├── prediction_pipeline.py ← Single-row inference
    │   └── stages/
    │       ├── data_ingestion.py      ← Load CSV → train/test split
    │       ├── data_transformation.py ← Preprocessor fit/transform
    │       └── model_trainer.py       ← GridSearchCV + best-model selection
    │
    └── analytics/
        ├── eda_engine.py         ← Statistical analysis for the dashboard
        └── model_insights.py     ← Inference metrics, leaderboard, feature importance
```

## Quickstart

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard (auto-trains on first launch)
streamlit run app.py
```

## Dataset

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | male / female |
| `race_ethnicity` | Categorical | group A–E |
| `parental_level_of_education` | Ordinal | some high school → master's degree |
| `lunch` | Categorical | standard / free/reduced |
| `test_preparation_course` | Categorical | none / completed |
| `reading_score` | Numeric | 0–100 |
| `writing_score` | Numeric | 0–100 |
| `math_score` | Numeric **(target)** | 0–100 |

## Models Evaluated

Linear Regression · Decision Tree · Random Forest · Gradient Boosting · K-Nearest Neighbors · XGBoost · CatBoost · AdaBoost

**Best model: Linear Regression — R² = 0.8804 on the test set**

## Dashboard Tabs

| Tab | Contents |
|---|---|
| 📊 Dashboard | KPI cards, score-band donut, violin chart, correlation matrix |
| 🔍 EDA | Distributions, gender/ethnicity charts, socioeconomic factors |
| 🤖 Model Performance | 8-model leaderboard, actual vs predicted, residual heatmap |
| 💡 Feature Impact | Feature importance or EDA-based factor lift analysis |
| 🎯 Predict | Styled form → gauge chart → score + percentile result |