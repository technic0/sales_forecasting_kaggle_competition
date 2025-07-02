# Retail Sales Forecasting using Machine Learning 📈

This repository contains the code for a sales forecasting project that achieved **3rd place** in the "Predykcja sprzedaży z pomocą Machine Learning" Kaggle competition. The goal was to predict weekly sales for various store-department combinations using historical data and supplementary features.

This project is designed to showcase a complete machine learning workflow, from data exploration and feature engineering to model tuning and ensembling, making it an ideal portfolio piece for a data scientist role.

---

## ✨ Key Features & Methodology

* **Feature Engineering**: The model's success is heavily reliant on a rich feature set, including:
    * **Time-Based Features**: Year, month, week of the year, and day of the year to capture seasonality.
    * **Lagged Sales Data**: Sales data from the same week and month of the previous year to capture historical trends.
    * **Statistical Features**: Rolling averages and aggregated statistics (mean, median, standard deviation) for each store and department.
    * **Interaction Features**: Combined features to capture unique store-department dynamics.

* **Modeling**:
    * **XGBoost**: A highly efficient and powerful gradient boosting library, utilized for its performance and speed.
    * **CatBoost**: Another robust gradient boosting library, which also performs exceptionally well on tabular data.
    * **GPU Acceleration**: Both models were configured to run on a GPU to significantly speed up the training process.

* **Hyperparameter Tuning**:
    * **Hyperopt**: Bayesian optimization was performed using the Hyperopt library to find the optimal hyperparameters for both XGBoost and CatBoost models, maximizing their predictive power.

* **Ensembling**:
    * **Averaging**: The final prediction is a simple, yet effective, arithmetic mean of the predictions from the optimized XGBoost and CatBoost models. This helps to reduce variance and improve generalization.

---

## 🛠️ Tech Stack

* **Data Manipulation**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn`, `xgboost`, `catboost`
* **Hyperparameter Optimization**: `hyperopt`
* **Model Interpretation**: `eli5`, `scikit-plot`
* **Visualization**: `matplotlib`, `seaborn`

---

## 📂 Repository Structure
```bash
.
├── input/                  # Data from the Kaggle competition should be placed here.
├── output/                 # (Not included) Directory where submission files are saved.
├── sales_forecast_final.ipynb # The original, working notebook from the competition.
└── README.md               # You are here!
```

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost catboost hyperopt eli5 scikit-plot matplotlib seaborn
    ```
    *Note: Ensure your environment supports GPU usage (e.g., CUDA) for the models to run as configured.*

3.  **Download the data:**
    * Download the competition data from the [Kaggle competition page](https://www.kaggle.com/competitions/predykcja-sprzeday/data).
    * Unzip the files and place all `.h5` and `.csv` files into a directory named `input/` at the root of the project.

4.  **Run the notebook:**
    * Open `sales_forecast_final.ipynb` in a Jupyter environment (like Jupyter Lab or VS Code).
    * Execute the cells sequentially to reproduce the data processing, model training, and prediction steps. The final submission file will be saved in the `output/` directory.
