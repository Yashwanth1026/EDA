Based on your provided files (`app.py` and `EDA.py`), here is a well-structured `README.md` description for your project:

---

# 📊 Advanced Automated EDA & Preprocessing Tool

An interactive **Streamlit web application** for performing **automated Exploratory Data Analysis (EDA)** and **advanced data preprocessing** on tabular datasets. It helps data scientists, analysts, and ML engineers clean, transform, and visualize data with minimal code using an intuitive GUI.

---

## 🚀 Features

### 🔍 Exploratory Data Analysis

* Dataset overview: shape, column types, and duplicates
* Data type inspection and issues detection
* Visual summary statistics for numerical & categorical variables
* Missing value visualization
* Column search and type filtering
* Scatter plots, distribution plots, pair plots, and heatmaps

### 🧹 Preprocessing Operations

#### Basic Preprocessing

* Remove columns and rows
* Fill or drop missing values
* Label and One-Hot Encoding
* Scaling: Standard, Min-Max

#### Advanced Imputation

* **KNN Imputation**
* **Interpolation (linear, spline, polynomial)**
* **Regression Imputation using Random Forest**

#### Encoding Techniques

* **Target Encoding** (with smoothing)
* **Frequency Encoding**

#### Feature Engineering

* **Datetime Feature Extraction** (year, month, weekday, etc.)
* **Text Preprocessing** (NLTK-based: stopwords, lemmatization, stemming, etc.)
* **Feature Binning** (equal width/frequency)
* **Feature Transformation** (log, sqrt, box-cox, etc.)

#### Feature Selection

* Variance Threshold
* Correlation Threshold
* Univariate Selection (ANOVA, F-regression)

#### Outlier Detection and Handling

* IQR-based and Z-score-based detection
* Remove or transform outliers (mean, median, capping)

#### Dimensionality Reduction

* **PCA (Principal Component Analysis)** with:

  * Fixed number of components
  * Variance retention threshold

#### Duplicate Management

* Detect and remove duplicate rows with column filters

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python (Pandas, Scikit-learn, NLTK, Plotly, Seaborn)
* **Libraries Used:**

  * `pandas`, `numpy`, `sklearn`, `nltk`, `plotly`, `seaborn`, `matplotlib`
  * `EDA.py`: All processing and visualization functions
  * `app.py`: Main Streamlit web interface

---



---




