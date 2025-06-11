# EDA.py - Enhanced version with all new features
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# NLTK imports with error handling
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.corpus import stopwords
    
    # Download required NLTK data only if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("NLTK not available. Text preprocessing features will be limited.")

from scipy import stats
from scipy.interpolate import interp1d

# ================= UTILITY FUNCTIONS =================
def safe_execute(func, *args, **kwargs):
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, str(e)
def load_data(file):
    return pd.read_csv(file)

# Function to find categorical and numerical columns/variables in dataset
def categorical_numerical(df):
    num_columns,cat_columns = [],[]
    for col in df.columns:
        if len(df[col].unique()) <= 30 or df[col].dtype== np.object_:
            cat_columns.append(col.strip())

        else:
            num_columns.append(col.strip())

    return num_columns,cat_columns
def load_data(file):
    return pd.read_csv(file)

# Function to find categorical and numerical columns/variables in dataset
def categorical_numerical(df):
    num_columns,cat_columns = [],[]
    for col in df.columns:
        if len(df[col].unique()) <= 30 or df[col].dtype== np.object_:
            cat_columns.append(col.strip())

        else:
            num_columns.append(col.strip())

    return num_columns,cat_columns


# Function to display dataset overview
def display_dataset_overview(df,cat_columns,num_columns):
    
    display_rows = st.slider("Display Rows", 1, len(df), len(df) if len(df) < 20 else 20)

    st.write(df.head(display_rows))

    st.subheader("2. Dataset Overview")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")
    st.write(f"**Duplicates:** {df.shape[0] - df.drop_duplicates().shape[0]}")
    st.write(f"**Categorical Columns:** {len(cat_columns)}")
    st.write(cat_columns)
    st.write(f"**Numerical Columns:** {len(num_columns)}")
    st.write(num_columns)
    

# Function to find the missing values in the dataset
def display_missing_values(df):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    if not missing_data.empty:
        st.write("Missing Data Summary:")
        st.write(missing_data)

    else:
        st.info("No Missing Value present in the Dataset")

# Function to display basic statistics and visualizations about the dataset
def display_statistics_visualization(df,cat_columns,num_columns):
    st.write("Summary Statistics for Numerical Columns")

    if len(num_columns)!=0:
        num_df = df[num_columns]
        st.write(num_df.describe())

    else:
        st.info("The dataset does not have any numerical columns")

    
    st.write("Statistics for Categorical Columns")
    if len(cat_columns)!=0:
        num_cat_columns = st.number_input("Select the number of categorical columns to visualize:",min_value=1,max_value=len(cat_columns))
        selected_cat_columns = st.multiselect("Select the Categorical Columns for bar chart",cat_columns,cat_columns[:num_cat_columns])

        for column in selected_cat_columns:
            st.write(f"**{column}**")
            value_counts = df[column].value_counts()
            st.bar_chart(value_counts)

            # display the value count in tabular format
            st.write(f"Value Count for {column}")
            value_counts_table = df[column].value_counts().reset_index()
            value_counts_table.columns = ['Value','Count']
            st.write(value_counts_table)

    else:
        st.info("The dataset does not have any categorical columns")

# Funciton to display the datatypes
def display_data_types(df):

    data_types_df = pd.DataFrame({'Data Type':df.dtypes})
    st.write(data_types_df)

# Function to search for a particular column or particular datatype in the dataset
def search_column(df):
    search_query = st.text_input("Search for a column:")

    selected_data_type = st.selectbox("Filter by Data Type:", ['All'] + df.dtypes.unique().tolist())

    # Apply filters to the DataFrame
    filtered_df = df.copy()

    # Filter by search query
    if search_query:
        filtered_df = filtered_df.loc[:, filtered_df.columns.str.contains(search_query, case=False)]

    # Filter by data type
    if selected_data_type != 'All':
        filtered_df = filtered_df.select_dtypes(include=[selected_data_type])

    # Display the filtered DataFrame
    st.write(filtered_df)



## FUNCTIONS FOR TAB2: Data Exploration and Visualization

def display_individual_feature_distribution(df,num_columns):
    st.subheader("Analyze Individual Feature Distribution")
    st.markdown("Here, you can explore individual numerical features, visualize their distributions, and analyze relationships between features.")

    if len(num_columns) == 0:
        st.info("The dataset does not have any numerical columns")
        return

    st.write("#### Understanding Numerical Features")
    feature = st.selectbox(label="Select Numerical Feature", options=num_columns, index=0)
    df_description = df.describe()

    # Display summary statistics
    null_count = df[feature].isnull().sum()
    st.write("Count: ", df_description[feature]['count'])
    st.write("Missing Count: ", null_count)
    st.write("Mean: ", df_description[feature]['mean'])
    st.write("Standard Deviation: ", df_description[feature]['std'])
    st.write("Minimum: ", df_description[feature]['min'])
    st.write("Maximum: ", df_description[feature]['max'])

    # create plots for distribution
    st.subheader("Distribution Plots")
    plot_type = st.selectbox(label="Select Plot Type",options=['Histogram','Scatter Plot','Density Plot','Box Plot'])

    if plot_type=='Histogram':
        fig=px.histogram(df,x=feature,title=f'Histogram of {feature}')

    elif plot_type=='Scatter Plot':
        fig = px.scatter(df,x=feature,y=feature,title=f'Scatter plot of {feature}')

    elif plot_type=='Density Plot':
        fig = px.density_contour(df,x=feature,title=f'Density plot of {feature}')

    elif plot_type=='Box Plot':
        fig = px.box(df,y=feature,title=f'Box plot of {feature}')

    st.plotly_chart(fig,use_container_width=True)


def display_scatter_plot_of_two_numeric_features(df,num_columns):

    if len(num_columns) == 0:
        st.info("The dataset does not have any numerical columns")
        return
    
    if len(num_columns)!=0:
        x_feature = st.selectbox(label="Select X-Axis Feature", options=num_columns, index=0)
        y_feature = st.selectbox(label="Select Y-Axis Feature", options=num_columns, index=1)

        scatter_fig = px.scatter(df, x=x_feature, y=y_feature, title=f'Scatter Plot: {x_feature} vs {y_feature}')
        st.plotly_chart(scatter_fig, use_container_width=True)



def categorical_variable_analysis(df,cat_columns):

    categorical_feature = st.selectbox(label="Select Categorical Feature",options=cat_columns)
    categorical_plot_type = st.selectbox(label="Select Plot Type",options=["Bar Chart","Pie Chart","Stacked Bar Chart","Frequency Count"])
    
    if categorical_plot_type =="Bar Chart":
        fig = px.bar(df,x=categorical_feature,title=f"Bar Chart of {categorical_feature}")

    elif categorical_plot_type == "Pie Chart":
        fig = px.pie(df,names=categorical_feature,title=f"Pie Chart of {categorical_feature}")

    elif categorical_plot_type == "Stacked Bar Chart":
        st.write("Select a second categorical feature for stacking")
        second_categorical_feature = st.selectbox(label="Select Second Categorical Feature",options=cat_columns)

        fig = px.bar(df,x=categorical_feature,color=second_categorical_feature,title=f"Stacked Bar Chart of {categorical_feature} by {second_categorical_feature}")

    elif categorical_plot_type == "Frequency Count":
        cat_value_counts = df[categorical_feature].value_counts()
        st.write(f"Frequency Count for {categorical_feature}: ")
        st.write(cat_value_counts)

    if categorical_plot_type!= "Frequency Count" and fig is not None:
        st.plotly_chart(fig,use_container_width=True) 


def feature_exploration_numerical_variables(df,num_columns):
    selected_features = st.multiselect("Select Features for Exploration:", num_columns, default=num_columns[:2], key="feature_exploration")

    if len(selected_features) < 2:
        st.warning("Please select at least two numerical features for exploration.")
    else:
        st.subheader("Explore Relationships Between Features")

        # Scatter Plot Matrix
        if st.button("Generate Scatter Plot Matrix"):
            scatter_matrix_fig = px.scatter_matrix(df, dimensions=selected_features, title="Scatter Plot Matrix")
            st.plotly_chart(scatter_matrix_fig, use_container_width=True)

        # Pair Plot
        if st.button("Generate Pair Plot"):
            pair_plot_fig = sns.pairplot(df[selected_features])
            st.pyplot(pair_plot_fig)

        # Correlation Heatmap
        if st.button("Generate Correlation Heatmap"):
            correlation_matrix = df[selected_features].corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("Correlation Heatmap")
            st.pyplot(plt)     


def categorical_numerical_variable_analysis(df,cat_columns,num_columns):
    categorical_feature_1 = st.selectbox(label="Categorical Feature", options=cat_columns)        
    numerical_feature_1 = st.selectbox(label="Numerical Feature", options=num_columns)

# Group by the selected categorical column and calculate the mean of the numerical column
    group_data = df.groupby(categorical_feature_1)[numerical_feature_1].mean().reset_index()

    st.subheader("Relationship between Categorical and Numerical Variables")
    st.write(f"Mean {numerical_feature_1} by {categorical_feature_1}")
    
    # Create a bar chart
    fig = px.bar(group_data, x=categorical_feature_1, y=numerical_feature_1, title=f"{numerical_feature_1} by {categorical_feature_1}")
    st.plotly_chart(fig, use_container_width=True)

# Function to display dataset overview
def display_dataset_overview(df,cat_columns,num_columns):
    
    display_rows = st.slider("Display Rows", 1, len(df), len(df) if len(df) < 20 else 20)

    st.write(df.head(display_rows))

    st.subheader("2. Dataset Overview")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")
    st.write(f"**Duplicates:** {df.shape[0] - df.drop_duplicates().shape[0]}")
    st.write(f"**Categorical Columns:** {len(cat_columns)}")
    st.write(cat_columns)
    st.write(f"**Numerical Columns:** {len(num_columns)}")
    st.write(num_columns)
    

# Function to find the missing values in the dataset
def display_missing_values(df):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    if not missing_data.empty:
        st.write("Missing Data Summary:")
        st.write(missing_data)

    else:
        st.info("No Missing Value present in the Dataset")

# Function to display basic statistics and visualizations about the dataset
def display_statistics_visualization(df,cat_columns,num_columns):
    st.write("Summary Statistics for Numerical Columns")

    if len(num_columns)!=0:
        num_df = df[num_columns]
        st.write(num_df.describe())

    else:
        st.info("The dataset does not have any numerical columns")

    
    st.write("Statistics for Categorical Columns")
    if len(cat_columns)!=0:
        num_cat_columns = st.number_input("Select the number of categorical columns to visualize:",min_value=1,max_value=len(cat_columns))
        selected_cat_columns = st.multiselect("Select the Categorical Columns for bar chart",cat_columns,cat_columns[:num_cat_columns])

        for column in selected_cat_columns:
            st.write(f"**{column}**")
            value_counts = df[column].value_counts()
            st.bar_chart(value_counts)

            # display the value count in tabular format
            st.write(f"Value Count for {column}")
            value_counts_table = df[column].value_counts().reset_index()
            value_counts_table.columns = ['Value','Count']
            st.write(value_counts_table)

    else:
        st.info("The dataset does not have any categorical columns")

# Funciton to display the datatypes
def display_data_types(df):

    data_types_df = pd.DataFrame({'Data Type':df.dtypes})
    st.write(data_types_df)

# Function to search for a particular column or particular datatype in the dataset
def search_column(df):
    search_query = st.text_input("Search for a column:")

    selected_data_type = st.selectbox("Filter by Data Type:", ['All'] + df.dtypes.unique().tolist())

    # Apply filters to the DataFrame
    filtered_df = df.copy()

    # Filter by search query
    if search_query:
        filtered_df = filtered_df.loc[:, filtered_df.columns.str.contains(search_query, case=False)]

    # Filter by data type
    if selected_data_type != 'All':
        filtered_df = filtered_df.select_dtypes(include=[selected_data_type])

    # Display the filtered DataFrame
    st.write(filtered_df)



## FUNCTIONS FOR TAB2: Data Exploration and Visualization

def display_individual_feature_distribution(df,num_columns):
    st.subheader("Analyze Individual Feature Distribution")
    st.markdown("Here, you can explore individual numerical features, visualize their distributions, and analyze relationships between features.")

    if len(num_columns) == 0:
        st.info("The dataset does not have any numerical columns")
        return

    st.write("#### Understanding Numerical Features")
    feature = st.selectbox(label="Select Numerical Feature", options=num_columns, index=0)
    df_description = df.describe()

    # Display summary statistics
    null_count = df[feature].isnull().sum()
    st.write("Count: ", df_description[feature]['count'])
    st.write("Missing Count: ", null_count)
    st.write("Mean: ", df_description[feature]['mean'])
    st.write("Standard Deviation: ", df_description[feature]['std'])
    st.write("Minimum: ", df_description[feature]['min'])
    st.write("Maximum: ", df_description[feature]['max'])

    # create plots for distribution
    st.subheader("Distribution Plots")
    plot_type = st.selectbox(label="Select Plot Type",options=['Histogram','Scatter Plot','Density Plot','Box Plot'])

    if plot_type=='Histogram':
        fig=px.histogram(df,x=feature,title=f'Histogram of {feature}')

    elif plot_type=='Scatter Plot':
        fig = px.scatter(df,x=feature,y=feature,title=f'Scatter plot of {feature}')

    elif plot_type=='Density Plot':
        fig = px.density_contour(df,x=feature,title=f'Density plot of {feature}')

    elif plot_type=='Box Plot':
        fig = px.box(df,y=feature,title=f'Box plot of {feature}')

    st.plotly_chart(fig,use_container_width=True)


def display_scatter_plot_of_two_numeric_features(df,num_columns):

    if len(num_columns) == 0:
        st.info("The dataset does not have any numerical columns")
        return
    
    if len(num_columns)!=0:
        x_feature = st.selectbox(label="Select X-Axis Feature", options=num_columns, index=0)
        y_feature = st.selectbox(label="Select Y-Axis Feature", options=num_columns, index=1)

        scatter_fig = px.scatter(df, x=x_feature, y=y_feature, title=f'Scatter Plot: {x_feature} vs {y_feature}')
        st.plotly_chart(scatter_fig, use_container_width=True)



def categorical_variable_analysis(df,cat_columns):

    categorical_feature = st.selectbox(label="Select Categorical Feature",options=cat_columns)
    categorical_plot_type = st.selectbox(label="Select Plot Type",options=["Bar Chart","Pie Chart","Stacked Bar Chart","Frequency Count"])
    
    if categorical_plot_type =="Bar Chart":
        fig = px.bar(df,x=categorical_feature,title=f"Bar Chart of {categorical_feature}")

    elif categorical_plot_type == "Pie Chart":
        fig = px.pie(df,names=categorical_feature,title=f"Pie Chart of {categorical_feature}")

    elif categorical_plot_type == "Stacked Bar Chart":
        st.write("Select a second categorical feature for stacking")
        second_categorical_feature = st.selectbox(label="Select Second Categorical Feature",options=cat_columns)

        fig = px.bar(df,x=categorical_feature,color=second_categorical_feature,title=f"Stacked Bar Chart of {categorical_feature} by {second_categorical_feature}")

    elif categorical_plot_type == "Frequency Count":
        cat_value_counts = df[categorical_feature].value_counts()
        st.write(f"Frequency Count for {categorical_feature}: ")
        st.write(cat_value_counts)

    if categorical_plot_type!= "Frequency Count" and fig is not None:
        st.plotly_chart(fig,use_container_width=True) 


def feature_exploration_numerical_variables(df,num_columns):
    selected_features = st.multiselect("Select Features for Exploration:", num_columns, default=num_columns[:2], key="feature_exploration")

    if len(selected_features) < 2:
        st.warning("Please select at least two numerical features for exploration.")
    else:
        st.subheader("Explore Relationships Between Features")

        # Scatter Plot Matrix
        if st.button("Generate Scatter Plot Matrix"):
            scatter_matrix_fig = px.scatter_matrix(df, dimensions=selected_features, title="Scatter Plot Matrix")
            st.plotly_chart(scatter_matrix_fig, use_container_width=True)

        # Pair Plot
        if st.button("Generate Pair Plot"):
            pair_plot_fig = sns.pairplot(df[selected_features])
            st.pyplot(pair_plot_fig)

        # Correlation Heatmap
        if st.button("Generate Correlation Heatmap"):
            correlation_matrix = df[selected_features].corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("Correlation Heatmap")
            st.pyplot(plt)     


def categorical_numerical_variable_analysis(df,cat_columns,num_columns):
    categorical_feature_1 = st.selectbox(label="Categorical Feature", options=cat_columns)        
    numerical_feature_1 = st.selectbox(label="Numerical Feature", options=num_columns)

# Group by the selected categorical column and calculate the mean of the numerical column
    group_data = df.groupby(categorical_feature_1)[numerical_feature_1].mean().reset_index()

    st.subheader("Relationship between Categorical and Numerical Variables")
    st.write(f"Mean {numerical_feature_1} by {categorical_feature_1}")
    
    # Create a bar chart
    fig = px.bar(group_data, x=categorical_feature_1, y=numerical_feature_1, title=f"{numerical_feature_1} by {categorical_feature_1}")
    st.plotly_chart(fig, use_container_width=True)
# ================= EXISTING PREPROCESSING FUNCTIONS =================
def remove_selected_columns(df, columns_remove):
    """Remove selected columns from dataframe"""
    if not columns_remove:
        return df
    return df.drop(columns=columns_remove, errors='ignore')

def remove_rows_with_missing_data(df, columns):
    """Remove rows with missing data in specified columns"""
    if columns:
        return df.dropna(subset=columns)
    return df

def fill_missing_data(df, columns, method):
    """Fill missing data using specified method"""
    df_copy = df.copy()
    for column in columns:
        if column in df_copy.columns:
            if method == 'mean' and pd.api.types.is_numeric_dtype(df_copy[column]):
                df_copy[column].fillna(df_copy[column].mean(), inplace=True)
            elif method == 'median' and pd.api.types.is_numeric_dtype(df_copy[column]):
                df_copy[column].fillna(df_copy[column].median(), inplace=True)
            elif method == 'mode':
                mode_val = df_copy[column].mode()
                if len(mode_val) > 0:
                    df_copy[column].fillna(mode_val.iloc[0], inplace=True)
    return df_copy

def one_hot_encode(df, columns):
    """Apply one-hot encoding to categorical columns"""
    if not columns:
        return df
    return pd.get_dummies(df, columns=columns, prefix=columns, drop_first=False)

def label_encode(df, columns):
    """Apply label encoding to categorical columns"""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    return df_copy

def standard_scale(df, columns):
    """Apply standard scaling to numerical columns"""
    df_copy = df.copy()
    if columns:
        scaler = StandardScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy

def min_max_scale(df, columns, feature_range=(0, 1)):
    """Apply min-max scaling to numerical columns"""
    df_copy = df.copy()
    if columns:
        scaler = MinMaxScaler(feature_range=feature_range)
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy

def detect_outliers_iqr(df, column_name):
    """Detect outliers using IQR method"""
    if column_name not in df.columns:
        return []
    data = df[column_name].dropna()
    if len(data) == 0:
        return []
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    outliers = data[(data < lower) | (data > upper)].tolist()
    return sorted(outliers)

def detect_outliers_zscore(df, column_name, threshold=3):
    """Detect outliers using Z-score method"""
    if column_name not in df.columns:
        return []
    data = df[column_name].dropna()
    if len(data) == 0:
        return []
    z_scores = np.abs(stats.zscore(data))
    return data[z_scores > threshold].tolist()

def remove_outliers(df, column_name, outliers):
    """Remove outliers from dataframe"""
    if not outliers or column_name not in df.columns:
        return df
    return df[~df[column_name].isin(outliers)]

def transform_outliers(df, column_name, outliers, method='median'):
    """Transform outliers using specified method"""
    df_copy = df.copy()
    if not outliers or column_name not in df_copy.columns:
        return df_copy
    
    if method == 'median':
        replacement_value = df_copy[~df_copy[column_name].isin(outliers)][column_name].median()
    elif method == 'mean':
        replacement_value = df_copy[~df_copy[column_name].isin(outliers)][column_name].mean()
    else:  # cap
        q25, q75 = np.percentile(df_copy[column_name].dropna(), [25, 75])
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        df_copy[column_name] = df_copy[column_name].clip(lower=lower, upper=upper)
        return df_copy
    
    df_copy.loc[df_copy[column_name].isin(outliers), column_name] = replacement_value
    return df_copy

# ================= NEW FEATURE 1: DATA TYPE CONVERSION =================
def convert_data_types(df, columns, target_type):
    """Convert data types of specified columns"""
    df_copy = df.copy()
    converted_cols = []
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        try:
            if target_type == 'datetime':
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            elif target_type == 'numeric':
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            elif target_type == 'category':
                df_copy[col] = df_copy[col].astype('category')
            elif target_type == 'string':
                df_copy[col] = df_copy[col].astype(str)
            elif target_type == 'boolean':
                df_copy[col] = df_copy[col].astype(bool)
            converted_cols.append(col)
        except Exception as e:
            st.warning(f"Could not convert {col} to {target_type}: {str(e)}")
    
    if converted_cols:
        st.success(f"Successfully converted {len(converted_cols)} columns to {target_type}")
    
    return df_copy

def detect_data_type_issues(df):
    """Detect potential data type issues"""
    issues = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if it could be numeric
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if numeric_count / len(df) > 0.8:
                issues.append(f"{col}: Likely numeric (stored as object)")
            
            # Check if it could be datetime
            try:
                datetime_count = pd.to_datetime(df[col], errors='coerce').notna().sum()
                if datetime_count / len(df) > 0.8:
                    issues.append(f"{col}: Likely datetime (stored as object)")
            except:
                pass
    
    return issues

# ================= NEW FEATURE 2: DATETIME FEATURE ENGINEERING =================
def datetime_feature_engineering(df, datetime_col, features_to_extract=None):
    """Extract datetime features from datetime column"""
    df_copy = df.copy()
    
    if datetime_col not in df_copy.columns:
        return df_copy
    
    # Ensure column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy[datetime_col]):
        df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col], errors='coerce')
    
    if features_to_extract is None:
        features_to_extract = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'quarter', 'dayofyear']
    
    extracted_features = []
    
    for feature in features_to_extract:
        try:
            if feature == 'year':
                df_copy[f'{datetime_col}_year'] = df_copy[datetime_col].dt.year
                extracted_features.append('year')
            elif feature == 'month':
                df_copy[f'{datetime_col}_month'] = df_copy[datetime_col].dt.month
                extracted_features.append('month')
            elif feature == 'day':
                df_copy[f'{datetime_col}_day'] = df_copy[datetime_col].dt.day
                extracted_features.append('day')
            elif feature == 'weekday':
                df_copy[f'{datetime_col}_weekday'] = df_copy[datetime_col].dt.weekday
                extracted_features.append('weekday')
            elif feature == 'hour':
                df_copy[f'{datetime_col}_hour'] = df_copy[datetime_col].dt.hour
                extracted_features.append('hour')
            elif feature == 'minute':
                df_copy[f'{datetime_col}_minute'] = df_copy[datetime_col].dt.minute
                extracted_features.append('minute')
            elif feature == 'quarter':
                df_copy[f'{datetime_col}_quarter'] = df_copy[datetime_col].dt.quarter
                extracted_features.append('quarter')
            elif feature == 'dayofyear':
                df_copy[f'{datetime_col}_dayofyear'] = df_copy[datetime_col].dt.dayofyear
                extracted_features.append('dayofyear')
        except Exception as e:
            st.warning(f"Could not extract {feature}: {str(e)}")
    
    if extracted_features:
        st.success(f"Extracted features: {', '.join(extracted_features)}")
    
    return df_copy

def create_date_differences(df, date_col1, date_col2, unit='days'):
    """Create date difference features"""
    df_copy = df.copy()
    
    if date_col1 not in df_copy.columns or date_col2 not in df_copy.columns:
        return df_copy
    
    # Ensure columns are datetime
    df_copy[date_col1] = pd.to_datetime(df_copy[date_col1], errors='coerce')
    df_copy[date_col2] = pd.to_datetime(df_copy[date_col2], errors='coerce')
    
    diff = df_copy[date_col2] - df_copy[date_col1]
    
    if unit == 'days':
        df_copy[f'{date_col2}_{date_col1}_days'] = diff.dt.days
    elif unit == 'hours':
        df_copy[f'{date_col2}_{date_col1}_hours'] = diff.dt.total_seconds() / 3600
    elif unit == 'minutes':
        df_copy[f'{date_col2}_{date_col1}_minutes'] = diff.dt.total_seconds() / 60
    
    return df_copy

# ================= NEW FEATURE 3: TEXT PREPROCESSING =================
def text_preprocessing(df, text_col, steps):
    """Comprehensive text preprocessing"""
    if not NLTK_AVAILABLE:
        st.error("NLTK not available for text preprocessing")
        return df
    
    df_copy = df.copy()
    
    if text_col not in df_copy.columns:
        return df_copy
    
    # Convert to string
    df_copy[text_col] = df_copy[text_col].astype(str)
    
    processed_steps = []
    
    try:
        if 'lowercase' in steps:
            df_copy[text_col] = df_copy[text_col].str.lower()
            processed_steps.append('lowercase')
        
        if 'remove_punct' in steps:
            df_copy[text_col] = df_copy[text_col].apply(lambda x: re.sub(r'[^\w\s]', '', x))
            processed_steps.append('remove_punct')
        
        if 'remove_stopwords' in steps:
            stop_words = set(stopwords.words('english'))
            df_copy[text_col] = df_copy[text_col].apply(
                lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words])
            )
            processed_steps.append('remove_stopwords')
        
        if 'stemming' in steps:
            ps = PorterStemmer()
            df_copy[text_col] = df_copy[text_col].apply(
                lambda x: ' '.join([ps.stem(word) for word in word_tokenize(x)])
            )
            processed_steps.append('stemming')
        
        if 'lemmatization' in steps:
            lemmatizer = WordNetLemmatizer()
            df_copy[text_col] = df_copy[text_col].apply(
                lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)])
            )
            processed_steps.append('lemmatization')
        
        if 'word_count' in steps:
            df_copy[f'{text_col}_word_count'] = df_copy[text_col].apply(
                lambda x: len(word_tokenize(x))
            )
            processed_steps.append('word_count')
        
        if 'char_count' in steps:
            df_copy[f'{text_col}_char_count'] = df_copy[text_col].apply(len)
            processed_steps.append('char_count')
        
        if 'remove_numbers' in steps:
            df_copy[text_col] = df_copy[text_col].apply(lambda x: re.sub(r'\d+', '', x))
            processed_steps.append('remove_numbers')
        
        if 'remove_whitespace' in steps:
            df_copy[text_col] = df_copy[text_col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
            processed_steps.append('remove_whitespace')
        
        if processed_steps:
            st.success(f"Applied text preprocessing steps: {', '.join(processed_steps)}")
    
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
    
    return df_copy

# ================= NEW FEATURE 4: FEATURE SELECTION =================
def feature_selection(df, method, target_col=None, threshold=0.8, k=10):
    """Advanced feature selection methods"""
    df_copy = df.copy()
    
    try:
        if method == 'variance':
            # Remove low variance features
            selector = VarianceThreshold(threshold=threshold)
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_data = selector.fit_transform(df_copy[numeric_cols])
                selected_cols = numeric_cols[selector.get_support()]
                
                # Keep non-numeric columns
                non_numeric_cols = df_copy.select_dtypes(exclude=[np.number]).columns
                result_df = pd.concat([
                    pd.DataFrame(selected_data, columns=selected_cols, index=df_copy.index),
                    df_copy[non_numeric_cols]
                ], axis=1)
                
                st.success(f"Removed {len(numeric_cols) - len(selected_cols)} low variance features")
                return result_df
        
        elif method == 'correlation':
            # Remove highly correlated features
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df_copy[numeric_cols].corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                to_drop = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > threshold)]
                
                result_df = df_copy.drop(columns=to_drop)
                st.success(f"Removed {len(to_drop)} highly correlated features")
                return result_df
        
        elif method == 'univariate' and target_col and target_col in df_copy.columns:
            # Univariate feature selection
            X = df_copy.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
            y = df_copy[target_col]
            
            if len(X.columns) > 0:
                # Determine if regression or classification
                if pd.api.types.is_numeric_dtype(y):
                    selector = SelectKBest(f_regression, k=min(k, len(X.columns)))
                else:
                    selector = SelectKBest(f_classif, k=min(k, len(X.columns)))
                
                X_selected = selector.fit_transform(X, y)
                selected_cols = X.columns[selector.get_support()]
                
                # Combine with non-numeric columns and target
                non_numeric_cols = df_copy.select_dtypes(exclude=[np.number]).columns
                result_df = pd.concat([
                    pd.DataFrame(X_selected, columns=selected_cols, index=df_copy.index),
                    df_copy[non_numeric_cols],
                    df_copy[[target_col]]
                ], axis=1)
                
                st.success(f"Selected top {len(selected_cols)} features using univariate selection")
                return result_df
    
    except Exception as e:
        st.error(f"Error in feature selection: {str(e)}")
    
    return df_copy

# ================= NEW FEATURE 5: ADVANCED IMPUTATION =================
def knn_imputation(df, columns, n_neighbors=5):
    """KNN imputation for missing values"""
    df_copy = df.copy()
    
    if not columns:
        return df_copy
    
    try:
        # Only apply to numeric columns
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df_copy[col])]
        
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
            st.success(f"Applied KNN imputation to {len(numeric_cols)} columns")
        else:
            st.warning("No numeric columns selected for KNN imputation")
    
    except Exception as e:
        st.error(f"Error in KNN imputation: {str(e)}")
    
    return df_copy

def interpolation_imputation(df, columns, method='linear'):
    """Interpolation imputation for time series data"""
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            try:
                if method == 'linear':
                    df_copy[col] = df_copy[col].interpolate(method='linear')
                elif method == 'polynomial':
                    df_copy[col] = df_copy[col].interpolate(method='polynomial', order=2)
                elif method == 'spline':
                    df_copy[col] = df_copy[col].interpolate(method='spline', order=2)
            except Exception as e:
                st.warning(f"Could not interpolate {col}: {str(e)}")
    
    return df_copy

def regression_imputation(df, columns, target_col):
    """Use regression to predict missing values"""
    df_copy = df.copy()
    
    if target_col not in df_copy.columns:
        return df_copy
    
    try:
        # Prepare data
        X = df_copy.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y = df_copy[target_col]
        
        # Find rows with missing target values
        missing_mask = y.isna()
        
        if missing_mask.sum() > 0 and not X.empty:
            # Train on complete cases
            X_train = X[~missing_mask].fillna(X.mean())
            y_train = y[~missing_mask]
            
            # Predict missing values
            X_predict = X[missing_mask].fillna(X.mean())
            
            if len(X_train) > 0 and len(X_predict) > 0:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                predictions = model.predict(X_predict)
                df_copy.loc[missing_mask, target_col] = predictions
                
                st.success(f"Imputed {missing_mask.sum()} missing values using regression")
    
    except Exception as e:
        st.error(f"Error in regression imputation: {str(e)}")
    
    return df_copy

# ================= NEW FEATURE 6: ADVANCED ENCODING =================
def target_encoding(df, categorical_cols, target_col, smoothing=1.0):
    """Target encoding for categorical variables"""
    df_copy = df.copy()
    
    if target_col not in df_copy.columns:
        return df_copy
    
    try:
        global_mean = df_copy[target_col].mean()
        
        for col in categorical_cols:
            if col in df_copy.columns:
                # Calculate target mean for each category
                target_means = df_copy.groupby(col)[target_col].agg(['mean', 'count'])
                
                # Apply smoothing
                smoothed_means = (target_means['mean'] * target_means['count'] + 
                                global_mean * smoothing) / (target_means['count'] + smoothing)
                
                # Create new encoded column
                df_copy[f'{col}_target_encoded'] = df_copy[col].map(smoothed_means).fillna(global_mean)
        
        st.success(f"Applied target encoding to {len(categorical_cols)} columns")
    
    except Exception as e:
        st.error(f"Error in target encoding: {str(e)}")
    
    return df_copy

def frequency_encoding(df, columns):
    """Frequency encoding for categorical variables"""
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            try:
                freq_map = df_copy[col].value_counts(normalize=True).to_dict()
                df_copy[f'{col}_freq_encoded'] = df_copy[col].map(freq_map)
            except Exception as e:
                st.warning(f"Could not apply frequency encoding to {col}: {str(e)}")
    
    return df_copy

# ================= NEW FEATURE 7: DUPLICATE DETECTION =================
def detect_duplicates(df):
    """Detect duplicate rows"""
    duplicates = df.duplicated()
    duplicate_count = duplicates.sum()
    
    return {
        'total_duplicates': duplicate_count,
        'duplicate_percentage': (duplicate_count / len(df)) * 100,
        'duplicate_indices': df[duplicates].index.tolist()
    }

def remove_duplicates(df, subset=None, keep='first'):
    """Remove duplicate rows"""
    original_length = len(df)
    df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = original_length - len(df_cleaned)
    
    if removed_count > 0:
        st.success(f"Removed {removed_count} duplicate rows")
    
    return df_cleaned

# ================= NEW FEATURE 8: PCA =================
def apply_pca(df, n_components=None, variance_threshold=0.95):
    """Apply PCA for dimensionality reduction"""
    try:
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for PCA")
            return df
        
        X = df[numeric_cols].fillna(0)  # Fill NaN values
        
        if n_components is None:
            # Find number of components for variance threshold
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= variance_threshold) + 1
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create new dataframe with PCA components
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        # Add non-numeric columns back
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            result_df = pd.concat([pca_df, df[non_numeric_cols]], axis=1)
        else:
            result_df = pca_df
        
        # Store PCA info in session state for visualization
        if 'pca_info' not in st.session_state:
            st.session_state.pca_info = {}
        
        st.session_state.pca_info = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'n_components': n_components
        }
        
        st.success(f"Applied PCA: {len(numeric_cols)} → {n_components} components "
                  f"(explaining {pca.explained_variance_ratio_.sum():.2%} of variance)")
        
        return result_df
    
    except Exception as e:
        st.error(f"Error in PCA: {str(e)}")
        return df

# ================= NEW FEATURE 9: FEATURE TRANSFORMATION =================
def transform_features(df, columns, method):
    """Transform features using various methods"""
    df_copy = df.copy()
    
    try:
        for col in columns:
            if col not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[col]):
                continue
            
            if method == 'log':
                # Add small constant to handle zeros
                df_copy[f'{col}_log'] = np.log1p(df_copy[col].clip(lower=0))
            
            elif method == 'sqrt':
                df_copy[f'{col}_sqrt'] = np.sqrt(df_copy[col].clip(lower=0))
            
            elif method == 'square':
                df_copy[f'{col}_square'] = df_copy[col] ** 2
            
            elif method == 'reciprocal':
                # Avoid division by zero
                df_copy[f'{col}_reciprocal'] = 1 / (df_copy[col] + 1e-8)
            
            elif method == 'boxcox':
                # Box-Cox requires positive values
                if (df_copy[col] > 0).all():
                    pt = PowerTransformer(method='box-cox')
                    df_copy[f'{col}_boxcox'] = pt.fit_transform(df_copy[[col]]).flatten()
                else:
                    st.warning(f"Box-Cox requires positive values. Skipping {col}")
            
            elif method == 'yeojohnson':
                pt = PowerTransformer(method='yeo-johnson')
                df_copy[f'{col}_yeojohnson'] = pt.fit_transform(df_copy[[col]]).flatten()
        
        st.success(f"Applied {method} transformation to {len(columns)} columns")
    
    except Exception as e:
        st.error(f"Error in feature transformation: {str(e)}")
    
    return df_copy

def bin_features(df, columns, method, bins=5):
    """Bin continuous features into categorical"""
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
        
        try:
            if method == 'equal_width':
                df_copy[f'{col}_binned'] = pd.cut(df_copy[col], bins=bins, labels=False)
            elif method == 'equal_freq':
                df_copy[f'{col}_binned'] = pd.qcut(df_copy[col], q=bins, labels=False, duplicates='drop')
            elif method == 'custom':
                # For custom binning, could add more sophisticated logic
                df_copy[f'{col}_binned'] = pd.cut(df_copy[col], bins=bins, labels=False)
        
        except Exception as e:
            st.warning(f"Could not bin {col}: {str(e)}")
    
    return df_copy

