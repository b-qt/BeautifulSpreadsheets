import matplotlib
import matplotlib.pyplot
from install_packages import import_packages
import_packages() # Import the necessary packages

import pandas as pd
import numpy as np
import numpy.core.multiarray

import matplotlib.pyplot as plt

import requests, subprocess, sys, re, warnings
warnings.filterwarnings('ignore')

# Convert the categorical columns to numerical columns
from sklearn.preprocessing import LabelEncoder # Encode target labels with value between 0 and n_classes-1.
from sklearn.preprocessing import StandardScaler # Removing the mean and scaling to unit variance
from sklearn.preprocessing import MinMaxScaler # Scaling features to a range

from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

from get_the_data import datasets as get_datasets
from explore_the_data import convert_categorical_columns, find_non_alphanumeric_data, replace_non_alphanumeric_data, clean_columns
from explore_the_data import handle_missing_values, remove_duplicates, remove_small_datasets, scale_datasets


# Now use streamlit to visualize the process
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'streamlit'])
import streamlit as st
from get_the_data import is_valid_url, convert_spreadsheet_to_csv, get_the_data, check_schema
from get_the_data import convert_bitly_to_url, convert_geni_to_url, convert_airtable_to_url, get_datasets


def plot_dataset_outliers(dataset: pd.DataFrame, dataset_outliers: list):
    """
        This function will plot the dataset and the outliers and display it:
        Parameters:
            dataset (pandas dataframe): The dataset
            dataset_outliers (list): The list of outliers
    """
    
    num_cols = 4
    num_rows = (len(dataset.columns) // num_cols) + (len(dataset.columns) % num_cols > 0)

    fig, axs = plt.subplots(num_rows, min(4, num_cols), figsize=(15, 5 * num_rows))
    axs = axs.flatten() if num_rows * num_cols > 1 else [axs]
    
    for idx, outliers in enumerate(dataset_outliers):
        col = dataset.columns[idx]
        # print(f"Dataset : {dataset.shape} {col}-Outliers: {len(outliers)}")

        axs[idx].plot(dataset[col], alpha=.9, linewidth=.8) # Plot the dataset values
        axs[idx].scatter(outliers, dataset[col][outliers], label='Outliers', alpha=.9, color='red', s=10) # Plot the outliers

        axs[idx].set_title(f'Dataset-{idx} {dataset.shape}', fontsize=8)
        axs[idx].set_xlabel(f'{col.replace("$", "")}', fontsize=8)
        axs[idx].set_ylabel('Value')
    
    plt.subplots_adjust(hspace=0.5) # Add a space between the plots
    plt.tight_layout()
    plt.show()

def detect_outliers(datasets: list):
    """
        This function will detect the outliers in the datasets.
        
        Parameters:
            datasets (list): The list of datasets
        
        Returns:
            list: The list of outlier scores
    """
    outlier_scores = []
    imputer = SimpleImputer(strategy='mean')  # Impute missing values with the mean
    model = IsolationForest(contamination=0.1) # 10% of the data are outliers

    with st.expander("Outliers detected"):
        for dataset in datasets: # For each dataset in the list of datasets
            dataset_outliers = [] # List to store the outliers for each dataset ... necessary?

            for col in dataset.columns: # For each column in the current dataset
                anomaly_indices = []
                
                imputer.fit(dataset[[col]]) # Train the imputer on the current column
                imputed_values = imputer.transform(dataset[[col]]) # Fill the missing values
                if imputed_values.shape[1] == 0: continue
                
                dataset[col] = pd.DataFrame(imputed_values, columns=[col]) # Replace the column

                model.fit(dataset[[col]]) # Fit the model to get baseline
                predictions = model.predict(dataset[[col]]) # The anomaly score of the input samples
                anomaly_indices = np.where(predictions == -1)[0] # Outliers are labeled -1 and inliers are labeled 1
                dataset_outliers.append(anomaly_indices) # Append the indices of the outliers

                # print(f'Dataset values and outliers: {dataset.shape} {len(anomaly_indices)} ..{.1*dataset.shape[0]}') 
            
            # Plot anomalies for each column
            fig=plot_dataset_outliers(dataset, dataset_outliers)
            with st.container():
                st.pyplot(fig)
            # break # <- Uncomment to plot all datasets

    outlier_scores.append(dataset_outliers)
    
    return outlier_scores

# detect_outliers(scaled_datasets) -> THIS WORKS

st.set_page_config(
    page_title="Anomalies detector",  # Change this to your desired title
    page_icon=":sparkles:",  # You can also set a custom icon (emoji or image)
)
st.header("Detecting Anomalies in Datasets")

with st.sidebar:
    st.help(matplotlib.pyplot)
    st.markdown("---")
    st.help(st)

with st.container():
    # Input: List of dataset links
    dataset_links = st.text_input("Enter dataset URL")
    submit_button = st.button('Detect Anomalies')

    if submit_button:
        st.balloons()
        
        if not dataset_links:
            dataset_links = 'https://docs.google.com/spreadsheets/d/173kXrmgG0K4Q_K0d2GFgADnW3wB66xh9R1IdyjB3eXY/edit?gid=1#gid=1'
            st.warning(f"Using default dataset link ... {dataset_links}")

        # Load the dataset
        df_base = get_the_data(dataset_links) # Get the dataset if its to a spreadheet
        # st.dataframe(df_base.head()) # Display the first 5 rows of the dataset

        # Initial data manipulation and cleaning
        multi_dfs = get_datasets(base_df=df_base)
        multi_dfs = [dataset['entry'] for dataset in multi_dfs]
        st.html(f"<h3>Number of datasets: {len(multi_dfs)}</h3><hr>")

        # Remove duplicates
        multi_dfs = remove_duplicates(multi_dfs)
        st.info("Duplicates removed")
        # Remove small datasets
        multi_dfs = remove_small_datasets(multi_dfs)
        st.info("Small datasets removed")
        # Handle missing values
        multi_dfs = handle_missing_values(multi_dfs)
        # st.info("{multi_df.shape[0]}")
        # Find non alphanumeric characters
        non_alphanumeric_chars = find_non_alphanumeric_data(multi_dfs)
        # Replace non alphanumeric characters
        multi_dfs = replace_non_alphanumeric_data(multi_dfs, non_alphanumeric_chars)
        st.info(f"{len(non_alphanumeric_chars)} non-alphanumeric characters replaced")
        # Clean the columns
        multi_dfs = clean_columns(multi_dfs)
        # Convert categorical columns to numerical columns
        numerically_encoded_datasets = convert_categorical_columns(multi_dfs)
        # Scale the datasets
        scaled_datasets = scale_datasets(numerically_encoded_datasets)
        st.info(f"{len(scaled_datasets)} datasets")
        
        # Detect anomalies
        outliers = detect_outliers(scaled_datasets)
        st.write('Outliers detected' if len(outliers) > 0 else 'No outliers detected')
    
    st.snow()
