from install_packages import import_packages
import_packages() # Import the necessary packages

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import requests, re, warnings
warnings.filterwarnings('ignore')

# Convert the categorical columns to numerical columns
from sklearn.preprocessing import LabelEncoder # Encode target labels with value between 0 and n_classes-1.
from sklearn.preprocessing import StandardScaler # Removing the mean and scaling to unit variance
from sklearn.preprocessing import MinMaxScaler # Scaling features to a range

from get_the_data import datasets as multiple_datasets

def remove_duplicates(datasets:list):
    """
        This function will remove the datasets with the same shapes, leaving a single copy.
        Parameters:
            datasets (list): The list of datasets
        Returns:
            list: The list of datasets without duplicates 
    """
    unique_datasets = []
    for dataset in datasets:
        if dataset.shape not in [data.shape for data in unique_datasets]:
            unique_datasets.append(dataset)
    return unique_datasets

def remove_small_datasets(datasets:list):
    """
        This function will remove the datasets with less than 10 rows.
        Parameters:
            datasets (list): The list of datasets
        Returns:
            list: The list of datasets without small datasets 
    """
    return [dataset for dataset in datasets if dataset.shape[0] > 10 & dataset.shape[1] > 2]

def handle_missing_values(datasets:list):
    """
        This function will handle the missing values in the datasets.
        Parameters:
            datasets (list): The list of datasets
        Returns:
            list: The list of datasets without missing values 
    """
    for dataset in datasets:
        dataset.fillna(method='ffill', inplace=True) 
    return datasets

def find_non_alphanumeric_data(datasets: list):
    """
    This function will list the datasets with columns containing non-alphanumeric data.
    
    Parameters:
        datasets (list): The list of datasets
    
    Returns:
        list: The list of non-alphanumeric characters found in the datasets
    """
    non_alphanumeric_chars = set()
    for idx, dataset in enumerate(datasets):
        for column in dataset.columns:
            for index, value in dataset[column].items():
                if isinstance(value, str):  # Check if the value is a string
                    non_alphanumeric = re.findall(r'[^a-zA-Z0-9]', value)
                    if non_alphanumeric:
                        non_alphanumeric_chars.update(non_alphanumeric)
                        # print(f"Dataset {idx}, Row {index}, Column '{column}': \
                        #       Non-alphanumeric characters: {non_alphanumeric}")
    return list(non_alphanumeric_chars)

def replace_non_alphanumeric_data(datasets: list, non_alphanumeric_chars: list):
    """
    This function will replace non-alphanumeric characters in the datasets.
    
    Parameters:
        datasets (list): The list of datasets
        non_alphanumeric_chars (list): The list of non-alphanumeric characters to replace
    
    Returns:
        datasets (list): The list of datasets with non-alphanumeric characters replaced
    """
    for dataset in datasets:
        for column in dataset.columns:
            for index, value in dataset[column].items():
                if isinstance(value, str):  # Check if the value is a string
                    for char in non_alphanumeric_chars:
                        value = value.replace(char, '')
                    dataset.at[index, column] = value
    return datasets

datasets = [dataset['entry'] for dataset in multiple_datasets]

print(f"Number of datasets: {len(datasets)}", end='\t')
datasets = remove_duplicates(datasets)
datasets = remove_small_datasets(datasets)
datasets = handle_missing_values(datasets)

non_alphanumeric_chars:list = find_non_alphanumeric_data(datasets)
datasets = replace_non_alphanumeric_data(datasets, non_alphanumeric_chars)

print(f'Editted number of datasets: {len(datasets)}')

def get_column_names(datasets:list=datasets):
    """
        This function will get the column names of each dataset in the list of datasets.
        Parameters:
            datasets (list): The list of datasets
        Returns:
            List of list: The list of column names of each dataset
    """
    for dataset in datasets:
        categorical_columns, numeric_columns = [],[]

        numeric_columns.append(dataset.select_dtypes(include=[np.number]).columns)
        categorical_columns.append(dataset.select_dtypes(include=[object]).columns)

        # print(f"Numeric Columns: {list(numeric_columns)}\nCategorical Columns: {list(categorical_columns)}\n\n")
    return [list(dataset.columns) for dataset in datasets]

def clean_columns(datasets:list=datasets):
    """
        This function will remove the columns with links
        Parameters:
            datasets (list): The list of datasets
        Returns:
            datasets (list): The list of datasets without columns with links
    """
    for dataset in datasets:
        for column in dataset.columns:
            if 'http' in column:
                dataset.drop(column, axis=1, inplace=True)
    return datasets

columns = get_column_names(datasets)
columns = clean_columns(datasets)

def plot_data_distribution(datasets: list):
    """
    This function will plot the data distribution of the datasets.
    For each dataset, plot a bunch of subplots for each column.
    
    Parameters:
        datasets (list): The list of datasets.
    """
    for idx, dataset in enumerate(datasets):
        if idx < 10:
            numeric_columns = dataset.select_dtypes(include=['number']).columns
            num_columns = len(numeric_columns)

            if num_columns == 0:
                print(f"Dataset {idx} has no numeric columns to plot.")
                continue
            
            num_rows = (num_columns // 4) + (num_columns % 4 > 0)
            fig, axs = plt.subplots(num_rows, min(4, num_columns), figsize=(15, 5 * num_rows))
            axs = axs.flatten() if num_columns > 1 else [axs]

            for col_idx, col_name in enumerate(numeric_columns):
                axs[col_idx].plot(dataset[col_name])
                axs[col_idx].set_title(f'Dataset-{idx} {dataset.shape}', fontsize=8)
                axs[col_idx].set_xlabel('{}'.format(col_name.replace('$', '\\$')), fontsize=8)
                # axs[col_idx].set_xlabel(f'{col_name.replace("$", "\\$")}', fontsize=8)
                axs[col_idx].set_ylabel('Value')
        
            plt.tight_layout()
            plt.show() 

def convert_categorical_columns(datasets:list=datasets):
    """
        This function will convert the categorical columns to numerical columns.
        Parameters:
            datasets (list): The list of datasets
        Returns:
            datasets (list): The list of datasets with numerical columns
    """
    for dataset in datasets:
        categorical_columns = dataset.select_dtypes(include=[object]).columns
        for column in categorical_columns:
            encoder = LabelEncoder()
            dataset[column] = encoder.fit_transform(dataset[column])
        
        # Replace the $ sign with empty space for values in each column
        for col in dataset.select_dtypes(include=['object']):  # Only process string columns
            # Replace dollar signs with spaces
            dataset[col] = dataset[col].str.replace('$', ' ', regex=False)
            cleaned_text = dataset[col].strip('$')  # Remove leading and trailing dollar signs
            cleaned_text = cleaned_text.replace('$', '\\$')  # Escape dollar signs within the text
            dataset[col] = cleaned_text    

    return datasets

numerically_datasets = convert_categorical_columns(datasets)

def scale_datasets(datasets:list=numerically_datasets):
    """
        This function will scale the datasets.
        Parameters:
            datasets (list): The list of datasets
        Returns:
            datasets (list): The list of scaled datasets
    """
    for dataset in datasets:
        scaler = MinMaxScaler(feature_range=(0,10)) #StandardScaler()
        dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
    return datasets

scaled_datasets = scale_datasets(numerically_datasets)
# plot_data_distribution(scaled_datasets) #-> THIS WORKS