from install_packages import install_packages
install_packages() # Install the necessary packages

from get_the_data import multi_datasets
### Identify missing values
# Deal with NaNs ; replace NaN with the median/mean/mode

def identify_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Identify NaNs and replace numeric NaNs with median and categorical NaNs with mode."""
    # Count NaNs before replacement
    # missing_before = data.isna().sum().sum()
    # print(f"Before: {missing_before}", end='\t') DEBUGGING

    # Replace NaNs in numeric columns with median+
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

    # Replace NaNs in categorical columns with mode
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].apply(lambda col: col.fillna(col.mode()[0]) if not col.mode().empty else col)

    # if missing values are more than 80% of the dataset, drop the column
    missing_threshold = 0.8
    data = data.dropna(thresh=int(missing_threshold*data.shape[0]), axis=1)
    
    # Count NaNs after replacement
    missing_after = data.isna().sum().sum()
    if missing_after > 0:
        # display(data[data.isna().any(axis=1)]) 
        print(f"After: {missing_after}", end='\n\n') ##DEBUGGING

    return data

for idx, dataset in enumerate(multi_datasets):
    """ Replace NaNs in each dataset """
    # print(f"Dataset {idx}: {dataset['title']}", end='\n')
    dataset['entry'] = identify_missing_values(dataset['entry'])


### Identify duplicates
# Remove duplicate values in the dataset

def remove_duplicates(title:str, data:pd.DataFrame):
    """Remove duplicates in the dataset."""
    # Count duplicates before removal
    duplicates_before = data.duplicated().sum()
    if duplicates_before > 0:
        print(f"{title}\t\t{duplicates_before} before ...") ##DEBUGGING

    # Remove duplicates
    data = data.drop_duplicates()
    return data

for idx, dataset in enumerate(multi_datasets):
    """ Remove duplicates in each dataset """
    dataset['entry'] = remove_duplicates(dataset['title'] ,dataset['entry'])


### Look at outliers
# Use _z-score_ to find outliers or _box (whisker) plots_ to visually identify the data distributions and outliers

from matplotlib.pylab import f
from scipy import stats
import pandas as pd

def check_for_outliers(data:pd.DataFrame)-> pd.DataFrame:
    """Calculate the z-score for each column in the dataset"""
    
    # display(data.dtypes) ##DEBUGGING
    z_scores = {'Column':[], 'Outliers':[], 'Percentage of Outliers':[]}
    # For each column, calculate the z-score
    for column in data.columns:
        # print(f'Column: {column} dtype: {data[column].dtype}') ##DEBUGGING
        if data[column].dtype == 'object':
            continue
        else:
            column_z_scores = stats.zscore(data[column].to_numpy())
            # Identify outliers higher than 3 standard deviations
            outlier_indices = np.where(np.abs(column_z_scores) > 3)[0]
            # print(f"{column}: {outliers} {data[column].iloc[outliers]}") ##DEBUGGING
            # Get the outliers
            outliers = data[column].iloc[outlier_indices].to_numpy()

            if outliers.size > 0:
                z_scores['Column'].append(column)
                z_scores['Outliers'].append(outliers)
                z_scores['Percentage of Outliers'].append(f"{(outliers.size/data[column].size)*100:.3f}%")
        
    return pd.DataFrame.from_dict(z_scores)

for idx, dataset in enumerate(multi_datasets):
    """ Check for outliers in each dataset """
    outliers = check_for_outliers(dataset['entry'])
    if outliers.empty:
        # print(f"No outliers found in {dataset['title']} dataset.")
        continue
    else:
        print(f"Dataset {idx}: {dataset['title']} with {dataset['entry'].shape[0]} rows", end='\n')
        for _, rows in outliers.iterrows():
            print(f"\t{rows['Column']} has {len(rows['Outliers'])} outliers", end='\n')
        # display(outliers) ##DEBUGGING

# --------- Maximum number of outliers is 27 ------------