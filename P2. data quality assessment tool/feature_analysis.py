from install_packages import install_packages
install_packages() # Install the necessary packages

from data_quality_checker import multi_datasets

## Feature Analysis
### Evaluate feature importance
# Evaluate feature importance by looking at the distributions to look at relationships
import seaborn as sns
import matplotlib.pyplot as plt

def plot_relationships(title: str, data: pd.DataFrame):
    """Plot relationships between columns in the dataset."""
    # Drop columns that start with 'Unnamed'
    to_plot = data.loc[:, ~data.columns.str.startswith('Unnamed')]
    
    # Check if there are at least two columns left to plot
    if to_plot.shape[1] < 2 or to_plot.shape[0] < 20:
        print(f"Not enough columns/rows to plot for dataset: {title}")
        return
    
    try:
        sns.set(style='ticks')
        sns.pairplot(to_plot, corner=True, diag_kind='kde', kind='scatter')

        plt.suptitle(title.replace("$", "\\$"))
        plt.show()
    except Exception as e:
        print(f"{title} error occurred: {e}")

for idx, dataset in enumerate(multi_datasets):
    """ Plot relationships between columns in each dataset """
    # print(f"Dataset {idx}: {dataset['title']}", end='\n')
    plot_relationships(dataset['title'], dataset['entry'])
    # break


### Feature Selection
# Perform feature selection to improve model accuracy/ boost model's performance on higher dimension datasets
#### Remove features with low variance 
# If the points are close to each other, they can be approximated as a singular point

from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

treated = []
def remove_low_varaince_points(data:pd.DataFrame)->pd.DataFrame:
    """Remove low variance points in the dataset."""
    if data.shape[1] < 10:
        return data
    
    # Check if there are numerical columns
    if data.select_dtypes(include=[np.number]).shape[1] == 0:
        return data
    
    selector = VarianceThreshold() # *By default, it removes features with zero variance
    selector.fit_transform(data.select_dtypes(include=[np.number])) # Fit to data, then transform it

    return data[data.columns[selector.get_support(indices=True)]] # Get the columns that have high variance

for idx, dataset in enumerate(multi_datasets):
    if dataset['title'] in treated:
        print(f"{dataset['title']} has already been treated.")
    else:
        treated.append(dataset['title'])
        # print(f'{dataset['title']} \n\tbefore: {dataset["entry"].shape}', end=' ')
        dataset['entry'] = remove_low_varaince_points(dataset['entry'])
        # print(f'After: {dataset["entry"].shape}')


#### Select features using Sequential Feature Selection
# - SFS is a greedy procedure where by at each iteration, it chooses the best new feature to add to selected features based on a cross-validation score until the desired number of selected features.   
# - It is a greedy search algorithm that selects features by adding or removing one feature at a time based on the classifier performance
# - _SFS can be backwards or forwards_

# - _Backwards_: Starts with all features and removes one feature at a time
# - _Forwards_: Starts with no features and adds one feature at a time
# - _Bidirectional_: Combines both forwards and backwards
# - _Floating_: Combines forwards and backwards but can add or remove more than one feature at a time

from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SequentialFeatureSelector 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold

forwarded = []

fwrd_data = multi_datasets.copy()
def forward_SFS(data: pd.DataFrame) -> pd.DataFrame:
    """Select features using forward sequential selection."""
    """ TODO:
        - Identify categorical columns
        - Encode categorical columns into numerical features
        - Fit the feature selector with RandomForestRegressor
        - Return the selected features
    """
    categorical_columns = data.select_dtypes(exclude=['number']).columns
    data_to_forward = data.copy()

    if categorical_columns.size > 0:
        encoder = OrdinalEncoder()
        data_to_forward[categorical_columns] = encoder.fit_transform(data_to_forward[categorical_columns])

    # Initialize the cross-validation
    cv = KFold(n_splits=max(2,
                            (data_to_forward.shape[1])//2), 
               shuffle=True, 
               random_state=42)

    # Initialize the feature selector
    # print(f'\t\t num features .. {data_to_forward.shape[1]} features to select... {max(2, data_to_forward.shape[1]//3)}', 
    #       end=' ')
    feature_selector = SequentialFeatureSelector(RandomForestRegressor(), 
                                                 n_features_to_select=max(1, 
                                                                          min(data_to_forward.shape[1] - 1, 
                                                                              data_to_forward.shape[1] // 2)),
                                                 direction='forward',
                                                 cv = cv
                                                                          )
    
    if data_to_forward.shape[1] < 3:
        return data_to_forward
    
    # Fit the feature selector
    feature_selector.fit(data_to_forward.iloc[:, :-1], # Get all columns except the target column
                         data_to_forward.iloc[:, -1] # Get the target column
                        )
    # Get the selected features
    selected_features = data_to_forward.iloc[:, feature_selector.get_support(indices=True)]

    return selected_features
    
for idx, dataset in enumerate(fwrd_data):
    """ Select features using forward sequential selection """
    if dataset['title'] in forwarded:
        # print(f"{dataset['title']} has already been forwarded.")
        continue
    else:
        rows, columns = dataset['entry'].shape[0], dataset['entry'].shape[1]
        
        if columns > 2 or rows > 10:
            print(f"Dataset {idx}: {dataset['title']} with {dataset['entry'].shape} ", end='\t')
            dataset['entry'] = forward_SFS(dataset['entry'])
            print(f"after: {dataset['entry'].shape} ")

        forwarded.append(dataset['title'])
    