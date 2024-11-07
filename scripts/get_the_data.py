from install_packages import import_packages
import_packages() # Import the necessary packages

import pandas as pd
import requests, re, warnings
warnings.filterwarnings('ignore')

def is_valid_url(url: str) -> bool:
    """
    Checks if the URL is valid.
    
    Parameters:
        url (str): The URL to check.
    
    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    return bool(re.match(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', url))

def convert_spreadsheet_to_csv(spreadsheet_url: str) -> str:
    """ 
    Converts a Google spreadsheet URL to a CSV export URL.
    
    Parameters:
        spreadsheet_url (str): The Google spreadsheet URL.
    
    Returns:
        str: The CSV export URL.
    """
    return re.sub(r'/edit.*$', '/export?format=csv', spreadsheet_url)

def get_the_data(sheet_url: str = 'https://docs.google.com/spreadsheets/d/173kXrmgG0K4Q_K0d2GFgADnW3wB66xh9R1IdyjB3eXY/edit?gid=1#gid=1') -> pd.DataFrame:
    """ 
    Retrieves data from a Google sheet and returns it as a pandas DataFrame.
    
    Parameters:
        sheet_url (str): The URL of the Google sheet.
    
    Returns:
        pd.DataFrame: The data from the Google sheet.
    """
    if not sheet_url:
        return pd.DataFrame()

    csv_export_url = convert_spreadsheet_to_csv(sheet_url)
    
    if not is_valid_url(sheet_url):
        print(f"Invalid URL: {sheet_url}")
        return pd.DataFrame()
    
    try:
        response = requests.get(csv_export_url, verify=False)
        response.raise_for_status()
        data = pd.read_csv(csv_export_url, on_bad_lines='skip')
        # print(f"\n\n\n\nRetrieving data ... {data.head(2)}\n\n\n\n")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve data from {csv_export_url}: {e}")
        return pd.DataFrame()

def check_schema(url: str) -> str:
    """
    Ensures the URL has a valid schema.
    
    Parameters:
        url (str): The URL to check.
    
    Returns:
        str: The URL with a valid schema.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url

def convert_bitly_to_url(bitly_url: str) -> str:
    """ 
    Converts a Bitly URL to the original Google sheet URL.
    
    Parameters:
        bitly_url (str): The Bitly URL.
    
    Returns:
        str: The original Google sheet URL.
    """
    try:
        return requests.get(bitly_url, verify=False, allow_redirects=True).url
    except requests.exceptions.RequestException as e:
        print(f"Failed to convert Bitly URL: {e}")
        return bitly_url

def convert_geni_to_url(geni_url: str) -> str:
    """ 
    Converts a Geni URL to the original Google sheet URL.
    
    Parameters:
        geni_url (str): The Geni URL.
    
    Returns:
        str: The original Google sheet URL.
    """
    try:
        return requests.get(geni_url, verify=False, allow_redirects=True).url
    except requests.exceptions.RequestException as e:
        print(f"Failed to convert Geni URL: {e}")
        return geni_url

def convert_airtable_to_url(airtable_url: str) -> str:
    """ 
    Converts an Airtable URL to the original Google sheet URL.
    
    Parameters:
        airtable_url (str): The Airtable URL.
    
    Returns:
        str: The original Google sheet URL.
    """
    return re.sub(r'/tbl([^/]+)/([^/]+)', r'/tbl\1/\2/csv', airtable_url)

dataset_base = get_the_data() # 151 rows, 6 columns
def get_datasets(base_df: pd.DataFrame = dataset_base, target_column: str = 'bitly ') -> list:
    """
    Retrieves datasets from the specified column in the DataFrame.
    
    Parameters:
        base_df (pd.DataFrame): The DataFrame containing the data.
        target_column (str): The name of the target column.
    
    Returns:
        list: A list of dictionaries with 'entry' and 'title' as keys.
    """
    datasets = []
    conversion_functions = {
        'spreadsheets': convert_spreadsheet_to_csv,
        'bit.ly': convert_bitly_to_url,
        'BIT.LY': convert_bitly_to_url,
        'airtable.com': convert_airtable_to_url,
        'geni': convert_geni_to_url,
    }


    for _, row in base_df.iterrows():
        entry = row[target_column].strip()
        title = row['title']
        entry = check_schema(entry)

        for keyword, func in conversion_functions.items():
            if keyword in entry:
                entry = func(entry)
                break

        if 'informationisbeautiful' in entry:
            continue

        dataset = get_the_data(entry)
        if not dataset.empty:
            datasets.append({'entry': dataset, 'title': title})

    return datasets

datasets = get_datasets()