from install_packages import install_packages
install_packages() # Install the necessary packages

base_url : str = 'https://docs.google.com/spreadsheets/d/173kXrmgG0K4Q_K0d2GFgADnW3wB66xh9R1IdyjB3eXY/edit?gid=1#gid=1'

def is_valid_url(url: str) -> bool:
    """Checks if the URL is valid."""
    return bool(re.match(r'https?://[^\s/$.?#].[^\s]*', url))

def ensure_schema(url: str) -> str:
    """Ensures the URL has a valid schema."""
    return url if url.startswith(('http://', 'https://')) else 'https://' + url

def convert_spreadsheet_to_csv(spreadsheet_url: str) -> str:
    """Converts a Google spreadsheet URL to a CSV export URL."""
    return re.sub(r'/edit.*$', '/export?format=csv', spreadsheet_url)

def convert_airtable_to_url(airtable_url: str) -> str:
    """Converts an Airtable URL to the original Google sheet URL."""
    return re.sub(r'/tbl([^/]+)/([^/]+)', r'/tbl\1/\2/csv', airtable_url)

def convert_url(url: str) -> str:
    """Converts various shortened URLs to their original form."""
    try:
        return requests.get(url, verify=False, allow_redirects=True).url
    except requests.exceptions.RequestException as e:
        print(f"Failed to convert URL: {e}")
        return url

from urllib.error import HTTPError
def get_the_data(sheet_url: str, visited: set = None, max_depth: int = 3, current_depth: int = 0) -> pd.DataFrame: # type: ignore
    """Retrieves data from a Google sheet and returns it as a pandas DataFrame."""
    if visited is None:
        visited = set()

    if not sheet_url or not is_valid_url(sheet_url):
        print(f"Invalid URL: {sheet_url}")
        return pd.DataFrame()

    # Prevent infinite loops by checking if the sheet has already been visited
    if sheet_url in visited:
        print(f"Already visited: {sheet_url}")
        return pd.DataFrame()

    visited.add(sheet_url)

    csv_export_url = convert_spreadsheet_to_csv(sheet_url)

    try:
        data = pd.read_csv(csv_export_url)

        # Check if the column has an entry 'See next sheet'
        if 'Unnamed: 1' in data.columns and data.shape[0] == 1:
            if current_depth < max_depth:
                next_sheet = 'https://docs.google.com/spreadsheets/d/1OvDq4_BtbR6nSnnHnjD5hVC3HQ-ulZPGbo0RDGbzM3Q/edit?usp=sharing'
                print(f"Found 'See next sheet' in {sheet_url} \n\t- fetching next sheet: {next_sheet}")
                return get_the_data(next_sheet, visited, max_depth, current_depth + 1)
            else:
                print(f"Max depth reached for {sheet_url}. Stopping recursion.")
                return data

        return data
    except HTTPError as e:
        # print(f"HTTP error occurred: {e}")
        return pd.DataFrame()
    except Exception as e:
        # print(f"An error occurred: {e}")
        return pd.DataFrame()

def get_datasets(base_df: pd.DataFrame, target_column: str = 'bitly ') -> list:
    """Retrieves datasets from the specified column in the DataFrame."""
    datasets = []
    conversion_functions = {
        'spreadsheets': convert_spreadsheet_to_csv,
        'bit.ly': convert_url,
        'airtable.com': convert_airtable_to_url,
        'geni': convert_url,
    }

    for _, row in base_df.iterrows():
        entry = ensure_schema(row[target_column].strip())
        title = row['title']

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

dataset_base = get_the_data(base_url)
multi_datasets = get_datasets(dataset_base)

# Clean the data
def clean_data(data:pd.DataFrame) -> pd.DataFrame:
    """ Cleans the data by checking if the first value is a number then converting the column to numeric """
    for column in data.columns:
        # check if the first value is not a letter
        first_entry = str(data[column].iloc[0])

        if not isinstance(first_entry, str) or not first_entry.isalpha():
            # print(f"First entry '{first_entry}' is not all letters. Converting column '{column}' to float.")
            # Convert the column to float
            data[column] = pd.to_numeric(data[column], errors='coerce')
        else:
            continue
    print(f'Types: {data.dtypes}')
    return data

for dataset in multi_datasets:
    dataset['entry'] = clean_data(dataset['entry'])