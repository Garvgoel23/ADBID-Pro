import pandas as pd
import os

# Automatically get absolute path to dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Backend folder
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset")  # Moves up to RTB-Hackathon

def load_data(file_path):
    """Loads and returns the dataset from a given file path."""
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, low_memory=False)
        print(f"Successfully loaded: {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_data(df):
    """Preprocesses the dataset."""
    column_names = [
        "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP", "Region", "City",
        "Adexchange", "Domain", "URL", "AnonymousURLID", "AdslotID", "Adslotwidth", "Adslotheight",
        "Adslotvisibility", "Adslotformat", "Adslotfloorprice", "CreativeID", "Biddingprice",
        "Payingprice", "KeypageURL", "AdvertiserID"
    ]

    if len(df.columns) != len(column_names):
        print(f"Column mismatch! Expected {len(column_names)}, found {len(df.columns)}")
        return None

    df.columns = column_names
    
    # Convert timestamp to datetime format
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format='%Y%m%d%H%M%S%f', errors='coerce')
    
    # Fill missing values
    df.fillna({'Payingprice': 0, 'AnonymousURLID': 'Unknown'}, inplace=True)
    
    # Convert numeric fields to appropriate data types
    df["Region"] = df["Region"].astype('category')
    df["City"] = df["City"].astype('category')
    df["Adexchange"] = df["Adexchange"].astype('category')
    df["Logtype"] = pd.to_numeric(df["Logtype"], errors='coerce').fillna(0).astype(int)
    df["Adslotfloorprice"] = pd.to_numeric(df["Adslotfloorprice"], errors='coerce').fillna(0.0)
    df["Biddingprice"] = pd.to_numeric(df["Biddingprice"], errors='coerce').fillna(0.0)
    df["Payingprice"] = pd.to_numeric(df["Payingprice"], errors='coerce').fillna(0.0)

    print(f"Preprocessing complete for {len(df)} rows")
    return df

def load_and_preprocess_all(data_dir=DATASET_PATH):
    """Loads and preprocesses all datasets in a given directory."""
    
    # Ensure dataset directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # List all .txt files in the dataset directory
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    print(f"Found {len(data_files)} dataset files in {data_dir}")

    dataframes = {}
    
    for file in data_files:
        file_path = os.path.join(data_dir, file)
        df = load_data(file_path)
        if df is not None:
            processed_df = preprocess_data(df)
            if processed_df is not None:
                dataframes[file] = processed_df
    
    print("Data processing complete!")
    return dataframes

if __name__ == "__main__":
    processed_data = load_and_preprocess_all()

