import pandas as pd
import os
from typing import List, Optional, Set
import gc
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataProcessor:
    def __init__(
        self,
        chunk_size: int = 100000,
        required_columns: Optional[List[str]] = None,
        output_dir: str = "processed_output"
    ):
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        
        # Define default required columns if none provided
        self.required_columns = required_columns or [
            "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP",
            "Region", "City", "Adexchange", "Domain", "URL", "AnonymousURLID",
            "AdslotID", "Adslotwidth", "Adslotheight", "Adslotvisibility",
            "Adslotformat", "Adslotfloorprice", "CreativeID", "Biddingprice",
            "Payingprice", "KeypageURL", "AdvertiserID"
        ]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def _preprocess_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a single chunk of data."""
        # Keep only required columns
        df = df[self.required_columns]

        # Convert timestamp
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"],
                format="%Y%m%d%H%M%S%f",
                errors="coerce"
            )

        # Handle categorical columns
        categorical_cols = ["Region", "City", "Adexchange"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Convert numeric fields efficiently
        numeric_cols = [
            "Logtype", "Adslotwidth", "Adslotheight",
            "Adslotfloorprice", "Biddingprice", "Payingprice"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def process_file(self, file_path: str, output_filename: Optional[str] = None) -> None:
        """Process a single file in chunks."""
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"processed_{os.path.basename(file_path)}_{timestamp}.csv"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        logging.info(f"Processing {file_path}")
        
        # Process the first chunk to get column names
        first_chunk = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            nrows=1,
            dtype=str
        )
        
        # Identify which required columns are actually present
        available_columns = set(range(len(first_chunk.columns)))
        columns_to_use = list(available_columns.intersection(
            set(range(len(self.required_columns)))
        ))
        
        # Create iterator for chunks
        chunks = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            dtype=str,
            chunksize=self.chunk_size,
            usecols=columns_to_use
        )
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Assign column names
                chunk.columns = [self.required_columns[i] for i in columns_to_use]
                
                # Process the chunk
                processed_chunk = self._preprocess_chunk(chunk)
                
                # Write to file
                mode = 'w' if i == 0 else 'a'
                header = i == 0
                processed_chunk.to_csv(
                    output_path,
                    mode=mode,
                    index=False,
                    header=header
                )
                
                # Clean up memory
                del processed_chunk
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing chunk {i}: {str(e)}")
                continue

    def process_directory(self, dataset_path: str) -> None:
        """Process all files in a directory."""
        for file_name in sorted(os.listdir(dataset_path)):
            if file_name.endswith(".txt"):
                file_path = os.path.join(dataset_path, file_name)
                self.process_file(file_path)

def main():

    processor = DataProcessor(
        chunk_size=100000,  # Adjust based on available RAM
        required_columns=[  # Specify only the columns you need
            "BidID", "Timestamp", "Region", "City",
            "Adexchange", "Biddingprice", "Payingprice"
        ]
    )
    
    dataset_path = os.path.join(os.path.dirname(__file__), "../dataset")
    processor.process_directory(dataset_path)

if __name__ == "__main__":
    main()
