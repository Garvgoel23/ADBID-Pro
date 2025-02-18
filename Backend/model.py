import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import gc
import random
from dataclasses import dataclass

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class ModelConfig:
    learning_rate: float = 0.1
    max_depth: int = 6
    num_boost_round: int = 100
    tree_method: str = "hist"
    random_state: int = 42
    
    def to_dict(self) -> Dict:
        return {
            "objective": "reg:squarederror",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "verbosity": 1,
            "tree_method": self.tree_method,
            "random_state": self.random_state
        }

class BidModel:
    def __init__(
        self,
        model_path: str = "xgboost_bid_model.pkl",
        chunk_size: int = 100000,
        features: Optional[List[str]] = None,
        config: Optional[ModelConfig] = None
    ):
        self.model_path = model_path
        self.model = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.chunk_size = chunk_size
        self.features = features or ["Region", "Adexchange", "City"]
        self.categorical_features = ["Region", "Adexchange", "City"]
        self.config = config or ModelConfig()
        self.valid_file_prefix = "processed_bid"  # Only process bid files, not click files

    def _validate_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Validate and clean the data before training."""
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Remove rows with NaN, infinity, or extremely large values in target column
        df = df[df[target_column].notna()]
        df = df[~df[target_column].isin([np.inf, -np.inf])]
        
        # Remove extreme outliers (values beyond 3 standard deviations)
        mean = df[target_column].mean()
        std = df[target_column].std()
        df = df[df[target_column].between(mean - 3*std, mean + 3*std)]
        
        # Ensure all values are positive
        df = df[df[target_column] > 0]
        
        # Cap maximum values at a reasonable threshold (e.g., 10000)
        df.loc[df[target_column] > 10000, target_column] = 10000
        
        return df

    def _initialize_encoders(self, data: pd.DataFrame) -> None:
        """Initialize label encoders for categorical features."""
        try:
            for feature in self.categorical_features:
                if feature in data.columns:
                    # Fill NA values and convert to string
                    feature_data = data[feature].fillna("Unknown").astype(str)
                    unique_values = feature_data.unique()
                    
                    if len(unique_values) > 10000:
                        logging.warning(f"High cardinality detected in {feature}: {len(unique_values)} unique values")
                    
                    # Initialize and fit the encoder
                    self.label_encoders[feature] = LabelEncoder()
                    self.label_encoders[feature].fit(unique_values)
                    
                    logging.info(f"Initialized encoder for {feature} with {len(unique_values)} unique values")
        except Exception as e:
            logging.error(f"Error initializing encoders: {str(e)}")
            raise

    def _transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features for model input."""
        if data.empty:
            raise ValueError("Empty DataFrame provided for feature transformation")
            
        transformed_data = data.copy()
        
        for feature in self.categorical_features:
            if feature not in transformed_data.columns:
                raise ValueError(f"Missing categorical feature: {feature}")
                
            # Fill NA values and convert to string
            transformed_data[feature] = transformed_data[feature].fillna("Unknown").astype(str)
            # Convert to categorical
            transformed_data[feature] = transformed_data[feature].astype("category")
                
        return transformed_data

    def _count_chunks(self, file_path: str) -> int:
        """Count total number of chunks in a file with proper encoding."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                total_rows = sum(1 for _ in f) - 1
        return (total_rows + self.chunk_size - 1) // self.chunk_size

    def train_model(self, data_dir: str, target_column: str = "Biddingprice") -> None:
        """Train the XGBoost model using one random chunk from each file."""
        logging.info("Starting model training process...")
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
            
        # Only process bid files, not click files
        bid_files = sorted([
            f for f in os.listdir(data_dir) 
            if f.endswith(".csv") and f.startswith(self.valid_file_prefix)
        ])
        
        if not bid_files:
            raise ValueError(f"No valid bid CSV files found in {data_dir}")
            
        # Initialize with first file
        first_file_path = os.path.join(data_dir, bid_files[0])
        try:
            sample_data = pd.read_csv(first_file_path, nrows=5, encoding='utf-8')
        except UnicodeDecodeError:
            sample_data = pd.read_csv(first_file_path, nrows=5, encoding='latin-1')
            
        logging.info(f"Available columns in data: {list(sample_data.columns)}")
        
        missing_columns = set(self.features + [target_column]) - set(sample_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Initialize encoders with first file
        try:
            initial_data = pd.read_csv(first_file_path, usecols=self.features + [target_column], encoding='utf-8')
        except UnicodeDecodeError:
            initial_data = pd.read_csv(first_file_path, usecols=self.features + [target_column], encoding='latin-1')
        
        self._initialize_encoders(initial_data)
        
        xgb_train = None
        processed_files = 0
        
        for file_num, file_name in enumerate(bid_files, 1):
            file_path = os.path.join(data_dir, file_name)
            logging.info(f"Processing file {file_num}/{len(bid_files)}: {file_name}")
            
            try:
                total_chunks = self._count_chunks(file_path)
                selected_chunk = random.randint(0, total_chunks - 1)
                logging.info(f"Selected chunk {selected_chunk} from {total_chunks} total chunks")
                
                try:
                    chunk_iterator = pd.read_csv(
                        file_path,
                        chunksize=self.chunk_size,
                        usecols=self.features + [target_column],
                        encoding='utf-8'
                    )
                except UnicodeDecodeError:
                    chunk_iterator = pd.read_csv(
                        file_path,
                        chunksize=self.chunk_size,
                        usecols=self.features + [target_column],
                        encoding='latin-1'
                    )
                
                for chunk_idx, chunk in enumerate(chunk_iterator):
                    if chunk_idx != selected_chunk:
                        continue
                    
                    logging.info(f"Processing chunk {chunk_idx} from {file_name}")
                    
                    # Validate and clean the data
                    chunk = self._validate_data(chunk, target_column)
                    
                    if len(chunk) == 0:
                        logging.warning(f"No valid data remaining in chunk after validation")
                        continue
                    
                    chunk = self._transform_features(chunk)
                    X_chunk = chunk[self.features]
                    y_chunk = chunk[target_column]
                    
                    try:
                        dmatrix = xgb.DMatrix(
                            data=X_chunk,
                            label=y_chunk,
                            enable_categorical=True
                        )
                        
                        if xgb_train is None:
                            xgb_train = dmatrix
                            self.model = xgb.train(self.config.to_dict(), dmatrix, num_boost_round=self.config.num_boost_round)
                        else:
                            self.model = xgb.train(
                                self.config.to_dict(),
                                dmatrix,
                                num_boost_round=self.config.num_boost_round,
                                xgb_model=self.model
                            )
                        
                        processed_files += 1
                        
                    except Exception as e:
                        logging.error(f"Error training on chunk: {str(e)}")
                        continue
                    
                    del chunk, X_chunk, y_chunk, dmatrix
                    gc.collect()
                    break
                    
            except Exception as e:
                logging.error(f"Error processing file {file_name}: {str(e)}")
                continue
        
        if processed_files == 0:
            raise RuntimeError("No valid data was processed for training")
        
        self._save_model()
        logging.info(f"Model training completed successfully! Processed {processed_files} files.")

    def _save_model(self) -> None:
        """Save the trained model and encoders to disk."""
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "label_encoders": self.label_encoders,
                "features": self.features
            }, f)
        logging.info(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load the trained model from disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
                self.model = model_data["model"]
                self.label_encoders = model_data["label_encoders"]
                self.features = model_data["features"]
            logging.info("Model loaded successfully!")
            return True
        logging.error("Model file not found!")
        return False

    def predict_bid(self, bid_request: Dict[str, Union[str, float, int]]) -> float:
        """Predict the bid price for a given bid request."""
        if not self.model:
            if not self.load_model():
                raise RuntimeError("Model not loaded and could not be loaded from disk")
        
        missing_features = set(self.features) - set(bid_request.keys())
        if missing_features:
            raise ValueError(f"Missing required features in bid request: {missing_features}")
        
        try:
            features_df = pd.DataFrame([bid_request])
            features_transformed = self._transform_features(features_df)
            dmatrix = xgb.DMatrix(features_transformed[self.features], enable_categorical=True)
            predicted_price = float(self.model.predict(dmatrix)[0])
            
            floor_price = float(bid_request.get("Adslotfloorprice", 0))
            return max(predicted_price, floor_price)
            
        except Exception as e:
            logging.error(f"Error during bid prediction: {str(e)}")
            raise

def main():
    random.seed(42)
    
    config = ModelConfig(
        learning_rate=0.05,
        max_depth=8,
        num_boost_round=150
    )
    
    model = BidModel(
        chunk_size=50000,
        config=config
    )
    
    try:
        model.train_model("processed_output")
        
        test_request = {
            "Region": "NA",
            "Adexchange": "Google",
            "City": "San Francisco",
            "Adslotfloorprice": 10.0,
        }
        
        predicted_bid = model.predict_bid(test_request)
        print(f"Predicted bid: ${predicted_bid:.2f}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
