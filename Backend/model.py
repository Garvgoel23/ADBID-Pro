import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Optional
import logging
from datetime import datetime
import gc
from DataProcessor import DataProcessor  # Import the optimized data processor

class BidModel:
    def __init__(
        self,
        model_path: str = "bid_model.pkl",
        chunk_size: int = 100000,
        features: Optional[List[str]] = None
    ):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.chunk_size = chunk_size
        
        # Define default features if none provided
        self.features = features or [
            "Adslotwidth", "Adslotheight", "Adslotfloorprice",
            "AdvertiserID", "Region"
        ]
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize categorical features
        self.categorical_features = ["Region", "AdvertiserID"]

    def _initialize_encoders(self, data: pd.DataFrame) -> None:
        """Initialize label encoders for categorical features."""
        for feature in self.categorical_features:
            if feature in data.columns:
                self.label_encoders[feature] = LabelEncoder()
                # Fit on unique values to save memory
                unique_values = data[feature].unique()
                self.label_encoders[feature].fit(unique_values)

    def _transform_features(self, data: pd.DataFrame) -> np.ndarray:
        """Transform features including encoding categoricals."""
        transformed_data = data.copy()
        
        # Encode categorical features
        for feature in self.categorical_features:
            if feature in transformed_data.columns:
                transformed_data[feature] = self.label_encoders[feature].transform(
                    transformed_data[feature].astype(str)
                )
        
        return transformed_data

    def train_model(self, data_path: str, target_column: str = "Biddingprice") -> None:
        """
        Trains the model using chunked data processing for large datasets.
        """
        logging.info("Initializing model training...")
        
        # Initialize DataProcessor with required columns
        required_columns = self.features + [target_column]
        data_processor = DataProcessor(
            chunk_size=self.chunk_size,
            required_columns=required_columns
        )
        
        # Process first chunk to initialize encoders and scaler
        first_chunk = pd.read_csv(
            data_path,
            nrows=self.chunk_size,
            usecols=required_columns
        )
        
        self._initialize_encoders(first_chunk)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        # Process data in chunks
        logging.info("Starting chunked training process...")
        
        for chunk_num, chunk in enumerate(pd.read_csv(
            data_path,
            chunksize=self.chunk_size,
            usecols=required_columns
        )):
            logging.info(f"Processing chunk {chunk_num + 1}")
            
            # Transform features
            X_chunk = self._transform_features(chunk[self.features])
            y_chunk = chunk[target_column]
            
            # Scale features
            if chunk_num == 0:
                X_chunk_scaled = self.scaler.fit_transform(X_chunk)
            else:
                X_chunk_scaled = self.scaler.transform(X_chunk)
            
            # Partial fit not available for RandomForest, so we'll use sample
            if chunk_num == 0:
                # Train on first chunk
                self.model.fit(X_chunk_scaled, y_chunk)
            else:
                # Sample and update model
                sample_mask = np.random.choice(
                    [True, False],
                    size=len(X_chunk_scaled),
                    p=[0.3, 0.7]  # Sample 30% of each chunk
                )
                if np.any(sample_mask):
                    self.model.fit(
                        X_chunk_scaled[sample_mask],
                        y_chunk[sample_mask]
                    )
            
            # Clean up memory
            del X_chunk, y_chunk, X_chunk_scaled
            gc.collect()
        
        # Save the trained model and preprocessing objects
        self._save_model()
        logging.info("Model training completed!")

    def _save_model(self) -> None:
        """Save the model and preprocessing objects."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "features": self.features,
            "categorical_features": self.categorical_features
        }
        
        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load the model and preprocessing objects."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
                self.model = model_data["model"]
                self.scaler = model_data["scaler"]
                self.label_encoders = model_data["label_encoders"]
                self.features = model_data["features"]
                self.categorical_features = model_data["categorical_features"]
            logging.info("Model loaded successfully!")
            return True
        else:
            logging.error("Model file not found! Please train the model first.")
            return False

    def predict_bid(self, bid_request: 'BidRequest') -> int:
        """
        Predict bid price for a given bid request.
        """
        if not self.model:
            if not self.load_model():
                return -1

        # Extract and transform features
        features_dict = {
            "Adslotwidth": bid_request.adSlotWidth,
            "Adslotheight": bid_request.adSlotHeight,
            "Adslotfloorprice": bid_request.adSlotFloorPrice,
            "AdvertiserID": str(bid_request.advertiserId),
            "Region": str(bid_request.region)
        }
        
        # Create DataFrame for consistent processing
        features_df = pd.DataFrame([features_dict])
        
        # Transform features
        features_transformed = self._transform_features(features_df)
        
        # Scale features
        features_scaled = self.scaler.transform(features_transformed)
        
        # Predict bid price
        predicted_price = int(self.model.predict(features_scaled)[0])
        
        # Ensure bid meets floor price
        return max(predicted_price, bid_request.adSlotFloorPrice)

def main():
    # Example usage
    model = BidModel(chunk_size=50000)  # Adjust chunk size based on available RAM
    
    # Train model
    data_path = "path_to_your_processed_data.csv"
    model.train_model(data_path)
    
    # Test prediction
    from BidRequest import BidRequest  # Import your BidRequest class
    test_request = BidRequest(
        adSlotWidth=300,
        adSlotHeight=250,
        adSlotFloorPrice=10,
        advertiserId=12345,
        region="NA"
    )
    
    predicted_bid = model.predict_bid(test_request)
    print(f"Predicted bid: {predicted_bid}")

if __name__ == "__main__":
    main()
