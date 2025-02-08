import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_processor import load_preprocessed_data  # Import preprocessed data
from BidRequest import BidRequest

class BidModel:
    def __init__(self, model_path="bid_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None  # For feature scaling

    def train_model(self):
        #Trains a RandomForest model on historical bid data.

        print("Loading preprocessed bid data...")
        data = load_preprocessed_data()  # Fetches cleaned dataset
        
        if data.empty:
            print("No data available for training!")
            return
        
        # Select features and target
        features = ["adSlotWidth", "adSlotHeight", "adSlotFloorPrice", "advertiserId", "region"]
        target = "bidPrice"

        X = data[features]
        y = data[target]

        # Split data for training & testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        print("Training RandomForest model...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Save the trained model
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

        print(f"Model trained and saved as '{self.model_path}'")

    def load_model(self):
        #Loads the trained model from disk.

        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
                self.model = model_data["model"]
                self.scaler = model_data["scaler"]
            print("Model loaded successfully!")
        else:
            print("Model file not found! Please train the model first.")

    def predict_bid(self, bid_request: BidRequest) -> int:
        #Predicts bid price for a given bid request.

        if not self.model:
            print("Model is not loaded. Loading now...")
            self.load_model()
            if not self.model:
                return -1

        # Extract relevant features
        features = np.array([
            bid_request.adSlotWidth,
            bid_request.adSlotHeight,
            bid_request.adSlotFloorPrice,
            bid_request.advertiserId,
            bid_request.region
        ]).reshape(1, -1)

        # Scale the input
        features_scaled = self.scaler.transform(features)

        # Predict bid price
        predicted_price = int(self.model.predict(features_scaled)[0])

        return max(predicted_price, bid_request.adSlotFloorPrice)  # Ensure bid >= floor price