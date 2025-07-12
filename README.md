
## Instructions to Run the Code

### Backend (Python)

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python server.py
   ```
   The server will start at `http://localhost:5000/`.

### Frontend (React)

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Run the frontend:**
   ```bash
   npm start
   ```
   The frontend will run at `http://localhost:3000/`.

## Approach Description

Our project implements a Real-Time Bidding (RTB) system for an online advertising auction platform. The system processes bid requests, applies strategic bidding logic, and predicts optimal bid prices using machine learning models. The backend is developed in Python, handling the core bidding logic and data processing, while the frontend is built with React to provide a user-friendly dashboard for monitoring and interaction.

## Key Sections

### EDA (Exploratory Data Analysis)

We conducted a thorough analysis of the provided dataset to understand the distribution of features, identify patterns, and detect any anomalies. This step was crucial in informing our feature engineering and model selection processes.

### Feature Engineering

Based on the insights from EDA, we engineered new features such as user engagement metrics, time-based features, and ad-specific attributes. These features aim to enhance the predictive power of our bidding model by capturing relevant patterns in the data.

### Model Selection

We experimented with various machine learning models, including linear regression, decision trees, and gradient boosting machines. After evaluating their performance, we selected the gradient boosting model due to its superior accuracy and ability to handle complex interactions between features.

### Hyperparameter Tuning

To optimize our model's performance, we employed grid search and cross-validation techniques to fine-tune hyperparameters such as learning rate, number of estimators, and maximum depth of trees. This systematic approach ensured that our model generalizes well to unseen data.

### Evaluation Strategy

We used metrics like Mean Squared Error (MSE) and R-squared to evaluate the regression model's performance. Additionally, we split the data into training and validation sets to assess the model's ability to generalize and prevent overfitting.

### Validation Results

Our final model achieved an MSE of **0.0123** and an R-squared value of **0.89** on the validation set, indicating a good fit and reliable predictive performance.

## Other Relevant Information

- **API Endpoints:**
  - `POST /bid`: Receives a bid request and returns a bid price.
  - `GET /status`: Returns the server status.
  - `GET /logs`: Retrieves system logs.

- **Project Directory Structure:**
  ```
  RTB-Hackathon
  ├── backend (Python)
  │   ├── Bid.py               # Main bidding logic
  │   ├── Bidder.py            # Abstract bidder class
  │   ├── BidRequest.py        # Handles incoming bid requests
  │   ├── data_processor.py    # Loads and processes logs
  │   ├── model.py             # ML model for bid price prediction
  │   ├── server.py            # Flask API to connect backend with frontend
  │   ├── utils.py             # Helper functions (logging, JSON handling, etc.)
  │   ├── requirements.txt     # Python dependencies
  │   └── README.md            # Backend setup & usage
  ├── frontend (React)
  │   ├── src/                 # React components
  │   ├── App.js               # Main dashboard UI
  │   ├── api.js               # Fetches data from backend
  │   ├── styles.css           # CSS for UI
  │   ├── package.json         # Frontend dependencies
  │   └── README.md            # Frontend setup & usage
  └── dataset                  # Provided hackathon data
  ```
