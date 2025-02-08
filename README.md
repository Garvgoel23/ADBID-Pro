# RTB-HACKATHON

## Overview

RTB-Hackathon is a **Real-Time Bidding (RTB) system** designed for an online advertising auction platform.\
It processes **bid requests**, applies **bidding logic**, and **predicts optimal bid prices** using a structured **backend (Python)** and **frontend (React).**

## Project Directory Structure

```
 RTB-Hackathon
  backend (Python)
     Bid.py               # Main bidding logic
     Bidder.py            # Abstract bidder class
     BidRequest.py        # Handles incoming bid requests
     data_processor.py    # Loads and processes logs
     model.py             # ML model for bid price prediction (if used)
     server.py            # Flask API to connect backend with frontend
     utils.py             # Helper functions (logging, JSON handling, etc.)
     requirements.txt     # Python dependencies
     README.md            # Backend setup & usage

  frontend (React)
     src/                 # React components
     App.js               # Main dashboard UI
     api.js               # Fetches data from backend
     styles.css           # CSS for UI
     package.json         # Frontend dependencies
     README.md            # Frontend setup & usage

  dataset                # Provided hackathon data

```

##  Setup Instructions

###  Backend (Python)

#### ** Install dependencies**

```bash
cd backend
pip install -r requirements.txt
```

####  Run the server

```bash
python server.py
```

 The server will start at ``

---

###  Frontend (React)

#### Install dependencies

```bash
cd frontend
npm install
```

#### Run the frontend

```bash
npm start
```

 The frontend will run at ``

---

##  How It Works

 **Bid Requests** → Incoming bid requests are received and processed.\
 **Bid Processing** → The system determines whether to place a bid.\
 **Bidding Logic** → `Bid.py` calculates bid prices based on advertiser type.\
 **Machine Learning (Optional)** → `model.py` can enhance bid pricing.\
 **Flask API** → `server.py` handles API endpoints for frontend interaction.\
 **React Dashboard** → The frontend displays live bidding statistics.

---

 API Endpoints (`server.py`)

| Method | Endpoint  | Description                                    |
| ------ | --------- | ---------------------------------------------- |
| `POST` | `/bid`    | Receives a bid request and returns a bid price |
| `GET`  | `/status` | Returns the server status                      |
| `GET`  | `/logs`   | Retrieves system logs                          |

---

##  Contributors

- Keshav Sharma
- Garv Goel
- Kush Aheer
- Naveen Verma


