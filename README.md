
# Sentify Server

## Overview
Sentify Server is a Flask-based application that integrates various APIs and libraries to fetch, analyze, and serve financial data and news. The application provides sentiment analysis using multiple models and serves data through RESTful endpoints and a Socket.IO connection.

## Project Structure
- **.env**: Environment file containing API keys and configurations.
- **.gitignore**: Specifies files and directories to be ignored by git.
- **app.py**: The main application script that contains the server logic and endpoints.
- **requirements.txt**: Lists the Python dependencies needed to run the application.

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package installer)

### Installation

1. **Clone the repository**
    ```sh
    git clone https://github.com/bdarko1125/Sentify-Server.git
    cd sentify-server
    ```

2. **Create and activate a virtual environment**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
    ```

3. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**
    - Create a `.env` file in the root directory.
    - Add your API keys and other configuration details as shown below:

    ```
    FINNHUB_API_KEY=innhub_api_key
    RAPIDAPI_KEY=rapidapi_key
    UNSPLASH_CLIENT_ID=unsplash_client_id
    UNSPLASH_CLIENT_SECRET=unsplash_client_secret
    UNSPLASH_REDIRECT_URI=unsplash_redirect_uri
    ```

## Running the Application

1. **Start the Flask server**
    ```sh
    python app.py
    ```

2. **Access the application**
    - The server will start on `http://localhost:5000` by default.

## API Endpoints

### `/search` [POST]
Fetch stock data and news for a given ticker.

- **Request Body**
    ```json
    {
        "ticker": "AAPL"
    }
    ```

### `/general-financial-news` [GET]
Fetch general financial news.

### `/upload` [POST]
Upload a CSV file and process news data.

- **Form Data**
    - `file`: CSV file containing tickers.
    - `startDate`: Start date in `YYYY-MM-DD` format.
    - `endDate`: End date in `YYYY-MM-DD` format.

## Libraries and Tools Used

- **Flask**: Web framework for Python.
- **Flask-CORS**: Handling Cross-Origin Resource Sharing (CORS).
- **Flask-SocketIO**: Adding WebSocket support to Flask applications.
- **Flask-Caching**: Caching support for Flask applications.
- **Finnhub**: Stock API client.
- **yfinance**: Yahoo Finance API client.
- **nltk**: Natural Language Toolkit for sentiment analysis.
- **transformers**: Huggingface library for FinBERT and RoBERTa models.
- **requests**: HTTP library for Python.
- **pandas**: Data analysis and manipulation library.
- **numpy**: Numerical operations library.
- **dotenv**: Loading environment variables from a `.env` file.





