import os
import datetime
import time
import random
import requests
import pandas as pd
import numpy as np
import finnhub
import torch
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_caching import Cache
from functools import wraps
from requests.exceptions import RequestException
from unsplash.api import Api
from unsplash.auth import Auth
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

# Configure caching
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Load API keys from environment variables
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY')
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY')
NEWS_API_URL = "https://mboum-finance.p.rapidapi.com/v1/markets/news"
RAPIDAPI_HOST = "mboum-finance.p.rapidapi.com"

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Set headers for RapidAPI requests
headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

# Initialize sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def truncate_and_pad_text(text, tokenizer, max_length=512):
    """Encode text with padding and truncation for model input."""
    encoded = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']

def analyze_sentiment(text, model, tokenizer):
    """Analyze sentiment using a specified model and tokenizer."""
    try:
        input_ids, attention_mask = truncate_and_pad_text(text, tokenizer)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        scores = outputs.logits.softmax(dim=1)
        return scores[0][1].item() - scores[0][0].item()
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0

def analyze_sentiment_finbert(text, model, tokenizer):
    """Analyze sentiment using the FinBERT model."""
    try:
        input_ids, attention_mask = truncate_and_pad_text(text, tokenizer)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        scores = outputs.logits.softmax(dim=1)
        return scores[0][2].item() - scores[0][0].item()
    except Exception as e:
        print(f"Error in FinBERT sentiment analysis: {e}")
        return 0

def fetch_yahoo_data(ticker):
    """Fetch stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1mo")
        current_volume = info.get('volume', 0)
        avg_volume = info.get('averageVolume', 0)
        relative_volume = current_volume / avg_volume if avg_volume else 0
        return {
            "volume": current_volume,
            "avg_volume": avg_volume,
            "relative_volume": relative_volume
        }
    except Exception as e:
        print(f"Error fetching Yahoo data for {ticker}: {e}")
        return {
            "volume": 0,
            "avg_volume": 0,
            "relative_volume": 0
        }

def fetch_finnhub_data(ticker):
    """Fetch stock data from Finnhub."""
    try:
        quote = finnhub_client.quote(ticker)
        profile = finnhub_client.company_profile2(symbol=ticker)
        metrics = finnhub_client.company_basic_financials(ticker, 'all')['metric']
        yahoo_data = fetch_yahoo_data(ticker)
        return {
            "name": profile.get("name", "N/A"),
            "price": quote.get("c", 0),
            "change": quote.get("d", 0),
            "percent_change": quote.get("dp", 0),
            "market_cap": metrics.get("marketCapitalization", 0),
            "pe_ratio": metrics.get("peBasicExclExtraTTM", 0),
            "avg_volume": yahoo_data["avg_volume"],
            "volume": yahoo_data["volume"],
            "relative_volume": yahoo_data["relative_volume"],
            "country": profile.get("country", "N/A"),
        }
    except Exception as e:
        print(f"Error fetching data for {ticker} from Finnhub: {e}")
        return {
            "name": "N/A", "price": 0, "change": 0, "percent_change": 0,
            "market_cap": 0, "pe_ratio": 0, "avg_volume": 0,
            "relative_volume": 0, "volume": 0, "country": "N/A",
        }

def fetch_news_data(ticker, start_date, end_date):
    """Fetch news data from RapidAPI."""
    querystring = {"symbol": ticker}
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(NEWS_API_URL, headers=headers, params=querystring)
            response.raise_for_status()
            data = response.json()
            if 'body' not in data or not isinstance(data['body'], list):
                print(f"Unexpected response format for {ticker}: {data}")
                return []
            news_data = data['body']
            filtered_news = []
            for news in news_data:
                if isinstance(news, dict) and 'pubDate' in news:
                    news_date = datetime.datetime.strptime(news['pubDate'], '%a, %d %b %Y %H:%M:%S %z').date()
                    if start_date <= news_date <= end_date:
                        filtered_news.append(news)
                else:
                    print(f"Unexpected news item format for {ticker}: {news}")
            return filtered_news
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                    continue
            print(f"Error fetching news data for {ticker}: {e}")
            return []
        except Exception as e:
            print(f"Error fetching news data for {ticker}: {e}")
            return []
    print(f"Failed to fetch news data for {ticker} after {max_retries} attempts")
    return []

def process_news_data(tickers, start_date, end_date):
    """Process and analyze news data for multiple tickers."""
    all_news = []
    for ticker in tickers:
        news_data = fetch_news_data(ticker, start_date, end_date)
        for news in news_data:
            headline = news.get('title', '')
            body = news.get('description', '')
            link = news.get('link', '')
            news_date = datetime.datetime.strptime(news.get('pubDate', ''), '%a, %d %b %Y %H:%M:%S %z')
            vader_score = vader_analyzer.polarity_scores(body)['compound']
            finbert_score = analyze_sentiment_finbert(body, finbert_model, finbert_tokenizer)
            roberta_score = analyze_sentiment(body, roberta_model, roberta_tokenizer)
            all_news.append({
                'Ticker': ticker,
                'Date': news_date.strftime('%Y-%m-%d'),
                'Time': news_date.strftime('%H:%M:%S'),
                'Headline': headline,
                'Link': link,
                'Body': body,
                'vader_score': vader_score,
                'vader': 'positive' if vader_score > 0.05 else ('negative' if vader_score < -0.05 else 'neutral'),
                'finbert_score': finbert_score,
                'roberta_score': roberta_score,
                'ensemble_score': (vader_score + finbert_score + roberta_score) / 3
            })
        time.sleep(1)
    if not all_news:
        return pd.DataFrame([{
            'Ticker': 'N/A', 'Date': 'N/A', 'Time': 'N/A', 'Headline': 'No News',
            'Link': 'N/A', 'Body': 'N/A', 'vader_score': 0, 'vader': 'neutral',
            'finbert_score': 0, 'roberta_score': 0, 'ensemble_score': 0
        }])
    return pd.DataFrame(all_news)

def convert_int64_to_int(obj):
    """Convert numpy int64 to Python int for JSON serialization."""
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_int64_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(i) for i in obj]
    return obj

def rate_limited(max_per_second):
    """Decorator to rate limit API calls."""
    min_interval = 1.0 / float(max_per_second)
    def decorator(func):
        last_time_called = [0.0]
        @wraps(func)
        def rate_limited_function(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return ret
        return rate_limited_function
    return decorator

def fetch_news_with_retry(max_retries=3, delay=5):
    """Fetch news with retries on failure."""
    for attempt in range(max_retries):
        try:
            response = requests.get(NEWS_API_URL, headers=headers)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)

@app.route('/search', methods=['POST'])
def search():
    """Search endpoint to fetch stock data and news."""
    try:
        data = request.json
        ticker = data.get('ticker', 'AAPL')  # Default to "AAPL" if no ticker is provided

        # Fetch stock data for chart
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        plot_stock = hist.reset_index().apply(lambda row: {
            'date': row['Date'].strftime('%Y-%m-%d'),
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': int(row['Volume'])
        }, axis=1).tolist()

        # Fetch company info and current stock data
        stock_info = fetch_finnhub_data(ticker)

        # Fetch news data
        start_date = datetime.date.today() - datetime.timedelta(days=30)
        end_date = datetime.date.today()
        news_data = fetch_news_data(ticker, start_date, end_date)

        # Process sentiment for each news item
        processed_news = []
        for news in news_data:
            headline = news.get('title', '')
            body = news.get('description', '')
            full_text = f"{headline} {body}"

            vader_score = vader_analyzer.polarity_scores(full_text)['compound']
            finbert_score = analyze_sentiment_finbert(full_text, finbert_model, finbert_tokenizer)
            roberta_score = analyze_sentiment(full_text, roberta_model, roberta_tokenizer)

            processed_news.append({
                'headline': headline,
                'url': news.get('link', ''),
                'date': news.get('pubDate', ''),
                'vader_sentiment': 'positive' if vader_score > 0.05 else ('negative' if vader_score < -0.05 else 'neutral'),
                'finbert_sentiment': 'positive' if finbert_score > 0.05 else ('negative' if finbert_score < -0.05 else 'neutral'),
                'roberta_sentiment': 'positive' if roberta_score > 0.05 else ('negative' if roberta_score < -0.05 else 'neutral')
            })

        response_data = {
            'stockData': plot_stock,
            'stockInfo': {
                'name': stock_info['name'],
                'ticker': ticker,
                'price': float(stock_info['price']),
                'change': float(stock_info['change']),
                'percentChange': float(stock_info['percent_change']),
                'marketCap': float(stock_info['market_cap']),
                'peRatio': float(stock_info['pe_ratio']),
                'avgVolume': int(stock_info['avg_volume']),
                'relativeVolume': float(stock_info['relative_volume']),
                'volume': int(stock_info['volume']),
            },
            'news': processed_news
        }

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error in search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/general-financial-news', methods=['GET'])
@cache.cached(timeout=300)
@rate_limited(1)
def get_general_financial_news():
    """Endpoint to fetch general financial news."""
    try:
        print("Fetching general financial news...")
        data = fetch_news_with_retry()
        if 'body' not in data or not isinstance(data['body'], list):
            print(f"Unexpected response format: {data}")
            return jsonify({'error': f'Unexpected response format: {data}'}), 500
        news_data = data['body']
        processed_news = []

        unsplash_client_id = os.environ.get('UNSPLASH_CLIENT_ID')
        unsplash_client_secret = os.environ.get('UNSPLASH_CLIENT_SECRET')
        unsplash_redirect_uri = os.environ.get('UNSPLASH_REDIRECT_URI')
        unsplash_auth = Auth(unsplash_client_id, unsplash_client_secret, unsplash_redirect_uri)
        unsplash_api = Api(unsplash_auth)

        for news in news_data:
            title = news.get('title', '')
            search_query = ' '.join(title.split()[:3])  # Use first 3 words of title
            images = unsplash_api.search.photos(search_query, page=1, per_page=1)
            image_url = images['results'][0].urls.small if images['results'] else 'https://example.com/default-financial-news-image.jpg'
            processed_news.append({
                'title': title,
                'summary': news.get('description', ''),
                'url': news.get('link', ''),
                'source': news.get('source', ''),
                'published_at': news.get('pubDate', ''),
                'tickers': news.get('tickers', []),
                'image_url': image_url
            })
        print(f"Returning {len(processed_news)} news items")
        return jsonify(processed_news)
    except RequestException as e:
        print(f"Error fetching news after retries: {str(e)}")
        return jsonify({'error': f'Failed to fetch news: {str(e)}'}), 500
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint to upload a CSV file and process news data."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    start_date = request.form.get('startDate')
    end_date = request.form.get('endDate')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and start_date and end_date:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        df = pd.read_csv(file)
        tickers = df['Ticker'].tolist()
        results = []
        news = process_news_data(tickers, start_date, end_date)
        for ticker in tickers:
            try:
                stock_info = fetch_finnhub_data(ticker)
                ticker_news = news[news['Ticker'] == ticker]
                vader_score = round(ticker_news['vader_score'].mean(), 2) if not ticker_news.empty else 0
                finbert_score = round(ticker_news['finbert_score'].mean(), 2) if not ticker_news.empty else 0
                roberta_score = round(ticker_news['roberta_score'].mean(), 2) if not ticker_news.empty else 0
                ensemble_score = round(ticker_news['ensemble_score'].mean(), 2) if not ticker_news.empty else 0
                results.append({
                    'No': len(results) + 1,
                    'Ticker': ticker,
                    'Company': stock_info['name'],
                    'Country': stock_info['country'],
                    'Market Cap': round(float(stock_info['market_cap'] or 0), 2),
                    'P/E': round(float(stock_info['pe_ratio'] or 0), 2),
                    'Average Volume': round(float(stock_info['avg_volume'] or 0), 2),
                    'Relative Volume': round(float(stock_info['relative_volume'] or 0), 2),
                    'Volume': round(float(stock_info['volume'] or 0), 2),
                    'Price': round(float(stock_info['price'] or 0), 2),
                    'Change %': round(float(stock_info['percent_change'] or 0), 2),
                    'Vader Score': vader_score,
                    'FinBERT Score': finbert_score,
                    'RoBERTa Score': roberta_score,
                    'Ensemble Score': ensemble_score
                })
            except Exception as e:
                print(f"Error processing ticker {ticker}: {e}")
                results.append({
                    'No': len(results) + 1,
                    'Ticker': ticker,
                    'Company': 'Error',
                    'Country': 'N/A',
                    'Market Cap': 0,
                    'P/E': 0,
                    'Average Volume': 0,
                    'Relative Volume': 0,
                    'Volume': 0,
                    'Price': 0,
                    'Change %': 0,
                    'Vader Score': 0,
                    'FinBERT Score': 0,
                    'RoBERTa Score': 0,
                    'Ensemble Score': 0
                })
        results = convert_int64_to_int(results)
        all_news = convert_int64_to_int(news.to_dict(orient='records'))
        return jsonify({'results': results, 'news': all_news})

@socketio.on('connect')
def test_connect():
    """Socket.IO connection event handler."""
    print("Client connected")

@socketio.on('disconnect')
def test_disconnect():
    """Socket.IO disconnection event handler."""
    print("Client disconnected")

if __name__ == '__main__':
    socketio.run(app, debug=True)


