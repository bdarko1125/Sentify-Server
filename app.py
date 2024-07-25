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

# Flask app setup
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')

# API configurations
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', 'cqfi8mpr01qle0e3p83gcqfi8mpr01qle0e3p840')
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY', '9601085229msh181f78af487f1dap109702jsn344b5d9a1892')
NEWS_API_URL = "https://mboum-finance.p.rapidapi.com/v1/markets/news"
RAPIDAPI_HOST = "mboum-finance.p.rapidapi.com"

# Finnhub API client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Headers for RapidAPI
headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

# Sentiment analysis setup
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize tokenizers and models
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")


def truncate_and_pad_text(text, tokenizer, max_length=512):
    encoded = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']


def analyze_sentiment(text, model, tokenizer):
    try:
        input_ids, attention_mask = truncate_and_pad_text(text, tokenizer)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        scores = outputs.logits.softmax(dim=1)
        return scores[0][1].item() - scores[0][0].item()  # Positive score minus negative score
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0  # Return neutral sentiment in case of error


def analyze_sentiment_finbert(text, model, tokenizer):
    try:
        input_ids, attention_mask = truncate_and_pad_text(text, tokenizer)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        scores = outputs.logits.softmax(dim=1)
        # FinBERT order: [negative, neutral, positive]
        return scores[0][2].item() - scores[0][0].item()  # Positive score minus negative score
    except Exception as e:
        print(f"Error in FinBERT sentiment analysis: {e}")
        return 0  # Return neutral sentiment in case of error


def fetch_yahoo_data(ticker):
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
    querystring = {"symbol": ticker}
    max_retries = 3
    retry_delay = 5  # seconds

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
            if e.response.status_code == 429:  # Too Many Requests
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

        time.sleep(1)  # Add delay between processing each ticker

    if not all_news:
        return pd.DataFrame([{
            'Ticker': 'N/A', 'Date': 'N/A', 'Time': 'N/A', 'Headline': 'No News',
            'Link': 'N/A', 'Body': 'N/A', 'vader_score': 0, 'vader': 'neutral',
            'finbert_score': 0, 'roberta_score': 0, 'ensemble_score': 0
        }])

    return pd.DataFrame(all_news)


def convert_int64_to_int(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_int64_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(i) for i in obj]
    return obj


@app.route('/upload', methods=['POST'])
def upload_file():
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
    print("Client connected")


@socketio.on('disconnect')
def test_disconnect():
    print("Client disconnected")


if __name__ == '__main__':
    socketio.run(app, debug=True)