from gevent import monkey

monkey.patch_all()

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import urllib.error
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import datetime
import finnhub
import threading
import time
import ssl
import socket

# Flask app setup
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')

# Finnhub API client
finnhub_client = finnhub.Client(api_key="cpr2japr01qifjjvdfegcpr2japr01qifjjvdff0")

# Sentiment analysis pipelines
vader_analyzer = SentimentIntensityAnalyzer()
finvader_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
finbert_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
roberta_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")


# Function to fetch news data with retry mechanism
def fetch_news_data(ticker, retries=3, backoff_factor=0.3):
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    url = finwiz_url + ticker
    req = Request(url=url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})

    for attempt in range(retries):
        try:
            resp = urlopen(req, timeout=10)
            html = BeautifulSoup(resp, features="lxml")
            news_table = html.find(id='news-table')
            return news_table
        except (urllib.error.URLError, ssl.SSLError, socket.error) as e:
            print(f"Error fetching data for {ticker} (attempt {attempt + 1}): {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    return None


# Function to fetch financial data using yfinance and Finnhub
def fetch_finnhub_yfinance_data(ticker):
    try:
        # Fetch data using Finnhub
        quote = finnhub_client.quote(ticker)
        company = finnhub_client.company_profile2(symbol=ticker)
        metrics = finnhub_client.company_basic_financials(ticker, 'all')['metric']

        price = quote.get("c", 0)
        change = quote.get("d", 0)
        percent_change = quote.get("dp", 0)
        name = company.get("name", "N/A")
        country = company.get("country", "N/A")

        # Fetch data using yfinance
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get("marketCap", 0)
        pe_ratio = stock.info.get("trailingPE", 0)
        avg_volume = stock.info.get("averageVolume", 0)
        volume = quote.get("v", 0)
        relative_volume = volume / avg_volume if avg_volume else 0

        return {
            "name": name,
            "price": price,
            "change": change,
            "percent_change": percent_change,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "avg_volume": avg_volume,
            "relative_volume": relative_volume,
            "volume": volume,
            "country": country,
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {
            "name": "N/A",
            "price": 0,
            "change": 0,
            "percent_change": 0,
            "market_cap": 0,
            "pe_ratio": 0,
            "avg_volume": 0,
            "relative_volume": 0,
            "volume": 0,
            "country": "N/A",
        }


# Convert sentiment labels to numeric scores for mode calculation
def sentiment_label_to_numeric(label):
    return {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }.get(label, 0)


# Function to process news data
def process_news_data(news_tables):
    parsed_news = []
    for ticker, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            if x.a:
                text = x.a.get_text()
                date_scrape = x.td.text.split()

                if len(date_scrape) == 1:
                    time = date_scrape[0]
                    date = datetime.datetime.now().date()
                else:
                    date = date_scrape[0]
                    time = date_scrape[1]

                parsed_news.append([ticker, date, time, text])

    columns = ['Ticker', 'Date', 'Time', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)

    # Specify the format of the date
    news['Date'] = pd.to_datetime(news['Date'], format='%b-%d-%y', errors='coerce').dt.date

    # Analyze sentiments
    news['vader_score'] = news['Headline'].apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])
    news['vader'] = news['vader_score'].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))

    news['finvader_score'] = news['Headline'].apply(lambda x: finvader_analyzer(x)[0]['label'])
    news['finbert_score'] = news['Headline'].apply(lambda x: finbert_analyzer(x)[0]['label'])
    news['roberta_score'] = news['Headline'].apply(lambda x: roberta_analyzer(x)[0]['label'])

    # Map RoBERTa labels to positive, neutral, negative
    news['roberta'] = news['roberta_score'].map(
        lambda x: 'positive' if x == 'LABEL_2' else ('negative' if x == 'LABEL_0' else 'neutral'))

    return news


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        df = pd.read_csv(file)
        tickers = df['Ticker'].tolist()
        results = []

        for ticker in tickers:
            try:
                stock_info = fetch_finnhub_yfinance_data(ticker)
                news_table = fetch_news_data(ticker)
                if not news_table:
                    continue

                news_tables = {ticker: news_table}
                news = process_news_data(news_tables)

                # Calculate the mean for vader_score
                vader_score = news['vader_score'].mean() if not news['vader_score'].empty else 0

                # Convert sentiment labels to numeric for mode calculation
                news['finvader_numeric'] = news['finvader_score'].apply(sentiment_label_to_numeric)
                news['finbert_numeric'] = news['finbert_score'].apply(sentiment_label_to_numeric)
                news['roberta_numeric'] = news['roberta'].apply(sentiment_label_to_numeric)

                finvader_score = news['finvader_numeric'].mode()[0] if not news['finvader_numeric'].empty else 0
                finbert_score = news['finbert_numeric'].mode()[0] if not news['finbert_numeric'].empty else 0
                roberta_score = news['roberta_numeric'].mode()[0] if not news['roberta_numeric'].empty else 0

                # Convert all numerical values to float or regular int
                result = {
                    'No': len(results) + 1,
                    'Ticker': ticker,
                    'Company': stock_info['name'],
                    'Country': stock_info['country'],
                    'Market Cap': float(stock_info['market_cap']) if stock_info['market_cap'] else 'N/A',
                    'P/E': float(stock_info['pe_ratio']) if stock_info['pe_ratio'] else 'N/A',
                    'Average Volume': float(stock_info['avg_volume']) if stock_info['avg_volume'] else 'N/A',
                    'Relative Volume': float(stock_info['relative_volume']) if stock_info['relative_volume'] else 'N/A',
                    'Volume': float(stock_info['volume']) if stock_info['volume'] else 'N/A',
                    'Price': float(stock_info['price']) if stock_info['price'] else 'N/A',
                    'Change %': float(stock_info['percent_change']) if stock_info['percent_change'] else 'N/A',
                    'Vader Score': float(vader_score),
                    'FinVader Score': float(finvader_score),
                    'FinBERT Score': float(finbert_score),
                    'RoBERTa Score': float(roberta_score)
                }

                for key, value in result.items():
                    if isinstance(value, pd._libs.tslibs.nattype.NaTType):
                        result[key] = 'N/A'
                    elif isinstance(value, pd._libs.tslibs.timestamps.Timestamp):
                        result[key] = str(value)
                    elif isinstance(value, (int, float)):
                        result[key] = float(value) if not pd.isnull(value) else 'N/A'

                results.append(result)
            except Exception as e:
                print(f"Error processing ticker {ticker}: {e}")

        return jsonify(results)


@socketio.on('connect')
def test_connect():
    print("Client connected")


@socketio.on('disconnect')
def test_disconnect():
    print("Client disconnected")


if __name__ == '__main__':
    socketio.run(app, debug=True)
