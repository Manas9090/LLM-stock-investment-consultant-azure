import os
import openai
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
import streamlit as st
import datetime
from pymongo import MongoClient

# -------------------------------
# --- Load Environment Variables
# -------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
twelvedata_api_key = os.getenv("TWELVEDATA_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
mongo_conn_str = os.getenv("MONGO_URI")

# -------------------------------
# --- Validate Environment Variables
# -------------------------------
missing_keys = []
for key, val in {
    "OPENAI_API_KEY": openai.api_key,
    "TWELVEDATA_API_KEY": twelvedata_api_key,
    "NEWS_API_KEY": news_api_key,
    "ALPHA_VANTAGE_API_KEY": alpha_vantage_api_key,
    "MONGO_URI": mongo_conn_str
}.items():
    if not val:
        missing_keys.append(key)

if missing_keys:
    st.error(f"❌ Missing environment variables: {', '.join(missing_keys)}")
    st.stop()

# -------------------------------
# --- MongoDB Connection (Azure Compatible)
# -------------------------------
try:
    client = MongoClient(
        mongo_conn_str,
        tls=True,
        tlsAllowInvalidCertificates=False,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000
    )
    client.admin.command('ping')
    db = client["stock_db"]
    collection = db["stock_symbols"]
    st.success("✅ Connected to MongoDB Atlas")
except Exception as e:
    st.error(f"❌ MongoDB connection failed: {e}")
    st.stop()

# -------------------------------
# --- Embedding Model
# -------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# --- MongoDB Utility Functions
# -------------------------------
def fetch_ticker_from_db(company_name):
    company = collection.find_one({"company_name": {"$regex": f"^{company_name}$", "$options": "i"}})
    if company:
        return company["symbol"], company["exchange"]
    return None, None

def match_company_in_db(user_query):
    all_companies = [doc["company_name"] for doc in collection.find({}, {"company_name": 1})]
    for name in all_companies:
        if name.lower() in user_query.lower():
            return name
    return None

# -------------------------------
# --- Fetch Live News
# -------------------------------
def fetch_live_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&sortBy=publishedAt&language=en&apiKey={news_api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [f"{a['title']}. {a.get('description', '')}" for a in articles if a.get('title')]

# -------------------------------
# --- Build FAISS Index
# -------------------------------
def build_faiss_index(corpus):
    embeddings = embedding_model.encode(corpus)
    embeddings = normalize(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, corpus

# -------------------------------
# --- Fetch Historical Data
# -------------------------------
def get_alpha_vantage_data(symbol):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": alpha_vantage_api_key,
        "outputsize": "full"
    }
    r = requests.get(url, params=params)
    data = r.json()
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Alpha Vantage error: {data}")

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
    df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.last("6M")[['close']].astype(float)

def get_twelve_data(symbol, exchange):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=180)
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "outputsize": 500,
        "apikey": twelvedata_api_key
    }
    if exchange:
        params["exchange"] = exchange
    response = requests.get(url, params=params)
    data = response.json()
    if "values" not in data:
        raise ValueError(f"Twelve Data error: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    return df[["close"]].astype(float)

# -------------------------------
# --- Generate Insight (RAG + LLM)
# -------------------------------
def get_stock_insight(query, corpus, index, symbol, hist_df):
    query_embedding = embedding_model.en
