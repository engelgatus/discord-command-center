"""
Simple Flask web server to keep bot alive on Render free tier
"""
from flask import Flask
from threading import Thread
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 Command Center Bot is running! ✅"

@app.route('/health')
def health():
    return "OK", 200

def run():
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    """Start web server in background thread"""
    t = Thread(target=run)
    t.start()

