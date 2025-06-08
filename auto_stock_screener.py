# Automated Stock Screener - Minimal Version
import pandas as pd
import numpy as np
import requests
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

# 🔧 AUTO-CONFIGURATION - Edit these values once
AUTO_CONFIG = {
    "upstox_token": "your_upstox_access_token_here",
    "telegram_bot_token": "your_telegram_bot_token_here", 
    "telegram_chat_id": "your_telegram_chat_id_here",
    "default_stocks": """instrument_key,tradingsymbol
NSE_EQ|INE585B01010,MARUTI
NSE_EQ|INE139A01034,NATIONALUM
NSE_EQ|INE763I01026,TARIL
NSE_EQ|INE970X01018,LEMONTREE
NSE_EQ|INE522D01027,MANAPPURAM
NSE_EQ|INE002A01018,RELIANCE
NSE_EQ|INE009A01021,INFY
NSE_EQ|INE467B01029,ASIANPAINT
NSE_EQ|INE040A01034,HDFCBANK
NSE_EQ|INE005A01025,WIPRO"""
}

class StockScreener:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {AUTO_CONFIG["upstox_token"]}',
            'Content-Type': 'application/json'
        })
    
    def get_data(self, instrument_key):
        """Get stock data"""
        try:
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
            url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{to_date}/{from_date}"
            
            response = self.session.get(url, timeout=10)
            data = response.json()
            
            if data.get('status') != 'success':
                return None
                
            candles = data.get('data', {}).get('candles', [])
            if not candles:
                return None
                
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
            return df.sort_values('timestamp').reset_index(drop=True)
            
        except:
            return None
    
    def calculate_indicators(self, df):
        """Calculate all indicators"""
        df['rsi'] = self.rsi(df['close'])
        df['atr'] = self.atr(df)
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['vol_ma'] = df['volume'].rolling(20).mean()
        return df
    
    def rsi(self, prices, period=14):
        """RSI calculation"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def atr(self, df, period=14):
        """ATR calculation"""
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def check_signal(self, df, symbol):
        """Check if stock meets signal criteria"""
        if len(df) < 50:
            return None
            
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        
        # Signal conditions
        rsi_ok = 30 <= latest['rsi'] <= 70
        trend_ok = latest['ema9'] > latest['ema21']
        volume_ok = latest['volume'] > latest['vol_ma'] * 1.3
        breakout_ok = latest['close'] > df.iloc[-3:-1]['high'].max()
        
        if all([rsi_ok, trend_ok, volume_ok, breakout_ok]):
            entry = latest['close']
            atr = latest['atr']
            return {
                'Stock': symbol,
                'Entry': f"₹{entry:.2f}",
                'Stop Loss': f"₹{entry - atr * 1.8:.2f}",
                'Target': f"₹{entry + atr * 2.5:.2f}",
                'RSI': f"{latest['rsi']:.1f}",
                'Volume': f"{latest['volume']/latest['vol_ma']:.1f}x"
            }
        return None
    
    def scan_stocks(self, stock_list):
        """Scan all stocks"""
        signals = []
        progress = st.progress(0)
        
        def analyze_stock(stock_data):
            instrument_key, symbol = stock_data
            df = self.get_data(instrument_key)
            if df is not None:
                return self.check_signal(df, symbol)
            return None
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(analyze_stock, stock) for stock in stock_list]
            
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        signals.append(result)
                        st.success(f"✅ {result['Stock']}")
                except:
                    pass
                progress.progress((i + 1) / len(stock_list))
        
        return signals
    
    def send_telegram(self, signals):
        """Send to Telegram"""
        if not AUTO_CONFIG["telegram_bot_token"]:
            return False
            
        if not signals:
            message = "📈 No signals found"
        else:
            message = f"📈 {len(signals)} Signals Found:\n\n"
            for i, s in enumerate(signals, 1):
                message += f"{i}. {s['Stock']}\nEntry: {s['Entry']} | SL: {s['Stop Loss']} | Target: {s['Target']}\n\n"
        
        url = f"https://api.telegram.org/bot{AUTO_CONFIG['telegram_bot_token']}/sendMessage"
        try:
            requests.post(url, data={
                "chat_id": AUTO_CONFIG["telegram_chat_id"],
                "text": message
            })
            return True
        except:
            return False

def main():
    st.set_page_config(page_title="Auto Stock Screener", layout="wide")
    st.title("🚀 Auto Stock Screener")
    st.info("✨ All tokens pre-configured! Just click scan.")
    
    # Load stocks
    uploaded_file = st.file_uploader("📁 Upload CSV (optional)", type=['csv'])
    if uploaded_file:
        csv_content = uploaded_file.getvalue().decode('utf-8')
    else:
        csv_content = AUTO_CONFIG["default_stocks"]
        st.info("📊 Using default 10 stocks")
    
    # Parse stocks
    try:
        df = pd.read_csv(StringIO(csv_content))
        stock_list = [(row['instrument_key'], row['tradingsymbol']) for _, row in df.iterrows()]
    except:
        st.error("❌ Invalid CSV format")
        return
    
    # Scan button
    if st.button("🔍 START SCAN", type="primary", use_container_width=True):
        screener = StockScreener()
        
        with st.spinner(f"Scanning {len(stock_list)} stocks..."):
            signals = screener.scan_stocks(stock_list)
        
        # Results
        if signals:
            st.success(f"🎯 Found {len(signals)} signals!")
            
            # Display table
            df_results = pd.DataFrame(signals)
            st.dataframe(df_results, use_container_width=True)
            
            # Download
            csv_data = df_results.to_csv(index=False)
            st.download_button("📥 Download CSV", csv_data, f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
            
            # Telegram
            if screener.send_telegram(signals):
                st.success("📱 Sent to Telegram!")
            else:
                st.warning("⚠️ Telegram failed")
        else:
            st.warning("😔 No signals found")

if __name__ == "__main__":
    main()