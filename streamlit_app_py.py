# app.py - Streamlit Cloud Ready Version

import pandas as pd
import numpy as np
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StockSignal:
    """Data class for stock signals"""
    stock_name: str
    close: float
    entry_price: float
    stop_loss: float
    take_profit: float
    rsi: float
    volume_ratio: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class UpstoxClient:
    """Handles all Upstox API communication"""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://api.upstox.com/v2"
        # Use session for connection pooling and better performance
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        })
        
    def get_historical_data(self, instrument_key: str, interval: str = 'day', days: int = 100) -> Optional[pd.DataFrame]:
        """Fetch historical data with proper error handling"""
        try:
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            url = f"{self.base_url}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') != 'success':
                logger.warning(f"API returned non-success status for {instrument_key}: {data}")
                return None
                
            candles = data.get('data', {}).get('candles', [])
            if not candles:
                logger.warning(f"No candle data for {instrument_key}")
                return None
                
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert to appropriate data types
            numeric_columns = ['open', 'high', 'low', 'close']
            df[numeric_columns] = df[numeric_columns].astype(float)
            df['volume'] = df['volume'].astype(int)
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching data for {instrument_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {instrument_key}: {e}")
            return None
    
    def close_session(self):
        """Close the session when done"""
        self.session.close()

class IndicatorCalculator:
    """Handles all technical indicator calculations"""
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14, smooth_period: int = 5) -> pd.Series:
        """Calculate Average True Range with smoothing"""
        high, low, close = df['high'], df['low'], df['close']
        prev_close = close.shift(1)
        
        true_ranges = pd.concat([
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        ], axis=1)
        
        tr = true_ranges.max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.rolling(window=smooth_period).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period).mean()
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume_price = typical_price * df['volume']
        return volume_price.cumsum() / df['volume'].cumsum()
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators to dataframe in one go"""
        df = df.copy()
        
        # Technical indicators
        df['atr_sma'] = IndicatorCalculator.calculate_atr(df)
        df['rsi'] = IndicatorCalculator.calculate_rsi(df['close'])
        df['ema_9'] = IndicatorCalculator.calculate_ema(df['close'], 9)
        df['ema_21'] = IndicatorCalculator.calculate_ema(df['close'], 21)
        df['vwap'] = IndicatorCalculator.calculate_vwap(df)
        
        # Volume indicators
        df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
        df['vol_ma_50'] = df['volume'].rolling(window=50).mean()
        
        # Price action indicators
        df['range'] = df['high'] - df['low']
        df['avg_range_10'] = df['range'].rolling(window=10).mean()
        
        return df

class StockAnalyzer:
    """Handles stock analysis and signal generation"""
    
    def __init__(self, upstox_client: UpstoxClient):
        self.upstox_client = upstox_client
        
    def check_breakout_condition(self, df: pd.DataFrame) -> bool:
        """Check breakout condition"""
        latest = df.iloc[-1]
        prev_2_high = df.iloc[-3:-1]['high'].max()
        
        if pd.isna(latest['atr_sma']) or latest['atr_sma'] == 0:
            return False
            
        breakout_strength = (latest['close'] - prev_2_high) / latest['atr_sma']
        return latest['close'] > prev_2_high and breakout_strength > 0.3
    
    def check_rsi_condition(self, rsi_value: float) -> bool:
        """Check RSI condition"""
        return not pd.isna(rsi_value) and 30 <= rsi_value <= 70
    
    def check_volume_condition(self, latest: pd.Series) -> bool:
        """Check volume condition"""
        vol_ratio_20 = latest['volume'] / latest['vol_ma_20'] if latest['vol_ma_20'] > 0 else 0
        return (vol_ratio_20 > 1.3 and 
                latest['volume'] > latest['vol_ma_50'])
    
    def check_trend_condition(self, latest: pd.Series) -> bool:
        """Check trend condition"""
        return (latest['ema_9'] > latest['ema_21'] and 
                latest['close'] > latest['vwap'])
    
    def check_price_action_condition(self, latest: pd.Series) -> bool:
        """Check price action condition"""
        if latest['avg_range_10'] == 0:
            return False
            
        range_condition = latest['range'] > latest['avg_range_10']
        body_condition = (latest['close'] - latest['low']) >= 0.75 * (latest['high'] - latest['low'])
        return range_condition and body_condition
    
    def analyze_stock(self, stock_data: Tuple[str, str]) -> Optional[StockSignal]:
        """Analyze a single stock and return signal if conditions are met"""
        instrument_key, symbol = stock_data
        
        try:
            # Fetch data
            df = self.upstox_client.get_historical_data(instrument_key)
            if df is None or len(df) < 60:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Add indicators
            df = IndicatorCalculator.add_all_indicators(df)
            latest = df.iloc[-1]
            
            # Early exit conditions - check fastest conditions first
            if not self.check_rsi_condition(latest['rsi']):
                return None
                
            if not self.check_trend_condition(latest):
                return None
                
            if not self.check_volume_condition(latest):
                return None
                
            if not self.check_price_action_condition(latest):
                return None
                
            if not self.check_breakout_condition(df):
                return None
            
            # All conditions met - calculate entry/exit levels
            entry = latest['close']
            atr = latest['atr_sma']
            
            if pd.isna(atr) or atr <= 0:
                logger.warning(f"Invalid ATR for {symbol}")
                return None
                
            sl = entry - atr * 1.8
            tp = entry + atr * 2.5
            vol_ratio = latest['volume'] / latest['vol_ma_20'] if latest['vol_ma_20'] > 0 else 0
            
            return StockSignal(
                stock_name=symbol,
                close=entry,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                rsi=latest['rsi'],
                volume_ratio=vol_ratio
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

class TelegramBot:
    """Handles Telegram communication with retry logic"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.session = requests.Session()
        
    def send_message(self, message: str, max_retries: int = 3) -> bool:
        """Send message to Telegram with retry logic"""
        # Split message if too long (Telegram limit is 4096 characters)
        if len(message) > 4000:
            return self._send_long_message(message, max_retries)
        
        return self._send_single_message(message, max_retries)
    
    def _send_single_message(self, message: str, max_retries: int) -> bool:
        """Send a single message with retry logic"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, data=payload, timeout=10)
                response.raise_for_status()
                
                if response.json().get('ok'):
                    return True
                else:
                    logger.warning(f"Telegram API returned not ok: {response.text}")
                    
            except requests.RequestException as e:
                logger.error(f"Telegram send attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return False
    
    def _send_long_message(self, message: str, max_retries: int) -> bool:
        """Split and send long messages"""
        lines = message.split('\n')
        current_message = ""
        success = True
        
        for line in lines:
            if len(current_message + line + '\n') > 4000:
                if current_message:
                    success &= self._send_single_message(current_message, max_retries)
                    current_message = line + '\n'
                else:
                    # Single line too long, truncate
                    success &= self._send_single_message(line[:4000], max_retries)
            else:
                current_message += line + '\n'
        
        if current_message:
            success &= self._send_single_message(current_message, max_retries)
            
        return success

class StockScreener:
    """Main screener class that orchestrates everything"""
    
    def __init__(self, access_token: str, telegram_bot_token: str = None, telegram_chat_id: str = None):
        self.upstox_client = UpstoxClient(access_token)
        self.analyzer = StockAnalyzer(self.upstox_client)
        
        if telegram_bot_token and telegram_chat_id:
            self.telegram_bot = TelegramBot(telegram_bot_token, telegram_chat_id)
        else:
            self.telegram_bot = None
    
    def load_instruments(self, csv_content: str = None) -> List[Tuple[str, str]]:
        """Load instruments from CSV content or default data"""
        if csv_content is None:
            # Default CSV data - expanded for better demo
            csv_content = """instrument_key,tradingsymbol
NSE_EQ|INE585B01010,MARUTI
NSE_EQ|INE139A01034,NATIONALUM
NSE_EQ|INE763I01026,TARIL
NSE_EQ|INE970X01018,LEMONTREE
NSE_EQ|INE522D01027,MANAPPURAM
NSE_EQ|INE427F01016,CHALET
NSE_EQ|INE00R701025,DALBHARAT
NSE_EQ|INE917I01010,BAJAJ-AUTO
NSE_EQ|INE146L01010,KIRLOSENG
NSE_EQ|INE267A01025,HINDZINC
NSE_EQ|INE002A01018,RELIANCE
NSE_EQ|INE009A01021,INFY
NSE_EQ|INE467B01029,ASIANPAINT
NSE_EQ|INE040A01034,HDFCBANK
NSE_EQ|INE005A01025,WIPRO"""
        
        try:
            df = pd.read_csv(StringIO(csv_content))
            return [(row['instrument_key'], row['tradingsymbol']) for _, row in df.iterrows()]
        except Exception as e:
            logger.error(f"Error loading instruments: {e}")
            return []
    
    def scan_stocks(self, stock_list: List[Tuple[str, str]], max_workers: int = 10) -> List[StockSignal]:
        """Scan stocks using threading"""
        results = []
        total = len(stock_list)
        
        # Streamlit progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.analyzer.analyze_stock, stock): stock[1] 
                for stock in stock_list
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        st.success(f"✅ Signal found: {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    st.error(f"❌ Error with {symbol}: {str(e)}")
                
                # Update progress
                progress = (i + 1) / total
                progress_bar.progress(progress)
                status_text.text(f"Processed: {i + 1}/{total} | Signals: {len(results)}")
        
        return results
    
    def format_telegram_message(self, signals: List[StockSignal]) -> str:
        """Format signals for Telegram"""
        if not signals:
            return "📈 *Stock Screener Results*\n\nNo signals found in this scan."
        
        header = f"📈 *{len(signals)} Entry Signals Found:*\n\n"
        
        signal_lines = []
        for i, signal in enumerate(signals, 1):
            line = (f"{i}. *{signal.stock_name}*\n"
                   f"Entry: ₹{signal.entry_price:.2f} | "
                   f"SL: ₹{signal.stop_loss:.2f} | "
                   f"TP: ₹{signal.take_profit:.2f}\n"
                   f"RSI: {signal.rsi:.1f} | Vol: {signal.volume_ratio:.1f}x\n")
            signal_lines.append(line)
        
        timestamp = f"\n🕐 Scan completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return header + "\n".join(signal_lines) + timestamp
    
    def send_to_telegram(self, signals: List[StockSignal]) -> bool:
        """Send results to Telegram"""
        if not self.telegram_bot:
            logger.warning("Telegram bot not configured")
            return False
            
        message = self.format_telegram_message(signals)
        return self.telegram_bot.send_message(message)
    
    def cleanup(self):
        """Cleanup resources"""
        self.upstox_client.close_session()
        if self.telegram_bot:
            self.telegram_bot.session.close()

def filter_signals(signals: List[StockSignal], max_rsi: float = None, min_volume_ratio: float = None) -> List[StockSignal]:
    """Filter signals based on criteria"""
    filtered = signals
    
    if max_rsi is not None:
        filtered = [s for s in filtered if s.rsi <= max_rsi]
    
    if min_volume_ratio is not None:
        filtered = [s for s in filtered if s.volume_ratio >= min_volume_ratio]
    
    return filtered

def signals_to_dataframe(signals: List[StockSignal]) -> pd.DataFrame:
    """Convert signals to DataFrame for display"""
    if not signals:
        return pd.DataFrame()
    
    data = []
    for signal in signals:
        data.append({
            'Stock': signal.stock_name,
            'Entry Price': f"₹{signal.entry_price:.2f}",
            'Stop Loss': f"₹{signal.stop_loss:.2f}",
            'Take Profit': f"₹{signal.take_profit:.2f}",
            'RSI': f"{signal.rsi:.1f}",
            'Volume Ratio': f"{signal.volume_ratio:.1f}x",
            'Risk-Reward': f"1:{((signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss)):.1f}",
            'Timestamp': signal.timestamp.strftime('%H:%M:%S')
        })
    
    return pd.DataFrame(data)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Advanced Stock Screener", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Advanced Nifty 500 Stock Screener")
    st.markdown("*Real-time technical analysis with Telegram integration*")
    
    # Show deployment info
    st.info("🌟 **Deployed on Streamlit Cloud** | Configure your API tokens in the sidebar")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Configuration
        access_token = st.text_input(
            "🔑 Upstox Access Token", 
            type="password",
            help="Your Upstox API access token"
        )
        
        # Telegram Configuration
        with st.expander("📱 Telegram Settings", expanded=False):
            telegram_token = st.text_input(
                "Bot Token", 
                type="password",
                help="Your Telegram bot token"
            )
            telegram_chat_id = st.text_input(
                "Chat ID", 
                help="Your Telegram chat ID"
            )
        
        # Scanning Parameters
        st.subheader("⚙️ Scan Parameters")
        max_workers = st.slider("Max Threads", 5, 20, 10, help="Number of parallel threads")
        
        # Post-scan Filters
        st.subheader("🔍 Result Filters")
        max_rsi_filter = st.slider("Max RSI", 30, 70, 70, help="Filter signals with RSI below this value")
        min_volume_filter = st.slider("Min Volume Ratio", 1.0, 3.0, 1.3, help="Filter signals with volume ratio above this")
    
    # File upload section
    st.subheader("📁 Instrument Data")
    uploaded_file = st.file_uploader(
        "Upload your instrument CSV file",
        type=['csv'],
        help="CSV should have columns: instrument_key, tradingsymbol"
    )
    
    csv_content = None
    if uploaded_file is not None:
        csv_content = uploaded_file.getvalue().decode('utf-8')
        st.success(f"✅ Loaded custom instrument file: {uploaded_file.name}")
    else:
        st.info("📝 Using default sample instruments (15 stocks)")
    
    # Main scanning section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scan_button = st.button("🚀 Start Stock Screening", type="primary", use_container_width=True)
    
    with col2:
        if st.button("📋 Download Template CSV"):
            template_csv = "instrument_key,tradingsymbol\nNSE_EQ|INE585B01010,MARUTI\nNSE_EQ|INE139A01034,NATIONALUM"
            st.download_button(
                label="📥 Download CSV Template",
                data=template_csv,
                file_name="instrument_template.csv",
                mime="text/csv"
            )
    
    # Main scanning logic
    if scan_button:
        if not access_token:
            st.error("🚫 Please provide Upstox Access Token")
            return
        
        scan_start_time = datetime.now()
        st.info(f"🕐 Scan started at: {scan_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize screener
        screener = StockScreener(
            access_token=access_token,
            telegram_bot_token=telegram_token if telegram_token else None,
            telegram_chat_id=telegram_chat_id if telegram_chat_id else None
        )
        
        try:
            # Load instruments
            stock_list = screener.load_instruments(csv_content)
            if not stock_list:
                st.error("🚫 No instruments loaded. Please check your CSV file.")
                return
            
            st.write(f"🔍 Scanning {len(stock_list)} stocks...")
            
            # Perform scan
            with st.spinner("Analyzing stocks..."):
                signals = screener.scan_stocks(stock_list, max_workers)
            
            # Apply filters
            filtered_signals = filter_signals(
                signals, 
                max_rsi=max_rsi_filter,
                min_volume_ratio=min_volume_filter
            )
            
            scan_duration = (datetime.now() - scan_start_time).total_seconds()
            
            # Display results
            st.success(f"✅ Scan completed in {scan_duration:.1f} seconds!")
            st.write(f"📊 Found {len(signals)} total signals, {len(filtered_signals)} after filtering")
            
            if filtered_signals:
                # Display results table
                df_display = signals_to_dataframe(filtered_signals)
                st.dataframe(df_display, use_container_width=True)
                
                # Download button for results
                csv_results = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_results,
                    file_name=f"stock_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Send to Telegram
                if telegram_token and telegram_chat_id:
                    with st.spinner("Sending to Telegram..."):
                        if screener.send_to_telegram(filtered_signals):
                            st.success("📩 Results sent to Telegram successfully!")
                        else:
                            st.warning("⚠️ Failed to send to Telegram")
                
            else:
                st.warning("⚠️ No signals found matching your criteria")
                
        except Exception as e:
            logger.error(f"Scanning error: {e}")
            st.error(f"🚫 Scanning failed: {str(e)}")
        
        finally:
            screener.cleanup()

if __name__ == "__main__":
    main()
