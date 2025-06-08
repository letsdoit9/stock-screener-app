# 📈 Advanced Stock Screener

A powerful stock screening application with technical analysis and Telegram integration, built with Streamlit and deployed on Streamlit Cloud.

## 🚀 Features

- **Technical Analysis**: RSI, EMA, ATR, VWAP calculations
- **Multi-threaded Scanning**: Fast parallel processing
- **Telegram Integration**: Automated signal alerts
- **Custom CSV Upload**: Use your own stock instruments
- **Real-time Progress**: Live scanning updates
- **Advanced Filters**: Post-scan filtering options
- **Results Export**: Download scan results as CSV

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Requests**: HTTP library for API calls
- **Upstox API**: Stock market data provider

## 📊 Technical Indicators

- **ATR (Average True Range)**: Volatility measurement
- **RSI (Relative Strength Index)**: Momentum oscillator
- **EMA (Exponential Moving Average)**: Trend indicator
- **VWAP (Volume Weighted Average Price)**: Price benchmark
- **Volume Analysis**: Trading volume patterns

## 🔧 Configuration

### Required Inputs:
1. **Upstox Access Token**: Get from Upstox Developer Console
2. **Telegram Bot Token** (Optional): Create via @BotFather
3. **Telegram Chat ID** (Optional): Get from @userinfobot

### Strategy Parameters:
- RSI Range: 30-70
- Volume Ratio: >1.3x
- Breakout Strength: >0.3 ATR
- Stop Loss: 1.8x ATR
- Take Profit: 2.5x ATR

## 📱 Usage

1. Enter your Upstox Access Token
2. (Optional) Configure Telegram settings
3. Upload custom instrument CSV or use defaults
4. Adjust scan parameters and filters
5. Click "Start Stock Screening"
6. View results and download CSV

## 📋 CSV Format

```csv
instrument_key,tradingsymbol
NSE_EQ|INE585B01010,MARUTI
NSE_EQ|INE139A01034,NATIONALUM
```

## 🔒 Security

- All sensitive tokens are input securely
- No data is stored permanently
- HTTPS encryption for all API calls

## 📞 Support

For issues or questions, please create an issue in this repository.

## ⚖️ Disclaimer

This tool is for educational purposes only. Not financial advice. Trade at your own risk.

---

**Deployed on Streamlit Cloud** 🌟
