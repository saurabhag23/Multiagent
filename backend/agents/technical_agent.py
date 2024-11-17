import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from scipy.signal import find_peaks
import io
import base64
import logging
from agents.serialization_utils import serialize_result
import json
class TechnicalAgent:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Perform a technical analysis on the given ticker.
        
        Args:
            ticker (str): The stock ticker symbol to analyze.
        
        Returns:
            Dict[str, Any]: A dictionary containing the analysis results and visualizations.
        """
        try:
            data = self.get_historical_data(ticker)
            indicators = self.calculate_indicators(data)
            patterns = self.detect_patterns(data)
            visualizations = self.create_visualizations(data, indicators)

            analysis_summary = self.interpret_technical_analysis(data, indicators, patterns)

            return serialize_result({
                "indicators": indicators,
                "patterns": patterns,
                "visualizations": visualizations,
                "analysis_summary": analysis_summary
            })

        except Exception as e:
            self.logger.error(f"Error in technical analysis for {ticker}: {str(e)}")
            return {"error": str(e)}

    def get_historical_data(self, ticker: str) -> pd.DataFrame:
        """Retrieve historical stock data from Yahoo Finance."""
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        return data

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators based on historical stock data."""
        indicators = {}
        
        # Simple Moving Average (SMA)
        indicators['SMA20'] = data['Close'].rolling(window=20).mean()
        indicators['SMA50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average (EMA)
        indicators['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        indicators['MACD'] = exp1 - exp2
        indicators['Signal_Line'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        indicators['BB_Middle'] = data['Close'].rolling(window=20).mean()
        indicators['BB_Upper'] = indicators['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
        indicators['BB_Lower'] = indicators['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
        
        return indicators

    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect technical patterns in the stock price."""
        patterns_found = {}
        
        # Detect Head and Shoulders pattern
        left_shoulder, head, right_shoulder = self.detect_head_and_shoulders(data['Close'])
        
        if left_shoulder is not None and head is not None and right_shoulder is not None:
            patterns_found["Head_and_Shoulders"] = {
                "left_shoulder": left_shoulder,
                "head": head,
                "right_shoulder": right_shoulder,
            }
        
        return patterns_found

    def detect_head_and_shoulders(self, prices: pd.Series) -> Tuple[int, int, int]:
        """Detect Head and Shoulders pattern in price series."""
        peaks, _ = find_peaks(prices)
        
        if len(peaks) < 3:
            return None, None, None
        
        head_index = peaks[np.argmax(prices[peaks])]
        
        left_peaks = peaks[peaks < head_index]
        right_peaks = peaks[peaks > head_index]
        
        if len(left_peaks) == 0 or len(right_peaks) == 0:
            return None, None, None
        
        left_shoulder_index = left_peaks[-1]
        right_shoulder_index = right_peaks[0]
        
        return left_shoulder_index, head_index, right_shoulder_index

    def create_visualizations(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, str]:
        """Create visualizations based on technical analysis."""
        visualizations = {}
        
        # Price with Moving Averages
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Close Price', color='blue')
        plt.plot(data.index, indicators['SMA20'], label='20-Day SMA', color='orange')
        plt.plot(data.index, indicators['SMA50'], label='50-Day SMA', color='green')
        plt.title('Stock Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        visualizations["price_moving_averages"] = self._fig_to_base64(plt)

        # RSI Plot
        plt.figure(figsize=(12, 4))
        plt.plot(data.index, indicators["RSI"], label='RSI', color='purple')
        plt.axhline(70, linestyle='--', color='red')
        plt.axhline(30, linestyle='--', color='green')
        plt.title('Relative Strength Index (RSI)')
        plt.xlabel('Date')
        plt.ylabel('RSI Value')
        plt.legend()
        visualizations["rsi_plot"] = self._fig_to_base64(plt)

        return visualizations

    def _fig_to_base64(self, fig) -> str:
        """Convert a matplotlib figure to a base64 encoded string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close('all')
        return img_str

    def interpret_technical_analysis(self, data: pd.DataFrame, indicators: Dict[str, pd.Series], patterns: Dict[str, Any]) -> str:
        """Interpret the technical analysis results."""
        latest_close = data['Close'].iloc[-1]
        latest_sma20 = indicators['SMA20'].iloc[-1]
        latest_sma50 = indicators['SMA50'].iloc[-1]
        latest_rsi = indicators['RSI'].iloc[-1]
        latest_macd = indicators['MACD'].iloc[-1]
        latest_signal = indicators['Signal_Line'].iloc[-1]

        analysis = f"Technical Analysis Summary:\n\n"

        # Trend Analysis
        if latest_close > latest_sma20 > latest_sma50:
            analysis += "The stock is in a strong uptrend. "
        elif latest_close < latest_sma20 < latest_sma50:
            analysis += "The stock is in a strong downtrend. "
        elif latest_close > latest_sma20 and latest_sma20 < latest_sma50:
            analysis += "The stock may be starting an uptrend. "
        elif latest_close < latest_sma20 and latest_sma20 > latest_sma50:
            analysis += "The stock may be starting a downtrend. "
        else:
            analysis += "The stock's trend is unclear. "

        # RSI Analysis
        if latest_rsi > 70:
            analysis += "The stock is currently overbought according to RSI. "
        elif latest_rsi < 30:
            analysis += "The stock is currently oversold according to RSI. "
        else:
            analysis += "The RSI indicates neutral momentum. "

        # MACD Analysis
        if latest_macd > latest_signal:
            analysis += "The MACD indicates bullish momentum. "
        else:
            analysis += "The MACD indicates bearish momentum. "

        # Pattern Analysis
        if "Head_and_Shoulders" in patterns:
            analysis += "A potential Head and Shoulders pattern has been detected, which could indicate a trend reversal. "

        return analysis