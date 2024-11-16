import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import io
import base64
from langchain_openai import OpenAI as LangChainOpenAI
from scipy.signal import find_peaks

class TechnicalAgent:
    def __init__(self):
        self.llm = LangChainOpenAI(temperature=0.7)

    def analyze(self, ticker: str) -> Dict[str, Any]:
        try:
            data = self.get_historical_data(ticker)
            indicators = self.calculate_indicators(data)
            patterns = self.detect_patterns(data)
            visualizations = self.create_visualizations(data, indicators, patterns)
            analysis = self.interpret_technical_data(data, indicators, patterns)
            
            return {
                "analysis": analysis,
                "visualizations": visualizations
            }
        except Exception as e:
            return {"error": str(e)}

    def get_historical_data(self, ticker: str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        return data

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
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
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        indicators['%K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        indicators['%D'] = indicators['%K'].rolling(window=3).mean()
        
        return indicators

    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        patterns = {}
        
        # Head and Shoulders pattern detection
        window = 60  # Adjust this value to change the detection window
        left_shoulder, head, right_shoulder = self.detect_head_and_shoulders(data['Close'], window)
        
        if left_shoulder and head and right_shoulder:
            patterns['Head_and_Shoulders'] = {
                'left_shoulder': left_shoulder,
                'head': head,
                'right_shoulder': right_shoulder
            }
        
        return patterns

    def detect_head_and_shoulders(self, prices: pd.Series, window: int) -> tuple:
        # Find peaks
        peaks, _ = find_peaks(prices, distance=window//3)
        
        if len(peaks) < 3:
            return None, None, None
        
        # Find the highest peak (head)
        head = peaks[np.argmax(prices[peaks])]
        
        # Find left and right shoulders
        left_peaks = peaks[peaks < head]
        right_peaks = peaks[peaks > head]
        
        if len(left_peaks) == 0 or len(right_peaks) == 0:
            return None, None, None
        
        left_shoulder = left_peaks[-1]
        right_shoulder = right_peaks[0]
        
        # Check if the pattern is valid
        if prices[left_shoulder] < prices[head] and prices[right_shoulder] < prices[head] and \
           abs(prices[left_shoulder] - prices[right_shoulder]) / prices[head] < 0.1:
            return left_shoulder, head, right_shoulder
        
        return None, None, None

    def create_visualizations(self, data: pd.DataFrame, indicators: Dict[str, pd.Series], patterns: Dict[str, Any]) -> Dict[str, str]:
        visualizations = {}
        
        # Price, Moving Averages, and Bollinger Bands
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Close Price')
        plt.plot(data.index, indicators['SMA20'], label='20-day SMA')
        plt.plot(data.index, indicators['SMA50'], label='50-day SMA')
        plt.plot(data.index, indicators['BB_Upper'], label='Upper BB', linestyle='--')
        plt.plot(data.index, indicators['BB_Middle'], label='Middle BB', linestyle='--')
        plt.plot(data.index, indicators['BB_Lower'], label='Lower BB', linestyle='--')
        plt.title('Price, Moving Averages, and Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        visualizations['price_ma_bb'] = self._fig_to_base64(plt)
        
        # RSI and Stochastic Oscillator
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(data.index, indicators['RSI'])
        ax1.set_title('Relative Strength Index (RSI)')
        ax1.set_ylabel('RSI')
        ax1.axhline(y=70, color='r', linestyle='--')
        ax1.axhline(y=30, color='g', linestyle='--')
        
        ax2.plot(data.index, indicators['%K'], label='%K')
        ax2.plot(data.index, indicators['%D'], label='%D')
        ax2.set_title('Stochastic Oscillator')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Stochastic')
        ax2.axhline(y=80, color='r', linestyle='--')
        ax2.axhline(y=20, color='g', linestyle='--')
        ax2.legend()
        visualizations['rsi_stochastic'] = self._fig_to_base64(plt)
        
        # MACD
        plt.figure(figsize=(12, 4))
        plt.plot(data.index, indicators['MACD'], label='MACD')
        plt.plot(data.index, indicators['Signal_Line'], label='Signal Line')
        plt.title('Moving Average Convergence Divergence (MACD)')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        visualizations['macd'] = self._fig_to_base64(plt)
        
        # Head and Shoulders pattern
        if 'Head_and_Shoulders' in patterns:
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['Close'])
            pattern = patterns['Head_and_Shoulders']
            plt.scatter(data.index[pattern['left_shoulder']], data['Close'][pattern['left_shoulder']], color='r', s=100, label='Left Shoulder')
            plt.scatter(data.index[pattern['head']], data['Close'][pattern['head']], color='g', s=100, label='Head')
            plt.scatter(data.index[pattern['right_shoulder']], data['Close'][pattern['right_shoulder']], color='b', s=100, label='Right Shoulder')
            plt.title('Head and Shoulders Pattern')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            visualizations['head_and_shoulders'] = self._fig_to_base64(plt)
        
        return visualizations

    def interpret_technical_data(self, data: pd.DataFrame, indicators: Dict[str, pd.Series], patterns: Dict[str, Any]) -> str:
        prompt = f"""
        Analyze the following technical indicators for a stock:

        1. Current Price: {data['Close'].iloc[-1]:.2f}
        2. 20-day SMA: {indicators['SMA20'].iloc[-1]:.2f}
        3. 50-day SMA: {indicators['SMA50'].iloc[-1]:.2f}
        4. Current RSI: {indicators['RSI'].iloc[-1]:.2f}
        5. Current MACD: {indicators['MACD'].iloc[-1]:.2f}
        6. Current MACD Signal Line: {indicators['Signal_Line'].iloc[-1]:.2f}
        7. Bollinger Bands:
           - Upper: {indicators['BB_Upper'].iloc[-1]:.2f}
           - Middle: {indicators['BB_Middle'].iloc[-1]:.2f}
           - Lower: {indicators['BB_Lower'].iloc[-1]:.2f}
        8. Stochastic Oscillator:
           - %K: {indicators['%K'].iloc[-1]:.2f}
           - %D: {indicators['%D'].iloc[-1]:.2f}
        9. Head and Shoulders Pattern: {'Detected' if 'Head_and_Shoulders' in patterns else 'Not Detected'}

        Please provide a comprehensive technical analysis based on this information. 
        Include insights on:
        1. The current trend (bullish, bearish, or neutral)
        2. Potential support and resistance levels
        3. Overbought or oversold conditions based on RSI and Stochastic Oscillator
        4. MACD signal (bullish or bearish crossover)
        5. Bollinger Bands squeeze or expansion
        6. Significance of the Head and Shoulders pattern (if detected)
        7. Overall technical outlook and potential trade setups

        Your analysis should be detailed, nuanced, and provide actionable insights for potential traders.
        """

        analysis = self.llm(prompt)
        return analysis

    def _fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)
        return img_str