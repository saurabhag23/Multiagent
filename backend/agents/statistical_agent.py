import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

class StatisticalAgent:
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis on the provided data.
        
        Args:
            data (Dict[str, Any]): The data to analyze, including stock prices and financial statements.
        
        Returns:
            Dict[str, Any]: Results of various statistical analyses.
        """
        stock_data = data['stock_data']
        financial_statements = data['financial_statements']

        results = {
            "stock_price_analysis": self.analyze_stock_prices(stock_data),
            "financial_statement_analysis": self.analyze_financial_statements(financial_statements),
            "correlation_analysis": self.perform_correlation_analysis(stock_data),
            "time_series_analysis": self.perform_time_series_analysis(stock_data['Close']),
            "regression_analysis": self.perform_regression_analysis(stock_data)
        }

        return results

    def analyze_stock_prices(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stock price data."""
        returns = stock_data['Close'].pct_change().dropna()
        
        return {
            "mean_return": returns.mean(),
            "std_dev_return": returns.std(),
            "annualized_volatility": returns.std() * np.sqrt(252),  # Assuming 252 trading days in a year
            "skewness": stats.skew(returns),
            "kurtosis": stats.kurtosis(returns),
            "jarque_bera_test": stats.jarque_bera(returns)
        }

    def analyze_financial_statements(self, financial_statements: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze financial statement data."""
        income_statement = financial_statements['income_statement']
        balance_sheet = financial_statements['balance_sheet']
        
        # Calculate year-over-year growth rates for key metrics
        revenue_growth = self.calculate_growth_rate(income_statement, 'Revenue')
        net_income_growth = self.calculate_growth_rate(income_statement, 'Net Income')
        asset_growth = self.calculate_growth_rate(balance_sheet, 'Total Assets')
        
        return {
            "revenue_growth": revenue_growth,
            "net_income_growth": net_income_growth,
            "asset_growth": asset_growth
        }

    def calculate_growth_rate(self, df: pd.DataFrame, metric: str) -> float:
        """Calculate the year-over-year growth rate for a given metric."""
        if metric not in df.index:
            return None
        values = df.loc[metric]
        growth_rate = (values.iloc[-1] - values.iloc[0]) / values.iloc[0]
        return growth_rate

    def perform_correlation_analysis(self, stock_data: pd.DataFrame) -> Dict[str, float]:
        """Perform correlation analysis on stock data."""
        correlation_matrix = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        return {
            "price_volume_correlation": correlation_matrix.loc['Close', 'Volume'],
            "high_low_correlation": correlation_matrix.loc['High', 'Low']
        }

    def perform_time_series_analysis(self, price_series: pd.Series) -> Dict[str, Any]:
        """Perform time series analysis on stock prices."""
        # Decompose the time series
        decomposition = seasonal_decompose(price_series, model='additive', period=252)  # Assuming 252 trading days in a year
        
        # Perform Augmented Dickey-Fuller test for stationarity
        adf_result = adfuller(price_series)
        
        return {
            "trend": decomposition.trend.iloc[-1],
            "seasonality": decomposition.seasonal.iloc[-1],
            "adf_statistic": adf_result[0],
            "adf_p_value": adf_result[1]
        }

    def perform_regression_analysis(self, stock_data: pd.DataFrame) -> Dict[str, float]:
        """Perform regression analysis to predict stock prices."""
        X = stock_data[['Open', 'High', 'Low', 'Volume']]
        y = stock_data['Close']
        
        model = LinearRegression()
        model.fit(X, y)
        
        return {
            "r_squared": model.score(X, y),
            "coefficients": dict(zip(['Open', 'High', 'Low', 'Volume'], model.coef_))
        }

    def interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret the statistical analysis results.
        
        Args:
            results (Dict[str, Any]): The results of the statistical analyses.
        
        Returns:
            str: An interpretation of the results.
        """
        interpretation = f"""
        Statistical Analysis Interpretation:

        1. Stock Price Analysis:
        - The mean daily return is {results['stock_price_analysis']['mean_return']:.4f}.
        - The annualized volatility is {results['stock_price_analysis']['annualized_volatility']:.4f}.
        - The returns distribution has a skewness of {results['stock_price_analysis']['skewness']:.4f} and kurtosis of {results['stock_price_analysis']['kurtosis']:.4f}.
        
        2. Financial Statement Analysis:
        - The revenue growth rate is {results['financial_statement_analysis']['revenue_growth']:.2%}.
        - The net income growth rate is {results['financial_statement_analysis']['net_income_growth']:.2%}.
        - The asset growth rate is {results['financial_statement_analysis']['asset_growth']:.2%}.
        
        3. Correlation Analysis:
        - The correlation between price and volume is {results['correlation_analysis']['price_volume_correlation']:.4f}.
        - The correlation between high and low prices is {results['correlation_analysis']['high_low_correlation']:.4f}.
        
        4. Time Series Analysis:
        - The current trend component is {results['time_series_analysis']['trend']:.4f}.
        - The ADF test statistic is {results['time_series_analysis']['adf_statistic']:.4f} with a p-value of {results['time_series_analysis']['adf_p_value']:.4f}.
        
        5. Regression Analysis:
        - The R-squared value of the price prediction model is {results['regression_analysis']['r_squared']:.4f}.
        
        Overall, these statistics provide insights into the stock's performance, volatility, growth, and predictability.
        """
        return interpretation