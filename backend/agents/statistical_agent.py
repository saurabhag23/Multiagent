import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any
import logging
import json
from agents.serialization_utils import serialize_result

class StatisticalAgent:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis on the provided data.
        
        Args:
            data (Dict[str, Any]): The data to analyze, including stock_data and financial_statements.
        
        Returns:
            Dict[str, Any]: A dictionary containing the results of various statistical analyses.
        """
        try:
            stock_data = data.get('stock_data')
            financial_statements = data.get('financial_statements')

            results = {}

            if stock_data is not None and not stock_data.empty:
                results['stock_statistics'] = self.analyze_stock_data(stock_data)

            if financial_statements:
                results['financial_statistics'] = self.analyze_financial_statements(financial_statements)

            results['interpretation'] = self.interpret_results(results)

            return results

        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {str(e)}")
            return {"error": str(e)}

    def analyze_stock_data(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stock price data."""
        returns = stock_data['Close'].pct_change().dropna()

        analysis = {
            'mean_return': returns.mean(),
            'std_dev': returns.std(),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
            'var_95': self.calculate_var(returns, 0.95),
            'autocorrelation': returns.autocorr(),
        }

        return serialize_result(analysis)

    def analyze_financial_statements(self, financial_statements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial statement data."""
        income_statement = financial_statements.get('income_statement', {})
        balance_sheet = financial_statements.get('balance_sheet', {})

        analysis = {
            'revenue_growth': self.calculate_growth_rate(income_statement.get('Revenues', [])),
            'profit_margin': self.calculate_profit_margin(income_statement),
            'debt_to_equity': self.calculate_debt_to_equity(balance_sheet),
            'current_ratio': self.calculate_current_ratio(balance_sheet),
        }

        return serialize_result(analysis)

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate the Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk (VaR)."""
        return np.percentile(returns, 100 * (1 - confidence_level))

    def calculate_growth_rate(self, values: list) -> float:
        """Calculate compound annual growth rate."""
        if len(values) < 2:
            return None
        start_value = float(values[0].replace(',', '').replace('$', ''))
        end_value = float(values[-1].replace(',', '').replace('$', ''))
        num_periods = len(values) - 1
        return (end_value / start_value) ** (1 / num_periods) - 1

    def calculate_profit_margin(self, income_statement: Dict[str, str]) -> float:
        """Calculate profit margin."""
        net_income = float(income_statement.get('NetIncomeLoss', '0').replace(',', '').replace('$', ''))
        revenue = float(income_statement.get('Revenues', '0').replace(',', '').replace('$', ''))
        return net_income / revenue if revenue != 0 else None

    def calculate_debt_to_equity(self, balance_sheet: Dict[str, str]) -> float:
        """Calculate debt to equity ratio."""
        total_debt = float(balance_sheet.get('Liabilities', '0').replace(',', '').replace('$', ''))
        equity = float(balance_sheet.get('StockholdersEquity', '0').replace(',', '').replace('$', ''))
        return total_debt / equity if equity != 0 else None

    def calculate_current_ratio(self, balance_sheet: Dict[str, str]) -> float:
        """Calculate current ratio."""
        current_assets = float(balance_sheet.get('CurrentAssets', '0').replace(',', '').replace('$', ''))
        current_liabilities = float(balance_sheet.get('CurrentLiabilities', '0').replace(',', '').replace('$', ''))
        return current_assets / current_liabilities if current_liabilities != 0 else None

    def interpret_results(self, results: Dict[str, Any]) -> str:
        """Interpret the statistical analysis results."""
        interpretation = "Statistical Analysis Interpretation:\n\n"

        stock_stats = results.get('stock_statistics', {})
        if stock_stats:
            interpretation += "Stock Performance:\n"
            interpretation += f"- Mean Daily Return: {stock_stats['mean_return']:.4f}\n"
            interpretation += f"- Annualized Volatility: {stock_stats['volatility']:.4f}\n"
            interpretation += f"- Sharpe Ratio: {stock_stats['sharpe_ratio']:.4f}\n"
            interpretation += f"- Value at Risk (95%): {stock_stats['var_95']:.4f}\n"

            if stock_stats['skewness'] > 0:
                interpretation += "- The returns distribution is positively skewed, indicating more extreme positive returns.\n"
            else:
                interpretation += "- The returns distribution is negatively skewed, indicating more extreme negative returns.\n"

            if stock_stats['kurtosis'] > 3:
                interpretation += "- The returns distribution has heavy tails, suggesting more outlier events than a normal distribution.\n"

        fin_stats = results.get('financial_statistics', {})
        if fin_stats:
            interpretation += "\nFinancial Health:\n"
            if fin_stats['revenue_growth'] is not None:
                interpretation += f"- Revenue Growth Rate: {fin_stats['revenue_growth']:.2%}\n"
            if fin_stats['profit_margin'] is not None:
                interpretation += f"- Profit Margin: {fin_stats['profit_margin']:.2%}\n"
            if fin_stats['debt_to_equity'] is not None:
                interpretation += f"- Debt to Equity Ratio: {fin_stats['debt_to_equity']:.2f}\n"
            if fin_stats['current_ratio'] is not None:
                interpretation += f"- Current Ratio: {fin_stats['current_ratio']:.2f}\n"

        return interpretation
