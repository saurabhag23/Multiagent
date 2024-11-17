import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from typing import Dict, Any
import logging
from agents.serialization_utils import serialize_result
import json
class VisualizationAgent:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    def create_visualizations(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Create visualizations based on the provided data.
        
        Args:
            data (Dict[str, Any]): The data to visualize.
        
        Returns:
            Dict[str, str]: A dictionary of base64 encoded images of the visualizations.
        """
        visualizations = {}
        
        try:
            if 'stock_data' in data and not data['stock_data'].empty:
                visualizations['stock_price_history'] = self.plot_stock_price_history(data['stock_data'])
                visualizations['volume_history'] = self.plot_volume_history(data['stock_data'])
            
            if 'financial_statements' in data:
                visualizations['financial_ratios'] = self.plot_financial_ratios(data['financial_statements'])
                visualizations['income_statement'] = self.plot_income_statement(data['financial_statements'])
                visualizations['balance_sheet'] = self.plot_balance_sheet(data['financial_statements'])
                visualizations['cash_flow'] = self.plot_cash_flow(data['financial_statements'])
            return serialize_result(visualizations)
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            
        

    def plot_stock_price_history(self, stock_data: pd.DataFrame) -> str:
        """Plot stock price history."""
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
        plt.title('Stock Price History')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        return self._fig_to_base64(plt)

    def plot_volume_history(self, stock_data: pd.DataFrame) -> str:
        """Plot volume history."""
        plt.figure(figsize=(12, 6))
        plt.bar(stock_data.index, stock_data['Volume'], alpha=0.7, color='orange')
        plt.title('Trading Volume History')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.grid(True)
        return self._fig_to_base64(plt)

    def plot_financial_ratios(self, financial_statements: Dict[str, Any]) -> str:
        """Plot key financial ratios."""
        ratios = {
            'P/E Ratio': financial_statements.get('P/E Ratio', 0),
            'P/B Ratio': financial_statements.get('P/B Ratio', 0),
            'Debt/Equity': financial_statements.get('Debt to Equity', 0),
            'ROE': financial_statements.get('Return on Equity', 0),
            'Profit Margin': financial_statements.get('Profit Margin', 0)
        }
        
        # Filter out None values
        ratios = {k: v for k, v in ratios.items() if v is not None}
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(ratios.keys(), ratios.values(), color='green')
        plt.title('Key Financial Ratios')
        plt.ylabel('Value')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        return self._fig_to_base64(plt)

    def plot_income_statement(self, financial_statements: Dict[str, Any]) -> str:
        """Plot income statement components."""
        income_statement = financial_statements.get('income_statement', {})
        
        if not income_statement:
            return ""

        plt.figure(figsize=(12, 6))
        plt.bar(income_statement.keys(), [float(v.replace(',', '').replace('$', '')) for v in income_statement.values()])
        plt.title('Income Statement Components')
        plt.ylabel('Amount ($)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return self._fig_to_base64(plt)

    def plot_balance_sheet(self, financial_statements: Dict[str, Any]) -> str:
        """Plot balance sheet components."""
        balance_sheet = financial_statements.get('balance_sheet', {})
        
        if not balance_sheet:
            return ""

        plt.figure(figsize=(12, 6))
        plt.bar(balance_sheet.keys(), [float(v.replace(',', '').replace('$', '')) for v in balance_sheet.values()])
        plt.title('Balance Sheet Components')
        plt.ylabel('Amount ($)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return self._fig_to_base64(plt)

    def plot_cash_flow(self, financial_statements: Dict[str, Any]) -> str:
        """Plot cash flow components."""
        cash_flow = financial_statements.get('cash_flow_statement', {})
        
        if not cash_flow:
            return ""

        plt.figure(figsize=(12, 6))
        plt.bar(cash_flow.keys(), [float(v.replace(',', '').replace('$', '')) for v in cash_flow.values()])
        plt.title('Cash Flow Components')
        plt.ylabel('Amount ($)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return self._fig_to_base64(plt)

    def _fig_to_base64(self, fig) -> str:
        """Convert a matplotlib figure to a base64 encoded string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close('all')
        return img_str