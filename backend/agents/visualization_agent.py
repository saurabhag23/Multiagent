import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any
import io
import base64

class VisualizationAgent:
    def create_visualizations(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Create visualizations based on the provided data.
        
        Args:
            data (Dict[str, Any]): The data to visualize.
        
        Returns:
            Dict[str, str]: A dictionary of base64 encoded images of the visualizations.
        """
        visualizations = {}
        
        # Stock price history
        visualizations['stock_price_history'] = self.plot_stock_price_history(data['stock_data'])
        
        # Volume history
        visualizations['volume_history'] = self.plot_volume_history(data['stock_data'])
        
        # Financial ratios
        visualizations['financial_ratios'] = self.plot_financial_ratios(data['company_profile'])
        
        return visualizations

    def plot_stock_price_history(self, stock_data: pd.DataFrame) -> str:
        """Plot stock price history."""
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data['Close'])
        plt.title('Stock Price History')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        return self._fig_to_base64(plt)

    def plot_volume_history(self, stock_data: pd.DataFrame) -> str:
        """Plot volume history."""
        plt.figure(figsize=(12, 6))
        plt.bar(stock_data.index, stock_data['Volume'])
        plt.title('Trading Volume History')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        return self._fig_to_base64(plt)

    def plot_financial_ratios(self, company_profile: Dict[str, Any]) -> str:
        """Plot key financial ratios."""
        ratios = {
            'P/E Ratio': company_profile.get('trailingPE', 0),
            'P/B Ratio': company_profile.get('priceToBook', 0),
            'Debt to Equity': company_profile.get('debtToEquity', 0) / 100 if company_profile.get('debtToEquity') else 0,
            'Return on Equity': company_profile.get('returnOnEquity', 0),
            'Profit Margin': company_profile.get('profitMargins', 0)
        }
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(ratios.keys()), y=list(ratios.values()))
        plt.title('Key Financial Ratios')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return self._fig_to_base64(plt)

    def _fig_to_base64(self, fig):
        """Convert a matplotlib figure to a base64 encoded string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)
        return img_str