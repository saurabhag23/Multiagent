import os
from langchain_openai import OpenAI as LangChainOpenAI
from sec_api import QueryApi
import yfinance as yf
import pandas as pd
from typing import Dict, Any
import logging
import io
import base64
import matplotlib.pyplot as plt
from agents.data_retrieval_agent import DataRetrievalAgent
from agents.serialization_utils import serialize_result
import json
class FundamentalAgent:
    def __init__(self):
        self.llm = LangChainOpenAI(temperature=0.7)
        self.sec_api = QueryApi(api_key=os.getenv('SEC_API_KEY'))
        logging.basicConfig(level=logging.INFO)

    
    def create_visualizations(self, financial_data: Dict[str, Any], ratios: Dict[str, float]) -> Dict[str, str]:
        """
        Create visualizations of key financial metrics and trends.
        
        Args:
            financial_data (Dict[str, Any]): The financial data dictionary.
            ratios (Dict[str, float]): The calculated financial ratios.
        
        Returns:
            Dict[str, str]: A dictionary of base64 encoded images of the visualizations.
        """
        visualizations = {}
        
        # Revenue and Net Income Trend
        visualizations['revenue_income_trend'] = self.plot_revenue_income_trend(financial_data)
        
        # Financial Ratios Comparison
        visualizations['financial_ratios'] = self.plot_financial_ratios(ratios)
        
        # Balance Sheet Composition
        visualizations['balance_sheet_composition'] = self.plot_balance_sheet_composition(financial_data)
        
        # Cash Flow Components
        visualizations['cash_flow_components'] = self.plot_cash_flow_components(financial_data)

        return visualizations

    def plot_revenue_income_trend(self, financial_data: Dict[str, Any]) -> str:
        """Plot revenue and net income trend."""
        income_statement = financial_data.get('income_statement', {})
        
        if not income_statement:
            return ""
        
        plt.figure(figsize=(10, 6))
        
        revenues = income_statement.get('Revenues', [])
        net_income = income_statement.get('NetIncomeLoss', [])
        
        if revenues and net_income:
            plt.plot(range(len(revenues)), [float(v.replace(',', '').replace('$', '')) for v in revenues], label='Revenue')
            plt.plot(range(len(net_income)), [float(v.replace(',', '').replace('$', '')) for v in net_income], label='Net Income')
            plt.title('Revenue and Net Income Trend')
            plt.xlabel('Fiscal Period')
            plt.ylabel('Amount ($)')
            plt.legend()
        
        return self._fig_to_base64(plt)

    def plot_financial_ratios(self, ratios: Dict[str, float]) -> str:
        """Plot financial ratios comparison."""
        plt.figure(figsize=(10, 6))
        
        if not ratios:
            return ""
        
        plt.bar(ratios.keys(), ratios.values())
        plt.title('Financial Ratios Comparison')
        plt.xlabel('Ratio')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        return self._fig_to_base64(plt)

    def plot_balance_sheet_composition(self, financial_data: Dict[str, Any]) -> str:
        """Plot balance sheet composition."""
        balance_sheet = financial_data.get('balance_sheet', {})
        
        if not balance_sheet:
            return ""
        
        assets = float(balance_sheet.get('Assets', '0').replace(',', '').replace('$', ''))
        liabilities = float(balance_sheet.get('Liabilities', '0').replace(',', '').replace('$', ''))
        equity = float(balance_sheet.get('StockholdersEquity', '0').replace(',', '').replace('$', ''))
        
        plt.figure(figsize=(10, 6))
        
        plt.pie([assets, liabilities, equity], labels=['Assets', 'Liabilities', 'Equity'], autopct='%1.1f%%')
        plt.title('Balance Sheet Composition')
        
        return self._fig_to_base64(plt)

    def plot_cash_flow_components(self, financial_data: Dict[str, Any]) -> str:
        """Plot cash flow components."""
        cash_flow = financial_data.get('cash_flow_statement', {})
        
        if not cash_flow:
            return ""
        
        operating = float(cash_flow.get('NetCashProvidedByUsedInOperatingActivities', '0').replace(',', '').replace('$', ''))
        investing = float(cash_flow.get('NetCashProvidedByUsedInInvestingActivities', '0').replace(',', '').replace('$', ''))
        financing = float(cash_flow.get('NetCashProvidedByUsedInFinancingActivities', '0').replace(',', '').replace('$', ''))
        
        plt.figure(figsize=(10, 6))
        
        plt.bar(['Operating', 'Investing', 'Financing'], [operating, investing, financing])
        plt.title('Cash Flow Components')
        plt.ylabel('Amount ($)')
        
        return self._fig_to_base64(plt)

    def _fig_to_base64(self, fig) -> str:
        """Convert a matplotlib figure to a base64 encoded string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)
        return img_str

    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Perform a comprehensive fundamental analysis on the given ticker.
        
        Args:
            ticker (str): The stock ticker symbol to analyze.
        
        Returns:
            Dict[str, Any]: A dictionary containing the analysis results and visualizations.
        """
        try:
            # Retrieve financial data
            financial_data = self.get_financial_data(ticker)
            
            # Calculate financial ratios
            ratios = self.calculate_financial_ratios(financial_data)
            
            # Create visualizations
            visualizations = self.create_visualizations(financial_data, ratios)
            
            # Analyze the data
            analysis = self.interpret_financial_data(financial_data, ratios)
            
            return serialize_result({
                "financial_data": financial_data,
                "ratios": ratios,
                "analysis": analysis,
                "visualizations": visualizations
            })
        except Exception as e:
            logging.error(f"Error analyzing {ticker}: {str(e)}")
            return {"error": str(e)}

    def get_financial_data(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieve financial data from SEC EDGAR and Yahoo Finance.
        
        Args:
            ticker (str): The stock ticker symbol.
        
        Returns:
            Dict[str, Any]: A dictionary containing financial data from various sources.
        """
        try:
            # Use DataRetrievalAgent to get financial data
            data_retrieval_agent = DataRetrievalAgent()
            return data_retrieval_agent.get_all_data(ticker)
        except Exception as e:
            logging.error(f"Error fetching financial data for {ticker}: {str(e)}")
            raise

    def calculate_financial_ratios(self, financial_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate key financial ratios based on the retrieved financial data.
        
        Args:
            financial_data (Dict[str, Any]): The financial data dictionary.
        
        Returns:
            Dict[str, float]: A dictionary of calculated financial ratios.
        """
        try:
            yf_data = financial_data['company_profile']
            fin_statements = financial_data['financial_statements']
            
            ratios = {
                'P/E Ratio': yf_data.get('trailingPE', 'N/A'),
                'P/B Ratio': yf_data.get('priceToBook', 'N/A'),
                'Debt to Equity': yf_data.get('debtToEquity', 'N/A'),
                'Return on Equity': yf_data.get('returnOnEquity', 'N/A'),
                'Profit Margin': yf_data.get('profitMargins', 'N/A'),
                'Current Ratio': self.calculate_current_ratio(fin_statements['balance_sheet']),
                'Quick Ratio': self.calculate_quick_ratio(fin_statements['balance_sheet']),
            }
            
            return ratios
        except Exception as e:
            logging.error(f"Error calculating financial ratios: {str(e)}")
            raise

    def calculate_current_ratio(self, balance_sheet: Dict[str, str]) -> float:
        """Calculate the current ratio."""
        current_assets = float(balance_sheet.get('CurrentAssets', '0').replace(',', '').replace('$', ''))
        current_liabilities = float(balance_sheet.get('CurrentLiabilities', '0').replace(',', '').replace('$', ''))
        return current_assets / current_liabilities if current_liabilities != 0 else None

    def calculate_quick_ratio(self, balance_sheet: Dict[str, str]) -> float:
        """Calculate the quick ratio."""
        current_assets = float(balance_sheet.get('CurrentAssets', '0').replace(',', '').replace('$', ''))
        inventory = float(balance_sheet.get('InventoryNet', '0').replace(',', '').replace('$', ''))
        current_liabilities = float(balance_sheet.get('CurrentLiabilities', '0').replace(',', '').replace('$', ''))
        return (current_assets - inventory) / current_liabilities if current_liabilities != 0 else None

    def interpret_financial_data(self, financial_data: Dict[str, Any], ratios: Dict[str, float]) -> str:
        """
        Interpret the financial data and ratios using the LLM.
        
        Args:
            financial_data (Dict[str, Any]): The financial data dictionary.
            ratios (Dict[str, float]): The calculated financial ratios.
        
        Returns:
            str: A comprehensive analysis of the company's financial health.
        """
        try:
            prompt = f"""
            Analyze the following financial data and ratios for a company:

            Financial Statements Summary:
            Income Statement: {financial_data['financial_statements']['income_statement']}
            Balance Sheet: {financial_data['financial_statements']['balance_sheet']}
            Cash Flow Statement: {financial_data['financial_statements']['cash_flow_statement']}

            Financial Ratios:
            {ratios}

            Please provide a comprehensive fundamental analysis based on this information. 
            Include insights on the company's:
            1. Financial health and stability
            2. Profitability and efficiency
            3. Growth prospects
            4. Potential risks and red flags
            5. Comparison to industry averages (if possible)
            6. Overall investment attractiveness

            Your analysis should be detailed, nuanced, and provide actionable insights for potential investors.
            """

            analysis = self.llm(prompt)

            return analysis
        except Exception as e:
            logging.error(f"Error interpreting financial data: {str(e)}")
            raise