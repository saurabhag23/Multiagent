import os
from crewai import Agent
from langchain_openai import OpenAI as LangChainOpenAI
from sec_api import QueryApi, RenderApi
import yfinance as yf
import pandas as pd
from typing import Dict, Any
import logging
import io
import base64
import matplotlib.pyplot as plt

class FundamentalAgent:
    def __init__(self):
        self.llm = LangChainOpenAI(temperature=0.7)
        self.sec_api = QueryApi(api_key=os.getenv('SEC_API_KEY'))
        self.sec_render_api = RenderApi(api_key=os.getenv('SEC_API_KEY'))
        logging.basicConfig(level=logging.INFO)

    def get_agent(self):
        return Agent(
            role='Fundamental Analyst',
            goal='Perform thorough fundamental analysis on companies',
            backstory='You are an expert in analyzing financial statements and company fundamentals',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

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
        income_statement = financial_data['financial_statements']['income_statement']
        
        plt.figure(figsize=(10, 6))
        plt.plot(income_statement['fiscalPeriod'], income_statement.loc[income_statement['index'] == 'Revenues', 'value'], label='Revenue')
        plt.plot(income_statement['fiscalPeriod'], income_statement.loc[income_statement['index'] == 'NetIncomeLoss', 'value'], label='Net Income')
        plt.title('Revenue and Net Income Trend')
        plt.xlabel('Fiscal Period')
        plt.ylabel('Amount ($)')
        plt.legend()
        
        return self._fig_to_base64(plt)

    def plot_financial_ratios(self, ratios: Dict[str, float]) -> str:
        """Plot financial ratios comparison."""
        plt.figure(figsize=(10, 6))
        plt.bar(ratios.keys(), ratios.values())
        plt.title('Financial Ratios Comparison')
        plt.xlabel('Ratio')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        return self._fig_to_base64(plt)

    def plot_balance_sheet_composition(self, financial_data: Dict[str, Any]) -> str:
        """Plot balance sheet composition."""
        balance_sheet = financial_data['financial_statements']['balance_sheet']
        
        assets = balance_sheet.loc[balance_sheet['index'] == 'Assets', 'value'].iloc[0]
        liabilities = balance_sheet.loc[balance_sheet['index'] == 'Liabilities', 'value'].iloc[0]
        equity = balance_sheet.loc[balance_sheet['index'] == 'StockholdersEquity', 'value'].iloc[0]
        
        plt.figure(figsize=(10, 6))
        plt.pie([assets, liabilities, equity], labels=['Assets', 'Liabilities', 'Equity'], autopct='%1.1f%%')
        plt.title('Balance Sheet Composition')
        
        return self._fig_to_base64(plt)

    def plot_cash_flow_components(self, financial_data: Dict[str, Any]) -> str:
        """Plot cash flow components."""
        cash_flow = financial_data['financial_statements']['cash_flow']
        
        operating = cash_flow.loc[cash_flow['index'] == 'NetCashProvidedByUsedInOperatingActivities', 'value'].iloc[0]
        investing = cash_flow.loc[cash_flow['index'] == 'NetCashProvidedByUsedInInvestingActivities', 'value'].iloc[0]
        financing = cash_flow.loc[cash_flow['index'] == 'NetCashProvidedByUsedInFinancingActivities', 'value'].iloc[0]
        
        plt.figure(figsize=(10, 6))
        plt.bar(['Operating', 'Investing', 'Financing'], [operating, investing, financing])
        plt.title('Cash Flow Components')
        plt.ylabel('Amount ($)')
        
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
            
            return {
                "financial_data": financial_data,
                "ratios": ratios,
                "analysis": analysis,
                "visualizations": visualizations
            }
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
            # Fetch latest 10-K filing
            query = {
                "query": {"query_string": {
                    "query": f"ticker:{ticker} AND formType:\"10-K\""
                }},
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            filings = self.sec_api.get_filings(query)
            
            if filings['total']['value'] == 0:
                raise ValueError(f"No 10-K filings found for {ticker}")

            latest_10k = filings['filings'][0]
            
            # Extract detailed financial statements
            financial_statements = self.extract_financial_statements(latest_10k['accessionNo'])

            # Use yfinance to get additional data
            yf_data = yf.Ticker(ticker).info
            
            return {
                'sec_data': latest_10k,
                'financial_statements': financial_statements,
                'yf_data': yf_data
            }
        except Exception as e:
            logging.error(f"Error fetching financial data for {ticker}: {str(e)}")
            raise

    def extract_financial_statements(self, accession_no: str) -> Dict[str, pd.DataFrame]:
        """
        Extract detailed financial statements from the 10-K filing.
        
        Args:
            accession_no (str): The SEC accession number of the filing.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing financial statements as DataFrames.
        """
        try:
            # Use SEC Render API to get XBRL data
            xbrl_json = self.sec_render_api.xbrl_to_json(accession_no)
            
            # Extract Income Statement, Balance Sheet, and Cash Flow Statement
            income_statement = pd.DataFrame(xbrl_json['IncomeStatement'])
            balance_sheet = pd.DataFrame(xbrl_json['BalanceSheet'])
            cash_flow = pd.DataFrame(xbrl_json['CashFlow'])
            
            return {
                'income_statement': income_statement,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
        except Exception as e:
            logging.error(f"Error extracting financial statements: {str(e)}")
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
            yf_data = financial_data['yf_data']
            fin_statements = financial_data['financial_statements']
            
            ratios = {
                'P/E Ratio': yf_data.get('trailingPE', 'N/A'),
                'P/B Ratio': yf_data.get('priceToBook', 'N/A'),
                'Debt to Equity': yf_data.get('debtToEquity', 'N/A'),
                'Return on Equity': yf_data.get('returnOnEquity', 'N/A'),
                'Profit Margin': yf_data.get('profitMargins', 'N/A'),
                'Current Ratio': self.calculate_current_ratio(fin_statements['balance_sheet']),
                'Quick Ratio': self.calculate_quick_ratio(fin_statements['balance_sheet']),
                'Inventory Turnover': self.calculate_inventory_turnover(fin_statements['income_statement'], fin_statements['balance_sheet']),
                'Asset Turnover': self.calculate_asset_turnover(fin_statements['income_statement'], fin_statements['balance_sheet']),
            }
            
            return ratios
        except Exception as e:
            logging.error(f"Error calculating financial ratios: {str(e)}")
            raise

    def calculate_current_ratio(self, balance_sheet: pd.DataFrame) -> float:
        """Calculate the current ratio."""
        current_assets = balance_sheet.loc[balance_sheet['index'] == 'CurrentAssets', 'value'].iloc[0]
        current_liabilities = balance_sheet.loc[balance_sheet['index'] == 'CurrentLiabilities', 'value'].iloc[0]
        return current_assets / current_liabilities if current_liabilities != 0 else 'N/A'

    def calculate_quick_ratio(self, balance_sheet: pd.DataFrame) -> float:
        """Calculate the quick ratio."""
        current_assets = balance_sheet.loc[balance_sheet['index'] == 'CurrentAssets', 'value'].iloc[0]
        inventory = balance_sheet.loc[balance_sheet['index'] == 'InventoryNet', 'value'].iloc[0]
        current_liabilities = balance_sheet.loc[balance_sheet['index'] == 'CurrentLiabilities', 'value'].iloc[0]
        return (current_assets - inventory) / current_liabilities if current_liabilities != 0 else 'N/A'

    def calculate_inventory_turnover(self, income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
        """Calculate the inventory turnover ratio."""
        cogs = income_statement.loc[income_statement['index'] == 'CostOfRevenue', 'value'].iloc[0]
        avg_inventory = balance_sheet.loc[balance_sheet['index'] == 'InventoryNet', 'value'].mean()
        return cogs / avg_inventory if avg_inventory != 0 else 'N/A'

    def calculate_asset_turnover(self, income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
        """Calculate the asset turnover ratio."""
        revenue = income_statement.loc[income_statement['index'] == 'Revenues', 'value'].iloc[0]
        avg_total_assets = balance_sheet.loc[balance_sheet['index'] == 'Assets', 'value'].mean()
        return revenue / avg_total_assets if avg_total_assets != 0 else 'N/A'

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
            # Prepare the prompt for the LLM
            prompt = f"""
            Analyze the following financial data and ratios for a company:

            Financial Statements Summary:
            Income Statement: {financial_data['financial_statements']['income_statement'].to_string()}
            Balance Sheet: {financial_data['financial_statements']['balance_sheet'].to_string()}
            Cash Flow Statement: {financial_data['financial_statements']['cash_flow'].to_string()}

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

            # Use the LLM to generate the analysis
            analysis = self.llm(prompt)

            return analysis
        except Exception as e:
            logging.error(f"Error interpreting financial data: {str(e)}")
            raise
