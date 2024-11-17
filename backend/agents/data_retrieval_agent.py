import yfinance as yf
from sec_api import QueryApi
import os
import logging
from typing import Dict, Any
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

class DataRetrievalAgent:
    def __init__(self):
        self.sec_api = QueryApi(api_key=os.getenv('SEC_API_KEY'))
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Retrieve historical stock data from Yahoo Finance."""
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data

    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """Retrieve financial statements from SEC filings."""
        query = {
            "query": {
                "query_string": {
                    "query": f"ticker:{ticker} AND formType:\"10-K\""
                }
            },
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}]
        }

        try:
            filings = self.sec_api.get_filings(query)
            if not filings['filings']:
                self.logger.error("No 10-K filings found for this company.")
                return {"error": "No 10-K filings found for this company."}

            latest_10k = filings['filings'][0]
            self.logger.debug(f"Latest 10-K Filing Data: {latest_10k}")

            accession_no = latest_10k.get('accessionNo')
            if not accession_no:
                self.logger.error("Accession number not found in the 10-K filing details.")
                return {"error": "Accession number not found in the 10-K filing details."}

            # Fetch filing HTML to extract financial data
            filing_url = latest_10k['linkToFilingDetails']
            html_content = self.get_filing_html(filing_url)

            # Parse financial statements from HTML content
            financial_data = self.parse_financial_statements(html_content)

            return {
                "income_statement": financial_data.get("income_statement", {}),
                "balance_sheet": financial_data.get("balance_sheet", {}),
                "cash_flow_statement": financial_data.get("cash_flow_statement", {}),
                "filing_date": latest_10k.get('filedAt'),
                "company_name": latest_10k.get('companyName', 'Unknown')
            }
        except Exception as e:
            self.logger.error(f"Error retrieving financial statements for {ticker}: {str(e)}")
            return {"error": str(e)}

    def get_filing_html(self, filing_url: str) -> str:
        """Fetch the HTML content of a filing."""
        response = requests.get(filing_url)
        return response.text

    def parse_financial_statements(self, html_content: str) -> Dict[str, Any]:
        """Parse financial statements from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Initialize empty dictionaries for each statement
        income_statement = {}
        balance_sheet = {}
        cash_flow_statement = {}

        # Example parsing logic (you may need to adjust based on actual HTML structure)
        
        # Extract Income Statement (example logic)
        income_table = soup.find('table', {'summary': 'Consolidated Statements of Operations'})
        if income_table:
            rows = income_table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) > 1:
                    label = cols[0].text.strip()
                    value = cols[1].text.strip()
                    income_statement[label] = value

        # Extract Balance Sheet (example logic)
        balance_table = soup.find('table', {'summary': 'Consolidated Balance Sheets'})
        if balance_table:
            rows = balance_table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) > 1:
                    label = cols[0].text.strip()
                    value = cols[1].text.strip()
                    balance_sheet[label] = value

        # Extract Cash Flow Statement (example logic)
        cash_flow_table = soup.find('table', {'summary': 'Consolidated Statements of Cash Flows'})
        if cash_flow_table:
            rows = cash_flow_table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) > 1:
                    label = cols[0].text.strip()
                    value = cols[1].text.strip()
                    cash_flow_statement[label] = value

        return {
            "income_statement": income_statement,
            "balance_sheet": balance_sheet,
            "cash_flow_statement": cash_flow_statement
        }

    def get_company_profile(self, ticker: str) -> Dict[str, Any]:
        """Retrieve company profile information from Yahoo Finance."""
        stock = yf.Ticker(ticker)
        return stock.info

    def get_all_data(self, ticker: str) -> Dict[str, Any]:
        """Retrieve all relevant data for a given company."""
        try:
            return {
                "stock_data": self.get_stock_data(ticker),
                "financial_statements": self.get_financial_statements(ticker),
                "company_profile": self.get_company_profile(ticker)
            }
        except Exception as e:
            self.logger.error(f"Error in get_all_data: {str(e)}")
            return {"error": f"An error occurred while retrieving data: {str(e)}"}