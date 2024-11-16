import yfinance as yf
from sec_api import QueryApi, ExtractorApi
import os
from typing import Dict, Any
import pandas as pd

class DataRetrievalAgent:
    def __init__(self):
        self.sec_api = QueryApi(api_key=os.getenv('SEC_API_KEY'))
        self.extractor_api = ExtractorApi(api_key=os.getenv('SEC_API_KEY'))

    def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Retrieve historical stock data from Yahoo Finance.
        
        Args:
            ticker (str): The stock ticker symbol.
            period (str): The time period to retrieve data for (e.g., "1y" for 1 year).
        
        Returns:
            pd.DataFrame: Historical stock data.
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data

    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieve financial statements from SEC filings.
        
        Args:
            ticker (str): The stock ticker symbol.
        
        Returns:
            Dict[str, Any]: Financial statements data.
        """
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

        filings = self.sec_api.get_filings(query)
        
        if filings['total']['value'] == 0:
            return {"error": "No 10-K filings found for this company."}

        latest_10k = filings['filings'][0]
        accession_no = latest_10k['accessionNo']

        # Extract the financial tables
        financial_data = self.extractor_api.get_financial_data(accession_no)

        # Parse the HTML content of the filing
        html_content = self.get_filing_html(latest_10k['linkToFilingDetails'])

        return {
            "income_statement": self.parse_financial_table(financial_data, 'income_statement'),
            "balance_sheet": self.parse_financial_table(financial_data, 'balance_sheet'),
            "cash_flow_statement": self.parse_financial_table(financial_data, 'cash_flow_statement'),
            "filing_date": latest_10k['filedAt'],
            "company_name": self.extract_company_name(html_content),
            "fiscal_year_end": self.extract_fiscal_year_end(html_content)
        }

    def parse_financial_table(self, financial_data: Dict[str, Any], statement_type: str) -> pd.DataFrame:
        """
        Parse financial data into a DataFrame.
        
        Args:
            financial_data (Dict[str, Any]): The financial data from SEC API.
            statement_type (str): The type of financial statement.
        
        Returns:
            pd.DataFrame: Parsed financial data.
        """
        if statement_type not in financial_data:
            return pd.DataFrame()

        data = financial_data[statement_type]
        df = pd.DataFrame(data)
        
        # Pivot the DataFrame to have years as columns
        df_pivoted = df.pivot(index='label', columns='frame', values='value')
        
        return df_pivoted

    def get_filing_html(self, filing_url: str) -> str:
        """
        Fetch the HTML content of a filing.
        
        Args:
            filing_url (str): The URL of the filing.
        
        Returns:
            str: The HTML content of the filing.
        """
        response = requests.get(filing_url)
        return response.text

    def extract_company_name(self, html_content: str) -> str:
        """
        Extract the company name from the filing HTML.
        
        Args:
            html_content (str): The HTML content of the filing.
        
        Returns:
            str: The extracted company name.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        company_name = soup.find('span', class_='companyName')
        return company_name.text.strip() if company_name else "Unknown"

    def extract_fiscal_year_end(self, html_content: str) -> str:
        """
        Extract the fiscal year end date from the filing HTML.
        
        Args:
            html_content (str): The HTML content of the filing.
        
        Returns:
            str: The extracted fiscal year end date.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        fiscal_year_end = soup.find('div', string=re.compile(r'Fiscal Year End:'))
        if fiscal_year_end:
            return fiscal_year_end.text.split(':')[1].strip()
        return "Unknown"

    def get_company_profile(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieve company profile information from Yahoo Finance.
        
        Args:
            ticker (str): The stock ticker symbol.
        
        Returns:
            Dict[str, Any]: Company profile information.
        """
        stock = yf.Ticker(ticker)
        return stock.info

    def get_all_data(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieve all relevant data for a given company.
        
        Args:
            ticker (str): The stock ticker symbol.
        
        Returns:
            Dict[str, Any]: All retrieved data.
        """
        return {
            "stock_data": self.get_stock_data(ticker),
            "financial_statements": self.get_financial_statements(ticker),
            "company_profile": self.get_company_profile(ticker)
        }