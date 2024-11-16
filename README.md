# Investment Analyst Multi-Agent System

This project implements a multi-agent system that functions as a junior investment analyst, assisting in the analysis of publicly traded companies.

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (On Windows: `venv\Scripts\activate`)
4. Install dependencies: `pip install -r requirements.txt`
5. Set up your `.env` file with necessary API keys
6. Run the application: `python backend/app.py`

## Usage

Send a POST request to `/analyze` with a JSON body containing the ticker symbol of the company you want to analyze:

```json
{
    "ticker": "AAPL"
}