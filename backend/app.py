from flask import Flask, request, jsonify, render_template_string
from agents.coordinator_agent import CoordinatorAgent
import logging
import os
import json
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from agents.serialization_utils import serialize_result,serialize_pandas
# Load environment variables
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

coordinator = CoordinatorAgent()

@app.route('/analyze', methods=['POST'])
def analyze_company():
    if request.is_json:
        data = request.json
    else:
        data = request.form
    ticker = data.get('ticker')
    
    
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        result = coordinator.analyze(ticker)
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 500
        serialized_result = json.loads(json.dumps(result, default=serialize_pandas))

        # HTML template for displaying results
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Investment Analysis for {{ result.company_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
                h1, h2, h3 { color: #333; }
                pre { white-space: pre-wrap; word-wrap: break-word; background-color: #f4f4f4; padding: 15px; }
                img { max-width: 100%; height: auto; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <h1>Investment Analysis for {{ result.company_name }}</h1>
            
            <h2>Fundamental Analysis</h2>
            <pre>{{ result.fundamental_analysis.analysis }}</pre>
            
            <h2>Technical Analysis</h2>
            <pre>{{ result.technical_analysis.analysis_summary }}</pre>
            
            <h2>Statistical Analysis</h2>
            <pre>{{ result.statistical_analysis.interpretation }}</pre>
            
            <h2>Financial Statements</h2>
            <h3>Income Statement</h3>
            <pre>{{ result.financial_statements.income_statement }}</pre>
            <h3>Balance Sheet</h3>
            <pre>{{ result.financial_statements.balance_sheet }}</pre>
            <h3>Cash Flow Statement</h3>
            <pre>{{ result.financial_statements.cash_flow_statement }}</pre>
            
            <h2>Visualizations</h2>
            {% for name, img_data in result.visualizations.items() %}
                <h3>{{ name }}</h3>
                <img src="data:image/png;base64,{{ img_data }}" alt="{{ name }}">
            {% endfor %}
            
            <h2>Additional Information</h2>
            <p>Fiscal Year End: {{ result.fiscal_year_end }}</p>
            <p>Filing Date: {{ result.filing_date }}</p>
        </body>
        </html>
        """

        return render_template_string(html_template, result=serialized_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/', methods=['GET'])
def index():
    return """
    <html>
        <head>
            <script>
                function submitForm(event) {
                    event.preventDefault();
                    var ticker = document.getElementById('ticker').value;
                    fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ticker: ticker})
                    })
                    .then(response => response.text())
                    .then(html => {
                        document.body.innerHTML = html;
                    })
                    .catch(error => console.error('Error:', error));
                }
            </script>
        </head>
        <body>
            <h1>Stock Analysis Tool</h1>
            <form onsubmit="submitForm(event)">
                <label for="ticker">Enter stock ticker:</label>
                <input type="text" id="ticker" name="ticker" required>
                <input type="submit" value="Analyze">
            </form>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6002, debug=True)