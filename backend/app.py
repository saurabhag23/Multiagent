from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from agents.coordinator_agent import CoordinatorAgent
import os
import base64

# Load environment variables
load_dotenv()

app = Flask(__name__)

coordinator = CoordinatorAgent()

@app.route('/analyze', methods=['POST'])
def analyze_company():
    data = request.json
    ticker = data.get('ticker')
    
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        result = coordinator.analyze(ticker)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Create an HTML template to display the results and visualizations
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Investment Analysis for {{ result.company }}</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            h1, h2, h3 { color: #333; }
            pre { white-space: pre-wrap; word-wrap: break-word; background-color: #f4f4f4; padding: 15px; }
            img { max-width: 100%; height: auto; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Investment Analysis for {{ result.company }}</h1>
        
        <h2>Executive Summary</h2>
        <pre>{{ result.coordinator_summary }}</pre>
        
        <h2>Final Report</h2>
        <pre>{{ result.final_report }}</pre>
        
        <h2>Fundamental Analysis</h2>
        <pre>{{ result.fundamental_analysis.analysis }}</pre>
        
        <h2>Technical Analysis</h2>
        <pre>{{ result.technical_analysis.analysis }}</pre>
        
        <h2>Statistical Analysis</h2>
        <pre>{{ result.statistical_interpretation }}</pre>
        
        <h2>Visualizations</h2>
        <h3>Fundamental Analysis Visualizations</h3>
        {% for name, img_data in result.fundamental_analysis.visualizations.items() %}
            <h4>{{ name }}</h4>
            <img src="data:image/png;base64,{{ img_data }}" alt="{{ name }}">
        {% endfor %}
        
        <h3>Technical Analysis Visualizations</h3>
        {% for name, img_data in result.technical_analysis.visualizations.items() %}
            <h4>{{ name }}</h4>
            <img src="data:image/png;base64,{{ img_data }}" alt="{{ name }}">
        {% endfor %}
        
        <h3>Additional Visualizations</h3>
        {% for name, img_data in result.visualizations.items() %}
            <h4>{{ name }}</h4>
            <img src="data:image/png;base64,{{ img_data }}" alt="{{ name }}">
        {% endfor %}
    </body>
    </html>
    """
    
    return render_template_string(html_template, result=result)

@app.route('/', methods=['GET'])
def index():
    return """
    <html>
        <body>
            <h1>Stock Analysis Tool</h1>
            <form action="/analyze" method="post">
                <label for="ticker">Enter stock ticker:</label>
                <input type="text" id="ticker" name="ticker" required>
                <input type="submit" value="Analyze">
            </form>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)