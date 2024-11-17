import datetime
from langchain_openai import OpenAI as LangChainOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
import os
from agents.data_retrieval_agent import DataRetrievalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.technical_agent import TechnicalAgent
from agents.visualization_agent import VisualizationAgent
from agents.statistical_agent import StatisticalAgent
from agents.serialization_utils import serialize_result
import logging
import json
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CoordinatorAgent:
    def __init__(self):
        self.llm = LangChainOpenAI(temperature=0.7)
        self.data_retrieval_agent = DataRetrievalAgent()
        self.visualization_agent = VisualizationAgent()
        self.fundamental_agent = FundamentalAgent()
        self.technical_agent = TechnicalAgent()
        self.statistical_agent = StatisticalAgent()
        self.tools = self._setup_tools()
        self.prompt = self._setup_prompt()
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def _setup_tools(self) -> List[Tool]:
        return [
            Tool(
                name="DataRetrieval",
                func=self.data_retrieval_agent.get_all_data,
                description="Retrieves all necessary data for a given company"
            ),
            Tool(
                name="FundamentalAnalysis",
                func=self.fundamental_agent.analyze,
                description="Performs fundamental analysis on a company"
            ),
            Tool(
                name="TechnicalAnalysis",
                func=self.technical_agent.analyze,
                description="Performs technical analysis on a company's stock"
            ),
            Tool(
                name="Visualization",
                func=self.visualization_agent.create_visualizations,
                description="Creates visualizations based on the provided data"
            ),
            Tool(
                name="StatisticalAnalysis",
                func=self.statistical_agent.analyze,
                description="Performs statistical analysis on company data"
            )
        ]

    def _setup_prompt(self) -> PromptTemplate:
        template = """You are a senior investment analyst coordinating a comprehensive analysis of a company and its stock. 
        Your job is to delegate tasks to specialized agents and compile their findings into a final report.

        Use the following tools to gather information and perform analyses:
        {tools}

        Use the following format:
        Question: the input question you must answer
        Thought: consider what information you need and which tools to use
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (usually the company ticker)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat as needed)
        Thought: I now have enough information to compile the final report
        Final Answer: provide a comprehensive investment analysis report

        Begin!

        Question: Provide a comprehensive investment analysis for {input}
        {agent_scratchpad}
        """

        return PromptTemplate(template=template, input_variables=["tools", "tool_names", "input", "agent_scratchpad"])
      
    def analyze(self, company: str) -> Dict[str, Any]:
       try:
           # Fetch data from various agents
           data_retrieval_result = self.data_retrieval_agent.get_all_data(company)
           
           if 'error' in data_retrieval_result:
               logger.error(data_retrieval_result['error'])
               return data_retrieval_result  # Return error if retrieval failed

           fundamental_result = self.fundamental_agent.analyze(company)
           technical_result = self.technical_agent.analyze(company)

           # Create visualizations based on retrieved data
           visualizations = self.visualization_agent.create_visualizations(data_retrieval_result)

           # Perform statistical analysis based on retrieved data
           statistical_result = self.statistical_agent.analyze({
               "stock_data": data_retrieval_result["stock_data"],
               "financial_statements": data_retrieval_result["financial_statements"]
           })

           # Compile final report (you can customize this further)
           final_report = {
               'company_name': company,
               'financial_statements': data_retrieval_result['financial_statements'],
               'fundamental_analysis': fundamental_result,
               'technical_analysis': technical_result,
               'visualizations': visualizations,
               'statistical_analysis': statistical_result,
               'fiscal_year_end': data_retrieval_result.get('fiscal_year_end', 'Unknown'),
               'filing_date': data_retrieval_result.get('filing_date', 'Unknown')
           }
           
            
           return serialize_result(final_report)
        
       except Exception as e:
           logger.error(f"Error in analyze method: {str(e)}")
           return {"error": f"An error occurred during analysis: {str(e)}"}