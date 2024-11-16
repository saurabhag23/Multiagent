from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, create_react_agent
from langchain.prompts import StringPromptTemplate, PromptTemplate
from langchain_openai import OpenAI as LangChainOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from typing import List, Dict, Any
import os
from agents.fundamental_agent import FundamentalAgent
from agents.technical_agent import TechnicalAgent
from agents.data_retrieval_agent import DataRetrievalAgent
from agents.visualization_agent import VisualizationAgent
from agents.statistical_agent import StatisticalAgent

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
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
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
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
        # Fetch data
        data = self.data_retrieval_agent.get_all_data(company)
        
        # Create visualizations
        visualizations = self.visualization_agent.create_visualizations(data)
        
        # Perform analyses
        fundamental_result = self.fundamental_agent.analyze(company)
        technical_result = self.technical_agent.analyze(company)
        
        # Perform statistical analysis
        statistical_result = self.statistical_agent.analyze({
            "stock_data": data["stock_data"],
            "financial_statements": data["financial_statements"]
        })
        statistical_interpretation = self.statistical_agent.interpret_results(statistical_result)

        # Run the agent executor for overall analysis and coordination
        coordinator_summary = self.agent_executor.run(company)

        # Compile final report
        final_report = self.compile_final_report(
            company, 
            coordinator_summary, 
            fundamental_result, 
            technical_result, 
            statistical_result, 
            statistical_interpretation
        )

        return {
            "company": company,
            "coordinator_summary": coordinator_summary,
            "fundamental_analysis": fundamental_result,
            "technical_analysis": technical_result,
            "statistical_analysis": statistical_result,
            "statistical_interpretation": statistical_interpretation,
            "visualizations": visualizations,
            "final_report": final_report
        }

    def compile_final_report(self, company: str, coordinator_summary: str, 
                             fundamental_result: Dict[str, Any], technical_result: Dict[str, Any], 
                             statistical_result: Dict[str, Any], statistical_interpretation: str) -> str:
        prompt = f"""
        As a senior investment analyst, compile a comprehensive final report for {company} based on the following information:

        Coordinator's Summary:
        {coordinator_summary}

        Fundamental Analysis:
        {fundamental_result['analysis']}

        Technical Analysis:
        {technical_result['analysis']}

        Statistical Analysis:
        {statistical_interpretation}

        Your report should include:
        1. An executive summary
        2. Key findings from the fundamental analysis
        3. Key findings from the technical analysis
        4. Key findings from the statistical analysis
        5. An overall investment recommendation (Buy, Hold, or Sell)
        6. Potential risks and opportunities
        7. A conclusion summarizing the main points

        The report should be well-structured, easy to read, and provide actionable insights for potential investors.
        """

        final_report = self.llm(prompt)
        return final_report
