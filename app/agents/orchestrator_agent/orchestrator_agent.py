"""
Orchestrator Agent for the Exam Helper System

Routes conversations to appropriate agent based on user requirement
"""

from typing import Any, Dict, List, Optional
 
import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.agents.agent_types import ORCHESTRATOR_NAME
from app.agents.base_agent import BaseAgent
from app.agents.llm_models import LLMModels
from app.agents.state import ExamHelperState
from app.tools.exam_helper_tools import get_agent_tools

logger = structlog.get_logger(__name__)


class OrchestratorResponse(BaseModel):
    """Response format for the orchestrator agent."""

    selected_agent: str = Field(description="The agent selected to handle this query")
    reasoning: str = Field(description="Why this agent was selected")
    context_summary: str = Field(description="Summary of conversation context")


ORCHESTRATOR_PROMPT = """
You are the ORCHESTRATOR of an AI Learning System.

YOUR PRIMARY RESPONSIBILITIES:
1. Understand the student’s topic and difficulty level
2. Identify their learning intent (basic understanding vs exam preparation)
3. Decide which teaching agent to delegate to
4. Maintain context across follow-up questions
5. Keep your own responses brief — let agents handle teaching

AVAILABLE TOOL AGENTS:

1) explainer_agent  
   - Explains concepts in the simplest possible way  
   - Uses analogies, stories, real-life examples  
   - Teaches like explaining to a 10-year-old  
   - Avoids jargon unless absolutely necessary  

2) learner_agent  
   - Provides in-depth explanations  
   - Includes structured notes  
   - Gives exam-focused content  
   - Provides 16-mark style answers  
   - Includes diagrams (described), bullet points, definitions, and key points  
   - Prepares student for competitive or university exams  

DECISION RULES:

If the student:
- Says “explain simply”, “I don’t understand”, “teach from basics”, “like I’m 5”, or sounds confused → delegate to explainer_agent
- Mentions exams, 16 marks, important questions, university, competitive exams, notes, revision, or deep understanding → delegate to learner_agent
- If unclear → ask:  
  “Would you like a simple explanation or an exam-focused detailed answer?”

CONVERSATION FLOW:

1. First interaction:
   - Greet briefly.
   - Ask what topic they need help with and their goal (understanding vs exams).

2. After user response:
   - Identify intent.
   - Delegate immediately to the correct agent.

3. Follow-ups:
   - Maintain topic continuity.
   - Switch agents only if the user explicitly changes learning style.

IMPORTANT:
- Do NOT explain the concept yourself.
- Do NOT mix both styles.
- Always delegate once intent is clear.
- Keep responses short and directive.
- Focus on routing, not teaching.

CURRENT STATE:
- Intent: {intent}
"""


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent for routing exam related conversations."""

    def __init__(
        self,
        agent_name: str = ORCHESTRATOR_NAME,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        model_name: str = LLMModels.GEMINI_2_5_FLASH,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            api_key=api_key,
            temperature=temperature,
            model_name=model_name,
        )

    def get_tools(self) -> List[BaseTool]:
        """Get agent-backed tools for the orchestrator."""
        return get_agent_tools()

    def get_result_key(self) -> str:
        return "orchestrator_result"

    def get_prompt(self, state: Optional[ExamHelperState] = None) -> str:
        intent = state.get("user_intent", "unknown") if state else "unknown"
        return ORCHESTRATOR_PROMPT.format(intent=intent)

    def get_response_format(self) -> type[BaseModel]:
        return OrchestratorResponse

    async def process_query(
        self,
        query: str,
        state: Optional[ExamHelperState] = None,
    ) -> Dict[str, Any]:
        """Process a query through the orchestrator."""
        try:
            from langgraph.prebuilt import create_react_agent

            tools = self.get_tools()
            prompt = self.get_prompt(state)

            agent = create_react_agent(self.model, tools, prompt=prompt)

            result = agent.invoke({"messages": state.get("messages", []) if state else []})

            return {
                "success": True,
                "orchestrator_result": result,
                "messages": result.get("messages", []),
                "error": [],
            }
        except Exception as e:
            logger.error("Orchestrator processing failed", error=str(e))
            return {
                "success": False,
                "orchestrator_result": None,
                "error": [str(e)],
            }
