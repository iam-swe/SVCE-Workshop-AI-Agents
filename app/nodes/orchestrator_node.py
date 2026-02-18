"""
Orchestrator Node for the Therapy Workflow.
"""

from typing import Any, Dict
import structlog

from app.agents.base_agent import BaseAgent
from app.agents.state import ExamHelperState
from app.utils.intent_detector import detect_intent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from app.tools.exam_helper_tools import get_agent_tools

logger = structlog.get_logger(__name__)


class OrchestratorNode:
    """Node for processing conversations through the orchestrator agent."""

    def __init__(self, orchestrator_agent: BaseAgent) -> None:
        self.orchestrator_agent = orchestrator_agent
        
    @staticmethod
    def _extract_text(content) -> str:
        """Extract text from content that may be a string or a list of content blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(content)

    def process(self, state: ExamHelperState) -> Dict[str, Any]:
        """Process the current state through the orchestrator."""
        try:

            user_msg = ""
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage):
                    user_msg = msg.content
                    break

            current_intent = state.get("user_intent", "unknown")

            if current_intent == "unknown" and user_msg:
                current_intent = detect_intent(user_msg)

            tools = get_agent_tools()
            prompt = self.orchestrator_agent.get_prompt(state)

            agent = create_react_agent(
                self.orchestrator_agent.model,
                tools,
                prompt=prompt,
            )

            result = agent.invoke({"messages": state.get("messages", [])})
            orchestrator_response = ""
            ai_message=""

            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, ToolMessage) and msg.content:
                    orchestrator_response = msg.content
                    break
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                    ai_message = self._extract_text(msg.content)

            if orchestrator_response == "":
                orchestrator_response = ai_message

            return {
                "messages": result.get("messages", []),
                "user_intent": current_intent,
                "orchestrator_result": orchestrator_response,
            }

        except Exception as e:
            error_msg = f"Orchestrator node failed: {str(e)}"
            logger.error("Orchestrator node failed", error=str(e))
            return {
                "orchestrator_result": None,
                "error": [error_msg],
            }
