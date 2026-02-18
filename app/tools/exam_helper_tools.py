"""
Agent tools for the multi-agent system.

These tools wrap the actual therapy agent instances so the orchestrator
delegates to them rather than duplicating agent logic inline.
"""

import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ExamHelperInput(BaseModel):
    """Input schema for agent tools."""

    message: str = Field(description="The user's message to respond to")
    context: str = Field(description="Conversation context/summary", default="")


_agent_cache = {}


def _get_agent(agent_class):
    """Lazily instantiate and cache agent instances."""
    name = agent_class.__name__
    if name not in _agent_cache:
        _agent_cache[name] = agent_class()
    return _agent_cache[name]


def _build_state_from_context(context: str) -> dict:
    """Build a minimal state dict from a context string for agent prompt formatting."""
    messages = []
    if context:
        messages.append(HumanMessage(content=context))
    return {"messages": messages}


def _create_agent_tool_fn(agent_class):
    """Create a tool function that delegates to an actual agent instance."""

    def agent_tool_fn(message: str, context: str = "") -> str:
        agent = _get_agent(agent_class)
        state = _build_state_from_context(context)

        result = asyncio.run(
             agent.process_query(message, state)
        )

        return result.get(agent.get_result_key(), "")

    return agent_tool_fn

def _build_tools():
    """Build all agent tools. Imports are deferred to avoid circular imports."""
    from app.agents.explainer_agent.explainer_agent import ExplainerAgent
    from app.agents.learner_agent.learner_agent import LearnerAgent


    explainer = StructuredTool.from_function(
        func=_create_agent_tool_fn(ExplainerAgent),
        name="explainer",
        description="Use when user wants a certain concept to be explained.",
        args_schema=ExamHelperInput,
    )

    learner = StructuredTool.from_function(
        func=_create_agent_tool_fn(LearnerAgent),
        name="learner",
        description="Use when user asks for material to study a certain topic",
        args_schema=ExamHelperInput,
    )

    return explainer,learner


def get_agent_tools():
    """Get all agent-backed tools for the orchestrator."""
    return list(_build_tools())

