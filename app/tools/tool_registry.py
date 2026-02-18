"""
Tool registry for managing available tools.
"""

from typing import Any, Dict, List

from langchain_core.tools import BaseTool

from app.agents.agent_types import (
        EXPLAINER_AGENT_NAME,LEARNER_AGENT_NAME
    )
from app.tools.exam_helper_tools import get_agent_tools
    
TOOL_REGISTRY: Dict[str, Any] = {}


def register_tool(name: str, tool_callable: Any) -> None:
    """Register a tool in the registry."""
    TOOL_REGISTRY[name] = tool_callable


def get_tool(name: str) -> Any:
    """Get a tool from the registry by name."""
    return TOOL_REGISTRY.get(name)


def get_all_tools() -> List[BaseTool]:
    """Get all registered agent tools."""

    return get_agent_tools()


def initialize_tools() -> None:
    """Initialize and register all agent tools."""
    
    tools = get_agent_tools()
    names = [EXPLAINER_AGENT_NAME,LEARNER_AGENT_NAME]
    for name, tool in zip(names, tools):
        register_tool(name, tool)