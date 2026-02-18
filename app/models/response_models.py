"""
Response models for agent outputs.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ExamHelperResponse(BaseModel):
    """Standard response format for agents."""

    response: str = Field(description="The exam helper response")


class OrchestratorResponse(BaseModel):
    """Response format for the orchestrator agent."""

    selected_agent: str = Field(description="The agent selected to handle this query")
    reasoning: str = Field(description="Why this agent was selected")
    context_summary: str = Field(description="Summary of conversation context")
    response: Optional[str] = Field(default=None, description="The final response")

