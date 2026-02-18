"""
Agent registry using Pydantic for type-safe agent definitions.
Single source of truth for all agent metadata.
Uses Gemini 2.5 Flash as the sole LLM provider.
"""

from typing import Any, Type

from pydantic import BaseModel, ConfigDict, Field

from app.agents.agent_types import (
    ORCHESTRATOR_NAME,
    LEARNER_AGENT_NAME,
    EXPLAINER_AGENT_NAME,
)
from app.agents.llm_models import LLMModels


class AgentDefinition(BaseModel):
    """Definition for an agent with its configuration."""

    name: str = Field(description="Canonical agent name")
    display_name: str = Field(description="Human-readable name")
    agent_class: Type[Any] = Field(description="Actual Python class")
    default_model: str = Field(default=LLMModels.GEMINI_2_5_FLASH, description="Default LLM model")
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    is_workflow: bool = Field(default=False, description="True for composite workflows")

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class AgentRegistry:
    """Registry of all available agents using Gemini 2.5 Flash."""

    @classmethod
    def get_orchestrator(cls) -> AgentDefinition:
        from app.agents.orchestrator_agent.orchestrator_agent import OrchestratorAgent

        return AgentDefinition(
            name=ORCHESTRATOR_NAME,
            display_name="Orchestrator Agent",
            agent_class=OrchestratorAgent,
            default_model=LLMModels.GEMINI_2_5_FLASH,
            default_temperature=0.7,
        )

    @classmethod
    def get_explainer_agent(cls) -> AgentDefinition:
        from app.agents.explainer_agent.explainer_agent import ExplainerAgent

        return AgentDefinition(
            name=EXPLAINER_AGENT_NAME,
            display_name="Explainer Agent",
            agent_class=ExplainerAgent,
            default_model=LLMModels.GEMINI_2_5_FLASH,
            default_temperature=0.7,
        )
        
    @classmethod
    def get_learner_agent(cls) -> AgentDefinition:
        from app.agents.learner_agent.learner_agent import LearnerAgent

        return AgentDefinition(
            name=LEARNER_AGENT_NAME,
            display_name="Learner Agent",
            agent_class=LearnerAgent,
            default_model=LLMModels.GEMINI_2_5_FLASH,
            default_temperature=0.7,
        )

    @classmethod
    def get_all_agents(cls) -> list[AgentDefinition]:
        return [
            cls.get_orchestrator(),
            cls.get_explainer_agent(),
            cls.get_learner_agent(),
        ]
