"""Agent configuration models using Pydantic."""

from pydantic import BaseModel, Field

from app.agents.llm_models import LLMModels


class AgentConfig(BaseModel):
    """Configuration for an individual agent."""

    model_name: str = Field(default=LLMModels.GEMINI_2_5_FLASH, description="The LLM model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class AgentFactoryConfig(BaseModel):
    """Factory configuration for all agents using Gemini 2.5 Flash."""

    orchestrator_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            model_name=LLMModels.GEMINI_2_5_FLASH,
            temperature=0.7,
        )
    )
    explainer_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            model_name=LLMModels.GEMINI_2_5_FLASH,
            temperature=0.7,
        )
    )
    learner_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            model_name=LLMModels.GEMINI_2_5_FLASH,
            temperature=0.7,
        )
    )
    def get_config(self, agent_name: str) -> AgentConfig:
        """Get config for a specific agent by name."""
        config: AgentConfig = getattr(self, agent_name, None)
        if config is None:
            raise ValueError(f"No config found for agent: {agent_name}")
        return config
