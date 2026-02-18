"""
Agent factory for creating and managing agent singletons.
"""

from typing import Any, Dict, cast

import structlog

from app.agents import (
    OrchestratorAgent,
    ExplainerAgent,
    LearnerAgent,
)
from app.agents.agent_types import (
    ORCHESTRATOR_NAME,
    EXPLAINER_AGENT_NAME,
    LEARNER_AGENT_NAME,
)
from app.agents.config import AgentConfig, AgentFactoryConfig

logger = structlog.get_logger(__name__)

_singletons: Dict[str, Any] = {}
_initialized: bool = False


def _create_agent_with_config(agent_name: str, agent_class: type, config: AgentConfig) -> Any:
    """Create an agent instance with the given configuration."""
    return agent_class(
        model_name=config.model_name,
        temperature=config.temperature,
    )


def initialize_agents(config: AgentFactoryConfig | None = None) -> None:
    """Initialize all agent singletons."""
    global _singletons, _initialized

    if _initialized:
        logger.info("Agents already initialized")
        return

    if config is None:
        config = AgentFactoryConfig()

    logger.info("Initializing agents")

    # Create all agents
    _singletons[ORCHESTRATOR_NAME] = _create_agent_with_config(
        ORCHESTRATOR_NAME,
        OrchestratorAgent,
        config.orchestrator_agent,
    )
    _singletons[EXPLAINER_AGENT_NAME] = _create_agent_with_config(
        EXPLAINER_AGENT_NAME,
        ExplainerAgent,
        config.explainer_agent,
    )
    _singletons[LEARNER_AGENT_NAME] = _create_agent_with_config(
        LEARNER_AGENT_NAME,
        LearnerAgent,
        config.learner_agent,
    )

    _initialized = True
    logger.info("All agents initialized successfully")


def get_agent(agent_name: str) -> Any:
    """Get an agent singleton by name."""
    if not _initialized:
        initialize_agents()
    return _singletons.get(agent_name)


def create_multi_agent_workflow(conversation_id: str | None = None) -> "MultiAgentWorkflow":
    """Create and return the multi-agent workflow.
    
    Args:
        conversation_id: Optional conversation ID to resume an existing conversation
    """
    from app.nodes.orchestrator_node import OrchestratorNode
    from app.workflows.multi_agentic_workflow import MultiAgentWorkflow

    if not _initialized:
        initialize_agents()

    orchestrator_agent = cast(OrchestratorAgent, _singletons.get(ORCHESTRATOR_NAME))

    orchestrator_node = OrchestratorNode(orchestrator_agent)

    return MultiAgentWorkflow(
        orchestrator_node=orchestrator_node,
        conversation_id=conversation_id,
    )
