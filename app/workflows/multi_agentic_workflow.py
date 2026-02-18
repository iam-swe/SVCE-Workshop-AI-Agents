"""
Multi-Agent Workflow for the Therapy System.

Implements a LangGraph workflow that orchestrates multiple therapy agents
in a coordinated manner
"""

import os
from typing import Any, Dict, List, Optional

import structlog
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.state import ExamHelperState, get_initial_state
from app.utils.conversation_store import get_conversation_store
from app.nodes.orchestrator_node import OrchestratorNode


logger = structlog.get_logger(__name__)


class MultiAgentWorkflow:
    """LangGraph workflow with multi-agent integration for exam helping conversations.

    Architecture:
        User Message
             |
             v
        Orchestrator (routes to appropriate agent)
             |
             v
        User Response
    """

    def __init__(
        self,
        orchestrator_node: OrchestratorNode,
        conversation_id: Optional[str] = None,
    ) -> None:
        self.orchestrator_node = orchestrator_node
        self.conversation_store = get_conversation_store()

        self.memory = MemorySaver()
        self.workflow = self._create_workflow()
        self.conversation_id = conversation_id or f"exam_helper_session_{hash(str(os.urandom(8)))}"
        self.thread_id = self.conversation_id
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self._state: Optional[ExamHelperState] = None

        self._load_conversation_history()

        logger.info("MultiAgentWorkflow initialized", conversation_id=self.conversation_id)

    def _create_workflow(self) -> CompiledStateGraph:
        """Create and compile the LangGraph workflow."""

        workflow = StateGraph(ExamHelperState)

        workflow.add_node("orchestrator", self.orchestrator_node.process)

        workflow.add_edge(START, "orchestrator")
        workflow.add_edge("orchestrator", END)

        return workflow.compile(checkpointer=self.memory)

    def _load_conversation_history(self) -> None:
        """Load conversation history from file storage."""
        stored_messages = self.conversation_store.get_messages(self.conversation_id)
        if stored_messages:
            self._state = get_initial_state()
            messages = []
            for msg in stored_messages:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
            self._state["messages"] = messages

            conversation_data = self.conversation_store.load_conversation(self.conversation_id)
            if conversation_data and conversation_data.get("metadata"):
                metadata = conversation_data["metadata"]
                self._state["user_intent"] = metadata.get("user_intent", "unknown")
                self._state["turn_count"] = metadata.get("turn_count", 0)

            logger.info("Loaded conversation history", conversation_id=self.conversation_id, message_count=len(messages))

    def _save_conversation(self) -> None:
        """Save current conversation to file storage."""
        if self._state is None:
            return

        messages: List[Dict[str, Any]] = []
        for msg in self._state.get("messages", []):
            if isinstance(msg, HumanMessage):
                messages.append({
                    "role": "user",
                    "content": msg.content,
                })
            elif isinstance(msg, AIMessage) and msg.content:
                if not getattr(msg, "tool_calls", None):
                    messages.append({
                        "role": "assistant",
                        "content": msg.content,
                    })

        metadata = {
            "user_intent": self._state.get("user_intent", "unknown"),
            "turn_count": self._state.get("turn_count", 0),
        }

        self.conversation_store.save_conversation(
            self.conversation_id,
            messages,
            metadata,
        )

    def _get_current_state(self) -> ExamHelperState:
        """Get the current state or initialize a new one."""
        if self._state is None:
            self._state = get_initial_state()
        return self._state

    async def process_query_async(
        self,
        user_message: str,
    ) -> Dict[str, Any]:
        """Process a query asynchronously through the workflow."""
        try:
            state = self._get_current_state()

            state["messages"] = list(state.get("messages", [])) + [
                HumanMessage(content=user_message)
            ]
            state["user_query"] = user_message

            final_state = await self.workflow.ainvoke(state, self.config)

            self._state = dict(final_state)

            self._save_conversation()

            response = final_state.get("orchestrator_result", "Hi there! What's up?")

            return {
                "success": True,
                "response": response,
                "state": final_state,
            }

        except Exception as e:
            logger.error("Workflow processing failed", error=str(e))
            return {
                "success": False,
                "response": "Hi there! What's up?",
                "error": str(e),
            }

    def process_query(self, user_message: str) -> Dict[str, Any]:
        """Process a query synchronously through the workflow."""
        try:
            state = self._get_current_state()

            state["messages"] = list(state.get("messages", [])) + [
                HumanMessage(content=user_message)
            ]
            state["user_query"] = user_message

            final_state = self.workflow.invoke(state, self.config)

            self._state = dict(final_state)

            self._save_conversation()

            response = final_state.get("orchestrator_result", "Hi there! What's up?")

            return {
                "success": True,
                "response": response,
                "state": final_state,
            }

        except Exception as e:
            logger.error("Workflow processing failed", error=str(e))
            return {
                "success": False,
                "response": "Hi there! What's up?",
                "error": str(e),
            }

    def chat(self, user_message: str) -> str:
        """Simple chat interface that returns just the response string."""
        result = self.process_query(user_message)
        return result.get("response", "Hi there! What's up?")

    def get_greeting(self) -> str:
        """Get initial greeting from the orchestrator."""
        try:
            model = self.orchestrator_node.orchestrator_agent.model

            response = model.invoke(
                "You are a supportive exam helper. Generate a brief, welcoming greeting for a new user who is about to get help for exam from you. Keep it to 1-2 sentences."
            )

            if response and response.content:
                return response.content


            return "Hi there! What's up?"

        except Exception as e:
            logger.error("Failed to get greeting", error=str(e))
            return "Hi there! What's up?"

    def reset(self) -> None:
        """Reset the conversation state and start a new conversation."""
        self._state = None
        self.conversation_id = f"therapy_session_{hash(str(os.urandom(8)))}"
        self.thread_id = self.conversation_id
        self.config = {"configurable": {"thread_id": self.thread_id}}
        logger.info("Workflow state reset", new_conversation_id=self.conversation_id)

    def delete_conversation(self) -> bool:
        """Delete the current conversation from storage."""
        return self.conversation_store.delete_conversation(self.conversation_id)

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all stored conversations."""
        return self.conversation_store.list_conversations()

    def load_conversation(self, conversation_id: str) -> bool:
        """Load a specific conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation to load
            
        Returns:
            True if conversation was loaded, False if not found
        """
        self.conversation_id = conversation_id
        self.thread_id = conversation_id
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self._state = None
        self._load_conversation_history()
        return self._state is not None

    def get_state(self) -> Optional[ExamHelperState]:
        """Get the current conversation state."""
        return self._state
