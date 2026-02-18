"""
Main entry point for the Exam helper System.
"""

import structlog
from dotenv import load_dotenv

load_dotenv()

from app.agents.agent_factory import create_multi_agent_workflow
from app.workflows.multi_agentic_workflow import MultiAgentWorkflow

logger = structlog.get_logger(__name__)


def create_app(conversation_id: str | None = None) -> MultiAgentWorkflow:
    """Create and configure the multi-agent workflow application.
    
    Args:
        conversation_id: Optional conversation ID to resume an existing conversation
    """
    logger.info("Intialising Exam Helper Application")
    workflow = create_multi_agent_workflow(conversation_id)
    logger.info("Exam Helper Application initialized successfully :))")
    return workflow


def run(query: str, conversation_id: str | None = None) -> str:
    """Run a single query through the therapy workflow.

    Args:
        query: The user's message/query
        conversation_id: Optional conversation ID to resume an existing conversation

    Returns:
        The assistant's response
    """
    workflow = create_app(conversation_id)
    response = workflow.chat(query)
    return response


def run_interactive_session(conversation_id: str | None = None) -> None:
    """Run an interactive session with continuous conversation."""
    workflow = create_app(conversation_id)

    print("\n" + "=" * 50)
    print("Welcome to Exam Helper System")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 50 + "\n")

    initial_response = workflow.get_greeting()
    print(f"Mentor: {initial_response}\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
                print("\nMentor: All the best :)) You will do well!! Dont worry : )) Goodbye!\n")
                break

            response = workflow.chat(user_input)
            print(f"\nMentor: {response}\n")

        except KeyboardInterrupt:
            print("\n\nMentor: All the best :)) You will do well!! Dont worry : )) Goodbye!\n")
            break
        except EOFError:
            print("\n\nSession ended.\n")
            break


def start_session(conversation_id: str | None = None) -> None:
    """Start a therapy session (alias for run_interactive_session)."""
    run_interactive_session(conversation_id)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        response = run(query)
        print(f"\nMentor: {response}\n")
    else:
        start_session()

