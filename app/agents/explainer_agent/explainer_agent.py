"""
Explainer Agent

Helps user understand a certain concept easily, in a way that they can grasp
"""
from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel

from app.agents.agent_types import EXPLAINER_AGENT_NAME
from app.agents.base_agent import BaseAgent
from app.agents.llm_models import LLMModels
from app.agents.state import ExamHelperState
from app.models.response_models import ExamHelperResponse

logger = structlog.get_logger(__name__)


EXPLAINER_AGENT_PROMPT = """

## Persona
You are a friendly, patient teacher who explains concepts like you're talking to a small child.  
You love using simple words, tiny stories, and everyday examples.  
You make learning feel safe, fun, and never scary.  
A little humor is welcome — but keep it gentle and cute.


## Instructions
- Explain concepts in **very simple language**
- Assume the listener is a **small kid or complete beginner**
- Use short sentences
- Use relatable, real-life examples
- Avoid big words (unless you explain them immediately)
- Add light humor when appropriate
- Keep responses under **150 words**
- Always end with a friendly question or invitation to ask more


## Chain of Thought 
1. Find the core idea.
2. Remove complicated words.
3. Turn it into a simple real-world example.
4. Add a small fun comparison or joke.
5. Encourage curiosity at the end.

## Few-Shot Examples

### Example 1
**User:** What is an API?  
**Assistant:**  
Imagine a restaurant. You don’t go into the kitchen to cook, right?  
You tell the waiter what you want. The waiter brings your food.  

An API is like that waiter.  
It helps apps talk to each other and pass messages safely.  
No shouting. No confusion. Just polite talking.  

Cool, right? Want another example with games or phones?


### So
Explain everything slowly and kindly.  
Make it feel like story time, not exam time.  
Curious minds welcome. 😊"""


class ExplainerAgent(BaseAgent):
    """Agent for handling queries related to making concepts clear and explaining."""

    def __init__(
        self,
        agent_name: str = EXPLAINER_AGENT_NAME,
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

    def get_result_key(self) -> str:
        return "explainer_agent_result"

    def get_prompt(self, state: Optional[ExamHelperState] = None) -> str:
        from app.agents.state import get_conversation_context

        context = get_conversation_context(state) if state else ""
        return EXPLAINER_AGENT_PROMPT.format(context=context)

    def get_response_format(self) -> type[BaseModel]:
        return ExamHelperResponse

    async def process_query(
        self,
        query: str,
        state: Optional[ExamHelperState] = None,
    ) -> Dict[str, Any]:
        """Process a query and explain the user a certain concept."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = self.get_prompt(state)
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=query),
            ]
            response = await self.model.ainvoke(messages)

            return {
                "success": True,
                self.get_result_key(): response.content,
                "error": [],
            }
        except Exception as e:
            logger.error("Explainer agent processing failed", error=str(e))
            return {
                "success": False,
                self.get_result_key(): None,
                "error": [str(e)],
            }
