# ==============================================================================
# File: agents/base_agent.py
# Description: 定义所有智能体的基类
# ==============================================================================

from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage
from utils.llm_clients import get_llm_client

class BaseAgent(ABC):
    def __init__(self, agent_id: str, config: dict):
        self.agent_id = agent_id
        self.config = config
        self.llm = get_llm_client(config)

    @abstractmethod
    def get_prompt(self):
        pass

    def invoke(self, query: str, conversation_history: str) -> str:
        prompt = self.get_prompt()
        chain = prompt | self.llm
        response = chain.invoke({
            "agent_id": self.agent_id,
            "query": query,
            "conversation_history": conversation_history
        })
        return response.content