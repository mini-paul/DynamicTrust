# ==============================================================================
# File: agents/adversarial_agent.py
# Description: 定义对抗性智能体
# ==============================================================================

from agents.base_agent import BaseAgent
from utils.prompts import ADVERSARIAL_AGENT_PROMPT

class AdversarialAgent(BaseAgent):
    def get_prompt(self):
        return ADVERSARIAL_AGENT_PROMPT