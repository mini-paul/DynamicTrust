# ==============================================================================
# File: agents/judge_agent.py
# Description: 定义评判者智能体，负责计算奖励和贡献分
# ==============================================================================

import json
import re
from agents.base_agent import BaseAgent
from utils.prompts import REWARD_JUDGE_PROMPT, CONTRIBUTION_JUDGE_PROMPT

def extract_final_answer(response: str) -> str:
    """提取 <think> 之后的内容作为最终答案"""
    # 方法 1: 删除所有 <think> 标签内容
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # 方法 2: 提取最后一个 </think> 后的内容
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    clean_response = clean_response.replace("\n","")

    return clean_response.strip()
class JudgeAgent(BaseAgent):
    def get_prompt(self):
        # Judge has multiple prompts, handled in specific methods
        pass

    def get_reward_score(self, query: str, final_answer: str, ground_truth: str) -> float:
        chain = REWARD_JUDGE_PROMPT | self.llm
        response = chain.invoke({
            "query": query,
            "final_answer": final_answer,
            "ground_truth": ground_truth
        })

        get_only_text = extract_final_answer(response.content.strip())
        print("get_reward_score 8888888888888 -- ", get_only_text)
        try:
            return float(get_only_text)
        except ValueError:
            print(f"Warning: Judge failed to produce a valid float for reward. Got: {get_only_text}")
            return 0.0

    def get_contribution_scores(self, query: str, final_answer: str, agent_outputs_and_history: str) -> dict:
        chain = CONTRIBUTION_JUDGE_PROMPT | self.llm
        response = chain.invoke({
            "query": query,
            "final_answer": final_answer,
            "agent_outputs_and_history": agent_outputs_and_history
        })


        get_only_text = extract_final_answer(response.content.strip())
        print("get_contribution_scores 99999999999999999999 -- ", get_only_text)
        try:
            # Clean the response content to ensure it's a valid JSON string
            content = get_only_text
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            scores = json.loads(content)

            print(scores)
            print(type(scores))
            print("12121212112121212")


            # Normalize scores to ensure they sum to 1.0
            total_score = sum(scores.values())
            if total_score > 0:
                normalized_scores = {agent: score / total_score for agent, score in scores.items()}
                return normalized_scores
            else:
                # If all scores are 0, distribute equally
                num_agents = len(scores)
                if num_agents > 0:
                    return {agent: 1.0 / num_agents for agent in scores}
                return {}


        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Judge failed to produce valid JSON for contributions. Error: {e}. Got: {response.content}")
            # Fallback: return equal scores
            try:
                # Attempt to parse agent IDs from the malformed string for fallback
                agent_ids = [line.split(":")[0].strip().strip('"') for line in
                             agent_outputs_and_history.strip().split('\n') if ":" in line]
                if agent_ids:
                    num_agents = len(agent_ids)
                    return {agent_id: 1.0 / num_agents for agent_id in agent_ids}
            except Exception:
                pass
            return {}
