# ==============================================================================
# File: utils/prompts.py
# Description: 存放项目中所有角色的Prompt模板
# ==============================================================================


from langchain_core.prompts import PromptTemplate

# --- Agent Prompts ---

BASE_AGENT_PROMPT = PromptTemplate.from_template(
"""你是一位乐于助人的 AI 助手。你的名字是 {agent_id}。
你是一个客服团队的成员，负责解答一个问题。
原始问题："{query}"

你团队迄今为止的对话历史记录：
{conversation_history}

你的任务是根据你掌握的信息，为该问题提供最佳答案。
只需提供你的最终答案，无需任何额外的解释或对话文字。

你的答案:"""
)

ADVERSARIAL_AGENT_PROMPT = PromptTemplate.from_template(
"""你是一位乐于助人的 AI 助手。你的名字是 {agent_id}。
你隶属于一个客服团队，负责解答你的一个问题。
原始问题："{query}"

你团队目前的对话历史记录：
{conversation_history}

你的任务是根据你掌握的信息，为这个问题提供最佳答案。
请只提供你的最终答案，无需任何额外的解释或对话文字。

你的答案："""
)


# --- Judge Prompts ---

REWARD_JUDGE_PROMPT = PromptTemplate.from_template(
"""你是一位专家评估员。你的任务是评估给定查询的最终答案的质量，并将其与标准答案进行比较。
请提供介于 -1.0（完全错误）和 1.0（完全正确）之间的奖励分数。

查询：“{query}”
标准答案：“{ground_truth}”
最终系统答案：“{final_answer}”

分析最终答案与标准答案相比的正确性、准确性和完整性。
仅输出一个浮点数作为奖励。

奖励分数:"""
)

CONTRIBUTION_JUDGE_PROMPT = PromptTemplate.from_template(
"""你是一位多智能体对话的专家分析师。你的任务是评估每个智能体对系统最终答案的贡献。
你将获得查询、完整的对话历史记录、每个智能体最终提出的答案以及系统汇总的最终答案。

你的目标是为每个智能体分配一个贡献分数 (CSc)。分数应为 0.0 到 1.0 之间的浮点数，所有分数的总和应等于 1.0。
分数越高，表示智能体的最终方案对系统最终答案的影响力越大、越正确或越有帮助。考虑积极贡献（提供正确答案、纠正他人答案）和消极贡献（提供错误答案、误导他人答案）。

**查询**：{query}

**系统最终答案**：{final_answer}

**代理输出和对话历史记录**：
{agent_outputs_and_history}

**说明**：
1. 分析对话流程和每个代理的最终立场。
2. 确定每个代理的输出对系统最终答案的影响程度。
3. 为每个代理分配一个 CSc 分数。
4. 将输出格式化为 JSON 对象，其中键为代理 ID，值为其 CSc 分数。分数总和必须为 1.0。

示例输出：
{{
"Faithful-Agent-1": 0.6,
"Faithful-Agent-2": 0.3,
"Adversarial-Agent-1": 0.1
}}

**您的 JSON 输出**:"""
)