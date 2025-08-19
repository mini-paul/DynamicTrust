# ==============================================================================
# File: main.py
# Description: 主运行脚本，用于配置和启动CrS实验
# ==============================================================================

import json
from langgraph.graph import StateGraph, END
from agents.faithful_agent import FaithfulAgent
from agents.adversarial_agent import AdversarialAgent
from agents.judge_agent import JudgeAgent
from graphs.crs_graph import (
    CrsGraphState,
    initialize_crs,
    agent_collaboration_step,
    crs_aware_aggregation,
    evaluate_and_reward,
    update_crs,
    should_continue_collaboration
)
from config import JUDGE_MODEL_CONFIG, AGENT_TEAM_CONFIG, INITIAL_CRS


def load_dataset(path="datasets/gsm8k_sample.json"):
    """Loads a sample dataset."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_agent_team():
    """Builds the team of agents based on the configuration."""
    team = {}
    # Add faithful agents
    for agent_conf in AGENT_TEAM_CONFIG["faithful_agents"]:
        team[agent_conf["id"]] = FaithfulAgent(agent_conf["id"], agent_conf["config"])
    # Add adversarial agents
    for agent_conf in AGENT_TEAM_CONFIG["adversarial_agents"]:
        team[agent_conf["id"]] = AdversarialAgent(agent_conf["id"], agent_conf["config"])

    # Create the judge agent separately
    judge = JudgeAgent("Judge-Agent", JUDGE_MODEL_CONFIG)
    return team, judge


def define_workflow():
    """Defines and compiles the LangGraph workflow."""
    workflow = StateGraph(CrsGraphState)
    workflow.add_node("initialize", initialize_crs)
    # Add nodes
    workflow.add_node("collaborate", agent_collaboration_step)
    workflow.add_node("aggregate", crs_aware_aggregation)
    workflow.add_node("evaluate", evaluate_and_reward)
    workflow.add_node("update_scores", update_crs)

    # Define edges
    workflow.set_entry_point("collaborate")

    # Conditional edge for collaboration loop (simplified to one iteration)
    workflow.add_conditional_edges(
        "collaborate",
        should_continue_collaboration,
        {
            "continue": "collaborate",
            "aggregate": "aggregate"
        }
    )

    workflow.add_edge("aggregate", "evaluate")
    workflow.add_edge("evaluate", "update_scores")
    workflow.add_edge("update_scores", END)  # End of one query round

    return workflow.compile()


def run_experiment():
    """Runs the full experiment over the dataset."""
    print(">>> Building Agent Team...")
    agent_team, judge_agent = build_agent_team()

    print(">>> Defining Workflow...")
    app = define_workflow()

    print(">>> Loading Dataset...")
    dataset = load_dataset()

    # Initialize the state for the first run
    # The credibility scores will be carried over between queries
    persistent_state = {
        "agent_team": agent_team,
        "judge_agent": judge_agent,
        "credibility_scores": {agent_id: INITIAL_CRS for agent_id in agent_team.keys()}
    }

    print(persistent_state)

    print("\n" + "=" * 50)
    print(">>> STARTING EXPERIMENT RUN")
    print("=" * 50 + "\n")
    print(f"Initial CrS: {persistent_state['credibility_scores']}")

    for i, item in enumerate(dataset):
        print(f"\n--- Processing Query {i + 1}/{len(dataset)} ---")
        print(f"Query: {item['question']}")

        # Prepare the input for this run, carrying over the persistent state
        run_input = {
            "query": item["question"],
            "ground_truth": item["answer"],
            **persistent_state
        }

        # Invoke the graph
        final_run_state = None
        for s in app.stream(run_input):
            # The final state is the one just before the END node
            if END in s:
                final_run_state = s[END]

        # Update the persistent state for the next run
        if final_run_state:
            updated_scores = final_run_state["credibility_scores"]
            persistent_state["credibility_scores"] = updated_scores
            print("\n--- End of Query Round ---")
            print(f"Final Answer Given: {final_run_state['final_answer']}")
            print(f"Ground Truth: {item['answer']}")
            print(f"Reward: {final_run_state['reward']}")
            print(f"Updated CrS for next round: { {k: round(v, 3) for k, v in updated_scores.items()} }")
        else:
            print("Error: Graph did not complete successfully.")
            break

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    run_experiment()


    """
    
    ### 如何运行这个项目

    1.  **环境设置**:
        * 确保您已经安装了 Python 3.8+。
        * 安装 Ollama 并拉取您想使用的模型，例如：
            ```bash
            ollama pull llama3
            ollama pull qwen:7b
            ```
        * 将上述所有代码文件按照目录结构保存好。
        * 在项目根目录创建一个名为 `.env` 的文件，并填入您的 DeepSeek API 密钥：
            ```
            DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
            ```
        * 安装所有依赖：
            ```bash
            pip install -r requirements.txt
            ```
    
    2.  **配置检查**:
        * 打开 `config.py` 文件。
        * 确认 `AGENT_TEAM_CONFIG` 中 `ollama` 模型的 `model_name` 与您在本地部署的Ollama模型名称一致。
        * 确认 `JUDGE_MODEL_CONFIG` 中的模型是您想使用的DeepSeek模型。
    
    3.  **运行实验**:
        * 在项目根目录下，运行主脚本：
            ```bash
            python main.py
            ```
    
    ### 代码实现要点说明
    
    * **状态传递**: `CrsGraphState` 是整个工作流的核心，它像一个数据管道，在不同的节点之间传递信息（如查询、CrS分数、历史记录等）。
    * **模块化智能体**: 将不同角色的智能体（忠实、对抗、评判）分离到不同的类中，每个类有自己独特的Prompt，这使得添加新角色或修改行为变得非常容易。
    * **CrS更新逻辑**: `update_crs` 节点严格按照论文中的公式 `NewCrS = OldCrS * (1 + eta * CSc * reward)` 来更新分数，并增加了边界处理（将分数限制在0.01到1.0之间）以防止分数变为0或无限大。
    * **简化的聚合**: `crs_aware_aggregation` 节点实现了一个简化的、基于CrS分数的加权随机选择。这模拟了论文中“质心法”的核心思想——高信誉的智能体有更大的概率决定最终答案。在实际应用中，您可以替换为更复杂的聚合逻辑，比如让另一个LLM来做聚合。
    * **迭代式实验**: `main.py` 中的循环模拟了论文中对一系列查询进行处理的过程。关键在于，上一轮查询更新后的 `credibility_scores` 会被用作下一轮查询的初始分数，这体现了框架的“学习”过程。
    
    这个实现为您提供了一个完整且可运行的框架，您可以基于此进行更深入的研究，例如：
    * 测试不同的通信拓扑。
    * 实现更复杂的聚合策略。
    * 在更大、更多样的数据集上进行测试。
    * 分析不同学习率 `eta` 对收敛速度的
    """