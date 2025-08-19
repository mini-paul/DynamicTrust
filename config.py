# config.py
# ==============================================================================
# File: config.py
# Description: 配置文件，用于管理API密钥和模型设置
# 注意：请在项目根目录下创建一个 .env 文件，并填入您的DeepSeek API Key
# .env 文件内容示例:
# DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
# ==============================================================================


import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

OLLAMA_BASE_URL = "http://10.10.206.138:11434"

# --- Model Configurations ---

# 强大的评判者模型配置
# JUDGE_MODEL_CONFIG = {
#     "type": "deepseek",
#     "model_name": "deepseek-chat",
#     "temperature": 0.7,
# }

# 智能体团队配置
# 您可以根据本地Ollama部署的模型进行修改
# AGENT_TEAM_CONFIG = {
#     "faithful_agents": [
#         {
#             "id": "Faithful-Agent-1",
#             "config": {
#                 "type": "ollama",
#                 "model_name": "qwen3:8b",
#                 "temperature": 0.7,
#             }
#         },
#         {
#             "id": "Faithful-Agent-2",
#             "config": {
#                 "type": "ollama",
#                 "model_name": "qwen3:8b",
#                 "temperature": 0.7,
#             }
#         }
#     ],
#     "adversarial_agents": [
#         {
#             "id": "Adversarial-Agent-1",
#             "config": {
#                 "type": "ollama",
#                 "model_name": "qwen3:8b",
#                 "temperature": 0.7,
#             }
#         },
#         {
#             "id": "Adversarial-Agent-2",
#             "config": {
#                 "type": "ollama",
#                 "model_name": "qwen3:8b",
#                 "temperature": 0.7,
#             }
#         },
#         {
#             "id": "Adversarial-Agent-3",
#             "config": {
#                 "type": "ollama",
#                 "model_name": "qwen3:8b",
#                 "temperature": 0.7,
#             }
#         }
#     ]
# }

JUDGE_MODEL_CONFIG = {
    "type": "ollama",
    "model_name": "llama3:8b",
    "temperature": 0.7,
}

AGENT_TEAM_CONFIG = {
    "faithful_agents": [
        {
            "id": "Faithful-Agent-1",
            "config": {
                "type": "ollama",
                "model_name": "qwen3:8b",
                "temperature": 0.7,
            }
        }
    ],
    "adversarial_agents": [
        {
            "id": "Adversarial-Agent-1",
            "config": {
                "type": "ollama",
                "model_name": "qwen3:8b",
                "temperature": 0.7,
            }
        }
    ]
}

# --- CrS Framework Settings ---
INITIAL_CRS = 0.5
LEARNING_RATE = 0.1 # 学习率 eta
MAX_ITERATIONS = 3 # 智能体内部协作的最大轮次