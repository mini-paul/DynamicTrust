# ==============================================================================
# File: utils/llm_clients.py
# Description: 封装用于与DeepSeek和Ollama模型交互的客户端
# ==============================================================================

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,OLLAMA_BASE_URL

def get_llm_client(config: dict):
    """
    根据配置获取LLM客户端实例
    """
    model_type = config.get("type")
    model_name = config.get("model_name")
    temperature = config.get("temperature", 0.7)

    if model_type == "deepseek":
        return ChatOpenAI(
            model=model_name,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=temperature,
        )
    elif model_type == "ollama":
        return ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")