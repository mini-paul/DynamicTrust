# llm_factory.py
import os
from langchain_deepseek import ChatDeepSeek
from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv

# Load environment variables from.env file
load_dotenv()

def create_llm_backend(config: dict):
    """
    Factory function to create an LLM backend based on a configuration dictionary.
    This allows for flexible switching between different LLM providers.
    """
    provider = config.get("provider")
    model_name = config.get("model_name")
    temperature = config.get("temperature", 0.7)

    if provider == "deepseek":
        return ChatDeepSeek(
            model=model_name,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=temperature
        )
    elif provider == "ollama":
        return ChatOllama(
            model=model_name,
            base_url=config.get("base_url", "http://localhost:11434"),
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")