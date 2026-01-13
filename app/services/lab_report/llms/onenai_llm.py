from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAILLM:
    
    def __init__(self):
        pass
    
    
    def get_llm_model(self):
        try:
            logger.info("OpenAI Model Initializing.....")
            
            self.llm = ChatOpenAI(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                api_key=OPENAI_API_KEY
            )
            return self.llm
        except Exception as e:
            raise ValueError(f"Error Occurred with Exception: {e}")