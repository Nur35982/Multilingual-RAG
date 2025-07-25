from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory
from typing import Dict, Any, List
from pydantic import BaseModel

class EnhancedConversationMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save conversation context to memory"""
        self.memory.save_context(inputs, outputs)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load conversation history"""
        return self.memory.load_memory_variables(inputs)

    def clear(self) -> None:
        """Clear conversation memory"""
        self.memory.clear()

    def get_recent_history(self, num_exchanges: int = 3) -> List[Dict[str, str]]:
        """Get recent conversation exchanges"""
        variables = self.memory.load_memory_variables({})
        history = variables.get("chat_history", [])
        return [{"role": msg.type, "content": msg.content} for msg in history[-num_exchanges*2:]]