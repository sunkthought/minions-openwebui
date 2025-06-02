# Partials File: partials/common_pipe_utils.py
from typing import List, Dict, Any # Added for type hints

class PipeBase:
    def __init__(self, name: str = "Base Pipe", id: str = "base-pipe"):
        """Base constructor for Pipe classes."""
        self.valves = None  # Must be initialized by the subclass
        self.name = name
        self.id = id

    def pipes(self) -> List[Dict[str, Any]]:
        """Defines the available models for this pipe."""
        if not self.valves:
            return [
                {
                    "id": self.id,
                    "name": f" (Configuration Error: Valves not initialized)",
                    "error": "Valves not initialized in pipe."
                }
            ]
        
        local_model_name = getattr(self.valves, 'local_model', 'local_model')
        remote_model_name = getattr(self.valves, 'remote_model', 'remote_model')

        return [
            {
                "id": self.id,
                "name": f" ({local_model_name} + {remote_model_name})",
            }
        ]

    async def pipe(self, body: Dict[str, Any], **kwargs) -> str:
        """Main pipe execution logic. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement the 'pipe' method.")