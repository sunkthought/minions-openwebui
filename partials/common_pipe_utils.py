from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# A placeholder for the actual Valves class that will be defined in specific pipe implementations
# Using Any for now, but could be a Protocol or a more specific BaseModel if common fields are known.
ValvesType = Any 

class PipeBase:
    valves: Optional[ValvesType] # Subclasses will define and initialize this with their specific Valves class
    name: str # Subclasses should set this
    id: str # Subclasses should set this for the pipe ID

    def __init__(self, pipe_name: str = "{PIPE_NAME}", pipe_id: str = "{PIPE_ID}"):
        """
        Base constructor for Pipe classes.
        Subclasses are expected to initialize self.valves with their specific Valves model instance.
        e.g., self.valves = self.Valves()
        """
        self.valves = None # Must be initialized by the subclass with its specific ValvesModel()
        self.name = pipe_name
        self.id = pipe_id # Store the pipe_id, useful for the pipes method

    def pipes(self) -> List[Dict[str, Any]]:
        """
        Defines the available models for this pipe.
        Relies on self.valves being initialized by the subclass.
        Relies on self.id being set.
        """
        if not self.valves:
            # This case should ideally not be reached if subclasses initialize valves correctly.
            # However, it's a safeguard.
            return [
                {
                    "id": self.id,
                    "name": f" (Configuration Error: Valves not initialized)",
                    "error": "Valves not initialized in pipe."
                }
            ]
        
        # These attributes are expected to be on the specific Valves model
        # defined and initialized in the subclass.
        local_model_name = getattr(self.valves, 'local_model', '{LOCAL_MODEL}')
        remote_model_name = getattr(self.valves, 'remote_model', '{REMOTE_MODEL}')

        return [
            {
                "id": self.id, # Use the stored pipe_id
                "name": f" ({local_model_name} + {remote_model_name})",
            }
        ]

    async def pipe_function(self, body: Dict[str, Any], **kwargs) -> str:
        """
        Placeholder for the main pipe execution logic.
        Subclasses must implement this method.
        The `**kwargs` can be used to catch specific arguments like 
        __user__, __request__, __files__, __pipe_id__ if needed, or they can be
        explicitly listed if always present.
        """
        raise NotImplementedError("Subclasses must implement the 'pipe_function' method.")

# Example of how a specific Valves model would look (defined in the actual pipe file, not here):
# class SpecificValves(BaseModel):
#     local_model: str = "default_local"
#     remote_model: str = "default_remote"
#     # ... other specific valves
#
# Example of how a specific Pipe would use PipeBase (defined in the actual pipe file):
# class SpecificPipe(PipeBase):
#     class Valves(SpecificValves): # Or just use SpecificValves directly
#         pass
#
#     def __init__(self):
#         super().__init__(pipe_name="My Specific Pipe", pipe_id="my-specific-pipe")
#         self.valves = self.Valves() # Initialize with its own Valves
#
#     async def pipe_function(self, body: Dict[str, Any], **kwargs) -> str:
#         # Actual implementation
#         return f"Processing with {self.valves.local_model} and {self.valves.remote_model}"

# To make it runnable, the main entry point for OpenWebUI is typically a top-level `pipe` async function
# or the `Pipe` class itself is instantiated and its `pipe` method is called.
# If the `Pipe` class's `pipe` method is the entry point, it would be named `pipe` in `PipeBase`
# and then overridden. Let's rename `pipe_function` to `pipe` to match OpenWebUI's expectation.

    async def pipe(
        self,
        body: Dict[str, Any],
        # These are typically injected by OpenWebUI
        # __user__: Dict[str, Any], 
        # __request__: Any, # fastapi.Request
        # __files__: List[Dict[str, Any]] = [],
        # __pipe_id__: str = "",
        **kwargs # Catch all other potential OpenWebUI arguments
    ) -> str:
        """
        Main pipe execution logic. Subclasses must implement this.
        This method is named 'pipe' to match the expected entry point for OpenWebUI.
        """
        raise NotImplementedError("Subclasses must implement the 'pipe' method.")

# Remove the pipe_function as 'pipe' is the standard name
del PipeBase.pipe_function
