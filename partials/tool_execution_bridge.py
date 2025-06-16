# Partials File: partials/tool_execution_bridge.py

import asyncio
import json
import uuid
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from .error_handling import MinionError

class ToolExecutionBridge:
    """
    Bridges MinionS protocol with Open WebUI's tool execution system.
    Handles the async communication between function calls and tool responses.
    """
    
    def __init__(self, valves, debug_mode: bool = False):
        self.valves = valves
        self.debug_mode = debug_mode
        self.pending_tool_calls = {}
        self.tool_results = {}
        self.max_wait_time = 30  # Maximum seconds to wait for tool response
    
    async def request_tool_execution(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Request tool execution from Open WebUI's pipeline.
        Returns a placeholder that will be replaced with actual results.
        
        Args:
            tool_name: Name of the tool to execute (e.g., "web_search")
            parameters: Tool parameters
            
        Returns:
            str: Tool call ID for tracking
        """
        # Generate unique ID for this tool call
        tool_call_id = str(uuid.uuid4())
        
        # Create tool call structure
        tool_call = {
            "id": tool_call_id,
            "name": tool_name,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Store in pending calls
        self.pending_tool_calls[tool_call_id] = tool_call
        
        if self.debug_mode:
            print(f"[ToolBridge] Requested tool execution: {tool_name} with ID: {tool_call_id}")
        
        return tool_call_id
    
    async def process_tool_response(self, tool_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the response from Open WebUI's tool execution.
        
        Args:
            tool_response: Response from tool execution
            
        Returns:
            Dict: Processed tool response
        """
        try:
            # Extract tool call ID if present
            tool_call_id = tool_response.get('tool_call_id')
            
            if not tool_call_id:
                # Try to match by tool name and parameters
                tool_name = tool_response.get('tool_name')
                if tool_name:
                    # Find matching pending call
                    for call_id, call in self.pending_tool_calls.items():
                        if call['name'] == tool_name and call['status'] == 'pending':
                            tool_call_id = call_id
                            break
            
            if tool_call_id and tool_call_id in self.pending_tool_calls:
                # Update call status
                self.pending_tool_calls[tool_call_id]['status'] = 'completed'
                self.pending_tool_calls[tool_call_id]['response'] = tool_response
                
                # Store result
                self.tool_results[tool_call_id] = {
                    'response': tool_response,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
                
                if self.debug_mode:
                    print(f"[ToolBridge] Processed tool response for ID: {tool_call_id}")
                
                return {
                    'tool_call_id': tool_call_id,
                    'success': True,
                    'data': tool_response
                }
            else:
                if self.debug_mode:
                    print(f"[ToolBridge] No matching tool call found for response")
                
                return {
                    'success': False,
                    'error': 'No matching tool call found',
                    'data': tool_response
                }
                
        except Exception as e:
            error_msg = f"Failed to process tool response: {str(e)}"
            if self.debug_mode:
                print(f"[ToolBridge] {error_msg}")
            
            return {
                'success': False,
                'error': error_msg
            }
    
    async def wait_for_tool_result(self, tool_call_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for a tool result to be available.
        
        Args:
            tool_call_id: ID of the tool call to wait for
            timeout: Maximum time to wait (uses max_wait_time if not specified)
            
        Returns:
            Dict: Tool result or timeout error
        """
        timeout = timeout or self.max_wait_time
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check if result is available
            if tool_call_id in self.tool_results:
                return self.tool_results[tool_call_id]
            
            # Check if call is still pending
            if tool_call_id in self.pending_tool_calls:
                call = self.pending_tool_calls[tool_call_id]
                if call['status'] == 'failed':
                    return {
                        'success': False,
                        'error': 'Tool execution failed'
                    }
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                if self.debug_mode:
                    print(f"[ToolBridge] Timeout waiting for tool result: {tool_call_id}")
                
                return {
                    'success': False,
                    'error': f'Timeout waiting for tool result after {timeout}s'
                }
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    def format_tool_call_for_pipeline(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Format a tool call in the way Open WebUI's pipeline expects.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            
        Returns:
            str: Formatted tool call
        """
        tool_call = {
            "name": tool_name,
            "parameters": parameters
        }
        
        # Use the __TOOL_CALL__ format that Open WebUI recognizes
        formatted = f"__TOOL_CALL__\n{json.dumps(tool_call, indent=2)}\n__TOOL_CALL__"
        
        if self.debug_mode:
            print(f"[ToolBridge] Formatted tool call:\n{formatted}")
        
        return formatted
    
    async def execute_tool_with_fallback(self, 
                                       tool_name: str, 
                                       parameters: Dict[str, Any],
                                       fallback_handler: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute a tool with fallback handling if execution fails.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            fallback_handler: Optional fallback function if tool execution fails
            
        Returns:
            Dict: Tool execution result or fallback result
        """
        try:
            # Request tool execution
            tool_call_id = await self.request_tool_execution(tool_name, parameters)
            
            # Wait for result
            result = await self.wait_for_tool_result(tool_call_id)
            
            if result.get('success'):
                return result['data']
            elif fallback_handler:
                if self.debug_mode:
                    print(f"[ToolBridge] Using fallback handler for {tool_name}")
                
                return await fallback_handler(parameters)
            else:
                raise MinionError(f"Tool execution failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            if fallback_handler:
                if self.debug_mode:
                    print(f"[ToolBridge] Exception in tool execution, using fallback: {str(e)}")
                
                return await fallback_handler(parameters)
            else:
                raise
    
    def get_pending_tools(self) -> List[Dict[str, Any]]:
        """Get list of pending tool calls."""
        return [
            call for call in self.pending_tool_calls.values()
            if call['status'] == 'pending'
        ]
    
    def clear_completed_tools(self) -> None:
        """Clear completed tool calls to free memory."""
        # Remove completed calls
        completed_ids = [
            call_id for call_id, call in self.pending_tool_calls.items()
            if call['status'] in ['completed', 'failed']
        ]
        
        for call_id in completed_ids:
            del self.pending_tool_calls[call_id]
            if call_id in self.tool_results:
                del self.tool_results[call_id]
        
        if self.debug_mode and completed_ids:
            print(f"[ToolBridge] Cleared {len(completed_ids)} completed tool calls")
    
    def inject_tool_call_into_message(self, message: str, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Inject a tool call into a message response.
        This allows the message to trigger Open WebUI's tool execution.
        
        Args:
            message: Original message
            tool_name: Name of the tool to call
            parameters: Tool parameters
            
        Returns:
            str: Message with embedded tool call
        """
        tool_call = self.format_tool_call_for_pipeline(tool_name, parameters)
        
        # Inject the tool call at the end of the message
        # Open WebUI will detect and execute it
        return f"{message}\n\n{tool_call}"
    
    async def handle_streaming_tool_execution(self, 
                                            tool_name: str,
                                            parameters: Dict[str, Any],
                                            stream_callback: Callable[[str], None]) -> Dict[str, Any]:
        """
        Handle tool execution with streaming updates.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            stream_callback: Callback for streaming updates
            
        Returns:
            Dict: Tool execution result
        """
        # Stream initial status
        await stream_callback(f"🔧 Executing {tool_name}...")
        
        try:
            # Execute tool
            result = await self.execute_tool_with_fallback(tool_name, parameters)
            
            # Stream success
            await stream_callback(f"✅ {tool_name} completed")
            
            return result
            
        except Exception as e:
            # Stream error
            await stream_callback(f"❌ {tool_name} failed: {str(e)}")
            raise