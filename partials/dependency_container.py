# Partials File: partials/dependency_container.py
from typing import Any, Callable, Dict, Optional, Type
import inspect
from dataclasses import dataclass

# Note: TypeVar not used in practice, simplified for compatibility
T = Any

@dataclass
class DependencyRegistration:
    """Represents a dependency registration."""
    factory: Callable
    singleton: bool = True
    dependencies: Optional[Dict[str, str]] = None


class DependencyContainer:
    """
    IoC container for managing dependencies across both protocols.
    Provides singleton instances and lazy initialization.
    """
    
    def __init__(self, valves: Any, debug_mode: bool = False):
        self.valves = valves
        self.debug_mode = debug_mode
        self._instances: Dict[str, Any] = {}
        self._registrations: Dict[str, DependencyRegistration] = {}
        
        # Register core dependencies
        self._register_core_dependencies()
        
    def _register_core_dependencies(self):
        """Register core dependencies that are always available."""
        # Lazy imports to avoid circular dependencies
        self.register_factory(
            'web_search', 
            lambda: self._create_web_search_integration(),
            singleton=True
        )
        
        self.register_factory(
            'tool_bridge',
            lambda: self._create_tool_execution_bridge(),
            singleton=True
        )
        
        self.register_factory(
            'streaming_manager',
            lambda: self._create_streaming_manager(),
            singleton=True
        )
        
        self.register_factory(
            'citation_manager',
            lambda: self._create_citation_manager(),
            singleton=True
        )
        
        self.register_factory(
            'rag_retriever',
            lambda: self._create_rag_retriever(),
            singleton=True
        )
        
        self.register_factory(
            'metrics_aggregator',
            lambda: self._create_metrics_aggregator(),
            singleton=True
        )
        
        self.register_factory(
            'unified_web_search_handler',
            lambda: self._create_unified_web_search_handler(),
            singleton=True
        )
        
    def register_factory(self, name: str, factory: Callable, singleton: bool = True, dependencies: Optional[Dict[str, str]] = None):
        """Register a factory function for creating instances."""
        self._registrations[name] = DependencyRegistration(
            factory=factory,
            singleton=singleton,
            dependencies=dependencies or {}
        )
        
    def get(self, name: str) -> Any:
        """Get or create an instance by name."""
        if name not in self._registrations:
            raise ValueError(f"Dependency '{name}' not registered")
            
        registration = self._registrations[name]
        
        # Return cached instance if singleton
        if registration.singleton and name in self._instances:
            return self._instances[name]
            
        # Create new instance
        instance = self._create_instance(name, registration)
        
        # Cache if singleton
        if registration.singleton:
            self._instances[name] = instance
            
        return instance
        
    def _create_instance(self, name: str, registration: DependencyRegistration) -> Any:
        """Create an instance using the factory."""
        try:
            # Check if factory needs dependency injection
            signature = inspect.signature(registration.factory)
            
            if len(signature.parameters) == 0:
                # No parameters needed
                return registration.factory()
            else:
                # Inject dependencies based on registration
                kwargs = {}
                for param_name, dependency_name in registration.dependencies.items():
                    kwargs[param_name] = self.get(dependency_name)
                
                return registration.factory(**kwargs)
                
        except Exception as e:
            if self.debug_mode:
                print(f"DEBUG [DependencyContainer]: Failed to create '{name}': {str(e)}")
            raise RuntimeError(f"Failed to create dependency '{name}': {str(e)}")
    
    def _create_web_search_integration(self):
        """Create WebSearchIntegration instance."""
        from .web_search_integration import WebSearchIntegration
        return WebSearchIntegration(self.valves, self.debug_mode)
        
    def _create_tool_execution_bridge(self):
        """Create ToolExecutionBridge instance."""
        from .tool_execution_bridge import ToolExecutionBridge
        return ToolExecutionBridge(self.valves, self.debug_mode)
        
    def _create_streaming_manager(self):
        """Create StreamingResponseManager instance."""
        from .streaming_support import StreamingResponseManager
        return StreamingResponseManager(self.valves, self.debug_mode)
        
    def _create_citation_manager(self):
        """Create CitationManager instance."""
        from .citation_manager import CitationManager
        return CitationManager(self.valves, self.debug_mode)
        
    def _create_rag_retriever(self):
        """Create RAGRetriever instance."""
        from .rag_retriever import RAGRetriever
        return RAGRetriever(self.valves, self.debug_mode)
        
    def _create_metrics_aggregator(self):
        """Create MetricsAggregator instance."""
        from .metrics_aggregator import MetricsAggregator
        protocol_name = getattr(self.valves, 'protocol_name', 'unknown')
        return MetricsAggregator(protocol_name, self.debug_mode)
        
    def _create_unified_web_search_handler(self):
        """Create UnifiedWebSearchHandler instance."""
        from .unified_web_search_handler import UnifiedWebSearchHandler
        return UnifiedWebSearchHandler(
            web_search=self.get('web_search'),
            tool_bridge=self.get('tool_bridge'),
            citation_manager=self.get('citation_manager'),
            streaming_manager=self.get('streaming_manager')
        )
    
    # Typed getters for better IDE support and type safety
    def get_web_search(self) -> Any:
        """Get WebSearchIntegration instance."""
        return self.get('web_search')
        
    def get_tool_bridge(self) -> Any:
        """Get ToolExecutionBridge instance."""
        return self.get('tool_bridge')
        
    def get_streaming_manager(self) -> Any:
        """Get StreamingResponseManager instance."""
        return self.get('streaming_manager')
        
    def get_citation_manager(self) -> Any:
        """Get CitationManager instance."""
        return self.get('citation_manager')
        
    def get_rag_retriever(self) -> Any:
        """Get RAGRetriever instance."""
        return self.get('rag_retriever')
        
    def get_metrics_aggregator(self) -> Any:
        """Get MetricsAggregator instance."""
        return self.get('metrics_aggregator')
        
    def get_unified_web_search_handler(self) -> Any:
        """Get UnifiedWebSearchHandler instance."""
        return self.get('unified_web_search_handler')
        
    def register_protocol_specific_dependencies(self, protocol_name: str):
        """Register protocol-specific dependencies."""
        if protocol_name.lower() == 'minion':
            self._register_minion_dependencies()
        elif protocol_name.lower() == 'minions':
            self._register_minions_dependencies()
            
    def _register_minion_dependencies(self):
        """Register Minion-specific dependencies."""
        self.register_factory(
            'conversation_state',
            lambda: self._create_minion_conversation_state(),
            singleton=False  # New instance for each conversation
        )
        
        self.register_factory(
            'question_deduplicator',
            lambda: self._create_question_deduplicator(),
            singleton=True
        )
        
        self.register_factory(
            'flow_controller',
            lambda: self._create_flow_controller(),
            singleton=True
        )
        
        self.register_factory(
            'answer_validator',
            lambda: self._create_answer_validator(),
            singleton=True
        )
        
    def _register_minions_dependencies(self):
        """Register MinionS-specific dependencies."""
        self.register_factory(
            'task_visualizer',
            lambda: self._create_task_visualizer(),
            singleton=True
        )
        
        self.register_factory(
            'decomposition_logic',
            lambda: self._create_decomposition_logic(),
            singleton=True
        )
        
        self.register_factory(
            'scaling_strategies',
            lambda: self._create_scaling_strategies(),
            singleton=True
        )
        
        self.register_factory(
            'adaptive_rounds',
            lambda: self._create_adaptive_rounds(),
            singleton=True
        )
        
    def _create_minion_conversation_state(self):
        """Create ConversationState instance for Minion."""
        from .minion_models import ConversationState
        return ConversationState()
        
    def _create_question_deduplicator(self):
        """Create QuestionDeduplicator instance."""
        from .minion_models import QuestionDeduplicator
        threshold = getattr(self.valves, 'deduplication_threshold', 0.8)
        return QuestionDeduplicator(threshold)
        
    def _create_flow_controller(self):
        """Create ConversationFlowController instance."""
        from .minion_models import ConversationFlowController
        return ConversationFlowController()
        
    def _create_answer_validator(self):
        """Create AnswerValidator instance."""
        from .minion_models import AnswerValidator
        return AnswerValidator()
        
    def _create_task_visualizer(self):
        """Create TaskVisualizer instance."""
        from .task_visualizer import TaskVisualizer
        return TaskVisualizer(self.valves, self.debug_mode)
        
    def _create_decomposition_logic(self):
        """Create decomposition logic instance."""
        # This would be refactored from existing logic
        pass
        
    def _create_scaling_strategies(self):
        """Create scaling strategies instance."""
        # This would be refactored from existing logic
        pass
        
    def _create_adaptive_rounds(self):
        """Create adaptive rounds instance."""
        # This would be refactored from existing logic
        pass
        
    def dispose(self):
        """Clean up resources."""
        for instance in self._instances.values():
            if hasattr(instance, 'dispose'):
                try:
                    instance.dispose()
                except Exception as e:
                    if self.debug_mode:
                        print(f"DEBUG [DependencyContainer]: Error disposing instance: {str(e)}")
        
        self._instances.clear()
        self._registrations.clear()