# Partials File: partials/task_visualizer.py

from typing import List, Dict, Optional, Any
from enum import Enum

class TaskStatus(Enum):
    """Status of individual tasks for visualization."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskType(Enum):
    """Types of tasks for different visualization styles."""
    DOCUMENT_ANALYSIS = "document_analysis"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"
    SYNTHESIS = "synthesis"

class TaskVisualizer:
    """
    Visual Task Decomposition UI using Mermaid diagrams.
    Shows the task decomposition process, execution flow, and real-time status updates.
    """
    
    def __init__(self, valves, debug_mode: bool = False):
        self.valves = valves
        self.debug_mode = debug_mode
        self.tasks = []
        self.task_relationships = []
        self.document_sources = []
        self.web_sources = []
    
    def is_visualization_enabled(self) -> bool:
        """Check if task visualization is enabled via valves."""
        return getattr(self.valves, 'show_task_visualization', True)
    
    def add_task(self, task_id: str, description: str, task_type: TaskType, 
                 status: TaskStatus = TaskStatus.PENDING, 
                 document_refs: List[str] = None,
                 web_query: str = None) -> None:
        """
        Add a task to the visualization.
        
        Args:
            task_id: Unique identifier for the task
            description: Human-readable task description
            task_type: Type of task (document_analysis, web_search, hybrid, synthesis)
            status: Current status of the task
            document_refs: List of document references for this task
            web_query: Web search query for this task
        """
        task = {
            "id": task_id,
            "description": description,
            "type": task_type,
            "status": status,
            "document_refs": document_refs or [],
            "web_query": web_query,
            "created_at": None,  # Could add timestamp
            "completed_at": None
        }
        
        self.tasks.append(task)
        
        if self.debug_mode:
            print(f"[Visualizer] Added task {task_id}: {description[:50]}...")
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update the status of a specific task."""
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = status
                if status == TaskStatus.COMPLETED:
                    task["completed_at"] = None  # Could add timestamp
                break
        
        if self.debug_mode:
            print(f"[Visualizer] Updated task {task_id} status to {status.value}")
    
    def add_task_relationship(self, parent_task_id: str, child_task_id: str, 
                            relationship_type: str = "depends_on") -> None:
        """
        Add a relationship between tasks.
        
        Args:
            parent_task_id: ID of the parent task
            child_task_id: ID of the child task
            relationship_type: Type of relationship (depends_on, feeds_into, etc.)
        """
        relationship = {
            "parent": parent_task_id,
            "child": child_task_id,
            "type": relationship_type
        }
        
        self.task_relationships.append(relationship)
    
    def add_document_source(self, doc_id: str, doc_name: str) -> None:
        """Add a document source to the visualization."""
        if doc_id not in [doc["id"] for doc in self.document_sources]:
            self.document_sources.append({
                "id": doc_id,
                "name": doc_name,
                "type": "document"
            })
    
    def add_web_source(self, query: str) -> None:
        """Add a web search source to the visualization."""
        web_source = {
            "id": f"web_{len(self.web_sources)}",
            "query": query,
            "type": "web_search"
        }
        self.web_sources.append(web_source)
    
    def generate_mermaid_diagram(self, include_status_colors: bool = True) -> str:
        """
        Generate a Mermaid diagram showing the task decomposition and execution flow.
        
        Args:
            include_status_colors: Whether to include status-based coloring
            
        Returns:
            str: Mermaid diagram syntax
        """
        if not self.is_visualization_enabled():
            return ""
        
        diagram_lines = [
            "```mermaid",
            "graph TD"
        ]
        
        # Add the user query as the root node
        diagram_lines.append("    Q[User Query] --> S[Supervisor Analysis]")
        
        # Add task nodes
        for task in self.tasks:
            task_id = task["id"]
            description = self._truncate_description(task["description"])
            task_type = task["type"]
            status = task["status"]
            
            # Create node with appropriate shape and label
            node_shape = self._get_node_shape(task_type)
            node_label = f"{task_id}[{description}]" if node_shape == "rect" else f"{task_id}({description})"
            
            # Add connection from supervisor to task
            diagram_lines.append(f"    S --> {node_label}")
            
            # Connect tasks to their data sources
            for doc_ref in task.get("document_refs", []):
                doc_id = self._sanitize_id(doc_ref)
                doc_name = self._truncate_description(doc_ref)
                diagram_lines.append(f"    {task_id} --> D{doc_id}[Document: {doc_name}]")
            
            if task.get("web_query"):
                web_id = f"W{task_id}"
                web_query = self._truncate_description(task["web_query"])
                diagram_lines.append(f"    {task_id} --> {web_id}[Web: {web_query}]")
        
        # Add synthesis node if we have multiple tasks
        if len(self.tasks) > 1:
            synthesis_inputs = " & ".join([task["id"] for task in self.tasks])
            diagram_lines.append(f"    {synthesis_inputs} --> F[Final Synthesis]")
        
        # Add status-based styling if enabled
        if include_status_colors:
            diagram_lines.extend(self._generate_status_styling())
        
        diagram_lines.append("```")
        
        return "\n".join(diagram_lines)
    
    def _get_node_shape(self, task_type: TaskType) -> str:
        """Get the appropriate node shape for a task type."""
        shape_mapping = {
            TaskType.DOCUMENT_ANALYSIS: "rect",
            TaskType.WEB_SEARCH: "round",
            TaskType.HYBRID: "diamond",
            TaskType.SYNTHESIS: "circle"
        }
        return shape_mapping.get(task_type, "rect")
    
    def _truncate_description(self, description: str, max_length: int = 30) -> str:
        """Truncate description for diagram readability."""
        if len(description) <= max_length:
            return description
        
        return description[:max_length-3] + "..."
    
    def _sanitize_id(self, text: str) -> str:
        """Sanitize text to be used as a node ID in Mermaid."""
        # Remove special characters and spaces, replace with underscores
        import re
        sanitized = re.sub(r'[^\w]', '_', text)
        return sanitized[:20]  # Limit length
    
    def _generate_status_styling(self) -> List[str]:
        """Generate CSS styling for task status colors."""
        styling_lines = []
        
        # Define color schemes
        color_mapping = {
            TaskStatus.PENDING: "#f9f9f9",      # Light gray
            TaskStatus.RUNNING: "#fff3cd",      # Light yellow
            TaskStatus.COMPLETED: "#d4edda",    # Light green
            TaskStatus.FAILED: "#f8d7da"        # Light red
        }
        
        # Add styling for each task based on status
        for task in self.tasks:
            task_id = task["id"]
            status = task["status"]
            color = color_mapping.get(status, "#f9f9f9")
            
            styling_lines.append(f"    style {task_id} fill:{color}")
        
        return styling_lines
    
    def generate_execution_timeline(self) -> str:
        """
        Generate a timeline view of task execution.
        
        Returns:
            str: Timeline diagram using Mermaid Gantt chart
        """
        if not self.is_visualization_enabled() or not self.tasks:
            return ""
        
        timeline_lines = [
            "```mermaid",
            "gantt",
            "    title Task Execution Timeline",
            "    dateFormat X",
            "    axisFormat %s"
        ]
        
        # Add sections for different types of tasks
        doc_tasks = [t for t in self.tasks if t["type"] == TaskType.DOCUMENT_ANALYSIS]
        web_tasks = [t for t in self.tasks if t["type"] == TaskType.WEB_SEARCH]
        hybrid_tasks = [t for t in self.tasks if t["type"] == TaskType.HYBRID]
        
        if doc_tasks:
            timeline_lines.append("    section Document Analysis")
            for i, task in enumerate(doc_tasks):
                task_name = self._truncate_description(task["description"], 20)
                status = task["status"].value
                timeline_lines.append(f"    {task_name} ({status}) : {i}, {i+1}")
        
        if web_tasks:
            timeline_lines.append("    section Web Search")
            for i, task in enumerate(web_tasks):
                task_name = self._truncate_description(task["description"], 20)
                status = task["status"].value
                timeline_lines.append(f"    {task_name} ({status}) : {i}, {i+1}")
        
        if hybrid_tasks:
            timeline_lines.append("    section Hybrid Tasks")
            for i, task in enumerate(hybrid_tasks):
                task_name = self._truncate_description(task["description"], 20)
                status = task["status"].value
                timeline_lines.append(f"    {task_name} ({status}) : {i}, {i+1}")
        
        timeline_lines.append("```")
        
        return "\n".join(timeline_lines)
    
    def generate_task_summary_table(self) -> str:
        """
        Generate a markdown table summarizing all tasks.
        
        Returns:
            str: Markdown table with task information
        """
        if not self.tasks:
            return ""
        
        table_lines = [
            "| Task ID | Type | Status | Description | Sources |",
            "|---------|------|--------|-------------|---------|"
        ]
        
        for task in self.tasks:
            task_id = task["id"]
            task_type = task["type"].value.replace("_", " ").title()
            status = task["status"].value.title()
            description = self._truncate_description(task["description"], 40)
            
            # Build sources column
            sources = []
            if task.get("document_refs"):
                sources.extend([f"Doc: {ref}" for ref in task["document_refs"][:2]])
            if task.get("web_query"):
                sources.append(f"Web: {task['web_query'][:20]}...")
            
            sources_str = ", ".join(sources) if sources else "None"
            
            table_lines.append(f"| {task_id} | {task_type} | {status} | {description} | {sources_str} |")
        
        return "\n".join(table_lines)
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current visualization state.
        
        Returns:
            Dict: Summary information about tasks and visualization
        """
        status_counts = {}
        type_counts = {}
        
        for task in self.tasks:
            status = task["status"].value
            task_type = task["type"].value
            
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
            "document_sources": len(self.document_sources),
            "web_sources": len(self.web_sources),
            "visualization_enabled": self.is_visualization_enabled()
        }
    
    def clear_visualization(self) -> None:
        """Clear all visualization data for a new session."""
        self.tasks.clear()
        self.task_relationships.clear()
        self.document_sources.clear()
        self.web_sources.clear()
        
        if self.debug_mode:
            print("[Visualizer] Visualization data cleared")
    
    def create_quick_visualization(self, tasks: List[Dict[str, Any]], 
                                 document_refs: List[str] = None,
                                 web_queries: List[str] = None) -> str:
        """
        Create a quick visualization from a list of tasks.
        
        Args:
            tasks: List of task dictionaries with id, description, type
            document_refs: List of document references
            web_queries: List of web search queries
            
        Returns:
            str: Complete visualization with diagram and summary
        """
        self.clear_visualization()
        
        # Add tasks
        for i, task in enumerate(tasks):
            task_id = task.get("id", f"task_{i+1}")
            description = task.get("description", f"Task {i+1}")
            task_type_str = task.get("type", "document_analysis")
            
            # Convert string to TaskType enum
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.DOCUMENT_ANALYSIS
            
            self.add_task(task_id, description, task_type)
        
        # Add document sources
        if document_refs:
            for doc_ref in document_refs:
                self.add_document_source(doc_ref, doc_ref)
        
        # Add web sources
        if web_queries:
            for query in web_queries:
                self.add_web_source(query)
        
        # Generate complete visualization
        visualization_parts = []
        
        if self.is_visualization_enabled():
            visualization_parts.append("## Task Decomposition Visualization")
            visualization_parts.append(self.generate_mermaid_diagram())
            visualization_parts.append("")
            visualization_parts.append("### Task Summary")
            visualization_parts.append(self.generate_task_summary_table())
        
        return "\n".join(visualization_parts)