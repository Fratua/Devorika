"""
Base tool classes and tool registry
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json


class Tool(ABC):
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool"
    parameters: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [tool.to_dict() for tool in self.tools.values()]

    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        return tool.execute(**kwargs)
