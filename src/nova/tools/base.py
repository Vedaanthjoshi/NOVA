from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel, Field

from nova.state import RiskLevel


class ToolResult(BaseModel):
    """Normalized output from any executed tool."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    success: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)


class ToolError(Exception):
    """Base exception for all tool operations."""
    pass


class UnknownToolError(ToolError):
    """Raised when an action requests a tool that is not in the approved registry."""
    pass


class ParameterValidationError(ToolError):
    """Raised when action parameters fail Pydantic validation against the tool schema."""
    pass


class BaseTool(ABC):
    """
    Abstract Base Class for all NOVA tools.
    Every tool must declare its name, documentation, risk level, and argument schema.
    """
    name: str
    description: str
    risk_level: RiskLevel
    args_schema: Type[BaseModel]

    @abstractmethod
    def _run(self, **kwargs: Any) -> ToolResult:
        """Core execution logic implemented by each tool subclass."""
        pass

    def run(self, **kwargs: Any) -> ToolResult:
        """Validates arguments against args_schema before invoking _run."""
        try:
            validated = self.args_schema.model_validate(kwargs)
        except Exception as e:
            raise ParameterValidationError(
                f"Parameter validation failed for tool '{self.name}': {str(e)}"
            ) from e

        return self._run(**validated.model_dump())

