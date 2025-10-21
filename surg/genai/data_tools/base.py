"""
Base Data Tool Architecture
==========================

This module defines the base architecture for all GenAI data tools with strict JSON
response formatting and consistent interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, asdict
from enum import Enum
import json
import jsonschema
from datetime import datetime
import pandas as pd
import logging


class DataToolType(Enum):
    """Types of data tools available."""
    QUALITY_ANALYSIS = "quality_analysis"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    ENRICHMENT = "enrichment"


class DataToolStatus(Enum):
    """Status of data tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL_SUCCESS = "partial_success"


@dataclass
class DataToolMetadata:
    """Metadata for data tool responses."""
    tool_name: str
    tool_version: str
    execution_time_ms: float
    timestamp: str
    data_size: int
    genai_model_used: Optional[str] = None
    confidence_score: Optional[float] = None


@dataclass
class DataToolResponse:
    """
    Standardized response format for all data tools.
    
    This ensures consistent JSON structure across all GenAI data operations.
    """
    status: DataToolStatus
    data: Dict[str, Any]
    metadata: DataToolMetadata
    errors: List[str]
    warnings: List[str]
    
    def to_json(self) -> str:
        """Convert response to JSON string."""
        response_dict = {
            "status": self.status.value,
            "data": self.data,
            "metadata": asdict(self.metadata),
            "errors": self.errors,
            "warnings": self.warnings
        }
        return json.dumps(response_dict, indent=2, default=str)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "status": self.status.value,
            "data": self.data,
            "metadata": asdict(self.metadata),
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DataToolResponse':
        """Create response from JSON string."""
        data = json.loads(json_str)
        return cls(
            status=DataToolStatus(data["status"]),
            data=data["data"],
            metadata=DataToolMetadata(**data["metadata"]),
            errors=data["errors"],
            warnings=data["warnings"]
        )


class BaseDataTool(ABC):
    """
    Abstract base class for all GenAI data tools.
    
    Provides consistent interface and JSON response formatting.
    """
    
    def __init__(
        self,
        name: str,
        version: str,
        tool_type: DataToolType,
        genai_provider: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.name = name
        self.version = version
        self.tool_type = tool_type
        self.genai_provider = genai_provider
        self.model_name = model_name
        self.logger = logging.getLogger(f"surg.genai.data_tools.{name}")
        
        # JSON schemas for request/response validation
        self.request_schema = self._get_request_schema()
        self.response_schema = self._get_response_schema()
    
    @abstractmethod
    def _get_request_schema(self) -> Dict[str, Any]:
        """Return JSON schema for request validation."""
        pass
    
    @abstractmethod
    def _get_response_schema(self) -> Dict[str, Any]:
        """Return JSON schema for response validation."""
        pass
    
    @abstractmethod
    def _execute_tool(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the core tool logic."""
        pass
    
    def validate_request(self, request: Dict[str, Any]) -> None:
        """Validate request against schema."""
        try:
            jsonschema.validate(request, self.request_schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid request format: {e.message}")
    
    def validate_response(self, response: Dict[str, Any]) -> None:
        """Validate response against schema."""
        try:
            jsonschema.validate(response, self.response_schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid response format: {e.message}")
    
    def execute(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> DataToolResponse:
        """
        Execute the data tool with standardized response formatting.
        
        Args:
            data: Input DataFrame
            params: Tool-specific parameters
            
        Returns:
            DataToolResponse with strict JSON formatting
        """
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            # Validate input parameters
            if params:
                self.validate_request(params)
            
            # Execute tool logic
            result_data = self._execute_tool(data, params or {})
            
            # Validate response format
            self.validate_response(result_data)
            
            status = DataToolStatus.SUCCESS
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            errors.append(str(e))
            result_data = {}
            status = DataToolStatus.ERROR
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Create metadata
        metadata = DataToolMetadata(
            tool_name=self.name,
            tool_version=self.version,
            execution_time_ms=execution_time_ms,
            timestamp=end_time.isoformat(),
            data_size=len(data),
            genai_model_used=self.model_name,
            confidence_score=result_data.get("confidence_score")
        )
        
        return DataToolResponse(
            status=status,
            data=result_data,
            metadata=metadata,
            errors=errors,
            warnings=warnings
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get tool information."""
        return {
            "name": self.name,
            "version": self.version,
            "type": self.tool_type.value,
            "genai_provider": self.genai_provider,
            "model_name": self.model_name,
            "request_schema": self.request_schema,
            "response_schema": self.response_schema
        }


class DataToolRegistry:
    """
    Registry for managing available data tools.
    
    Provides discovery, registration, and instantiation of tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, Type[BaseDataTool]] = {}
        self._instances: Dict[str, BaseDataTool] = {}
    
    def register_tool(self, tool_class: Type[BaseDataTool]) -> None:
        """Register a new data tool."""
        tool_name = tool_class.__name__
        self._tools[tool_name] = tool_class
        self.logger.info(f"Registered tool: {tool_name}")
    
    def get_tool(self, tool_name: str, **kwargs) -> BaseDataTool:
        """Get tool instance by name."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        # Create instance if not cached
        instance_key = f"{tool_name}_{hash(str(kwargs))}"
        if instance_key not in self._instances:
            tool_class = self._tools[tool_name]
            self._instances[instance_key] = tool_class(**kwargs)
        
        return self._instances[instance_key]
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        tools_info = []
        for tool_name, tool_class in self._tools.items():
            # Create temporary instance to get info
            try:
                temp_instance = tool_class()
                tools_info.append(temp_instance.get_info())
            except Exception as e:
                tools_info.append({
                    "name": tool_name,
                    "error": f"Failed to get info: {str(e)}"
                })
        
        return tools_info
    
    def get_tools_by_type(self, tool_type: DataToolType) -> List[str]:
        """Get tools by type."""
        matching_tools = []
        for tool_name, tool_class in self._tools.items():
            try:
                temp_instance = tool_class()
                if temp_instance.tool_type == tool_type:
                    matching_tools.append(tool_name)
            except Exception:
                pass  # Skip tools that can't be instantiated
        
        return matching_tools


# Global registry instance
registry = DataToolRegistry()


def register_tool(tool_class: Type[BaseDataTool]) -> Type[BaseDataTool]:
    """Decorator for registering data tools."""
    registry.register_tool(tool_class)
    return tool_class


# JSON Schema definitions for common data types
COMMON_SCHEMAS = {
    "data_quality_metrics": {
        "type": "object",
        "properties": {
            "completeness": {"type": "number", "minimum": 0, "maximum": 1},
            "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
            "consistency": {"type": "number", "minimum": 0, "maximum": 1},
            "validity": {"type": "number", "minimum": 0, "maximum": 1},
            "overall_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["completeness", "accuracy", "consistency", "validity", "overall_score"]
    },
    
    "column_profile": {
        "type": "object",
        "properties": {
            "column_name": {"type": "string"},
            "data_type": {"type": "string"},
            "null_count": {"type": "integer"},
            "null_percentage": {"type": "number"},
            "unique_count": {"type": "integer"},
            "unique_percentage": {"type": "number"},
            "min_value": {"type": ["string", "number", "null"]},
            "max_value": {"type": ["string", "number", "null"]},
            "mean_value": {"type": ["number", "null"]},
            "std_dev": {"type": ["number", "null"]},
            "quality_issues": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["column_name", "data_type", "null_count", "null_percentage"]
    },
    
    "anomaly_detection": {
        "type": "object",
        "properties": {
            "anomalies_detected": {"type": "integer"},
            "anomaly_score": {"type": "number", "minimum": 0, "maximum": 1},
            "anomalous_rows": {
                "type": "array",
                "items": {"type": "integer"}
            },
            "anomaly_reasons": {
                "type": "array",
                "items": {"type": "string"}
            },
            "recommended_actions": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["anomalies_detected", "anomaly_score"]
    }
}


# Response validation helpers
def create_response_schema(data_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Create a complete response schema with the given data schema."""
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error", "warning", "partial_success"]},
            "data": data_schema,
            "metadata": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "tool_version": {"type": "string"},
                    "execution_time_ms": {"type": "number"},
                    "timestamp": {"type": "string"},
                    "data_size": {"type": "integer"},
                    "genai_model_used": {"type": ["string", "null"]},
                    "confidence_score": {"type": ["number", "null"]}
                },
                "required": ["tool_name", "tool_version", "execution_time_ms", "timestamp", "data_size"]
            },
            "errors": {
                "type": "array",
                "items": {"type": "string"}
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["status", "data", "metadata", "errors", "warnings"]
    }