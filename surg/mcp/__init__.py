"""
Model Context Protocol (MCP) Integration Layer

This module implements the Model Context Protocol specification for SURG,
enabling seamless integration between AI models and external tools/resources.

The MCP layer provides:
- Standardized protocol for AI-tool communication
- Resource discovery and management
- Tool execution coordination
- Security and access control
- Session and state management

Key Components:
- ProtocolEngine: Core MCP protocol implementation
- ToolRegistry: Central tool discovery and management
- ResourceManager: Resource access and lifecycle management
- SecurityManager: Authentication and authorization
- SessionManager: Session state and context management

Protocol Features:
1. Resource Management:
   - Dataset resource discovery and access
   - Model resource registration and lifecycle
   - External API and service integration
   - File system and storage resource management

2. Tool Execution:
   - Tool invocation and parameter validation
   - Execution environment management
   - Result collection and standardization
   - Error handling and recovery

3. Communication Protocol:
   - JSON-RPC based message exchange
   - Request routing and response handling
   - Streaming and batch operations
   - Progress tracking and notifications

4. Security and Control:
   - Client authentication and authorization
   - Resource access permissions
   - Rate limiting and quotas
   - Audit logging and monitoring

Integration Benefits:
- Standardized AI-tool integration
- Extensible tool ecosystem
- Secure resource access
- Scalable architecture
- Cross-platform compatibility

This MCP implementation enables SURG to leverage external tools and resources
while maintaining security, performance, and reliability standards.

Legacy Tool System:
The original tool-based architecture remains available for:
- Data preprocessing tools (cleaning, transformation, validation)
- Feature engineering tools (selection, creation, encoding)
- Algorithm implementations (training, prediction, evaluation)
- Visualization tools (plotting, reporting, dashboards)
- Analysis tools (statistics, pattern detection, bias analysis)
- Optimization tools (hyperparameter tuning, architecture search)
- Deployment tools (model serving, monitoring, scaling)
"""

from .protocol_engine import ProtocolEngine
from .tool_registry import ToolRegistry

__all__ = [
    'ProtocolEngine',
    'ToolRegistry'
]