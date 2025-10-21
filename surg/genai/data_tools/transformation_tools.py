"""
Data Transformation Tools
========================

GenAI-powered tools for intelligent data transformation including schema mapping,
data cleaning, normalization, and enrichment with AI-driven suggestions.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re
import json

from .base import (
    BaseDataTool, 
    DataToolType, 
    register_tool,
    create_response_schema
)


@register_tool
class SchemaMapper(BaseDataTool):
    """
    GenAI-powered schema mapping tool.
    
    Intelligently maps between different data schemas using AI to understand
    semantic relationships and suggest mappings.
    """
    
    def __init__(self, genai_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        super().__init__(
            name="SchemaMapper",
            version="1.0.0",
            tool_type=DataToolType.TRANSFORMATION,
            genai_provider=genai_provider,
            model_name=model_name
        )
    
    def _get_request_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_schema": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "required": {"type": "boolean", "default": False},
                            "constraints": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "mapping_confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.7},
                "include_transformation_rules": {"type": "boolean", "default": True},
                "suggest_new_columns": {"type": "boolean", "default": True}
            },
            "required": ["target_schema"]
        }
    
    def _get_response_schema(self) -> Dict[str, Any]:
        data_schema = {
            "type": "object",
            "properties": {
                "column_mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_column": {"type": "string"},
                            "target_column": {"type": "string"},
                            "mapping_type": {"type": "string", "enum": ["direct", "transformation", "combination", "derivation"]},
                            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "transformation_rule": {"type": "string"},
                            "rationale": {"type": "string"}
                        }
                    }
                },
                "unmapped_source_columns": {
                    "type": "array", 
                    "items": {"type": "string"}
                },
                "missing_target_columns": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "suggested_transformations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_column": {"type": "string"},
                            "transformation": {"type": "string"},
                            "description": {"type": "string"}
                        }
                    }
                },
                "schema_compatibility_score": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["column_mappings", "unmapped_source_columns", "missing_target_columns", "schema_compatibility_score"]
        }
        
        return create_response_schema(data_schema)
    
    def _execute_tool(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute schema mapping analysis."""
        target_schema = params["target_schema"]
        confidence_threshold = params.get("mapping_confidence_threshold", 0.7)
        
        # Analyze source schema
        source_schema = self._analyze_source_schema(data)
        
        # Generate column mappings
        column_mappings = self._generate_column_mappings(
            source_schema, target_schema, confidence_threshold
        )
        
        # Identify unmapped and missing columns
        mapped_source_cols = {mapping["source_column"] for mapping in column_mappings}
        mapped_target_cols = {mapping["target_column"] for mapping in column_mappings}
        
        unmapped_source_columns = [col for col in data.columns if col not in mapped_source_cols]
        missing_target_columns = [col for col in target_schema.keys() if col not in mapped_target_cols]
        
        # Generate transformation suggestions
        suggested_transformations = self._suggest_transformations(
            data, unmapped_source_columns, missing_target_columns, target_schema, params
        ) if params.get("suggest_new_columns", True) else []
        
        # Calculate compatibility score
        schema_compatibility_score = self._calculate_compatibility_score(
            column_mappings, len(data.columns), len(target_schema)
        )
        
        return {
            "column_mappings": column_mappings,
            "unmapped_source_columns": unmapped_source_columns,
            "missing_target_columns": missing_target_columns,
            "suggested_transformations": suggested_transformations,
            "schema_compatibility_score": schema_compatibility_score
        }
    
    def _analyze_source_schema(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze the source data schema."""
        schema = {}
        
        for column in data.columns:
            column_info = {
                "type": str(data[column].dtype),
                "null_percentage": data[column].isnull().sum() / len(data),
                "unique_count": data[column].nunique(),
                "sample_values": data[column].dropna().head(5).tolist(),
                "inferred_semantic_type": self._infer_semantic_type(column, data[column])
            }
            schema[column] = column_info
        
        return schema
    
    def _infer_semantic_type(self, column_name: str, column_data: pd.Series) -> str:
        """Infer semantic type of column based on name and data patterns."""
        column_name_lower = column_name.lower()
        sample_values = column_data.dropna().astype(str).head(20)
        
        # ID patterns
        if any(keyword in column_name_lower for keyword in ['id', 'key', 'pk']):
            return "identifier"
        
        # Name patterns
        if any(keyword in column_name_lower for keyword in ['name', 'title', 'label']):
            return "name"
        
        # Email patterns
        if 'email' in column_name_lower or any('@' in str(val) for val in sample_values):
            return "email"
        
        # Phone patterns
        if 'phone' in column_name_lower or any(
            re.match(r'^[\+]?[1-9][\d]{0,15}$', str(val).replace('-', '').replace(' ', '').replace('(', '').replace(')', ''))
            for val in sample_values
        ):
            return "phone"
        
        # Date patterns
        if any(keyword in column_name_lower for keyword in ['date', 'time', 'created', 'updated']):
            return "datetime"
        
        # Address patterns
        if any(keyword in column_name_lower for keyword in ['address', 'street', 'city', 'zip', 'postal']):
            return "address"
        
        # Age patterns
        if 'age' in column_name_lower:
            return "age"
        
        # Amount/Price patterns
        if any(keyword in column_name_lower for keyword in ['price', 'amount', 'cost', 'salary', 'revenue']):
            return "money"
        
        # Category patterns
        if column_data.dtype == 'object' and column_data.nunique() / len(column_data) < 0.1:
            return "category"
        
        # Default based on data type
        if column_data.dtype in ['int64', 'float64']:
            return "numeric"
        elif column_data.dtype == 'bool':
            return "boolean"
        else:
            return "text"
    
    def _generate_column_mappings(
        self, 
        source_schema: Dict[str, Dict[str, Any]], 
        target_schema: Dict[str, Dict[str, Any]],
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Generate intelligent column mappings."""
        mappings = []
        
        for source_col, source_info in source_schema.items():
            best_match = None
            best_score = 0.0
            best_rationale = ""
            best_transformation = ""
            
            for target_col, target_info in target_schema.items():
                score, rationale, transformation = self._calculate_mapping_score(
                    source_col, source_info, target_col, target_info
                )
                
                if score > best_score:
                    best_match = target_col
                    best_score = score
                    best_rationale = rationale
                    best_transformation = transformation
            
            if best_match and best_score >= confidence_threshold:
                mapping_type = self._determine_mapping_type(source_info, target_schema[best_match], best_transformation)
                
                mappings.append({
                    "source_column": source_col,
                    "target_column": best_match,
                    "mapping_type": mapping_type,
                    "confidence_score": round(best_score, 4),
                    "transformation_rule": best_transformation,
                    "rationale": best_rationale
                })
        
        return mappings
    
    def _calculate_mapping_score(
        self, 
        source_col: str, 
        source_info: Dict[str, Any], 
        target_col: str, 
        target_info: Dict[str, Any]
    ) -> Tuple[float, str, str]:
        """Calculate mapping score between source and target columns."""
        score = 0.0
        rationale_parts = []
        transformation = ""
        
        # Exact name match
        if source_col.lower() == target_col.lower():
            score += 0.4
            rationale_parts.append("exact name match")
        
        # Partial name match
        elif source_col.lower() in target_col.lower() or target_col.lower() in source_col.lower():
            score += 0.3
            rationale_parts.append("partial name match")
        
        # Semantic type match
        source_semantic = source_info.get("inferred_semantic_type", "unknown")
        target_type = target_info.get("type", "unknown")
        
        if source_semantic == "identifier" and "id" in target_col.lower():
            score += 0.3
            rationale_parts.append("identifier type match")
        elif source_semantic == "email" and "email" in target_col.lower():
            score += 0.3
            rationale_parts.append("email type match")
        elif source_semantic == "phone" and "phone" in target_col.lower():
            score += 0.3
            rationale_parts.append("phone type match")
        elif source_semantic == "datetime" and any(kw in target_col.lower() for kw in ["date", "time"]):
            score += 0.3
            rationale_parts.append("datetime type match")
        
        # Data type compatibility
        source_type = source_info.get("type", "")
        if self._are_types_compatible(source_type, target_type):
            score += 0.2
            rationale_parts.append("compatible data types")
            if source_type != target_type:
                transformation = f"convert {source_type} to {target_type}"
        
        # Similarity in unique values (for categorical data)
        if (source_info.get("unique_count", 0) < 50 and 
            source_info.get("unique_count", 0) > 1):
            score += 0.1
            rationale_parts.append("categorical data pattern")
        
        rationale = ", ".join(rationale_parts) if rationale_parts else "low similarity"
        
        return score, rationale, transformation
    
    def _are_types_compatible(self, source_type: str, target_type: str) -> bool:
        """Check if source and target data types are compatible."""
        # Numeric type compatibility
        numeric_types = ["int64", "float64", "int32", "float32", "number", "integer", "float"]
        if any(st in source_type for st in numeric_types) and any(tt in target_type for tt in numeric_types):
            return True
        
        # String type compatibility
        string_types = ["object", "string", "text", "varchar", "char"]
        if any(st in source_type for st in string_types) and any(tt in target_type for tt in string_types):
            return True
        
        # Boolean compatibility
        if "bool" in source_type and "bool" in target_type:
            return True
        
        # Date/time compatibility
        datetime_types = ["datetime", "timestamp", "date", "time"]
        if any(dt in source_type for dt in datetime_types) and any(dt in target_type for dt in datetime_types):
            return True
        
        return False
    
    def _determine_mapping_type(self, source_info: Dict[str, Any], target_info: Dict[str, Any], transformation: str) -> str:
        """Determine the type of mapping required."""
        if not transformation:
            return "direct"
        elif "convert" in transformation:
            return "transformation"
        else:
            return "transformation"
    
    def _suggest_transformations(
        self, 
        data: pd.DataFrame, 
        unmapped_columns: List[str], 
        missing_columns: List[str], 
        target_schema: Dict[str, Dict[str, Any]],
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest transformations to create missing target columns."""
        suggestions = []
        
        for missing_col in missing_columns:
            target_info = target_schema[missing_col]
            
            # Try to derive from existing columns
            derivation = self._suggest_column_derivation(missing_col, unmapped_columns, data)
            
            if derivation:
                suggestions.append({
                    "target_column": missing_col,
                    "transformation": derivation["rule"],
                    "description": derivation["description"]
                })
        
        return suggestions
    
    def _suggest_column_derivation(self, target_col: str, available_cols: List[str], data: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Suggest how to derive a target column from available columns."""
        target_col_lower = target_col.lower()
        
        # Full name from first and last name
        if "full_name" in target_col_lower or "name" == target_col_lower:
            first_name_col = None
            last_name_col = None
            
            for col in available_cols:
                if "first" in col.lower() and "name" in col.lower():
                    first_name_col = col
                elif "last" in col.lower() and "name" in col.lower():
                    last_name_col = col
            
            if first_name_col and last_name_col:
                return {
                    "rule": f"concat({first_name_col}, ' ', {last_name_col})",
                    "description": f"Combine {first_name_col} and {last_name_col} to create full name"
                }
        
        # Age from birth date
        if "age" in target_col_lower:
            for col in available_cols:
                if any(kw in col.lower() for kw in ["birth", "dob", "born"]):
                    return {
                        "rule": f"calculate_age({col})",
                        "description": f"Calculate age from birth date in {col}"
                    }
        
        # Email domain extraction
        if "domain" in target_col_lower:
            for col in available_cols:
                if "email" in col.lower():
                    return {
                        "rule": f"extract_domain({col})",
                        "description": f"Extract domain from email address in {col}"
                    }
        
        return None
    
    def _calculate_compatibility_score(self, mappings: List[Dict[str, Any]], source_cols: int, target_cols: int) -> float:
        """Calculate overall schema compatibility score."""
        if target_cols == 0:
            return 1.0 if source_cols == 0 else 0.0
        
        # Weight by confidence scores
        total_confidence = sum(mapping["confidence_score"] for mapping in mappings)
        mapped_ratio = len(mappings) / target_cols
        
        # Penalize for unmapped required columns
        compatibility_score = (mapped_ratio * 0.7) + (total_confidence / len(mappings) * 0.3 if mappings else 0)
        
        return round(min(1.0, compatibility_score), 4)


@register_tool
class DataCleaner(BaseDataTool):
    """
    GenAI-powered data cleaning tool.
    
    Intelligently cleans data by identifying and fixing common data quality issues
    with AI-driven suggestions for cleaning strategies.
    """
    
    def __init__(self, genai_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        super().__init__(
            name="DataCleaner",
            version="1.0.0",
            tool_type=DataToolType.TRANSFORMATION,
            genai_provider=genai_provider,
            model_name=model_name
        )
    
    def _get_request_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cleaning_operations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "remove_duplicates", "handle_missing_values", "fix_data_types",
                            "standardize_formats", "remove_outliers", "normalize_text",
                            "validate_constraints", "auto_clean"
                        ]
                    },
                    "default": ["auto_clean"]
                },
                "missing_value_strategy": {
                    "type": "string",
                    "enum": ["drop", "mean", "median", "mode", "forward_fill", "backward_fill", "interpolate", "auto"],
                    "default": "auto"
                },
                "outlier_method": {
                    "type": "string",
                    "enum": ["iqr", "z_score", "isolation_forest", "auto"],
                    "default": "auto"
                },
                "preserve_original": {"type": "boolean", "default": True},
                "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8}
            }
        }
    
    def _get_response_schema(self) -> Dict[str, Any]:
        data_schema = {
            "type": "object",
            "properties": {
                "cleaning_summary": {
                    "type": "object",
                    "properties": {
                        "operations_performed": {"type": "array", "items": {"type": "string"}},
                        "rows_affected": {"type": "integer"},
                        "columns_affected": {"type": "integer"},
                        "data_quality_improvement": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "operation_details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string"},
                            "column": {"type": "string"},
                            "changes_made": {"type": "integer"},
                            "method_used": {"type": "string"},
                            "rationale": {"type": "string"}
                        }
                    }
                },
                "cleaned_data_stats": {
                    "type": "object",
                    "properties": {
                        "original_shape": {"type": "array", "items": {"type": "integer"}},
                        "cleaned_shape": {"type": "array", "items": {"type": "integer"}},
                        "missing_values_before": {"type": "integer"},
                        "missing_values_after": {"type": "integer"},
                        "duplicates_removed": {"type": "integer"}
                    }
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["cleaning_summary", "operation_details", "cleaned_data_stats"]
        }
        
        return create_response_schema(data_schema)
    
    def _execute_tool(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data cleaning operations."""
        # Store original data stats
        original_stats = self._get_data_stats(data)
        
        # Make a copy for cleaning
        cleaned_data = data.copy() if params.get("preserve_original", True) else data
        
        # Determine operations to perform
        operations = params.get("cleaning_operations", ["auto_clean"])
        if "auto_clean" in operations:
            operations = self._determine_auto_cleaning_operations(cleaned_data)
        
        # Perform cleaning operations
        operation_details = []
        for operation in operations:
            details = self._perform_cleaning_operation(cleaned_data, operation, params)
            operation_details.extend(details)
        
        # Calculate stats and improvements
        cleaned_stats = self._get_data_stats(cleaned_data)
        
        # Generate summary
        cleaning_summary = self._generate_cleaning_summary(
            original_stats, cleaned_stats, operation_details
        )
        
        # Generate recommendations
        recommendations = self._generate_cleaning_recommendations(
            cleaned_data, operation_details
        )
        
        return {
            "cleaning_summary": cleaning_summary,
            "operation_details": operation_details,
            "cleaned_data_stats": {
                "original_shape": list(original_stats["shape"]),
                "cleaned_shape": list(cleaned_stats["shape"]),
                "missing_values_before": original_stats["missing_values"],
                "missing_values_after": cleaned_stats["missing_values"],
                "duplicates_removed": original_stats["duplicates"] - cleaned_stats["duplicates"]
            },
            "recommendations": recommendations
        }
    
    def _get_data_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about the data."""
        return {
            "shape": data.shape,
            "missing_values": data.isnull().sum().sum(),
            "duplicates": data.duplicated().sum(),
            "memory_usage": data.memory_usage(deep=True).sum()
        }
    
    def _determine_auto_cleaning_operations(self, data: pd.DataFrame) -> List[str]:
        """Automatically determine which cleaning operations to perform."""
        operations = []
        
        # Check for duplicates
        if data.duplicated().sum() > 0:
            operations.append("remove_duplicates")
        
        # Check for missing values
        if data.isnull().sum().sum() > 0:
            operations.append("handle_missing_values")
        
        # Check for data type issues
        for col in data.select_dtypes(include=['object']).columns:
            # Check if numeric data stored as string
            sample_values = data[col].dropna().head(100)
            numeric_count = 0
            for val in sample_values:
                try:
                    float(str(val))
                    numeric_count += 1
                except:
                    pass
            
            if numeric_count / len(sample_values) > 0.8:
                operations.append("fix_data_types")
                break
        
        # Check for format inconsistencies
        operations.append("standardize_formats")
        
        # Check for outliers in numeric columns
        if len(data.select_dtypes(include=[np.number]).columns) > 0:
            operations.append("remove_outliers")
        
        # Text normalization for string columns
        if len(data.select_dtypes(include=['object']).columns) > 0:
            operations.append("normalize_text")
        
        return list(set(operations))  # Remove duplicates
    
    def _perform_cleaning_operation(
        self, 
        data: pd.DataFrame, 
        operation: str, 
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Perform a specific cleaning operation."""
        details = []
        
        if operation == "remove_duplicates":
            initial_count = len(data)
            data.drop_duplicates(inplace=True)
            removed_count = initial_count - len(data)
            
            if removed_count > 0:
                details.append({
                    "operation": "remove_duplicates",
                    "column": "all_columns",
                    "changes_made": removed_count,
                    "method_used": "pandas_drop_duplicates",
                    "rationale": f"Removed {removed_count} duplicate rows to ensure data uniqueness"
                })
        
        elif operation == "handle_missing_values":
            strategy = params.get("missing_value_strategy", "auto")
            
            for col in data.columns:
                missing_count = data[col].isnull().sum()
                if missing_count == 0:
                    continue
                
                # Determine strategy
                if strategy == "auto":
                    col_strategy = self._determine_missing_strategy(data[col])
                else:
                    col_strategy = strategy
                
                # Apply strategy
                initial_missing = missing_count
                if col_strategy == "drop":
                    data.dropna(subset=[col], inplace=True)
                elif col_strategy == "mean" and data[col].dtype in ['int64', 'float64']:
                    data[col].fillna(data[col].mean(), inplace=True)
                elif col_strategy == "median" and data[col].dtype in ['int64', 'float64']:
                    data[col].fillna(data[col].median(), inplace=True)
                elif col_strategy == "mode":
                    mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else "Unknown"
                    data[col].fillna(mode_val, inplace=True)
                elif col_strategy == "forward_fill":
                    data[col].fillna(method='ffill', inplace=True)
                
                final_missing = data[col].isnull().sum()
                changes_made = initial_missing - final_missing
                
                if changes_made > 0:
                    details.append({
                        "operation": "handle_missing_values",
                        "column": col,
                        "changes_made": changes_made,
                        "method_used": col_strategy,
                        "rationale": f"Applied {col_strategy} strategy to handle {initial_missing} missing values"
                    })
        
        elif operation == "fix_data_types":
            for col in data.select_dtypes(include=['object']).columns:
                # Try to convert to numeric
                original_dtype = str(data[col].dtype)
                try:
                    # Check if it's actually numeric data
                    numeric_data = pd.to_numeric(data[col], errors='coerce')
                    non_null_ratio = numeric_data.notna().sum() / len(data[col])
                    
                    if non_null_ratio > 0.8:  # 80% can be converted to numeric
                        data[col] = numeric_data
                        details.append({
                            "operation": "fix_data_types",
                            "column": col,
                            "changes_made": 1,
                            "method_used": "to_numeric",
                            "rationale": f"Converted {col} from {original_dtype} to numeric (80%+ values were numeric)"
                        })
                except:
                    pass
        
        elif operation == "standardize_formats":
            for col in data.select_dtypes(include=['object']).columns:
                # Email standardization
                if any('@' in str(val) for val in data[col].dropna().head(10)):
                    original_values = data[col].copy()
                    data[col] = data[col].astype(str).str.lower().str.strip()
                    
                    changes = (original_values != data[col]).sum()
                    if changes > 0:
                        details.append({
                            "operation": "standardize_formats",
                            "column": col,
                            "changes_made": changes,
                            "method_used": "email_normalization",
                            "rationale": f"Standardized email format (lowercase, trimmed) for {changes} values"
                        })
                
                # Phone number standardization
                elif any(str(val).replace('-', '').replace(' ', '').isdigit() for val in data[col].dropna().head(10)):
                    original_values = data[col].copy()
                    # Remove common phone formatting
                    data[col] = data[col].astype(str).str.replace(r'[^\d]', '', regex=True)
                    
                    changes = (original_values != data[col]).sum()
                    if changes > 0:
                        details.append({
                            "operation": "standardize_formats",
                            "column": col,
                            "changes_made": changes,
                            "method_used": "phone_normalization",
                            "rationale": f"Standardized phone number format (digits only) for {changes} values"
                        })
        
        elif operation == "remove_outliers":
            outlier_method = params.get("outlier_method", "auto")
            
            for col in data.select_dtypes(include=[np.number]).columns:
                initial_count = len(data)
                
                if outlier_method in ["iqr", "auto"]:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Remove outliers
                    outlier_mask = (data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)
                    outlier_count = outlier_mask.sum()
                    
                    # Only remove if outliers are < 5% of data
                    if outlier_count / len(data) < 0.05 and outlier_count > 0:
                        data = data[~outlier_mask]
                        
                        details.append({
                            "operation": "remove_outliers",
                            "column": col,
                            "changes_made": outlier_count,
                            "method_used": "iqr_method",
                            "rationale": f"Removed {outlier_count} outliers using IQR method (< 5% of data)"
                        })
        
        elif operation == "normalize_text":
            for col in data.select_dtypes(include=['object']).columns:
                if data[col].dtype == 'object':
                    original_values = data[col].copy()
                    
                    # Basic text normalization
                    data[col] = (data[col]
                                .astype(str)
                                .str.strip()
                                .str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
                                .str.title())  # Title case
                    
                    changes = (original_values != data[col]).sum()
                    if changes > 0:
                        details.append({
                            "operation": "normalize_text",
                            "column": col,
                            "changes_made": changes,
                            "method_used": "basic_text_normalization",
                            "rationale": f"Normalized text format (trimmed, title case) for {changes} values"
                        })
        
        return details
    
    def _determine_missing_strategy(self, series: pd.Series) -> str:
        """Determine the best strategy for handling missing values in a column."""
        missing_ratio = series.isnull().sum() / len(series)
        
        # If too many missing values, consider dropping
        if missing_ratio > 0.7:
            return "drop"
        
        # For numeric data
        if series.dtype in ['int64', 'float64']:
            # If data is normally distributed, use mean
            # If skewed, use median
            # For simplicity, use median as it's more robust
            return "median"
        
        # For categorical data, use mode
        else:
            return "mode"
    
    def _generate_cleaning_summary(
        self, 
        original_stats: Dict[str, Any], 
        cleaned_stats: Dict[str, Any], 
        operation_details: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary of cleaning operations."""
        operations_performed = list(set(detail["operation"] for detail in operation_details))
        rows_affected = original_stats["shape"][0] - cleaned_stats["shape"][0]
        columns_affected = len(set(detail["column"] for detail in operation_details if detail["column"] != "all_columns"))
        
        # Calculate quality improvement (simplified)
        original_quality = 1 - (original_stats["missing_values"] / (original_stats["shape"][0] * original_stats["shape"][1]))
        cleaned_quality = 1 - (cleaned_stats["missing_values"] / (cleaned_stats["shape"][0] * cleaned_stats["shape"][1])) if cleaned_stats["shape"][0] > 0 else 1.0
        
        quality_improvement = max(0, cleaned_quality - original_quality)
        
        return {
            "operations_performed": operations_performed,
            "rows_affected": rows_affected,
            "columns_affected": columns_affected,
            "data_quality_improvement": round(quality_improvement, 4)
        }
    
    def _generate_cleaning_recommendations(
        self, 
        cleaned_data: pd.DataFrame, 
        operation_details: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for further data cleaning."""
        recommendations = []
        
        # Check remaining issues
        remaining_missing = cleaned_data.isnull().sum().sum()
        if remaining_missing > 0:
            recommendations.append(f"Consider addressing {remaining_missing} remaining missing values")
        
        # Check for potential data type optimizations
        for col in cleaned_data.select_dtypes(include=['int64']).columns:
            max_val = cleaned_data[col].max()
            if max_val < 32767:  # Can use int16
                recommendations.append(f"Consider converting {col} to int16 to save memory")
        
        # Check for high cardinality categorical columns
        for col in cleaned_data.select_dtypes(include=['object']).columns:
            unique_ratio = cleaned_data[col].nunique() / len(cleaned_data)
            if 0.05 < unique_ratio < 0.5:
                recommendations.append(f"Consider encoding categorical column {col}")
        
        # Performance recommendations
        if len(cleaned_data) > 100000:
            recommendations.append("Consider indexing for large dataset performance")
        
        return recommendations


@register_tool
class DataNormalizer(BaseDataTool):
    """
    GenAI-powered data normalization tool.
    
    Normalizes data using various techniques with AI-driven selection
    of appropriate normalization methods.
    """
    
    def __init__(self, genai_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        super().__init__(
            name="DataNormalizer",
            version="1.0.0",
            tool_type=DataToolType.TRANSFORMATION,
            genai_provider=genai_provider,
            model_name=model_name
        )
    
    def _get_request_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "normalization_method": {
                    "type": "string",
                    "enum": ["min_max", "z_score", "robust", "quantile", "auto"],
                    "default": "auto"
                },
                "columns_to_normalize": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "preserve_distribution": {"type": "boolean", "default": False},
                "handle_outliers": {"type": "boolean", "default": True}
            }
        }
    
    def _get_response_schema(self) -> Dict[str, Any]:
        data_schema = {
            "type": "object",
            "properties": {
                "normalization_summary": {
                    "type": "object",
                    "properties": {
                        "columns_normalized": {"type": "integer"},
                        "method_used": {"type": "string"},
                        "outliers_handled": {"type": "integer"}
                    }
                },
                "column_transformations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "method": {"type": "string"},
                            "parameters": {"type": "object"},
                            "distribution_change": {"type": "string"}
                        }
                    }
                },
                "transformation_parameters": {"type": "object"}
            },
            "required": ["normalization_summary", "column_transformations"]
        }
        
        return create_response_schema(data_schema)
    
    def _execute_tool(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data normalization."""
        method = params.get("normalization_method", "auto")
        columns_to_normalize = params.get("columns_to_normalize", [])
        
        if not columns_to_normalize:
            # Auto-select numeric columns
            columns_to_normalize = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Perform normalization
        column_transformations = []
        transformation_parameters = {}
        outliers_handled = 0
        
        for col in columns_to_normalize:
            if col not in data.columns or data[col].dtype not in ['int64', 'float64']:
                continue
            
            # Determine method for this column
            if method == "auto":
                col_method = self._determine_normalization_method(data[col])
            else:
                col_method = method
            
            # Apply normalization
            original_data = data[col].copy()
            transformation_info = self._apply_normalization(data, col, col_method, params)
            
            column_transformations.append({
                "column": col,
                "method": col_method,
                "parameters": transformation_info["parameters"],
                "distribution_change": transformation_info["distribution_change"]
            })
            
            transformation_parameters[col] = transformation_info["parameters"]
        
        return {
            "normalization_summary": {
                "columns_normalized": len(column_transformations),
                "method_used": method,
                "outliers_handled": outliers_handled
            },
            "column_transformations": column_transformations,
            "transformation_parameters": transformation_parameters
        }
    
    def _determine_normalization_method(self, series: pd.Series) -> str:
        """Determine the best normalization method for a column."""
        # Check distribution characteristics
        skewness = abs(series.skew()) if len(series) > 3 else 0
        
        # Check for outliers
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
        outlier_ratio = len(outliers) / len(series)
        
        # Decision logic
        if outlier_ratio > 0.1:  # High outlier presence
            return "robust"  # Robust scaling
        elif skewness > 1:  # Highly skewed
            return "quantile"  # Quantile transformation
        else:
            return "z_score"  # Standard scaling
    
    def _apply_normalization(self, data: pd.DataFrame, column: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply normalization method to a column."""
        original_series = data[column].copy()
        transformation_info = {"parameters": {}, "distribution_change": ""}
        
        if method == "min_max":
            # Min-Max scaling to [0, 1]
            min_val = original_series.min()
            max_val = original_series.max()
            
            if max_val != min_val:
                data[column] = (original_series - min_val) / (max_val - min_val)
            
            transformation_info["parameters"] = {"min": float(min_val), "max": float(max_val)}
            transformation_info["distribution_change"] = "scaled to [0, 1] range"
        
        elif method == "z_score":
            # Standard scaling (mean=0, std=1)
            mean_val = original_series.mean()
            std_val = original_series.std()
            
            if std_val != 0:
                data[column] = (original_series - mean_val) / std_val
            
            transformation_info["parameters"] = {"mean": float(mean_val), "std": float(std_val)}
            transformation_info["distribution_change"] = "standardized (mean=0, std=1)"
        
        elif method == "robust":
            # Robust scaling using median and IQR
            median_val = original_series.median()
            Q1, Q3 = original_series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            
            if IQR != 0:
                data[column] = (original_series - median_val) / IQR
            
            transformation_info["parameters"] = {"median": float(median_val), "iqr": float(IQR)}
            transformation_info["distribution_change"] = "robust scaling (median=0, IQR=1)"
        
        elif method == "quantile":
            # Quantile transformation (simplified)
            # In a real implementation, this would use sklearn's QuantileTransformer
            ranks = original_series.rank(method='average')
            data[column] = (ranks - 1) / (len(ranks) - 1)
            
            transformation_info["parameters"] = {"method": "uniform_quantile"}
            transformation_info["distribution_change"] = "uniform distribution via quantile transformation"
        
        return transformation_info


@register_tool
class DataEnricher(BaseDataTool):
    """
    GenAI-powered data enrichment tool.
    
    Enriches data by adding derived features, external data integration,
    and AI-generated insights.
    """
    
    def __init__(self, genai_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        super().__init__(
            name="DataEnricher",
            version="1.0.0",
            tool_type=DataToolType.ENRICHMENT,
            genai_provider=genai_provider,
            model_name=model_name
        )
    
    def _get_request_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enrichment_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["derived_features", "categorical_encoding", "temporal_features", "text_features", "statistical_features"]
                    },
                    "default": ["derived_features"]
                },
                "feature_suggestions": {"type": "boolean", "default": True},
                "max_new_features": {"type": "integer", "minimum": 1, "default": 20}
            }
        }
    
    def _get_response_schema(self) -> Dict[str, Any]:
        data_schema = {
            "type": "object",
            "properties": {
                "enrichment_summary": {
                    "type": "object",
                    "properties": {
                        "features_added": {"type": "integer"},
                        "enrichment_types": {"type": "array", "items": {"type": "string"}},
                        "original_columns": {"type": "integer"},
                        "enriched_columns": {"type": "integer"}
                    }
                },
                "new_features": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature_name": {"type": "string"},
                            "feature_type": {"type": "string"},
                            "description": {"type": "string"},
                            "source_columns": {"type": "array", "items": {"type": "string"}},
                            "creation_method": {"type": "string"}
                        }
                    }
                },
                "feature_importance_estimates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature_name": {"type": "string"},
                            "importance_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "rationale": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["enrichment_summary", "new_features"]
        }
        
        return create_response_schema(data_schema)
    
    def _execute_tool(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data enrichment."""
        enrichment_types = params.get("enrichment_types", ["derived_features"])
        max_features = params.get("max_new_features", 20)
        
        original_columns = len(data.columns)
        new_features = []
        features_added = 0
        
        for enrichment_type in enrichment_types:
            if features_added >= max_features:
                break
            
            type_features = self._create_enrichment_features(data, enrichment_type, max_features - features_added)
            new_features.extend(type_features)
            features_added += len(type_features)
        
        # Generate feature importance estimates
        feature_importance = self._estimate_feature_importance(data, new_features) if params.get("feature_suggestions", True) else []
        
        return {
            "enrichment_summary": {
                "features_added": features_added,
                "enrichment_types": enrichment_types,
                "original_columns": original_columns,
                "enriched_columns": len(data.columns)
            },
            "new_features": new_features,
            "feature_importance_estimates": feature_importance
        }
    
    def _create_enrichment_features(self, data: pd.DataFrame, enrichment_type: str, max_features: int) -> List[Dict[str, Any]]:
        """Create enrichment features of a specific type."""
        features_created = []
        
        if enrichment_type == "derived_features":
            features_created.extend(self._create_derived_features(data, max_features))
        elif enrichment_type == "categorical_encoding":
            features_created.extend(self._create_categorical_features(data, max_features))
        elif enrichment_type == "temporal_features":
            features_created.extend(self._create_temporal_features(data, max_features))
        elif enrichment_type == "text_features":
            features_created.extend(self._create_text_features(data, max_features))
        elif enrichment_type == "statistical_features":
            features_created.extend(self._create_statistical_features(data, max_features))
        
        return features_created
    
    def _create_derived_features(self, data: pd.DataFrame, max_features: int) -> List[Dict[str, Any]]:
        """Create derived features from existing columns."""
        features = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ratios between numeric columns
        for i, col1 in enumerate(numeric_cols):
            if len(features) >= max_features:
                break
            
            for col2 in numeric_cols[i+1:]:
                if len(features) >= max_features:
                    break
                
                # Avoid division by zero
                if (data[col2] != 0).all():
                    feature_name = f"{col1}_{col2}_ratio"
                    data[feature_name] = data[col1] / data[col2]
                    
                    features.append({
                        "feature_name": feature_name,
                        "feature_type": "derived_numeric",
                        "description": f"Ratio of {col1} to {col2}",
                        "source_columns": [col1, col2],
                        "creation_method": "division"
                    })
        
        # Sums and differences
        for i, col1 in enumerate(numeric_cols):
            if len(features) >= max_features:
                break
            
            for col2 in numeric_cols[i+1:]:
                if len(features) >= max_features:
                    break
                
                # Sum
                sum_feature = f"{col1}_{col2}_sum"
                data[sum_feature] = data[col1] + data[col2]
                features.append({
                    "feature_name": sum_feature,
                    "feature_type": "derived_numeric",
                    "description": f"Sum of {col1} and {col2}",
                    "source_columns": [col1, col2],
                    "creation_method": "addition"
                })
                
                if len(features) >= max_features:
                    break
                
                # Difference
                diff_feature = f"{col1}_{col2}_diff"
                data[diff_feature] = data[col1] - data[col2]
                features.append({
                    "feature_name": diff_feature,
                    "feature_type": "derived_numeric",
                    "description": f"Difference between {col1} and {col2}",
                    "source_columns": [col1, col2],
                    "creation_method": "subtraction"
                })
        
        return features[:max_features]
    
    def _create_categorical_features(self, data: pd.DataFrame, max_features: int) -> List[Dict[str, Any]]:
        """Create features from categorical columns."""
        features = []
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if len(features) >= max_features:
                break
            
            # One-hot encoding for low cardinality columns
            unique_values = data[col].nunique()
            if unique_values <= 10:  # Low cardinality
                for value in data[col].unique():
                    if pd.isna(value):
                        continue
                    
                    feature_name = f"{col}_{str(value)}_flag"
                    data[feature_name] = (data[col] == value).astype(int)
                    
                    features.append({
                        "feature_name": feature_name,
                        "feature_type": "categorical_encoding",
                        "description": f"Binary indicator for {col} = {value}",
                        "source_columns": [col],
                        "creation_method": "one_hot_encoding"
                    })
                    
                    if len(features) >= max_features:
                        break
            
            # Frequency encoding
            elif unique_values > 10:
                freq_feature = f"{col}_frequency"
                freq_map = data[col].value_counts().to_dict()
                data[freq_feature] = data[col].map(freq_map)
                
                features.append({
                    "feature_name": freq_feature,
                    "feature_type": "categorical_encoding",
                    "description": f"Frequency encoding of {col}",
                    "source_columns": [col],
                    "creation_method": "frequency_encoding"
                })
        
        return features[:max_features]
    
    def _create_temporal_features(self, data: pd.DataFrame, max_features: int) -> List[Dict[str, Any]]:
        """Create temporal features from datetime columns."""
        features = []
        
        # Look for datetime columns or date-like strings
        for col in data.columns:
            if len(features) >= max_features:
                break
            
            # Try to convert to datetime
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    dt_series = pd.to_datetime(data[col], errors='coerce')
                    
                    if dt_series.notna().sum() > len(data) * 0.5:  # More than 50% valid dates
                        # Extract temporal features
                        temporal_features = [
                            ('year', dt_series.dt.year, "Year component"),
                            ('month', dt_series.dt.month, "Month component"),
                            ('day', dt_series.dt.day, "Day component"),
                            ('weekday', dt_series.dt.weekday, "Day of week (0=Monday)"),
                            ('quarter', dt_series.dt.quarter, "Quarter of year"),
                            ('is_weekend', (dt_series.dt.weekday >= 5).astype(int), "Weekend indicator")
                        ]
                        
                        for suffix, values, description in temporal_features:
                            if len(features) >= max_features:
                                break
                            
                            feature_name = f"{col}_{suffix}"
                            data[feature_name] = values
                            
                            features.append({
                                "feature_name": feature_name,
                                "feature_type": "temporal",
                                "description": f"{description} from {col}",
                                "source_columns": [col],
                                "creation_method": "datetime_extraction"
                            })
                
                except:
                    continue
        
        return features[:max_features]
    
    def _create_text_features(self, data: pd.DataFrame, max_features: int) -> List[Dict[str, Any]]:
        """Create features from text columns."""
        features = []
        text_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        for col in text_cols:
            if len(features) >= max_features:
                break
            
            # Check if column contains text (not just categorical values)
            sample_values = data[col].dropna().astype(str).head(10)
            avg_length = sample_values.str.len().mean()
            
            if avg_length > 10:  # Likely text content
                # Text length
                length_feature = f"{col}_length"
                data[length_feature] = data[col].astype(str).str.len()
                features.append({
                    "feature_name": length_feature,
                    "feature_type": "text_feature",
                    "description": f"Character length of {col}",
                    "source_columns": [col],
                    "creation_method": "string_length"
                })
                
                if len(features) >= max_features:
                    break
                
                # Word count
                word_count_feature = f"{col}_word_count"
                data[word_count_feature] = data[col].astype(str).str.split().str.len()
                features.append({
                    "feature_name": word_count_feature,
                    "feature_type": "text_feature", 
                    "description": f"Word count in {col}",
                    "source_columns": [col],
                    "creation_method": "word_count"
                })
                
                if len(features) >= max_features:
                    break
                
                # Contains digits
                digits_feature = f"{col}_has_digits"
                data[digits_feature] = data[col].astype(str).str.contains(r'\d').astype(int)
                features.append({
                    "feature_name": digits_feature,
                    "feature_type": "text_feature",
                    "description": f"Whether {col} contains digits",
                    "source_columns": [col],
                    "creation_method": "pattern_detection"
                })
        
        return features[:max_features]
    
    def _create_statistical_features(self, data: pd.DataFrame, max_features: int) -> List[Dict[str, Any]]:
        """Create statistical aggregation features."""
        features = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Row-wise statistics
            stat_features = [
                ('row_mean', data[numeric_cols].mean(axis=1), "Mean of numeric columns per row"),
                ('row_std', data[numeric_cols].std(axis=1), "Standard deviation of numeric columns per row"),
                ('row_min', data[numeric_cols].min(axis=1), "Minimum of numeric columns per row"),
                ('row_max', data[numeric_cols].max(axis=1), "Maximum of numeric columns per row")
            ]
            
            for suffix, values, description in stat_features:
                if len(features) >= max_features:
                    break
                
                feature_name = f"numeric_{suffix}"
                data[feature_name] = values
                
                features.append({
                    "feature_name": feature_name,
                    "feature_type": "statistical",
                    "description": description,
                    "source_columns": numeric_cols,
                    "creation_method": "row_aggregation"
                })
        
        return features[:max_features]
    
    def _estimate_feature_importance(self, data: pd.DataFrame, new_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Estimate importance of new features."""
        importance_estimates = []
        
        for feature_info in new_features:
            feature_name = feature_info["feature_name"]
            
            if feature_name not in data.columns:
                continue
            
            # Simple importance estimation based on variance and correlation
            feature_data = data[feature_name]
            
            # Variance-based importance (higher variance = potentially more informative)
            if feature_data.dtype in ['int64', 'float64']:
                variance = feature_data.var()
                normalized_variance = min(1.0, variance / (feature_data.mean() + 1e-8))
            else:
                normalized_variance = 0.5  # Default for categorical
            
            # Correlation with other numeric features (lower correlation = more unique information)
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if feature_name in numeric_cols and len(numeric_cols) > 1:
                other_cols = [col for col in numeric_cols if col != feature_name]
                correlations = [abs(data[feature_name].corr(data[col])) for col in other_cols[:5]]  # Limit to 5
                avg_correlation = np.mean([c for c in correlations if not np.isnan(c)])
                uniqueness_score = 1 - avg_correlation if not np.isnan(avg_correlation) else 0.5
            else:
                uniqueness_score = 0.5
            
            # Combine scores
            importance_score = (normalized_variance * 0.6 + uniqueness_score * 0.4)
            importance_score = max(0.1, min(1.0, importance_score))  # Bound between 0.1 and 1.0
            
            # Generate rationale
            rationale = self._generate_importance_rationale(feature_info, importance_score)
            
            importance_estimates.append({
                "feature_name": feature_name,
                "importance_score": round(importance_score, 4),
                "rationale": rationale
            })
        
        # Sort by importance
        importance_estimates.sort(key=lambda x: x["importance_score"], reverse=True)
        
        return importance_estimates
    
    def _generate_importance_rationale(self, feature_info: Dict[str, Any], importance_score: float) -> str:
        """Generate explanation for feature importance score."""
        feature_type = feature_info["feature_type"]
        creation_method = feature_info["creation_method"]
        
        if importance_score > 0.8:
            strength = "High"
        elif importance_score > 0.6:
            strength = "Medium-High"
        elif importance_score > 0.4:
            strength = "Medium"
        else:
            strength = "Low"
        
        base_rationales = {
            "derived_numeric": f"{strength} potential due to mathematical relationship between source columns",
            "categorical_encoding": f"{strength} discriminative power based on categorical distribution",
            "temporal": f"{strength} relevance for time-based patterns and seasonality",
            "text_feature": f"{strength} information content based on text characteristics",
            "statistical": f"{strength} aggregated information value from multiple numeric features"
        }
        
        base_rationale = base_rationales.get(feature_type, f"{strength} estimated importance")
        
        return f"{base_rationale} (created via {creation_method})"