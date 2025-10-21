"""
Data Quality Analysis Tools
==========================

GenAI-powered tools for comprehensive data quality analysis including profiling,
validation, anomaly detection, and quality scoring.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json

from .base import (
    BaseDataTool, 
    DataToolType, 
    register_tool,
    COMMON_SCHEMAS,
    create_response_schema
)


@register_tool
class DataQualityAnalyzer(BaseDataTool):
    """
    Comprehensive data quality analysis using GenAI.
    
    Analyzes completeness, accuracy, consistency, validity, and provides
    an overall quality score with recommendations.
    """
    
    def __init__(self, genai_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        super().__init__(
            name="DataQualityAnalyzer",
            version="1.0.0",
            tool_type=DataToolType.QUALITY_ANALYSIS,
            genai_provider=genai_provider,
            model_name=model_name
        )
    
    def _get_request_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "quality_dimensions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["completeness", "accuracy", "consistency", "validity", "uniqueness"]
                    },
                    "default": ["completeness", "accuracy", "consistency", "validity"]
                },
                "column_types": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                },
                "business_rules": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "include_recommendations": {"type": "boolean", "default": True}
            }
        }
    
    def _get_response_schema(self) -> Dict[str, Any]:
        data_schema = {
            "type": "object",
            "properties": {
                "quality_metrics": COMMON_SCHEMAS["data_quality_metrics"],
                "column_quality": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column_name": {"type": "string"},
                            "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "issues": {"type": "array", "items": {"type": "string"}},
                            "recommendations": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "data_issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue_type": {"type": "string"},
                            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                            "description": {"type": "string"},
                            "affected_columns": {"type": "array", "items": {"type": "string"}},
                            "affected_rows": {"type": "integer"},
                            "recommendation": {"type": "string"}
                        }
                    }
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["quality_metrics", "column_quality", "data_issues", "confidence_score"]
        }
        
        return create_response_schema(data_schema)
    
    def _execute_tool(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data quality analysis."""
        # Basic quality metrics calculation
        quality_metrics = self._calculate_quality_metrics(data, params)
        
        # Column-level quality analysis
        column_quality = self._analyze_column_quality(data, params)
        
        # Identify data issues
        data_issues = self._identify_data_issues(data, params)
        
        # Generate GenAI-powered recommendations
        recommendations = self._generate_recommendations(data, quality_metrics, data_issues, params)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(data, quality_metrics)
        
        return {
            "quality_metrics": quality_metrics,
            "column_quality": column_quality,
            "data_issues": data_issues,
            "recommendations": recommendations,
            "confidence_score": confidence_score
        }
    
    def _calculate_quality_metrics(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall data quality metrics."""
        total_cells = data.shape[0] * data.shape[1]
        
        # Completeness: percentage of non-null values
        null_cells = data.isnull().sum().sum()
        completeness = (total_cells - null_cells) / total_cells if total_cells > 0 else 0
        
        # Accuracy: estimated based on data type consistency
        accuracy = self._estimate_accuracy(data)
        
        # Consistency: based on format patterns and value distributions
        consistency = self._estimate_consistency(data)
        
        # Validity: based on business rules and constraints
        validity = self._estimate_validity(data, params)
        
        # Overall score (weighted average)
        overall_score = (
            completeness * 0.3 +
            accuracy * 0.3 +
            consistency * 0.2 +
            validity * 0.2
        )
        
        return {
            "completeness": round(completeness, 4),
            "accuracy": round(accuracy, 4),
            "consistency": round(consistency, 4),
            "validity": round(validity, 4),
            "overall_score": round(overall_score, 4)
        }
    
    def _estimate_accuracy(self, data: pd.DataFrame) -> float:
        """Estimate data accuracy based on type consistency."""
        accuracy_scores = []
        
        for column in data.columns:
            column_data = data[column].dropna()
            if len(column_data) == 0:
                continue
            
            # Check if numeric columns contain actual numbers
            if data[column].dtype in ['int64', 'float64']:
                # Check for infinity, NaN in numeric data
                invalid_numeric = np.isinf(column_data).sum() + np.isnan(column_data).sum()
                accuracy = 1 - (invalid_numeric / len(column_data))
            
            # Check string data for obvious issues
            elif data[column].dtype == 'object':
                # Look for mixed types, encoding issues, etc.
                string_data = column_data.astype(str)
                # Simple heuristic: check for consistent string patterns
                avg_length = string_data.str.len().mean()
                std_length = string_data.str.len().std()
                
                # If standard deviation is very high compared to mean, likely inconsistent
                if avg_length > 0:
                    consistency_score = 1 - min(std_length / avg_length, 1) * 0.5
                else:
                    consistency_score = 1.0
                
                accuracy = consistency_score
            
            else:
                accuracy = 1.0  # Assume other types are accurate
            
            accuracy_scores.append(accuracy)
        
        return np.mean(accuracy_scores) if accuracy_scores else 1.0
    
    def _estimate_consistency(self, data: pd.DataFrame) -> float:
        """Estimate data consistency based on patterns and formats."""
        consistency_scores = []
        
        for column in data.columns:
            column_data = data[column].dropna()
            if len(column_data) == 0:
                continue
            
            if data[column].dtype == 'object':
                # Check for consistent formats in string data
                string_data = column_data.astype(str)
                
                # Email format consistency
                if any('@' in str(val) for val in string_data):
                    email_pattern_matches = string_data.str.contains(r'^[^@]+@[^@]+\.[^@]+$', regex=True).sum()
                    email_total = string_data.str.contains('@').sum()
                    if email_total > 0:
                        consistency = email_pattern_matches / email_total
                    else:
                        consistency = 1.0
                
                # Phone number consistency
                elif any(str(val).replace('-', '').replace(' ', '').replace('(', '').replace(')', '').isdigit() 
                        for val in string_data.head(10)):
                    # Simple phone number pattern check
                    phone_like = string_data.str.replace(r'[^\d]', '', regex=True).str.len()
                    consistent_length = (phone_like == phone_like.mode().iloc[0] if len(phone_like.mode()) > 0 else 10).sum()
                    consistency = consistent_length / len(string_data)
                
                else:
                    # General string consistency (length and character patterns)
                    lengths = string_data.str.len()
                    length_consistency = 1 - (lengths.std() / lengths.mean()) if lengths.mean() > 0 else 1.0
                    consistency = max(0, min(1, length_consistency))
            
            else:
                # Numeric data consistency
                consistency = 1.0  # Numeric data is generally consistent
            
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _estimate_validity(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Estimate data validity based on business rules and constraints."""
        validity_scores = []
        business_rules = params.get('business_rules', [])
        
        for column in data.columns:
            column_data = data[column].dropna()
            if len(column_data) == 0:
                continue
            
            validity = 1.0
            
            # Check basic constraints
            if data[column].dtype in ['int64', 'float64']:
                # Check for negative values where they shouldn't be (e.g., age, price)
                if 'age' in column.lower() or 'price' in column.lower() or 'amount' in column.lower():
                    negative_count = (column_data < 0).sum()
                    validity = 1 - (negative_count / len(column_data))
                
                # Check for reasonable ranges
                if 'age' in column.lower():
                    unreasonable_age = ((column_data < 0) | (column_data > 150)).sum()
                    validity = min(validity, 1 - (unreasonable_age / len(column_data)))
            
            # Apply custom business rules if provided
            for rule in business_rules:
                # Simple rule evaluation (would be more sophisticated in real implementation)
                try:
                    rule_violations = eval(f"(data['{column}'] {rule}).sum()")
                    validity = min(validity, 1 - (rule_violations / len(column_data)))
                except:
                    pass  # Skip invalid rules
            
            validity_scores.append(validity)
        
        return np.mean(validity_scores) if validity_scores else 1.0
    
    def _analyze_column_quality(self, data: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze quality for each column."""
        column_quality = []
        
        for column in data.columns:
            column_data = data[column]
            
            # Calculate column-specific quality metrics
            null_percentage = column_data.isnull().sum() / len(column_data)
            
            issues = []
            recommendations = []
            
            # Identify issues
            if null_percentage > 0.5:
                issues.append("High missing value rate")
                recommendations.append("Consider imputation or data collection improvement")
            
            if column_data.dtype == 'object':
                # Check for mixed data types
                numeric_count = 0
                for val in column_data.dropna().head(100):
                    try:
                        float(str(val))
                        numeric_count += 1
                    except:
                        pass
                
                if 0.1 < numeric_count / len(column_data.dropna().head(100)) < 0.9:
                    issues.append("Mixed data types detected")
                    recommendations.append("Standardize data types")
            
            # Calculate quality score
            quality_score = 1.0
            quality_score -= null_percentage * 0.5  # Penalize missing values
            quality_score -= len(issues) * 0.1  # Penalize each issue
            quality_score = max(0, quality_score)
            
            column_quality.append({
                "column_name": column,
                "quality_score": round(quality_score, 4),
                "issues": issues,
                "recommendations": recommendations
            })
        
        return column_quality
    
    def _identify_data_issues(self, data: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific data quality issues."""
        issues = []
        
        # Missing values
        missing_counts = data.isnull().sum()
        for column, missing_count in missing_counts.items():
            if missing_count > 0:
                severity = "high" if missing_count / len(data) > 0.5 else "medium" if missing_count / len(data) > 0.1 else "low"
                issues.append({
                    "issue_type": "missing_values",
                    "severity": severity,
                    "description": f"Column '{column}' has {missing_count} missing values ({missing_count/len(data)*100:.1f}%)",
                    "affected_columns": [column],
                    "affected_rows": int(missing_count),
                    "recommendation": "Consider imputation, data collection improvement, or handling strategy"
                })
        
        # Duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            issues.append({
                "issue_type": "duplicate_rows",
                "severity": "medium" if duplicate_count / len(data) > 0.05 else "low",
                "description": f"Found {duplicate_count} duplicate rows ({duplicate_count/len(data)*100:.1f}%)",
                "affected_columns": list(data.columns),
                "affected_rows": int(duplicate_count),
                "recommendation": "Remove duplicates or verify if they are legitimate"
            })
        
        # Outliers in numeric columns
        for column in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[column] < Q1 - 1.5 * IQR) | (data[column] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                severity = "high" if len(outliers) / len(data) > 0.1 else "medium" if len(outliers) / len(data) > 0.05 else "low"
                issues.append({
                    "issue_type": "outliers",
                    "severity": severity,
                    "description": f"Column '{column}' has {len(outliers)} potential outliers",
                    "affected_columns": [column],
                    "affected_rows": len(outliers),
                    "recommendation": "Investigate outliers for data entry errors or legitimate extreme values"
                })
        
        return issues
    
    def _generate_recommendations(
        self, 
        data: pd.DataFrame, 
        quality_metrics: Dict[str, float], 
        data_issues: List[Dict[str, Any]], 
        params: Dict[str, Any]
    ) -> List[str]:
        """Generate GenAI-powered recommendations for data quality improvement."""
        recommendations = []
        
        # Based on overall quality score
        overall_score = quality_metrics["overall_score"]
        
        if overall_score < 0.5:
            recommendations.append("Critical data quality issues detected. Comprehensive data cleaning required.")
        elif overall_score < 0.7:
            recommendations.append("Moderate data quality issues. Focus on high-impact improvements.")
        elif overall_score < 0.9:
            recommendations.append("Good data quality with room for improvement. Address specific issues.")
        else:
            recommendations.append("Excellent data quality. Maintain current standards.")
        
        # Specific recommendations based on quality dimensions
        if quality_metrics["completeness"] < 0.8:
            recommendations.append("Improve data completeness by enhancing data collection processes")
        
        if quality_metrics["accuracy"] < 0.8:
            recommendations.append("Implement data validation rules to improve accuracy")
        
        if quality_metrics["consistency"] < 0.8:
            recommendations.append("Standardize data formats and establish data entry guidelines")
        
        if quality_metrics["validity"] < 0.8:
            recommendations.append("Define and enforce business rules for data validity")
        
        # Issue-specific recommendations
        high_severity_issues = [issue for issue in data_issues if issue["severity"] == "high"]
        if high_severity_issues:
            recommendations.append(f"Address {len(high_severity_issues)} high-severity issues immediately")
        
        # Data size recommendations
        if len(data) < 100:
            recommendations.append("Consider collecting more data for robust analysis")
        elif len(data) > 1000000:
            recommendations.append("Consider data sampling for improved processing performance")
        
        return recommendations
    
    def _calculate_confidence_score(self, data: pd.DataFrame, quality_metrics: Dict[str, float]) -> float:
        """Calculate confidence score for the analysis."""
        # Base confidence on data size and quality
        size_factor = min(1.0, len(data) / 1000)  # More data = higher confidence
        quality_factor = quality_metrics["overall_score"]
        
        # Adjust for number of columns (more features = higher confidence)
        feature_factor = min(1.0, len(data.columns) / 10)
        
        confidence = (size_factor * 0.4 + quality_factor * 0.4 + feature_factor * 0.2)
        return round(confidence, 4)


@register_tool
class DataProfiler(BaseDataTool):
    """
    Comprehensive data profiling tool using GenAI for insights.
    
    Generates detailed profiles for each column including statistics,
    distributions, patterns, and GenAI-powered insights.
    """
    
    def __init__(self, genai_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        super().__init__(
            name="DataProfiler",
            version="1.0.0",
            tool_type=DataToolType.QUALITY_ANALYSIS,
            genai_provider=genai_provider,
            model_name=model_name
        )
    
    def _get_request_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "include_distributions": {"type": "boolean", "default": True},
                "include_patterns": {"type": "boolean", "default": True},
                "include_correlations": {"type": "boolean", "default": True},
                "sample_size": {"type": "integer", "minimum": 100, "default": 1000}
            }
        }
    
    def _get_response_schema(self) -> Dict[str, Any]:
        data_schema = {
            "type": "object",
            "properties": {
                "dataset_summary": {
                    "type": "object",
                    "properties": {
                        "total_rows": {"type": "integer"},
                        "total_columns": {"type": "integer"},
                        "memory_usage_mb": {"type": "number"},
                        "data_types": {"type": "object"},
                        "missing_data_summary": {"type": "object"}
                    }
                },
                "column_profiles": {
                    "type": "array",
                    "items": COMMON_SCHEMAS["column_profile"]
                },
                "correlations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column1": {"type": "string"},
                            "column2": {"type": "string"},
                            "correlation": {"type": "number"},
                            "strength": {"type": "string"}
                        }
                    }
                },
                "insights": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["dataset_summary", "column_profiles"]
        }
        
        return create_response_schema(data_schema)
    
    def _execute_tool(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data profiling."""
        # Dataset summary
        dataset_summary = self._generate_dataset_summary(data)
        
        # Column profiles
        column_profiles = self._generate_column_profiles(data, params)
        
        # Correlations
        correlations = self._analyze_correlations(data, params) if params.get("include_correlations", True) else []
        
        # GenAI insights
        insights = self._generate_insights(data, dataset_summary, column_profiles)
        
        # Recommendations
        recommendations = self._generate_profiling_recommendations(data, column_profiles, correlations)
        
        return {
            "dataset_summary": dataset_summary,
            "column_profiles": column_profiles,
            "correlations": correlations,
            "insights": insights,
            "recommendations": recommendations
        }
    
    def _generate_dataset_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate high-level dataset summary."""
        memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        data_types = {}
        for dtype in data.dtypes.value_counts().index:
            data_types[str(dtype)] = int(data.dtypes.value_counts()[dtype])
        
        missing_summary = {}
        total_missing = data.isnull().sum().sum()
        missing_summary["total_missing_values"] = int(total_missing)
        missing_summary["missing_percentage"] = round(total_missing / (data.shape[0] * data.shape[1]) * 100, 2)
        missing_summary["columns_with_missing"] = int((data.isnull().sum() > 0).sum())
        
        return {
            "total_rows": int(data.shape[0]),
            "total_columns": int(data.shape[1]),
            "memory_usage_mb": round(memory_usage, 2),
            "data_types": data_types,
            "missing_data_summary": missing_summary
        }
    
    def _generate_column_profiles(self, data: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed profiles for each column."""
        profiles = []
        
        for column in data.columns:
            profile = {
                "column_name": column,
                "data_type": str(data[column].dtype),
                "null_count": int(data[column].isnull().sum()),
                "null_percentage": round(data[column].isnull().sum() / len(data) * 100, 2),
                "unique_count": int(data[column].nunique()),
                "unique_percentage": round(data[column].nunique() / len(data) * 100, 2),
                "quality_issues": []
            }
            
            # Type-specific statistics
            if data[column].dtype in ['int64', 'float64']:
                valid_data = data[column].dropna()
                if len(valid_data) > 0:
                    profile.update({
                        "min_value": float(valid_data.min()),
                        "max_value": float(valid_data.max()),
                        "mean_value": float(valid_data.mean()),
                        "std_dev": float(valid_data.std()) if len(valid_data) > 1 else 0.0
                    })
                else:
                    profile.update({
                        "min_value": None,
                        "max_value": None,
                        "mean_value": None,
                        "std_dev": None
                    })
                
                # Identify quality issues for numeric columns
                if len(valid_data) > 0:
                    if (valid_data < 0).any() and ('age' in column.lower() or 'count' in column.lower()):
                        profile["quality_issues"].append("Negative values in positive-only field")
                    
                    # Check for outliers
                    Q1, Q3 = valid_data.quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    outliers = valid_data[(valid_data < Q1 - 1.5 * IQR) | (valid_data > Q3 + 1.5 * IQR)]
                    if len(outliers) / len(valid_data) > 0.1:
                        profile["quality_issues"].append("High outlier rate")
            
            else:
                # String/object columns
                valid_data = data[column].dropna()
                if len(valid_data) > 0:
                    # Sample values
                    sample_values = valid_data.head(5).tolist()
                    profile["min_value"] = str(min(sample_values, key=len)) if sample_values else None
                    profile["max_value"] = str(max(sample_values, key=len)) if sample_values else None
                    profile["mean_value"] = None
                    profile["std_dev"] = None
                    
                    # String-specific quality issues
                    str_data = valid_data.astype(str)
                    
                    # Check for mixed formats
                    lengths = str_data.str.len()
                    if lengths.std() / lengths.mean() > 1.0:  # High variation in length
                        profile["quality_issues"].append("Inconsistent string lengths")
                    
                    # Check for encoding issues
                    if any('?' in str(val) for val in str_data.head(100)):
                        profile["quality_issues"].append("Potential encoding issues")
                else:
                    profile.update({
                        "min_value": None,
                        "max_value": None,
                        "mean_value": None,
                        "std_dev": None
                    })
            
            profiles.append(profile)
        
        return profiles
    
    def _analyze_correlations(self, data: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze correlations between numeric columns."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return []
        
        correlations = []
        corr_matrix = numeric_data.corr()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if not np.isnan(corr_value):
                    # Determine correlation strength
                    abs_corr = abs(corr_value)
                    if abs_corr >= 0.8:
                        strength = "very_strong"
                    elif abs_corr >= 0.6:
                        strength = "strong"
                    elif abs_corr >= 0.4:
                        strength = "moderate"
                    elif abs_corr >= 0.2:
                        strength = "weak"
                    else:
                        strength = "very_weak"
                    
                    correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(float(corr_value), 4),
                        "strength": strength
                    })
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return correlations[:20]  # Return top 20 correlations
    
    def _generate_insights(
        self, 
        data: pd.DataFrame, 
        dataset_summary: Dict[str, Any], 
        column_profiles: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate GenAI-powered insights about the data."""
        insights = []
        
        # Dataset size insights
        total_rows = dataset_summary["total_rows"]
        total_cols = dataset_summary["total_columns"]
        
        if total_rows < 100:
            insights.append("Small dataset detected - consider collecting more samples for robust analysis")
        elif total_rows > 1000000:
            insights.append("Large dataset detected - consider sampling for exploratory analysis")
        
        if total_cols > 50:
            insights.append("High-dimensional dataset - consider feature selection or dimensionality reduction")
        
        # Missing data insights
        missing_pct = dataset_summary["missing_data_summary"]["missing_percentage"]
        if missing_pct > 20:
            insights.append(f"High missing data rate ({missing_pct}%) - investigate data collection processes")
        elif missing_pct > 5:
            insights.append(f"Moderate missing data ({missing_pct}%) - consider imputation strategies")
        
        # Column-specific insights
        high_cardinality_cols = [p for p in column_profiles if p["unique_percentage"] > 95]
        if high_cardinality_cols:
            insights.append(f"Potential identifier columns detected: {[c['column_name'] for c in high_cardinality_cols]}")
        
        constant_cols = [p for p in column_profiles if p["unique_count"] == 1]
        if constant_cols:
            insights.append(f"Constant columns detected (consider removal): {[c['column_name'] for c in constant_cols]}")
        
        # Data quality insights
        problematic_cols = [p for p in column_profiles if len(p["quality_issues"]) > 0]
        if problematic_cols:
            insights.append(f"{len(problematic_cols)} columns have quality issues requiring attention")
        
        # Data type insights
        data_types = dataset_summary["data_types"]
        if "object" in data_types and data_types["object"] > data_types.get("int64", 0) + data_types.get("float64", 0):
            insights.append("Predominantly text data - consider text processing and feature extraction")
        
        return insights
    
    def _generate_profiling_recommendations(
        self, 
        data: pd.DataFrame, 
        column_profiles: List[Dict[str, Any]], 
        correlations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on profiling results."""
        recommendations = []
        
        # Missing data recommendations
        high_missing_cols = [p for p in column_profiles if p["null_percentage"] > 50]
        if high_missing_cols:
            recommendations.append(f"Consider dropping columns with >50% missing data: {[c['column_name'] for c in high_missing_cols]}")
        
        # Data type recommendations
        potential_numeric = []
        for profile in column_profiles:
            if profile["data_type"] == "object" and profile["unique_count"] < profile["null_count"] + 100:
                # Might be categorical
                recommendations.append(f"Consider encoding categorical column: {profile['column_name']}")
        
        # Correlation recommendations
        strong_correlations = [c for c in correlations if c["strength"] in ["strong", "very_strong"]]
        if strong_correlations:
            recommendations.append("Strong correlations detected - consider feature selection to reduce multicollinearity")
        
        # Quality issue recommendations
        for profile in column_profiles:
            if profile["quality_issues"]:
                recommendations.append(f"Address quality issues in {profile['column_name']}: {', '.join(profile['quality_issues'])}")
        
        # Performance recommendations
        if len(data) > 100000:
            recommendations.append("Consider data sampling or chunked processing for large dataset")
        
        return recommendations


@register_tool
class AnomalyDetector(BaseDataTool):
    """
    GenAI-powered anomaly detection tool.
    
    Detects anomalies in data using statistical methods enhanced with
    GenAI reasoning for context-aware anomaly interpretation.
    """
    
    def __init__(self, genai_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        super().__init__(
            name="AnomalyDetector",
            version="1.0.0",
            tool_type=DataToolType.QUALITY_ANALYSIS,
            genai_provider=genai_provider,
            model_name=model_name
        )
    
    def _get_request_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["statistical", "isolation_forest", "local_outlier_factor", "auto"],
                    "default": "auto"
                },
                "contamination": {"type": "number", "minimum": 0, "maximum": 0.5, "default": 0.1},
                "include_explanations": {"type": "boolean", "default": True},
                "columns_to_analyze": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    
    def _get_response_schema(self) -> Dict[str, Any]:
        return create_response_schema(COMMON_SCHEMAS["anomaly_detection"])
    
    def _execute_tool(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute anomaly detection."""
        method = params.get("method", "auto")
        contamination = params.get("contamination", 0.1)
        columns_to_analyze = params.get("columns_to_analyze", [])
        
        if not columns_to_analyze:
            # Auto-select numeric columns
            columns_to_analyze = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns_to_analyze:
            return {
                "anomalies_detected": 0,
                "anomaly_score": 0.0,
                "anomalous_rows": [],
                "anomaly_reasons": [],
                "recommended_actions": ["No numeric columns available for anomaly detection"]
            }
        
        # Detect anomalies
        anomalous_rows, anomaly_scores = self._detect_anomalies(
            data[columns_to_analyze], method, contamination
        )
        
        # Generate explanations
        anomaly_reasons = self._explain_anomalies(
            data, anomalous_rows, columns_to_analyze, params
        ) if params.get("include_explanations", True) else []
        
        # Generate recommendations
        recommended_actions = self._generate_anomaly_recommendations(
            data, anomalous_rows, anomaly_reasons
        )
        
        return {
            "anomalies_detected": len(anomalous_rows),
            "anomaly_score": float(np.mean(anomaly_scores)) if anomaly_scores else 0.0,
            "anomalous_rows": [int(row) for row in anomalous_rows],
            "anomaly_reasons": anomaly_reasons,
            "recommended_actions": recommended_actions
        }
    
    def _detect_anomalies(
        self, 
        data: pd.DataFrame, 
        method: str, 
        contamination: float
    ) -> tuple:
        """Detect anomalies using specified method."""
        if method == "auto":
            method = "statistical"  # Default to statistical for simplicity
        
        if method == "statistical":
            return self._statistical_anomaly_detection(data, contamination)
        else:
            # Placeholder for other methods (would require sklearn)
            return self._statistical_anomaly_detection(data, contamination)
    
    def _statistical_anomaly_detection(self, data: pd.DataFrame, contamination: float) -> tuple:
        """Simple statistical anomaly detection using IQR method."""
        anomalous_rows = set()
        anomaly_scores = []
        
        for column in data.columns:
            column_data = data[column].dropna()
            if len(column_data) == 0:
                continue
            
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            anomalous_rows.update(outliers.index.tolist())
            
            # Calculate anomaly scores
            for idx in outliers.index:
                value = data.loc[idx, column]
                if value < lower_bound:
                    score = (lower_bound - value) / IQR if IQR > 0 else 1.0
                else:
                    score = (value - upper_bound) / IQR if IQR > 0 else 1.0
                anomaly_scores.append(min(score, 1.0))
        
        # Limit to contamination percentage
        max_anomalies = int(len(data) * contamination)
        anomalous_rows = list(anomalous_rows)[:max_anomalies]
        
        return anomalous_rows, anomaly_scores[:len(anomalous_rows)]
    
    def _explain_anomalies(
        self, 
        data: pd.DataFrame, 
        anomalous_rows: List[int], 
        columns_analyzed: List[str],
        params: Dict[str, Any]
    ) -> List[str]:
        """Generate explanations for detected anomalies."""
        explanations = []
        
        for row_idx in anomalous_rows[:10]:  # Limit to first 10 for performance
            row_data = data.loc[row_idx]
            reasons = []
            
            for column in columns_analyzed:
                value = row_data[column]
                if pd.isna(value):
                    continue
                
                column_data = data[column].dropna()
                mean_val = column_data.mean()
                std_val = column_data.std()
                
                # Check how many standard deviations away from mean
                if std_val > 0:
                    z_score = abs(value - mean_val) / std_val
                    if z_score > 2:
                        reasons.append(f"{column}: {value} (z-score: {z_score:.2f})")
            
            if reasons:
                explanations.append(f"Row {row_idx}: Anomalous values in {', '.join(reasons)}")
        
        return explanations
    
    def _generate_anomaly_recommendations(
        self, 
        data: pd.DataFrame, 
        anomalous_rows: List[int], 
        anomaly_reasons: List[str]
    ) -> List[str]:
        """Generate recommendations for handling anomalies."""
        recommendations = []
        
        anomaly_rate = len(anomalous_rows) / len(data)
        
        if anomaly_rate > 0.1:
            recommendations.append("High anomaly rate detected - review data collection process")
        elif anomaly_rate > 0.05:
            recommendations.append("Moderate anomaly rate - investigate potential data quality issues")
        else:
            recommendations.append("Low anomaly rate - spot-check detected anomalies for validity")
        
        if len(anomalous_rows) > 0:
            recommendations.append("Review anomalous records for data entry errors")
            recommendations.append("Consider domain expert validation for flagged records")
            recommendations.append("Implement automated monitoring for similar anomalies")
        
        return recommendations


@register_tool  
class DataValidator(BaseDataTool):
    """
    GenAI-powered data validation tool.
    
    Validates data against business rules, constraints, and expectations
    with intelligent rule interpretation and suggestion.
    """
    
    def __init__(self, genai_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        super().__init__(
            name="DataValidator",
            version="1.0.0", 
            tool_type=DataToolType.VALIDATION,
            genai_provider=genai_provider,
            model_name=model_name
        )
    
    def _get_request_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "validation_rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rule_name": {"type": "string"},
                            "rule_type": {"type": "string", "enum": ["range", "format", "uniqueness", "completeness", "custom"]},
                            "column": {"type": "string"},
                            "condition": {"type": "string"},
                            "severity": {"type": "string", "enum": ["error", "warning", "info"]}
                        },
                        "required": ["rule_name", "rule_type", "column", "condition"]
                    }
                },
                "auto_generate_rules": {"type": "boolean", "default": False},
                "include_suggestions": {"type": "boolean", "default": True}
            },
            "required": ["validation_rules"]
        }
    
    def _get_response_schema(self) -> Dict[str, Any]:
        data_schema = {
            "type": "object",
            "properties": {
                "validation_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rule_name": {"type": "string"},
                            "status": {"type": "string", "enum": ["passed", "failed", "warning"]},
                            "violations_count": {"type": "integer"},
                            "violation_percentage": {"type": "number"},
                            "sample_violations": {"type": "array"},
                            "recommendation": {"type": "string"}
                        }
                    }
                },
                "overall_validation_score": {"type": "number", "minimum": 0, "maximum": 1},
                "suggested_rules": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["validation_results", "overall_validation_score"]
        }
        
        return create_response_schema(data_schema)
    
    def _execute_tool(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data validation."""
        validation_rules = params.get("validation_rules", [])
        
        # Auto-generate rules if requested
        if params.get("auto_generate_rules", False):
            auto_rules = self._auto_generate_validation_rules(data)
            validation_rules.extend(auto_rules)
        
        # Execute validation rules
        validation_results = self._execute_validation_rules(data, validation_rules)
        
        # Calculate overall score
        overall_score = self._calculate_validation_score(validation_results)
        
        # Generate suggested rules
        suggested_rules = self._suggest_additional_rules(data, validation_rules) if params.get("include_suggestions", True) else []
        
        return {
            "validation_results": validation_results,
            "overall_validation_score": overall_score,
            "suggested_rules": suggested_rules
        }
    
    def _auto_generate_validation_rules(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Auto-generate basic validation rules based on data analysis."""
        rules = []
        
        for column in data.columns:
            column_data = data[column].dropna()
            
            # Completeness rule
            null_percentage = data[column].isnull().sum() / len(data)
            if null_percentage < 0.05:  # Less than 5% null
                rules.append({
                    "rule_name": f"{column}_completeness",
                    "rule_type": "completeness",
                    "column": column,
                    "condition": "not null",
                    "severity": "warning"
                })
            
            # Range rules for numeric columns
            if data[column].dtype in ['int64', 'float64'] and len(column_data) > 0:
                min_val = column_data.min()
                max_val = column_data.max()
                
                # Age-specific rules
                if 'age' in column.lower():
                    rules.append({
                        "rule_name": f"{column}_age_range",
                        "rule_type": "range",
                        "column": column,
                        "condition": f">= 0 and <= 150",
                        "severity": "error"
                    })
                
                # General range rule based on data
                elif min_val >= 0:  # Positive values
                    rules.append({
                        "rule_name": f"{column}_positive_range",
                        "rule_type": "range", 
                        "column": column,
                        "condition": f">= 0",
                        "severity": "warning"
                    })
            
            # Uniqueness rules
            if data[column].nunique() == len(data):  # Unique column
                rules.append({
                    "rule_name": f"{column}_uniqueness",
                    "rule_type": "uniqueness",
                    "column": column,
                    "condition": "unique",
                    "severity": "error"
                })
        
        return rules
    
    def _execute_validation_rules(self, data: pd.DataFrame, validation_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute validation rules against the data."""
        results = []
        
        for rule in validation_rules:
            rule_name = rule["rule_name"]
            rule_type = rule["rule_type"]
            column = rule["column"]
            condition = rule["condition"]
            severity = rule.get("severity", "error")
            
            if column not in data.columns:
                results.append({
                    "rule_name": rule_name,
                    "status": "failed",
                    "violations_count": 0,
                    "violation_percentage": 0.0,
                    "sample_violations": [],
                    "recommendation": f"Column '{column}' not found in data"
                })
                continue
            
            violations = []
            
            try:
                if rule_type == "completeness":
                    if "not null" in condition.lower():
                        violations = data[data[column].isnull()].index.tolist()
                
                elif rule_type == "range":
                    # Parse range conditions
                    if ">=" in condition and "<=" in condition:
                        # Format: >= X and <= Y
                        parts = condition.split(" and ")
                        min_val = float(parts[0].replace(">=", "").strip())
                        max_val = float(parts[1].replace("<=", "").strip())
                        violations = data[(data[column] < min_val) | (data[column] > max_val)].index.tolist()
                    elif ">=" in condition:
                        min_val = float(condition.replace(">=", "").strip())
                        violations = data[data[column] < min_val].index.tolist()
                    elif "<=" in condition:
                        max_val = float(condition.replace("<=", "").strip())
                        violations = data[data[column] > max_val].index.tolist()
                
                elif rule_type == "uniqueness":
                    if "unique" in condition.lower():
                        duplicated_mask = data.duplicated(subset=[column], keep=False)
                        violations = data[duplicated_mask].index.tolist()
                
                elif rule_type == "format":
                    # Basic format validation (would be more sophisticated in real implementation)
                    if "email" in condition.lower():
                        email_pattern = data[column].astype(str).str.contains(r'^[^@]+@[^@]+\.[^@]+$', regex=True, na=False)
                        violations = data[~email_pattern].index.tolist()
                
                elif rule_type == "custom":
                    # Custom rule evaluation (simplified)
                    try:
                        eval_condition = condition.replace(column, f"data['{column}']")
                        mask = eval(eval_condition)
                        violations = data[~mask].index.tolist()
                    except:
                        violations = []  # Skip invalid custom rules
                
            except Exception as e:
                violations = []
            
            violations_count = len(violations)
            violation_percentage = violations_count / len(data) * 100 if len(data) > 0 else 0
            
            status = "passed"
            if violations_count > 0:
                status = "warning" if severity == "warning" else "failed"
            
            # Sample violations (first 5)
            sample_violations = violations[:5]
            
            # Generate recommendation
            recommendation = self._generate_validation_recommendation(
                rule_type, violations_count, violation_percentage, severity
            )
            
            results.append({
                "rule_name": rule_name,
                "status": status,
                "violations_count": violations_count,
                "violation_percentage": round(violation_percentage, 2),
                "sample_violations": sample_violations,
                "recommendation": recommendation
            })
        
        return results
    
    def _calculate_validation_score(self, validation_results: List[Dict[str, Any]]) -> float:
        """Calculate overall validation score."""
        if not validation_results:
            return 1.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in validation_results:
            # Weight by severity
            weight = 1.0 if result["status"] == "failed" else 0.5 if result["status"] == "warning" else 0.0
            
            # Score based on violation percentage
            violation_pct = result["violation_percentage"] / 100
            rule_score = 1.0 - violation_pct
            
            total_score += rule_score * weight
            total_weight += weight
        
        return round(total_score / total_weight if total_weight > 0 else 1.0, 4)
    
    def _suggest_additional_rules(self, data: pd.DataFrame, existing_rules: List[Dict[str, Any]]) -> List[str]:
        """Suggest additional validation rules based on data analysis."""
        suggestions = []
        existing_columns = {rule["column"] for rule in existing_rules}
        
        for column in data.columns:
            if column in existing_columns:
                continue
            
            # Suggest completeness rules for columns with low null rates
            null_pct = data[column].isnull().sum() / len(data)
            if null_pct < 0.1:
                suggestions.append(f"Add completeness rule for {column} (current null rate: {null_pct:.1%})")
            
            # Suggest format rules for string columns
            if data[column].dtype == 'object':
                sample_values = data[column].dropna().astype(str).head(10)
                
                # Email detection
                if any('@' in val for val in sample_values):
                    suggestions.append(f"Add email format validation for {column}")
                
                # Phone number detection
                if any(val.replace('-', '').replace(' ', '').replace('(', '').replace(')', '').isdigit() for val in sample_values):
                    suggestions.append(f"Add phone number format validation for {column}")
            
            # Suggest range rules for numeric columns
            elif data[column].dtype in ['int64', 'float64']:
                if 'id' in column.lower():
                    suggestions.append(f"Add uniqueness rule for {column} (appears to be an identifier)")
                elif data[column].min() >= 0:
                    suggestions.append(f"Add positive range rule for {column}")
        
        return suggestions[:10]  # Limit to 10 suggestions
    
    def _generate_validation_recommendation(
        self, 
        rule_type: str, 
        violations_count: int, 
        violation_percentage: float, 
        severity: str
    ) -> str:
        """Generate recommendation for validation rule result."""
        if violations_count == 0:
            return "Rule passed successfully"
        
        base_recommendations = {
            "completeness": "Review data collection process for missing values",
            "range": "Verify data entry procedures and implement input validation",
            "uniqueness": "Investigate duplicate records and establish unique constraints",
            "format": "Implement format validation at data entry point",
            "custom": "Review business logic and data consistency"
        }
        
        base_rec = base_recommendations.get(rule_type, "Review and address violations")
        
        if violation_percentage > 50:
            return f"Critical: {base_rec} - {violation_percentage:.1f}% violations detected"
        elif violation_percentage > 10:
            return f"High priority: {base_rec} - {violation_percentage:.1f}% violations"
        else:
            return f"{base_rec} - {violation_percentage:.1f}% violations found"