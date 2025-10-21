"""
Custom exceptions for the SURG recommendation system.

This module defines all custom exceptions used throughout the SURG package,
providing clear error types for different failure scenarios and enabling
better error handling and debugging.
"""


class SURGException(Exception):
    """
    Base exception class for all SURG-related errors.
    
    All custom exceptions in the SURG package should inherit from this class
    to provide a consistent exception hierarchy and enable catch-all error handling.
    """
    pass


class DataValidationError(SURGException):
    """
    Raised when input data fails validation checks.
    
    This exception is raised when user-provided data (users, items, interactions)
    doesn't meet the expected format, contains invalid values, or fails
    consistency checks.
    """
    pass


class ModelError(SURGException):
    """
    Raised when model training or inference fails.
    
    This exception covers errors during model training, loading, saving,
    or recommendation generation that aren't related to data validation.
    """
    pass


class GenAIError(SURGException):
    """
    Raised when GenAI backend operations fail.
    
    This exception is raised when interactions with the GenAI backend
    (e.g., OpenAI API) fail due to network issues, API limits, or
    invalid responses.
    """
    pass


class ConfigurationError(SURGException):
    """
    Raised when configuration is invalid or missing.
    
    This exception is raised when required configuration values are missing,
    invalid, or when environment setup is incomplete.
    """
    pass


class EmbeddingError(SURGException):
    """
    Raised when embedding generation fails.
    
    This exception covers errors during embedding generation, including
    model loading failures, dimension mismatches, or processing errors.
    """
    pass