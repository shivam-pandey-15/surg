"""
Base Recommendation Algorithm Interface

This module defines the base interface that all recommendation algorithms
should implement for consistency across traditional, matrix factorization,
and deep learning approaches.

Design Principles:
- Consistent API across all algorithms
- Support for both explicit and implicit feedback
- Extensible for custom algorithms
- Integration hooks for GenAI enhancements

Key Interface Methods:
- fit(): Train the model on interaction data
- predict(): Predict scores for user-item pairs
- recommend(): Generate top-k recommendations
- get_embeddings(): Extract learned representations
- explain_recommendation(): Provide explanations

All algorithms should inherit from BaseRecommender and implement
the abstract methods according to their specific approach.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass,field
import pickle
import warnings

class RecommendationResult:
    """
    Structured result for recommendation outputs.
    Attributes:
        user_id: User identifier
        item_ids: List of recommended item identifiers
        scores: List of scores for the recommended items
        metadata: Optional dictionary for additional information
            - explanations: List of explanations for each recommendation
            - confidence: Confidence scores for recommendations
            - method: Algorithm/method used for generation
            - reasoning: Natural language reasoning (GenAI)
            - features: Contributing features and weights
            - context: Context information used
    """

    user_id: Any
    item_ids: List[Any]
    scores: List[Any]
    metadata: Optional[Dict[str,Any]] = field(deffault_factory=dict)

    def __post_init__(self):
        '''Validate recommendation result structure.'''
        if len(self.item_ids) != len(self.scores):
            raise ValueError("Length of item_ids and scores must match.")
        if not self.metadata:
            self.metadata = {}

class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation algorithms.
    
    This class defines the core interface that all recommendation
    algorithms must implement to ensure consistency and interoperability
    across different approaches. (collaborative filtering, matrix factorization, deep learning).
    
    Supports data-driven workflows and GenAI integration.
    1. Initialize with dataset (optional)
    2. Analyze data characteristics
    3. Understand problem type
    4. Think and plan approach (GenAI)
    5. Fit model with execution plan
    6. Generate recommendations

    Attributes:
        config: Configuration dictionary for algorithm 
        dataset:  Optional dataset provided at initialization
        is_fitted: Flag indicating if the model is trained
        user_id_mapping: Mapping of user IDs to internal indices
        item_id_mapping: Mapping of item IDs to internal indices
        n_users: Number of unique users
        n_items: Number of unique items
        enable_genai: Flag to enable GenAI features
        problem_type: Detected problem type (rating prediction, ranking, etc.)
        interactions: Interaction data used for training
    """

    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 dataset: Optional[pd.DataFrame] = None,
                 enable_genai: bool = False):
        """
        Initialize the recommender with configuration and optional dataset.
        
        Args:
            config: Configuration dictionary with algorithm-specific parameters
            dataset: Optional dataset for data-driven initialization
                     Expected columns: user_id, item_id, rating/interaction
            enable_genai: Whether to enable GenAI-powered analysis and planning
        """
        self.config = config or {}
        self.dataset = dataset
        self.enable_genai = enable_genai
        
        # Model state
        self.is_fitted = False
        self.user_id_map = {}
        self.item_id_map = {}
        self.n_users = 0
        self.n_items = 0
        
        # Data-driven attributes
        self.problem_type = None
        self.interactions = None
        self._analysis_cache = None
        self._problem_understanding = None
        
        # Reverse mappings for decoding
        self._user_id_reverse = {}
        self._item_id_reverse = {}

    @abstractmethod
    def fit(self,
            interactions: Union[pd.DataFrame, np.ndarray] = None,
            user_features: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            item_features: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            execution_plan: Optional[Dict[str, Any]] = None) -> 'BaseRecommender':
        """
        Train the recommendation model on interaction data.
        
        Args:
            interactions: User-item interaction data
                - DataFrame: columns [user_id, item_id, rating/interaction, ...]
                - ndarray: sparse or dense interaction matrix (users x items)
            user_features: Optional user feature matrix
                - DataFrame: user_id as index, features as columns
                - ndarray: (n_users, n_features)
            item_features: Optional item feature matrix
                - DataFrame: item_id as index, features as columns
                - ndarray: (n_items, n_features)
            execution_plan: Optional execution plan from think_and_plan()
                Contains strategy, hyperparameters, preprocessing steps
            
        Returns:
            self: Fitted recommender instance
            
        Raises:
            ValueError: If interaction data format is invalid
            
        Notes:
            - Must set self.is_fitted = True after successful training
            - Must populate self.user_id_map and self.item_id_map
            - Should use execution_plan if provided (from GenAI planning)
            - Store interactions for later analysis/explanation
        """
        pass

    @abstractmethod
    def predict(self,
                  user_ids: Union[List[Any], Any],
                  item_ids: Union[List[Any], Any]) -> Union[float, np.ndarray, List[RecommendationResult]]:
        """
        Predict scores for specific user-item pairs.
        
        Args:
            user_ids: Single user ID or list of user IDs
            item_ids: Single item ID or list of item IDs
                     Must be same length as user_ids for pairwise prediction
            
        Returns:
            Predicted score(s) for the user-item pair(s)
            - Single float if scalar inputs
            - np.ndarray if list/array inputs
            
        Raises:
            ValueError: If model is not fitted or IDs are invalid
            
        Notes:
            - Supports both scalar and vectorized predictions
            - Use _encode_ids() to convert external IDs to internal indices
            - Should handle unknown users/items gracefully (e.g., return mean score)
        """
        pass

    @abstractmethod
    def recommend(self,
                  user_ids: Union[Any, List[Any]],
                  k: int = 10,
                  exclude_seen: bool = True,
                  filter_items: Optional[List[Any]] = None,
                  context: Optional[Dict[str, Any]] = None) -> Union[RecommendationResult, List[RecommendationResult]]:
        """
        Generate top-k recommendations for user(s).
        
        Args:
            user_ids: Single user ID or list of user IDs
            k: Number of recommendations to generate
            exclude_seen: Whether to exclude items user has already interacted with
            filter_items: Optional list of items to exclude from recommendations
            context: Optional context information (for GenAI enhancement)
                - user_intent: Current user intent/mood
                - session_history: Recent interactions
                - constraints: Business rules or constraints
            
        Returns:
            RecommendationResult or list of RecommendationResult objects
            
        Raises:
            ValueError: If model is not fitted or user_ids are invalid
            
        Notes:
            - If enable_genai=True and context provided, enrich recommendations
            - Populate metadata with explanations, confidence scores
            - Use _decode_ids() to convert internal indices to external IDs
        """
        pass

    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze dataset characteristics for problem understanding.
        
        Returns:
            Dictionary containing:
                - n_users: Number of unique users
                - n_items: Number of unique items
                - n_interactions: Total interactions
                - sparsity: Dataset sparsity (1 - density)
                - rating_distribution: Distribution of ratings/interactions
                - temporal_info: Temporal patterns if timestamps present
                - user_activity: User activity statistics
                - item_popularity: Item popularity statistics
                - problem_indicators: Signals for problem type detection
                
        Notes:
            - Called before fit() in data-driven workflow
            - Results cached in self._analysis_cache
            - Should work with self.dataset if provided
            - Used by understand_problem() for strategy recommendation
        """
        pass

    def describe(self, analysis: Optional[Dict[str,Any]] = None) -> str:
        """
        Generate natural language description of dataset analysis.
        
        Args:
            analysis: Optional analysis dict (uses self._analysis_cache if None)
            
        Returns:
            Natural language description of dataset characteristics
            
        Notes:
            - If enable_genai=True, uses LLM for rich description
            - Otherwise, returns template-based description
            - Useful for data exploration and reporting
        """
        if analysis is None:
            analysis = self._analysis_cache or self.analyze()
        
        if self.enable_genai:
            return self._generate_genai_description(analysis)
        else:
            return self._generate_template_description(analysis)
    
    def _generate_template_description(self, analysis: Dict[str, Any]) -> str:
        """Generate template-based description."""
        desc = f"""
        Dataset Analysis Summary:
        - Users: {analysis.get('n_users', 'N/A')}
        - Items: {analysis.get('n_items', 'N/A')}
        - Interactions: {analysis.get('n_interactions', 'N/A')}
        - Sparsity: {analysis.get('sparsity', 'N/A'):.2%}
        - Problem Type: {analysis.get('problem_type', 'Unknown')}
        """
        return desc.strip()

    def _generate_genai_description(self, analysis: Dict[str, Any]) -> str:
        """Generate GenAI-powered description (placeholder for now)."""
        # TODO: Integrate with GenAI layer for natural language generation
        return self._generate_template_description(analysis) + "\n(GenAI enhancement available)"
    
    def understand_problem(self, 
                          problem_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Understand the recommendation problem type and suggest strategy.
        
        Args:
            problem_description: Optional natural language problem description
                                Enhances analysis with domain context
        
        Returns:
            Dictionary containing:
                - problem_type: Detected type (rating, implicit, categorical, temporal)
                - recommended_algorithms: List of suitable algorithms
                - evaluation_metrics: Suggested evaluation metrics
                - preprocessing_steps: Recommended preprocessing
                - reasoning: Explanation of recommendations
                
        Notes:
            - Uses analyze() results to detect problem characteristics
            - If enable_genai=True, uses LLM reasoning for strategy
            - Caches result in self._problem_understanding
        """
        # Use cached analysis or perform new one
        analysis = self._analysis_cache or self.analyze()
        
        # Detect problem type from data characteristics
        problem_type = self._detect_problem_type(analysis)
        
        understanding = {
            'problem_type': problem_type,
            'recommended_algorithms': self._recommend_algorithms(problem_type, analysis),
            'evaluation_metrics': self._recommend_metrics(problem_type),
            'preprocessing_steps': self._recommend_preprocessing(problem_type, analysis),
            'reasoning': self._generate_reasoning(problem_type, analysis, problem_description)
        }
        
        self._problem_understanding = understanding
        return understanding
    
    def _detect_problem_type(self, analysis: Dict[str, Any]) -> str:
        """Detect problem type from analysis."""
        # Simple heuristics (can be overridden by subclasses)
        if 'rating_distribution' in analysis:
            # Check if ratings are discrete or continuous
            ratings = analysis.get('rating_distribution', {})
            if len(ratings) <= 5:
                return 'explicit_rating'
            else:
                return 'implicit_feedback'
        else:
            return 'implicit_feedback'
    
    def _recommend_algorithms(self, problem_type: str, analysis: Dict[str, Any]) -> List[str]:
        """Recommend suitable algorithms based on problem type."""
        recommendations = {
            'explicit_rating': ['SVD', 'NMF', 'LightFM', 'NeuralCF'],
            'implicit_feedback': ['ALS', 'BPR', 'LightFM', 'CDAE'],
            'categorical': ['LightFM', 'FactorizationMachines'],
            'temporal': ['RNN', 'Transformer', 'SASRec']
        }
        return recommendations.get(problem_type, ['LightFM', 'NeuralCF'])
    
    def _recommend_metrics(self, problem_type: str) -> List[str]:
        """Recommend evaluation metrics for problem type."""
        metrics = {
            'explicit_rating': ['RMSE', 'MAE', 'R2'],
            'implicit_feedback': ['Precision@K', 'Recall@K', 'NDCG@K', 'MAP@K'],
            'categorical': ['Precision@K', 'Recall@K', 'F1@K'],
            'temporal': ['Hit Rate', 'MRR', 'NDCG@K']
        }
        return metrics.get(problem_type, ['Precision@K', 'NDCG@K'])
    
    def _recommend_preprocessing(self, problem_type: str, analysis: Dict[str, Any]) -> List[str]:
        """Recommend preprocessing steps."""
        steps = ['normalize_ids', 'handle_cold_start']
        
        sparsity = analysis.get('sparsity', 0)
        if sparsity > 0.99:
            steps.append('aggressive_filtering')
        
        if problem_type == 'explicit_rating':
            steps.append('normalize_ratings')
        
        return steps
    
    def _generate_reasoning(self, 
                          problem_type: str, 
                          analysis: Dict[str, Any],
                          problem_description: Optional[str]) -> str:
        """Generate reasoning for recommendations."""
        reasoning = f"Based on analysis, this appears to be a {problem_type} problem. "
        reasoning += f"With {analysis.get('n_users', 0)} users and {analysis.get('n_items', 0)} items, "
        reasoning += f"and sparsity of {analysis.get('sparsity', 0):.2%}."
        
        if problem_description:
            reasoning += f" User context: {problem_description}"
        
        return reasoning
    
    def think_and_plan(self, 
                      objective: str,
                      max_steps: int = 10,
                      constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        GenAI-powered recursive thinking and planning for optimization.
        
        Args:
            objective: High-level objective (e.g., "maximize NDCG@10")
            max_steps: Maximum recursive thinking steps
            constraints: Optional constraints (time, memory, accuracy requirements)
            
        Returns:
            Dictionary containing:
                - execution_plan: Detailed execution plan with steps
                - hyperparameters: Recommended hyperparameters
                - strategies: Alternative strategies considered
                - reasoning_trace: Step-by-step reasoning
                - confidence: Confidence in plan
                
        Notes:
            - Only works if enable_genai=True, otherwise returns basic plan
            - Uses recursive reasoning through thinking engine
            - Plan can be passed to fit() for execution
            - Considers problem understanding and constraints
        """
        if not self.enable_genai:
            warnings.warn("GenAI is disabled. Returning basic plan.")
            return self._generate_basic_plan(objective)
        
        # TODO: Integrate with ThinkingEngine for recursive planning
        # For now, return structured plan
        return self._generate_basic_plan(objective)
    
    def _generate_basic_plan(self, objective: str) -> Dict[str, Any]:
        """Generate basic execution plan without GenAI."""
        problem = self._problem_understanding or self.understand_problem()
        
        return {
            'execution_plan': {
                'steps': [
                    'preprocess_data',
                    'initialize_model',
                    'train_model',
                    'evaluate_model'
                ],
                'preprocessing': problem.get('preprocessing_steps', []),
                'algorithm': problem.get('recommended_algorithms', ['LightFM'])[0]
            },
            'hyperparameters': self._get_default_hyperparameters(),
            'strategies': ['default_strategy'],
            'reasoning_trace': [f"Objective: {objective}", "Using standard workflow"],
            'confidence': 0.7
        }
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for algorithm."""
        return {
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 256,
            'embedding_dim': 64
        }
    
    def get_similar_items(self,
                         item_ids: Union[Any, List[Any]],
                         k: int = 10) -> Union[List[Tuple[Any, float]], List[List[Tuple[Any, float]]]]:
        """
        Find similar items based on learned representations.
        
        Args:
            item_ids: Single item ID or list of item IDs
            k: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples for each input item
            
        Raises:
            NotImplementedError: If algorithm doesn't support item similarity
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support item similarity computation")
    
    def get_similar_users(self,
                         user_ids: Union[Any, List[Any]],
                         k: int = 10) -> Union[List[Tuple[Any, float]], List[List[Tuple[Any, float]]]]:
        """
        Find similar users based on learned representations.
        
        Args:
            user_ids: Single user ID or list of user IDs
            k: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples for each input user
            
        Raises:
            NotImplementedError: If algorithm doesn't support user similarity
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support user similarity computation")
    
    def get_user_embeddings(self, user_ids: Optional[List[Any]] = None) -> np.ndarray:
        """
        Extract learned user embeddings/representations.
        
        Args:
            user_ids: Optional list of specific user IDs (if None, return all)
            
        Returns:
            User embedding matrix (n_users x embedding_dim)
            
        Raises:
            NotImplementedError: If algorithm doesn't learn user embeddings
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support user embeddings")
    
    def get_item_embeddings(self, item_ids: Optional[List[Any]] = None) -> np.ndarray:
        """
        Extract learned item embeddings/representations.
        
        Args:
            item_ids: Optional list of specific item IDs (if None, return all)
            
        Returns:
            Item embedding matrix (n_items x embedding_dim)
            
        Raises:
            NotImplementedError: If algorithm doesn't learn item embeddings
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support item embeddings")
    
    def explain_recommendation(self,
                             user_id: Any,
                             item_id: Any,
                             top_k_features: int = 5) -> Dict[str, Any]:
        """
        Provide explanation for why an item was recommended to a user.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            top_k_features: Number of top contributing features to return
            
        Returns:
            Dictionary containing explanation details:
                - score: Predicted score
                - rank: Rank in user's recommendations
                - features: Top contributing features and their weights
                - reasoning: Human-readable explanation (if GenAI enabled)
                - confidence: Confidence in explanation
                - similar_items: Similar items user liked
                
        Raises:
            NotImplementedError: If algorithm doesn't support explanations
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support explanations")
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseRecommender':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded recommender instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def _validate_fitted(self) -> None:
        """
        Check if model has been fitted.
        
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} must be fitted before making predictions")
    
    def _build_id_mappings(self, interactions: pd.DataFrame) -> None:
        """
        Build internal ID mappings from interaction data.
        
        Args:
            interactions: DataFrame with user_id and item_id columns
        """
        # Build user ID mapping
        unique_users = interactions['user_id'].unique()
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self._user_id_reverse = {idx: uid for uid, idx in self.user_id_map.items()}
        self.n_users = len(unique_users)
        
        # Build item ID mapping
        unique_items = interactions['item_id'].unique()
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self._item_id_reverse = {idx: iid for iid, idx in self.item_id_map.items()}
        self.n_items = len(unique_items)
    
    def _encode_ids(self, 
                    ids: Union[Any, List[Any]], 
                    id_mapping: Dict[Any, int],
                    id_type: str = "user") -> Union[int, np.ndarray]:
        """
        Convert external IDs to internal indices.
        
        Args:
            ids: Single ID or list of IDs
            id_mapping: Mapping from external IDs to internal indices
            id_type: Type of ID ("user" or "item") for error messages
            
        Returns:
            Internal index or array of indices
            
        Raises:
            ValueError: If ID is not found in mapping
        """
        if isinstance(ids, (list, np.ndarray)):
            encoded = []
            for id_ in ids:
                if id_ not in id_mapping:
                    raise ValueError(f"Unknown {id_type} ID: {id_}")
                encoded.append(id_mapping[id_])
            return np.array(encoded)
        else:
            if ids not in id_mapping:
                raise ValueError(f"Unknown {id_type} ID: {ids}")
            return id_mapping[ids]
    
    def _decode_ids(self,
                    indices: Union[int, np.ndarray],
                    reverse_mapping: Dict[int, Any],
                    id_type: str = "user") -> Union[Any, List[Any]]:
        """
        Convert internal indices to external IDs.
        
        Args:
            indices: Single index or array of indices
            reverse_mapping: Mapping from internal indices to external IDs
            id_type: Type of ID ("user" or "item") for error messages
            
        Returns:
            External ID or list of IDs
        """
        if isinstance(indices, (list, np.ndarray)):
            return [reverse_mapping[idx] for idx in indices]
        else:
            return reverse_mapping[indices]
    
    def __repr__(self) -> str:
        """String representation of the recommender."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        genai_status = "GenAI enabled" if self.enable_genai else "GenAI disabled"
        return f"{self.__class__.__name__}({fitted_status}, {genai_status}, n_users={self.n_users}, n_items={self.n_items})"


    
