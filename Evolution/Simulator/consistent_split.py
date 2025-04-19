"""
Consistent Train/Validation Split Module

This module provides functions to ensure consistent train/validation splits
across different strategies in the same simulation run.
"""

import numpy as np
import pandas as pd
import logging
import os
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Global variables to store the current split
_training_data = None
_validation_data = None
_split_initialized = False

def initialize_data_split(historical_data_df: pd.DataFrame, 
                         validation_split: float = 0.3, 
                         random_seed: int = 42,
                         use_cross_validation: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Initialize a consistent train/validation split that can be reused across strategies.
    
    Args:
        historical_data_df: DataFrame with historical data
        validation_split: Fraction of data to use for validation (default: 0.3)
        random_seed: Random seed for reproducibility (default: 42)
        use_cross_validation: Whether to use cross-validation (default: True)
        
    Returns:
        tuple: (training_data, validation_data) - DataFrames for training and validation
    """
    global _training_data, _validation_data, _split_initialized
    
    # If already initialized, return the existing split
    if _split_initialized:
        logger.info("Using existing train/validation split")
        return _training_data, _validation_data
    
    # Initialize the split
    training_data = historical_data_df
    validation_data = None
    
    if use_cross_validation and len(historical_data_df) > 10:  # Only use cross-validation if we have enough data
        # Check if we have timestamp column for time-based split
        if 'timestamp1' in historical_data_df.columns:
            # Time-based split (more appropriate for financial data)
            logger.info("Using time-based cross-validation (more realistic for financial data)")
            
            # Sort by timestamp
            sorted_df = historical_data_df.sort_values('timestamp')
            
            # Calculate split point
            split_idx = int(len(sorted_df) * (1 - validation_split))
            
            # Split the data
            training_data = sorted_df.iloc[:split_idx].reset_index(drop=True)
            validation_data = sorted_df.iloc[split_idx:].reset_index(drop=True)
            
            logger.info(f"Cross-validation enabled: Split data into {len(training_data)} training samples (older data) and {len(validation_data)} validation samples (newer data)")
        else:
            # Random split (less ideal for financial data but works if no timestamp)
            logger.info("Using random cross-validation (timestamp column not found)")
            
            # Set random seed for reproducibility
            np.random.seed(random_seed)
            
            # Shuffle the data
            shuffled_indices = np.random.permutation(len(historical_data_df))
            
            # Calculate split point
            split_idx = int(len(historical_data_df) * (1 - validation_split))
            
            # Split the data
            training_indices = shuffled_indices[:split_idx]
            validation_indices = shuffled_indices[split_idx:]
            
            training_data = historical_data_df.iloc[training_indices].reset_index(drop=True)
            validation_data = historical_data_df.iloc[validation_indices].reset_index(drop=True)
            
            logger.info(f"Cross-validation enabled: Split data into {len(training_data)} training samples and {len(validation_data)} validation samples (random split)")
    
    # Store the split for future use
    _training_data = training_data
    _validation_data = validation_data
    _split_initialized = True
    
    return training_data, validation_data

def get_training_data() -> pd.DataFrame:
    """
    Get the training data from the current split.
    
    Returns:
        DataFrame: Training data
    """
    global _training_data, _split_initialized
    
    if not _split_initialized:
        logger.warning("Data split not initialized. Call initialize_data_split first.")
        return None
    
    return _training_data

def get_validation_data() -> Optional[pd.DataFrame]:
    """
    Get the validation data from the current split.
    
    Returns:
        DataFrame: Validation data (or None if cross-validation is disabled)
    """
    global _validation_data, _split_initialized
    
    if not _split_initialized:
        logger.warning("Data split not initialized. Call initialize_data_split first.")
        return None
    
    return _validation_data

def reset_data_split():
    """Reset the data split to allow re-initialization."""
    global _training_data, _validation_data, _split_initialized
    
    _training_data = None
    _validation_data = None
    _split_initialized = False
    
    logger.info("Data split has been reset")
