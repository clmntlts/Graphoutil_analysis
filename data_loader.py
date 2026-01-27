"""
Data loading and preprocessing utilities.
Handles Excel file loading, column mapping, and data cleaning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of handwriting data"""
    
    def __init__(self, config):
        """
        Initialize DataLoader with configuration
        
        Args:
            config: AnalysisConfig object
        """
        self.config = config
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess main data from Excel file
        
        Returns:
            Cleaned DataFrame with standardized columns
        """
        logger.info(f"Loading data from {self.config.input_file}")
        
        # Load Excel file
        df = pd.read_excel(self.config.input_file)
        
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # Map columns
        column_mapping = {
            self.config.col_time: "PacketTime",
            self.config.col_x: "X",
            self.config.col_y: "Y",
            self.config.col_pressure: "NormalPressure"
        }
        
        # Select and rename columns
        keep_cols = [self.config.col_time, self.config.col_x, 
                    self.config.col_y, self.config.col_pressure]
        
        try:
            df = df[[c for c in df.columns if any(k in c for k in keep_cols)]]
            df.columns = ["PacketTime", "X", "Y", "NormalPressure"]
        except KeyError as e:
            logger.error(f"Could not find required columns: {e}")
            logger.info(f"Available columns: {df.columns.tolist()}")
            raise
        
        # Clean data
        df = self._clean_data(df)
        
        logger.info(f"Loaded {len(df)} data points")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with missing values
        initial_count = len(df)
        df = df.dropna(subset=["PacketTime", "X", "Y", "NormalPressure"])
        
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} rows with missing values")
        
        # Remove negative pressure values
        negative_pressure = df["NormalPressure"] < 0
        if negative_pressure.any():
            logger.warning(f"Removed {negative_pressure.sum()} rows with negative pressure")
            df = df[~negative_pressure]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def load_segmentation_sheet(self) -> Optional[pd.DataFrame]:
        """
        Load segmentation sheet if available
        
        Returns:
            Segmentation DataFrame or None if not found
        """
        try:
            seg = pd.read_excel(self.config.input_file, 
                              sheet_name="Segmentation", 
                              header=None)
            seg.columns = ["Type", "Start", "End", "WordIndex"]
            seg["WordIndex"] = seg["WordIndex"].ffill().astype(int)
            logger.info("Loaded segmentation sheet")
            return seg
        except Exception as e:
            logger.warning(f"Could not load segmentation sheet: {e}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate loaded data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for empty dataframe
        if len(df) == 0:
            issues.append("DataFrame is empty")
        
        # Check for required columns
        required_cols = ["PacketTime", "X", "Y", "NormalPressure"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for temporal consistency
        if "PacketTime" in df.columns:
            time_diffs = df["PacketTime"].diff()
            if (time_diffs < 0).any():
                issues.append("Time values are not monotonically increasing")
        
        # Check for spatial outliers
        if "X" in df.columns and "Y" in df.columns:
            x_range = df["X"].max() - df["X"].min()
            y_range = df["Y"].max() - df["Y"].min()
            if x_range == 0 or y_range == 0:
                issues.append("No spatial variation detected")
        
        # Check pressure values
        if "NormalPressure" in df.columns:
            if df["NormalPressure"].max() == 0:
                issues.append("All pressure values are zero")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Data validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues


def load_and_validate(config) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Convenience function to load and validate data
    
    Args:
        config: AnalysisConfig object
        
    Returns:
        Tuple of (main_df, segmentation_df)
    """
    loader = DataLoader(config)
    
    # Load main data
    df = loader.load_data()
    
    # Validate
    is_valid, issues = loader.validate_data(df)
    if not is_valid:
        raise ValueError(f"Data validation failed: {issues}")
    
    # Load segmentation if available
    seg = loader.load_segmentation_sheet()
    
    return df, seg
