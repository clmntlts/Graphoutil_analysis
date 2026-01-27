"""
Example unit tests for handwriting analysis modules.
Run with: pytest test_analysis.py
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Import modules to test
from pause_detection import detect_pauses, min_duration_mask
from config import AnalysisConfig


class TestPauseDetection:
    """Tests for pause detection functionality"""
    
    def test_min_duration_mask_basic(self):
        """Test basic filtering of short segments"""
        mask = np.array([True, True, False, True, True, True, False])
        result = min_duration_mask(mask, min_samples=3)
        
        # Only the 3-sample True segment should remain
        expected = np.array([False, False, False, True, True, True, False])
        np.testing.assert_array_equal(result, expected)
    
    def test_min_duration_mask_all_short(self):
        """Test when all segments are too short"""
        mask = np.array([True, False, True, False, True, False])
        result = min_duration_mask(mask, min_samples=2)
        
        # All segments should be filtered out
        expected = np.zeros_like(mask, dtype=bool)
        np.testing.assert_array_equal(result, expected)
    
    def test_detect_pauses_no_pauses(self):
        """Test pause detection with high speed and pressure"""
        t = pd.Series([0.0, 0.005, 0.010, 0.015, 0.020])
        speed = np.array([100, 100, 100, 100, 100])
        pressure = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
        
        mask, pauses = detect_pauses(t, speed, pressure, 
                                     speed_threshold=50, 
                                     min_samples=2)
        
        assert len(pauses) == 0
        assert not mask.any()
    
    def test_detect_pauses_zero_pressure(self):
        """Test pause detection with zero pressure (pen lift)"""
        t = pd.Series([0.0, 0.005, 0.010, 0.015, 0.020])
        speed = np.array([100, 100, 0, 0, 100])
        pressure = pd.Series([0.5, 0.5, 0.0, 0.0, 0.5])
        
        mask, pauses = detect_pauses(t, speed, pressure,
                                     speed_threshold=50,
                                     min_samples=2)
        
        assert len(pauses) == 1
        assert pauses[0]['type'] == 'pen_lift'
    
    def test_detect_pauses_low_speed(self):
        """Test pause detection with low speed"""
        t = pd.Series([0.0, 0.005, 0.010, 0.015, 0.020])
        speed = np.array([100, 30, 30, 30, 100])
        pressure = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
        
        mask, pauses = detect_pauses(t, speed, pressure,
                                     speed_threshold=50,
                                     min_samples=2)
        
        assert len(pauses) == 1
        assert pauses[0]['type'] == 'low_speed'


class TestConfig:
    """Tests for configuration module"""
    
    def test_config_creation(self):
        """Test basic config creation"""
        config = AnalysisConfig(
            input_file=Path("test.xlsx"),
            n_trials=10,
            sampling_rate=200
        )
        
        assert config.input_file == Path("test.xlsx")
        assert config.n_trials == 10
        assert config.sampling_rate == 200
        assert config.expected_interval == 5.0  # 1000/200
    
    def test_config_derived_parameters(self):
        """Test that derived parameters are calculated correctly"""
        config = AnalysisConfig(
            input_file=Path("test.xlsx"),
            sampling_rate=200,
            min_pause_duration_ms=20,
            min_separation_ms=500
        )
        
        assert config.expected_interval == 5.0
        assert config.min_pause_samples == 4  # 20/5
        assert config.min_sep_samples == 100  # 500/5
    
    def test_config_output_paths(self):
        """Test that output paths are generated correctly"""
        config = AnalysisConfig(
            input_file=Path("data/test.xlsx"),
            output_dir=Path("output")
        )
        
        assert config.summary_csv == Path("output/test_summary.csv")
        assert config.pauses_csv == Path("output/test_pauses.csv")
        assert config.pdf_output == Path("output/test_report.pdf")
    
    def test_config_to_dict(self):
        """Test config serialization"""
        config = AnalysisConfig(
            input_file=Path("test.xlsx"),
            n_trials=10
        )
        
        config_dict = config.to_dict()
        
        assert 'input_file' in config_dict
        assert 'n_trials' in config_dict
        assert config_dict['n_trials'] == 10


class TestDataValidation:
    """Tests for data validation"""
    
    def test_empty_dataframe(self):
        """Test validation of empty dataframe"""
        from data_loader import DataLoader
        
        config = AnalysisConfig(input_file=Path("test.xlsx"))
        loader = DataLoader(config)
        
        df = pd.DataFrame()
        is_valid, issues = loader.validate_data(df)
        
        assert not is_valid
        assert "empty" in issues[0].lower()
    
    def test_missing_columns(self):
        """Test validation with missing columns"""
        from data_loader import DataLoader
        
        config = AnalysisConfig(input_file=Path("test.xlsx"))
        loader = DataLoader(config)
        
        df = pd.DataFrame({'X': [1, 2, 3], 'Y': [1, 2, 3]})
        is_valid, issues = loader.validate_data(df)
        
        assert not is_valid
        assert any("missing" in issue.lower() for issue in issues)
    
    def test_valid_data(self):
        """Test validation with valid data"""
        from data_loader import DataLoader
        
        config = AnalysisConfig(input_file=Path("test.xlsx"))
        loader = DataLoader(config)
        
        df = pd.DataFrame({
            'PacketTime': [0, 5, 10, 15, 20],
            'X': [100, 110, 120, 130, 140],
            'Y': [200, 210, 220, 230, 240],
            'NormalPressure': [0.5, 0.6, 0.5, 0.4, 0.5]
        })
        
        is_valid, issues = loader.validate_data(df)
        
        assert is_valid
        assert len(issues) == 0


class TestRobustZ:
    """Tests for robust z-score calculation"""
    
    def test_robust_z_normal_data(self):
        """Test robust z-score with normal data"""
        from trial_detector import robust_z
        
        data = pd.Series([1, 2, 3, 4, 5])
        z_scores = robust_z(data)
        
        # Median should have z-score near 0
        assert abs(z_scores.iloc[2]) < 0.1
    
    def test_robust_z_with_outlier(self):
        """Test that robust z-score handles outliers"""
        from trial_detector import robust_z
        
        data = pd.Series([1, 2, 3, 4, 100])
        z_scores = robust_z(data)
        
        # Outlier should have high z-score
        assert z_scores.iloc[-1] > 5
        
        # Other values should have moderate z-scores
        assert all(abs(z_scores.iloc[:-1]) < 2)


# Fixtures for integration tests
@pytest.fixture
def sample_trial_data():
    """Create sample trial data for testing"""
    n_points = 100
    t = np.linspace(0, 1, n_points)
    
    return pd.DataFrame({
        'PacketTime': t * 1000,  # ms
        'X': 100 + 50 * np.sin(2 * np.pi * t),
        'Y': 200 + 30 * np.cos(2 * np.pi * t),
        'NormalPressure': 0.5 + 0.2 * np.random.randn(n_points).clip(-1, 1)
    })


@pytest.fixture
def sample_config(tmp_path):
    """Create sample configuration for testing"""
    return AnalysisConfig(
        input_file=tmp_path / "test.xlsx",
        output_dir=tmp_path,
        n_trials=5,
        sampling_rate=200
    )


class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_full_analysis_pipeline(self, sample_trial_data, sample_config):
        """Test complete analysis pipeline (without interactive parts)"""
        from trial_analyzer import TrialAnalyzer
        
        # Add Trial column
        sample_trial_data['Trial'] = 0
        
        # Analyze
        analyzer = TrialAnalyzer(sample_config)
        summary, pauses = analyzer.analyze_trial(sample_trial_data, trial_id=0)
        
        # Check results
        assert 'Trial' in summary
        assert 'MeanSpeed' in summary
        assert isinstance(pauses, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
