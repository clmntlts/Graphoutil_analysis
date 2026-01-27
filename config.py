"""
Configuration module for handwriting analysis.
Centralizes all parameters and settings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AnalysisConfig:
    """Configuration for handwriting analysis pipeline"""
    
    # File paths
    input_file: Path
    output_dir: Optional[Path] = None
    
    # Trial detection parameters
    n_trials: Optional[int] = None
    min_separation_ms: float = 500
    min_gap_ms: float = 20
    
    # Sampling and timing
    sampling_rate: float = 200  # Hz
    
    # Pause detection parameters
    min_pause_duration_ms: float = 20
    speed_threshold_px_per_s: float = 50
    
    # Letter segmentation parameters
    min_letter_duration_ms: float = 100
    smoothing_window: int = 5
    
    # Visualization parameters
    trajectory_aspect_ratio: float = 224 / 140
    pdf_dpi: int = 150
    
    # Column names mapping
    col_time: str = "PacketTime"
    col_x: str = "X"
    col_y: str = "Y"
    col_pressure: str = "NormalPressure"
    
    def __post_init__(self):
        """Initialize derived parameters"""
        self.input_file = Path(self.input_file)
        
        # ===== Output directory logic =====
        if self.output_dir is None:
            self.output_dir = Path.cwd() / "results"

        # Create directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Derived parameters
        self.expected_interval = 1000 / self.sampling_rate  # ms
        self.min_pause_samples = int(self.min_pause_duration_ms / self.expected_interval)
        self.min_sep_samples = int(self.min_separation_ms / self.expected_interval)
    
    @property
    def summary_csv(self) -> Path:
        """Path to summary CSV output"""
        return self.output_dir / f"{self.input_file.stem}_summary.csv"
    
    @property
    def pauses_csv(self) -> Path:
        """Path to pauses CSV output"""
        return self.output_dir / f"{self.input_file.stem}_pauses.csv"
    
    @property
    def segments_csv(self) -> Path:
        """Path to segments CSV output"""
        return self.output_dir / f"{self.input_file.stem}_segments.csv"
    
    @property
    def letters_csv(self) -> Path:
        """Path to letters CSV output"""
        return self.output_dir / f"{self.input_file.stem}_letters.csv"
    
    @property
    def letters_summary_csv(self) -> Path:
        """Path to letters summary CSV output"""
        return self.output_dir / f"{self.input_file.stem}_letters_summary.csv"
    
    @property
    def annotated_csv(self) -> Path:
        """Path to annotated raw data CSV output"""
        return self.output_dir / f"{self.input_file.stem}_annotated.csv"
    
    @property
    def pdf_output(self) -> Path:
        """Path to PDF report output"""
        return self.output_dir / f"{self.input_file.stem}_report.pdf"
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return {
            'input_file': str(self.input_file),
            'output_dir': str(self.output_dir),
            'n_trials': self.n_trials,
            'sampling_rate': self.sampling_rate,
            'min_pause_duration_ms': self.min_pause_duration_ms,
            'speed_threshold_px_per_s': self.speed_threshold_px_per_s,
            'min_letter_duration_ms': self.min_letter_duration_ms,
        }


def load_config_from_file(config_path: Path) -> AnalysisConfig:
    """Load configuration from YAML or JSON file"""
    import json
    
    with open(config_path, 'r') as f:
        if config_path.suffix == '.json':
            params = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            import yaml
            params = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return AnalysisConfig(**params)


# Example default configuration
DEFAULT_CONFIG = AnalysisConfig(
    input_file=Path("data.xlsx"),
    n_trials=30,
    sampling_rate=200,
    min_pause_duration_ms=20,
    speed_threshold_px_per_s=50,
    min_letter_duration_ms=100
)
