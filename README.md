# Handwriting Analysis - Modular Structure

This project analyzes handwriting data with trial detection, pause detection, and interactive segmentation capabilities.

## File Structure

```
project/
│
├── main_analysis.py                    # Main execution script
├── pause_detection.py                  # Pause detection utilities
├── letter_segmenter.py                 # Automatic letter segmentation
├── interactive_letter_segmenter.py     # Interactive letter segmentation interface
├── interactive_word_segmenter.py       # Interactive word/segment marking
└── README.md                           # This file
```

## Module Descriptions

### 1. `pause_detection.py`
Contains utilities for detecting pauses in handwriting based on Critten (2023) methodology:
- `min_duration_mask()` - Filters boolean masks to keep only segments above minimum duration
- `detect_pauses()` - Main pause detection function using speed and pressure criteria

### 2. `letter_segmenter.py`
Automatic segmentation of cursive handwriting into letters:
- `LetterSegmenter` class - Analyzes speed, curvature, and pressure to detect letter boundaries
- Multiple detection methods: speed minima, curvature peaks, multi-criteria (recommended)
- Computes features for each detected letter (duration, size, speed, etc.)

### 3. `interactive_letter_segmenter.py`
Interactive GUI for manual correction of letter segmentation:
- `InteractiveLetterSegmenter` class - Provides visualization and manual editing tools
- Click to add/remove boundaries
- Automatic segmentation with manual refinement
- Letter labeling after validation

### 4. `interactive_word_segmenter.py`
Interactive GUI for marking word segments manually:
- `InteractiveSegmenter` class - Mark word boundaries with clicks
- Creates segments between two marked points
- Computes segment statistics (duration, pauses, speed, etc.)
- Label segments after session

### 5. `main_analysis.py`
Main orchestration script that:
1. Loads Excel data
2. Detects trials automatically
3. Analyzes each trial (pauses, speed, spatial metrics)
4. Optionally launches interactive word segmentation
5. Optionally launches interactive letter segmentation
6. Generates comprehensive PDF report
7. Exports all data to CSV files

## How to Use

### Basic Usage

Simply run the main script:

```python
python main_analysis.py
```

The script will:
1. Load your data file (update the `file_path` variable)
2. Detect trials automatically
3. Analyze each trial
4. Ask if you want to do interactive segmentation
5. Generate outputs

### Customizing Parameters

Edit these parameters in `main_analysis.py`:

```python
# File and trial settings
file_path = Path(r"your/path/to/data.xlsx")
N_trials = 30  # Expected number of trials

# Pause detection settings
min_pause_duration_ms = 20  # Minimum pause duration
speed_threshold_px_per_s = 50  # Speed threshold for low-speed pauses
```

### Using Modules Independently

You can also import and use individual modules:

```python
from pause_detection import detect_pauses
from letter_segmenter import LetterSegmenter
from interactive_letter_segmenter import InteractiveLetterSegmenter

# Example: Detect pauses
pause_mask, pauses = detect_pauses(time, speed, pressure, 
                                   speed_threshold=50, 
                                   min_samples=4)

# Example: Automatic letter segmentation
segmenter = LetterSegmenter(trial_data)
boundaries = segmenter.detect_letter_boundaries()
letters = segmenter.segment_into_letters(boundaries)

# Example: Interactive letter segmentation
interactive = InteractiveLetterSegmenter(trial_data, trial_id=1)
labeled_letters = interactive.start_interactive()
```

## Output Files

The analysis generates several output files:

1. **`*_summary.csv`** - Trial-level statistics (RT, duration, speed, spatial metrics)
2. **`*_pauses.csv`** - Detailed pause information (type, duration, location)
3. **`*_segments.csv`** - Word segments (if interactive segmentation was used)
4. **`*_letters.csv`** - Letter-level data (if letter segmentation was used)
5. **`*_letters_summary.csv`** - Letter statistics aggregated by trial
6. **`*_annotated.csv`** - Raw data with segment and letter IDs added
7. **`*.pdf`** - Comprehensive visualization report

## Dependencies

Required packages:
```bash
pip install pandas numpy matplotlib scipy scikit-learn openpyxl
```

## Interactive Segmentation Controls

### Word Segmentation
- **Click** on trajectory to place markers
- **Press 'm'** to create segment between last 2 markers
- **Press 'u'** to undo last marker
- **Press 'r'** to reset all markers
- **Close window** to finish and label segments

### Letter Segmentation
- **Click** on trajectory to add/move boundary
- **Click** on red point then **press 'd'** to delete boundary
- **Auto Segment** button to re-compute automatically
- **Validate & Label** button to proceed to labeling
- **Clear All** button to remove all boundaries

## Adapting to Your Data

To adapt this code to your own data:

1. **File path**: Update `file_path` in `main_analysis.py`
2. **Column names**: Adjust column mapping if your Excel has different names
3. **Sampling rate**: Update `sampling_rate` based on your device (Hz)
4. **Number of trials**: Set `N_trials` to expected number
5. **Pause thresholds**: Adjust `speed_threshold_px_per_s` and `min_pause_duration_ms`
6. **Aspect ratio**: Update `set_aspect(224/140)` if your tablet has different dimensions

## Troubleshooting

**Q: Trial detection fails**
A: Adjust `min_separation_ms` or check that your data has clear breaks between trials

**Q: Too many/few pauses detected**
A: Adjust `speed_threshold_px_per_s` (higher = fewer pauses) or `min_pause_duration_ms`

**Q: Letter segmentation produces wrong boundaries**
A: Use the interactive interface to manually correct, or adjust `min_letter_duration_ms`

**Q: Import errors**
A: Make sure all module files are in the same directory as `main_analysis.py`

## Contact

For questions or issues, please refer to the original script documentation or contact the developer.
