# Improved Trial Detector for PsychoPy Experiments

## Overview

This improved trial detector is specifically designed for handwriting data collected using PsychoPy with start/stop commands. It addresses the key requirements:

1. **Mandatory timestamp jumps**: Uses temporal gaps as PRIMARY criterion
2. **Known number of trials**: Must find exactly N trials (hard constraint)
3. **Interactive validation**: Visual tool to review and correct boundaries
4. **Confidence scores**: Each candidate boundary has reliability metrics

## Key Improvements

### 1. Timestamp-First Approach

The detector prioritizes temporal gaps (from PsychoPy start/stop) over other features:

- **Temporal confidence: 3x weight** - Most important for PsychoPy data
- **Spatial jump: 1x weight** - Secondary supporting evidence  
- **Pressure drop: 0.5x weight** - Additional validation

### 2. Mandatory Trial Count

The detector MUST find exactly `n_trials`:

- Uses greedy selection with separation constraints
- Falls back to relaxed constraints if needed
- Throws clear error if impossible to find N trials
- No "best effort" - either succeeds or fails explicitly

### 3. Interactive Validation Interface

New `InteractiveTrialValidator` class provides:

- **Visual overview**: See all candidates with confidence scores
- **Click to toggle**: Add/remove boundaries interactively
- **Live feedback**: Shows current trial count vs required
- **Auto-select**: Press 'a' to automatically select top N candidates
- **Candidate ranking**: Top 15 candidates shown with metrics

### 4. Confidence Scoring System

Each candidate boundary receives multiple confidence scores:

- **Overall Confidence**: 0-100% combined score
- **Temporal Confidence**: Based on timestamp gap size
- **Spatial Confidence**: Based on coordinate jump
- **Pressure Confidence**: Based on pressure changes

## Usage

### Basic Automatic Detection

```python
from config import AnalysisConfig
from data_loader import load_and_validate
from trial_detector import detect_trials_auto

config = AnalysisConfig(
    input_file=Path("data.xlsx"),
    n_trials=30,  # CRITICAL: Must match experiment
    sampling_rate=200,
    min_gap_ms=20,
    min_separation_ms=500
)

df, seg = load_and_validate(config)

# Automatic detection with report
df = detect_trials_auto(df, config, interactive=False)
```

Output:
```
================================================================================
TRIAL BOUNDARY CANDIDATES REPORT
================================================================================
Total candidates found: 35
Required boundaries: 29
Selected boundaries: 29
================================================================================

Top 20 Candidates (sorted by confidence):
--------------------------------------------------------------------------------
 Rank  Time_s  DeltaT  DistJump  Confidence  TemporalConfidence  SpatialConfidence  IsSelected
    1    12.5  1250.0     450.2        98.5                99.8               89.2        True
    2    25.3  1180.0     523.1        96.2                98.1               95.3        True
    ...
```

### Interactive Validation

```python
# Launch interactive mode
df = detect_trials_auto(df, config, interactive=True)
```

This opens an interactive window with:
- **Top panel**: Temporal gaps with all candidates marked
- **Candidate list**: Top 15 candidates with confidence scores
- **Spatial trajectory**: Shows how trials are currently segmented
- **Spatial jumps**: Validates spatial discontinuities

**Controls**:
- Click on temporal gap plot to toggle boundaries
- Press `a`: Auto-select top N candidates
- Press `r`: Reset to initial selection
- Press `c`: Clear all boundaries
- Close window when satisfied

### Using the Detector Object

```python
from trial_detector import TrialDetector

detector = TrialDetector(config)

# Detect with options
df = detector.detect_trials(df, interactive=True)

# Print detailed report
detector.print_candidate_report()

# Visualize results
detector.visualize_detection(df, save_path="detection.png")

# Access candidate data
print(detector.candidates_info.head())
print(f"Selected: {detector.selected_boundaries}")
```

## Configuration Parameters

### Critical Parameters

**`n_trials`** (int, required)
- Number of trials in your experiment
- Detector MUST find exactly this many trials
- Example: `n_trials=30`

**`min_gap_ms`** (float, default=20)
- Minimum temporal gap to consider as candidate
- Auto-adjusted upward for trial boundaries
- Example: `min_gap_ms=20`

**`min_separation_ms`** (float, default=500)
- Minimum time between consecutive trials
- Prevents detecting boundaries too close together
- Example: `min_separation_ms=500`

### Derived Parameters

**`min_sep_samples`** (int, automatic)
- Calculated from `min_separation_ms` and `sampling_rate`
- Used internally for separation constraint

**`expected_interval`** (float, automatic)
- Calculated from `sampling_rate`
- Normal time between samples (e.g., 5ms at 200Hz)

## Understanding Confidence Scores

### Overall Confidence (0-100%)
Normalized combined score from all features. Higher is better.

- **> 90%**: Very strong candidate (clear temporal gap + spatial jump)
- **70-90%**: Good candidate (strong temporal gap)
- **50-70%**: Moderate candidate (visible gap but weaker evidence)
- **< 50%**: Weak candidate (small gap or ambiguous)

### Temporal Confidence (0-100%)
Based on timestamp gap size relative to other candidates.

- **> 95%**: Clear trial boundary (gap >> typical interval)
- **80-95%**: Strong trial boundary
- **< 80%**: Weaker evidence from timing alone

### Spatial Confidence (0-100%)
Based on coordinate jump size.

- **> 90%**: Large spatial jump (strong supporting evidence)
- **50-90%**: Moderate jump
- **< 50%**: Small jump (typical within-trial movement)

## Visualization

### `visualize_detection(df, save_path=None)`

Creates a 3-panel figure:

1. **Temporal Gaps (log scale)**
   - Shows all time gaps in data
   - Red circles: Selected boundaries
   - Orange X's: Other candidates
   - Green line: Minimum gap threshold

2. **Spatial Jumps**
   - Shows coordinate jumps over time
   - Red circles: Selected boundaries
   - Validates that boundaries coincide with spatial discontinuities

3. **Spatial Trajectory**
   - Shows all trials in different colors
   - Visual check that segmentation makes sense
   - Each trial should be a distinct writing segment

## Error Handling

### "Not enough candidates" Error

```
RuntimeError: Not enough candidates (25) to create 30 trials.
```

**Causes**:
- `n_trials` is set too high
- Data doesn't have clear temporal gaps
- `min_gap_ms` is set too high

**Solutions**:
1. Check your `n_trials` parameter
2. Lower `min_gap_ms` threshold
3. Use interactive mode to inspect candidates
4. Verify data quality (check for missing PsychoPy events)

### "Could not find exactly N trials" Error

```
RuntimeError: Could not find exactly 30 trials. Found 28 trials.
```

**Causes**:
- Candidates too close together (violate separation constraint)
- Some trial boundaries have ambiguous gaps

**Solutions**:
1. Try interactive mode: `interactive=True`
2. Adjust `min_separation_ms` parameter
3. Check candidate report for clues
4. Manually inspect problematic regions

## Best Practices

### 1. Always Validate

```python
df = detect_trials_auto(df, config, interactive=False)

n_detected = df['Trial'].max() + 1
assert n_detected == config.n_trials, f"Expected {config.n_trials}, got {n_detected}"
```

### 2. Review Candidate Report

The automatic report shows:
- How many candidates were found
- Which ones were selected
- Confidence scores for top candidates

Look for:
- Are selected boundaries high-confidence? (> 70%)
- Are there high-confidence candidates that weren't selected?
- Is there a clear gap between selected and rejected?

### 3. Use Interactive Mode When Unsure

If automatic detection gives warnings or seems wrong:

```python
try:
    df = detect_trials_auto(df, config, interactive=False)
except RuntimeError:
    print("Automatic failed, switching to interactive...")
    df = detect_trials_auto(df, config, interactive=True)
```

### 4. Visualize Results

Always visualize to verify:

```python
detector = TrialDetector(config)
df = detector.detect_trials(df)
detector.visualize_detection(df, save_path="check_trials.png")
```

Look for:
- Do temporal gaps align with trial boundaries?
- Do spatial trajectories show distinct trials?
- Are trial sizes reasonably consistent?

### 5. Check Trial Size Distribution

```python
trial_sizes = df.groupby('Trial').size()
print(f"Mean: {trial_sizes.mean():.0f}")
print(f"Std: {trial_sizes.std():.0f}")
print(f"CV: {trial_sizes.std() / trial_sizes.mean():.2f}")

# High coefficient of variation (> 0.5) suggests problems
```

## Troubleshooting

### Problem: Getting 1 fewer trial than expected

**Cause**: One boundary is being filtered due to separation constraint

**Solution**: 
1. Check `min_separation_ms` - might be too large
2. Use interactive mode to see which boundary is missing
3. Look at candidate report to see if a good candidate was skipped

### Problem: Boundaries don't align with visual gaps

**Cause**: Other features (spatial, pressure) are overriding temporal

**Solution**: The new detector strongly prioritizes temporal gaps, but if this still happens:
1. Increase temporal weight in `_calculate_confidence_scores`
2. Check if your data has unusual characteristics
3. Use interactive mode to manually correct

### Problem: Too many low-confidence candidates selected

**Cause**: Not enough high-confidence candidates available

**Solution**:
1. Lower `min_separation_ms` to allow closer boundaries
2. Check your data - may have inconsistent trial spacing
3. Verify `n_trials` is correct
4. Use interactive mode to review and select manually

## API Reference

### `TrialDetector` Class

#### `__init__(config)`
Initialize detector with configuration.

#### `detect_trials(df, interactive=False) -> pd.DataFrame`
Main detection method. Returns df with 'Trial' column added.

**Parameters**:
- `df`: Input DataFrame
- `interactive`: If True, launch validation interface

**Returns**: DataFrame with Trial column

#### `print_candidate_report()`
Print detailed report of all candidates with scores.

#### `visualize_detection(df, save_path=None)`
Create visualization of detection results.

**Parameters**:
- `df`: DataFrame with Trial column
- `save_path`: Optional path to save figure

### `detect_trials_auto(df, config, interactive=False)` Function

Convenience function that creates detector, runs detection, and prints report.

**Parameters**:
- `df`: Input DataFrame
- `config`: AnalysisConfig object
- `interactive`: Launch interactive validation

**Returns**: DataFrame with Trial column

## Comparison with Previous Version

| Feature | Old Version | New Version |
|---------|------------|-------------|
| Primary criterion | Multi-weighted | **Timestamp-first** |
| Trial count | Best effort | **Mandatory exact match** |
| Confidence scores | No | **Yes, multi-component** |
| Interactive mode | No | **Yes, full interface** |
| Candidate report | No | **Yes, detailed** |
| Error handling | Generic | **Specific, actionable** |
| PsychoPy optimized | No | **Yes** |

## Examples of Good Detection

A successful detection will show:

```
Successfully detected 30 trials

Top candidates:
- Rank 1-29: All selected, confidence > 85%
- Clear temporal gaps (> 1000ms)
- Spatial jumps align with temporal gaps
- Trial sizes relatively consistent (CV < 0.3)
```

## Examples of Problematic Detection

Watch out for:

```
Warning: Expected 30 trials but found 28

Top candidates:
- Rank 5 (conf=92%) NOT selected (too close to rank 4)
- Rank 12 (conf=88%) NOT selected (too close to rank 11)
```

This suggests `min_separation_ms` might be too strict.

## Summary

The improved detector provides:

1. ✅ **Mandatory timestamp jumps**: Temporal gaps are primary criterion
2. ✅ **Exact trial count**: Must find exactly N trials, no approximation
3. ✅ **Interactive validation**: Visual tool with click-to-toggle
4. ✅ **Confidence scoring**: Multi-component reliability metrics
5. ✅ **Clear errors**: Actionable messages when detection fails
6. ✅ **Comprehensive reporting**: Detailed candidate analysis
7. ✅ **PsychoPy optimized**: Designed for start/stop command data
