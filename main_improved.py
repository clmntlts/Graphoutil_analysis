"""
Main execution script for handwriting analysis pipeline.
Improved version with better organization, error handling, and logging.
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Import custom modules
from config import AnalysisConfig
from data_loader import load_and_validate
from trial_detector import detect_trials_auto
from trial_analyzer import analyze_trials
from visualization import ReportGenerator
from interactive_word_segmenter import InteractiveSegmenter
from interactive_letter_segmenter import InteractiveLetterSegmenter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('analysis.log')
    ]
)
logger = logging.getLogger(__name__)


def run_letter_segmentation(df: pd.DataFrame, config, trials_to_segment='all'):
    """
    Launch interactive letter segmentation
    
    Args:
        df: DataFrame with Trial column
        config: AnalysisConfig object
        trials_to_segment: 'all' or list of trial IDs
    
    Returns:
        DataFrame with all detected letters or None
    """
    all_letters = []
    available_trials = sorted(df['Trial'].unique())
    
    if trials_to_segment == 'all':
        trials = available_trials
    else:
        trials = [t for t in trials_to_segment if t in available_trials]
    
    for trial_id in trials:
        logger.info(f"Segmenting letters for trial {trial_id}")
        
        trial_data = df[df['Trial'] == trial_id]
        segmenter = InteractiveLetterSegmenter(trial_data, trial_id)
        
        try:
            letters = segmenter.start_interactive()
            all_letters.extend(letters)
        except Exception as e:
            logger.error(f"Error in letter segmentation for trial {trial_id}: {e}")
            continue
    
    return pd.DataFrame(all_letters) if all_letters else None


def run_word_segmentation(df: pd.DataFrame, seg, config, trials_to_segment='all'):
    """
    Launch interactive word segmentation
    
    Args:
        df: DataFrame with Trial column
        seg: Segmentation sheet DataFrame (optional)
        config: AnalysisConfig object
        trials_to_segment: 'all' or list of trial IDs
    
    Returns:
        List of segment dictionaries
    """
    all_segments = []
    available_trials = sorted(df['Trial'].unique())
    
    if trials_to_segment == 'all':
        trials = available_trials
    else:
        trials = [t for t in trials_to_segment if t in available_trials]
    
    for trial_id in trials:
        logger.info(f"Segmenting words for trial {trial_id}")
        
        trial_data = df[df['Trial'] == trial_id]
        segmenter = InteractiveSegmenter(
            trial_data, trial_id, seg,
            config.speed_threshold_px_per_s,
            config.min_pause_samples
        )
        
        try:
            segments = segmenter.start_interactive()
            all_segments.extend(segments)
        except Exception as e:
            logger.error(f"Error in word segmentation for trial {trial_id}: {e}")
            continue
    
    return all_segments


def save_results(df, summary_df, pauses_df, segments_df, letters_df, config):
    """
    Save all results to CSV files
    
    Args:
        df: Main DataFrame (possibly with annotations)
        summary_df: Trial summary DataFrame
        pauses_df: Pauses DataFrame
        segments_df: Segments DataFrame (optional)
        letters_df: Letters DataFrame (optional)
        config: AnalysisConfig object
    """
    # Save trial summary
    summary_df.to_csv(config.summary_csv, index=False)
    logger.info(f"✅ Saved trial summary: {config.summary_csv}")
    
    # Save pauses
    if pauses_df is not None and len(pauses_df) > 0:
        pauses_df.to_csv(config.pauses_csv, index=False)
        logger.info(f"✅ Saved pauses: {config.pauses_csv}")
    
    # Save segments
    if segments_df is not None and len(segments_df) > 0:
        segments_df.to_csv(config.segments_csv, index=False)
        logger.info(f"✅ Saved segments: {config.segments_csv}")
    
    # Save letters
    if letters_df is not None and len(letters_df) > 0:
        letters_df.to_csv(config.letters_csv, index=False)
        logger.info(f"✅ Saved letters: {config.letters_csv}")
        
        # Save letters summary
        letters_summary = letters_df.groupby('trial').agg({
            'letter_id': 'count',
            'duration_ms': ['mean', 'std', 'sum'],
            'path_length': ['mean', 'std'],
            'mean_speed': ['mean', 'std'],
            'mean_pressure': ['mean', 'std'],
            'width': ['mean', 'std'],
            'height': ['mean', 'std']
        }).round(2)
        
        letters_summary.columns = ['_'.join(col).strip('_') for col in letters_summary.columns]
        letters_summary = letters_summary.rename(columns={'letter_id_count': 'n_letters'})
        letters_summary.to_csv(config.letters_summary_csv)
        logger.info(f"✅ Saved letters summary: {config.letters_summary_csv}")
    
    # Save annotated data if we have segments or letters
    if (segments_df is not None and len(segments_df) > 0) or \
       (letters_df is not None and len(letters_df) > 0):
        df.to_csv(config.annotated_csv, index=False)
        logger.info(f"✅ Saved annotated data: {config.annotated_csv}")


def print_summary(summary_df, pauses_df, segments_df, letters_df, config):
    """Print analysis summary"""
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"📊 Trials analyzed: {len(summary_df)}")
    print(f"⏸️  Pauses detected: {len(pauses_df) if pauses_df is not None else 0}")
    
    if segments_df is not None and len(segments_df) > 0:
        print(f"📝 Word segments: {len(segments_df)}")
    
    if letters_df is not None and len(letters_df) > 0:
        print(f"🔤 Letters segmented: {len(letters_df)}")
    
    print(f"\n⚙️  Parameters:")
    print(f"   • Speed threshold: {config.speed_threshold_px_per_s} px/s")
    print(f"   • Min pause duration: {config.min_pause_duration_ms} ms")
    print(f"   • Sampling rate: {config.sampling_rate} Hz")
    
    print(f"\n📂 Output files:")
    print(f"   • {config.summary_csv.name}")
    print(f"   • {config.pauses_csv.name}")
    if segments_df is not None and len(segments_df) > 0:
        print(f"   • {config.segments_csv.name}")
    if letters_df is not None and len(letters_df) > 0:
        print(f"   • {config.letters_csv.name}")
        print(f"   • {config.letters_summary_csv.name}")
    print(f"   • {config.pdf_output.name}")
    print("="*70 + "\n")


def main():
    """Main execution function"""
    try:
        # ============================================
        # 1. CONFIGURATION
        # ============================================
        logger.info("Starting handwriting analysis pipeline")
        
        config = AnalysisConfig(
            input_file=Path(r"C:\Users\cletesson\Downloads\Hurisah-Erpent-prétest.xlsx"),

            n_trials=30,

            # Trial detection
            min_separation_ms=500,
            min_gap_ms=20,

            # Sampling
            sampling_rate=200,

            # Pause detection
            min_pause_duration_ms=20,
            speed_threshold_px_per_s=50,

            # Letter segmentation
            min_letter_duration_ms=100,
            smoothing_window=5,

            # Visualization
            trajectory_aspect_ratio=224 / 140,
            pdf_dpi=300
        )
     
        logger.info(f"Configuration: {config.to_dict()}")
        
        # ============================================
        # 2. DATA LOADING
        # ============================================
        logger.info("Loading data...")
        df, seg = load_and_validate(config)
        
        # ============================================
        # 3. TRIAL DETECTION
        # ============================================
        logger.info("Detecting trials...")
        df = detect_trials_auto(df, config, interactive=True)
        
        # Optional: Visualize detection
        # from trial_detector import TrialDetector
        # detector = TrialDetector(config)
        # detector.visualize_detection(df, save_path=config.output_dir / "trial_detection.png")
        
        # ============================================
        # 4. TRIAL ANALYSIS
        # ============================================
        logger.info("Analyzing trials...")
        summary_df, all_pauses = analyze_trials(df, config)
        pauses_df = pd.DataFrame(all_pauses) if all_pauses else None
        
        # ============================================
        # 5. INTERACTIVE SEGMENTATION (OPTIONAL)
        # ============================================
        
        # Initialize columns for segmentation
        df['segment_id'] = -1
        df['segment_label'] = ''
        df['letter_id'] = -1
        df['letter_label'] = ''
        
        segments_df = None
        letters_df = None
        all_segments = []
        all_letters = []
        
        # Word segmentation
        print("\n" + "="*70)
        print("WORD SEGMENTATION")
        print("="*70)
        response = input("Do you want to segment words interactively? (y/n): ")
        
        if response.lower() == 'y':
            available_trials = sorted(df['Trial'].unique())
            print(f"\nAvailable trials: {available_trials}")
            trials_input = input("Enter trial numbers (comma-separated, or 'all'): ")
            
            trials_to_segment = 'all' if trials_input.lower() == 'all' else \
                               [int(x.strip()) for x in trials_input.split(',')]
            
            all_segments = run_word_segmentation(df, seg, config, trials_to_segment)
            
            if all_segments:
                segments_df = pd.DataFrame(all_segments)
                segments_df['unique_segment_id'] = segments_df.apply(
                    lambda row: f"T{row['trial']}_S{row['segment_id']}", axis=1
                )
                
                # Annotate raw data with segments
                for _, seg_row in segments_df.iterrows():
                    trial_id = seg_row['trial']
                    idx1, idx2 = int(seg_row['idx1']), int(seg_row['idx2'])
                    
                    trial_mask = df['Trial'] == trial_id
                    trial_indices = df[trial_mask].index
                    
                    if idx1 < len(trial_indices) and idx2 < len(trial_indices):
                        segment_indices = trial_indices[idx1:idx2+1]
                        df.loc[segment_indices, 'segment_id'] = seg_row['segment_id']
                        df.loc[segment_indices, 'segment_label'] = seg_row['label']
        
        # Letter segmentation
        print("\n" + "="*70)
        print("LETTER SEGMENTATION")
        print("="*70)
        response = input("Do you want to segment letters automatically? (y/n): ")
        
        if response.lower() == 'y':
            available_trials = sorted(df['Trial'].unique())
            print(f"\nAvailable trials: {available_trials}")
            trials_input = input("Enter trial numbers (comma-separated, or 'all'): ")
            
            trials_to_segment = 'all' if trials_input.lower() == 'all' else \
                               [int(x.strip()) for x in trials_input.split(',')]
            
            letters_df = run_letter_segmentation(df, config, trials_to_segment)
            
            if letters_df is not None and len(letters_df) > 0:
                all_letters = letters_df.to_dict('records')
                
                # Annotate raw data with letters
                for _, letter_row in letters_df.iterrows():
                    trial_id = letter_row['trial']
                    idx1, idx2 = int(letter_row['idx1']), int(letter_row['idx2'])
                    
                    trial_mask = df['Trial'] == trial_id
                    trial_indices = df[trial_mask].index
                    
                    if idx1 < len(trial_indices) and idx2 < len(trial_indices):
                        letter_indices = trial_indices[idx1:idx2+1]
                        df.loc[letter_indices, 'letter_id'] = letter_row['letter_id']
                        df.loc[letter_indices, 'letter_label'] = letter_row['label']
        
        # ============================================
        # 6. SAVE RESULTS
        # ============================================
        logger.info("Saving results...")
        save_results(df, summary_df, pauses_df, segments_df, letters_df, config)
        
        # ============================================
        # 7. GENERATE PDF REPORT
        # ============================================
        logger.info("Generating PDF report...")
        report_gen = ReportGenerator(config)
        report_gen.generate_report(df, all_letters, all_segments)
        
        # ============================================
        # 8. PRINT SUMMARY
        # ============================================
        print_summary(summary_df, pauses_df, segments_df, letters_df, config)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
