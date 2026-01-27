"""
Comprehensive visualization module for handwriting analysis.
Generates PDF reports with multiple visualization types.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from typing import List, Dict, Optional
import logging

from pause_detection import detect_pauses

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive PDF reports for handwriting analysis
    """
    
    def __init__(self, config):
        """
        Initialize ReportGenerator
        
        Args:
            config: AnalysisConfig object
        """
        self.config = config
    
    def generate_report(self, df: pd.DataFrame, 
                       all_letters: List[Dict] = None,
                       all_segments: List[Dict] = None):
        """
        Generate complete PDF report
        
        Args:
            df: Main DataFrame with Trial column
            all_letters: Optional list of letter data
            all_segments: Optional list of segment data
        """
        logger.info(f"Generating PDF report to {self.config.pdf_output}")
        
        with PdfPages(self.config.pdf_output) as pdf:
            # Generate page for each trial
            for trial_id, trial in df.groupby("Trial"):
                try:
                    self._generate_trial_page(
                        pdf, trial, trial_id, 
                        all_letters, all_segments
                    )
                except Exception as e:
                    logger.error(f"Error generating page for trial {trial_id}: {e}")
                    continue
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = f'Handwriting Analysis Report - {self.config.input_file.stem}'
            d['Author'] = 'Handwriting Analysis Pipeline'
            d['Subject'] = 'Automated handwriting analysis with pause detection and segmentation'
            d['Keywords'] = 'handwriting, analysis, kinematics, pauses, letters'
        
        logger.info(f"Report saved successfully: {self.config.pdf_output}")
    
    def _generate_trial_page(self, pdf: PdfPages, trial: pd.DataFrame, 
                            trial_id: int,
                            all_letters: List[Dict] = None,
                            all_segments: List[Dict] = None):
        """Generate a single page for one trial"""
        trial = trial.reset_index(drop=True)
        t = (trial["PacketTime"] - trial["PacketTime"].iloc[0]) / 1000
        p = trial["NormalPressure"]
        
        # Skip empty trials
        if p.max() == 0:
            logger.warning(f"Skipping trial {trial_id} - no pressure data")
            return
        
        # Calculate speed
        vx = np.gradient(trial["X"], t)
        vy = np.gradient(trial["Y"], t)
        speed = np.sqrt(vx**2 + vy**2)
        
        # Detect pauses
        pause_mask, pauses = detect_pauses(
            t, speed, p,
            self.config.speed_threshold_px_per_s,
            self.config.min_pause_samples
        )
        
        # Filter letters and segments for this trial
        trial_letters = [l for l in all_letters if l['trial'] == trial_id] if all_letters else []
        trial_segments = [s for s in all_segments if s['trial'] == trial_id] if all_segments else []
        
        # Create figure
        fig = plt.figure(figsize=(12, 14))
        gs = fig.add_gridspec(7, 2, height_ratios=[4, 0.3, 0.3, 2, 2, 1, 0.5],
                             width_ratios=[3, 1], hspace=0.4, wspace=0.3)
        
        # Plot 1: Spatial trajectory
        self._plot_trajectory(fig, gs[0, :], trial, trial_letters, trial_segments)
        
        # Plot 2: Pressure heatmap
        self._plot_pressure_heatmap(fig, gs[1, :], t, p)
        
        # Plot 3: Speed heatmap
        self._plot_speed_heatmap(fig, gs[2, :], t, speed)
        
        # Plot 4: X/Y temporal
        self._plot_xy_temporal(fig, gs[3, :], t, trial, pauses, trial_letters)
        
        # Plot 5: Speed & Pressure
        self._plot_speed_pressure(fig, gs[4, :], t, speed, p, pauses)
        
        # Plot 6: Statistics table
        self._plot_statistics_table(fig, gs[5, 0], trial, t, speed, p, pauses)
        
        # Plot 7: Pause distribution
        self._plot_pause_distribution(fig, gs[5, 1], pauses)
        
        # Plot 8: Letter table (if available)
        if trial_letters:
            self._plot_letter_table(fig, gs[6, :], trial_letters)
        
        # Title
        title = f"Trial {trial_id}"
        if trial_letters:
            title += f" | {len(trial_letters)} letters"
        if trial_segments:
            title += f" | {len(trial_segments)} segments"
        title += f" | {len(pauses)} pauses"
        
        plt.suptitle(title, fontsize=14, weight="bold")
        
        # Save to PDF
        pdf.savefig(fig, dpi=self.config.pdf_dpi, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_trajectory(self, fig, gs_cell, trial, letters, segments):
        """Plot spatial trajectory"""
        ax = fig.add_subplot(gs_cell)
        
        # Base trajectory
        ax.plot(trial["X"], trial["Y"], color="lightgray", linewidth=1, alpha=0.5)
        
        # Color by letters
        if letters:
            colors = plt.cm.tab20.colors
            for i, letter in enumerate(letters):
                idx1, idx2 = letter['idx1'], letter['idx2']
                subset = trial.iloc[idx1:idx2+1]
                ax.plot(subset["X"], subset["Y"],
                       color=colors[i % len(colors)],
                       linewidth=2.5, alpha=0.9)
                
                # Label
                mid_x = subset["X"].mean()
                mid_y = subset["Y"].mean()
                label_text = letter.get('label', str(i+1))
                ax.text(mid_x, mid_y, label_text,
                       fontsize=10, weight='bold',
                       bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        # Mark segments
        for seg in segments:
            idx1, idx2 = seg['idx1'], seg['idx2']
            ax.plot([trial["X"].iloc[idx1], trial["X"].iloc[idx2]],
                   [trial["Y"].iloc[idx1], trial["Y"].iloc[idx2]],
                   'r-', linewidth=3, alpha=0.4)
        
        ax.set_aspect(self.config.trajectory_aspect_ratio)
        ax.set_title("Spatial Trajectory", fontsize=11, weight="bold")
        ax.axis("off")
    
    def _plot_pressure_heatmap(self, fig, gs_cell, t, p):
        """Plot pressure as color gradient"""
        ax = fig.add_subplot(gs_cell)
        norm = Normalize(vmin=p.min(), vmax=p.max())
        pressure_gradient = p.values.reshape(1, -1)
        ax.imshow(pressure_gradient, aspect='auto', cmap='YlOrRd',
                 extent=[t.iloc[0], t.iloc[-1], 0, 1], norm=norm)
        ax.set_ylabel('Pressure', fontsize=8)
        ax.set_yticks([])
        ax.set_xticks([])
    
    def _plot_speed_heatmap(self, fig, gs_cell, t, speed):
        """Plot speed as color gradient"""
        ax = fig.add_subplot(gs_cell)
        norm = Normalize(vmin=speed.min(), vmax=speed.max())
        speed_gradient = speed.reshape(1, -1)
        ax.imshow(speed_gradient, aspect='auto', cmap='viridis',
                 extent=[t.iloc[0], t.iloc[-1], 0, 1], norm=norm)
        ax.set_ylabel('Speed', fontsize=8)
        ax.set_yticks([])
        ax.set_xlabel('Time (s)', fontsize=8)
    
    def _plot_xy_temporal(self, fig, gs_cell, t, trial, pauses, letters):
        """Plot X and Y over time"""
        ax = fig.add_subplot(gs_cell)
        ax.plot(t, trial["X"], color="steelblue", lw=1, label="X")
        ax.set_ylabel("X (px)", color="steelblue")
        
        ax2 = ax.twinx()
        ax2.plot(t, trial["Y"], color="darkorange", lw=1, label="Y")
        ax2.set_ylabel("Y (px)", color="darkorange")
        
        # Mark letter boundaries
        if letters:
            for letter in letters:
                ax.axvline(letter['t1'], color='green', linestyle=':', alpha=0.5, lw=1)
                ax.axvline(letter['t2'], color='green', linestyle=':', alpha=0.5, lw=1)
        
        # Mark pauses
        for pause in pauses:
            color = {'pen_lift': 'red', 'low_speed': 'blue', 'mixed': 'purple'}[pause['type']]
            ax.axvspan(pause['start_time'], pause['end_time'],
                      color=color, alpha=0.2, zorder=0)
        
        ax.set_xlabel("Time (s)")
        ax.set_title("X & Y Trajectories", fontsize=10, weight="bold")
        ax.grid(True, alpha=0.3)
    
    def _plot_speed_pressure(self, fig, gs_cell, t, speed, p, pauses):
        """Plot speed and pressure over time"""
        ax = fig.add_subplot(gs_cell)
        ax.plot(t, speed, color="purple", lw=1)
        ax.axhline(self.config.speed_threshold_px_per_s, 
                  color='purple', linestyle='--', alpha=0.5)
        ax.set_ylabel("Speed (px/s)", color="purple")
        
        ax2 = ax.twinx()
        ax2.plot(t, p, color="gray", lw=1)
        ax2.set_ylabel("Pressure", color="gray")
        
        ax.set_xlabel("Time (s)")
        ax.set_title("Speed & Pressure", fontsize=10, weight="bold")
        ax.grid(True, alpha=0.3)
    
    def _plot_statistics_table(self, fig, gs_cell, trial, t, speed, p, pauses):
        """Plot summary statistics table"""
        ax = fig.add_subplot(gs_cell)
        ax.axis('off')
        
        # Calculate statistics
        duration = t.iloc[-1]
        total_pause_time = sum([p['duration_ms'] for p in pauses]) / 1000
        
        stats_data = [
            ['Duration', f"{duration:.2f}s"],
            ['Pause Time', f"{total_pause_time:.2f}s"],
            ['# Pauses', f"{len(pauses)}"],
            ['Mean Speed', f"{speed.mean():.1f} px/s"],
            ['Mean Pressure', f"{p[p>0].mean():.3f}" if len(p[p>0]) > 0 else "N/A"],
            ['Path Length', f"{np.sum(np.sqrt(np.diff(trial['X'])**2 + np.diff(trial['Y'])**2)):.1f} px"],
        ]
        
        table = ax.table(cellText=stats_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        ax.set_title("Statistics", fontsize=10, weight="bold")
    
    def _plot_pause_distribution(self, fig, gs_cell, pauses):
        """Plot pause duration distribution"""
        ax = fig.add_subplot(gs_cell)
        
        if pauses:
            durations = [p['duration_ms'] for p in pauses]
            ax.hist(durations, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Duration (ms)', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.set_title('Pause Distribution', fontsize=10, weight="bold")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No pauses detected', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    def _plot_letter_table(self, fig, gs_cell, letters):
        """Plot table of letter properties"""
        ax = fig.add_subplot(gs_cell)
        ax.axis('off')
        
        # Limit to first 10 letters
        table_data = []
        for letter in letters[:10]:
            table_data.append([
                letter.get('label', str(letter['letter_id'])),
                f"{letter['duration_ms']:.0f}",
                f"{letter['mean_speed']:.0f}",
                f"{letter['width']:.0f}×{letter['height']:.0f}"
            ])
        
        if table_data:
            table = ax.table(cellText=table_data,
                           colLabels=['Letter', 'Duration (ms)', 'Speed (px/s)', 'Size (px)'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
