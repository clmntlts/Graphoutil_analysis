"""
Interactive interface for manual word/segment marking with zoomable trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from pause_detection import detect_pauses


class InteractiveSegmenter:
    """
    Interface interactive pour marquer manuellement des segments de mots.
    """
    
    def __init__(self, trial_data, trial_id, seg=None, 
                 speed_threshold_px_per_s=50, min_pause_samples=4):
        self.trial_offset = trial_data.index[0]  # ← ADD THIS LINE
        self.trial = trial_data.reset_index(drop=True)
        self.trial_id = trial_id
        self.t = (self.trial["PacketTime"] - self.trial["PacketTime"].iloc[0]) / 1000
        self.markers = []
        self.segments = []
        self.seg = seg  # Segmentation sheet data (optional)
        
        # Filter data to only include points with pressure
        self.pressure_mask = self.trial["NormalPressure"] > 0
        
        # Paramètres de détection des pauses
        self.speed_threshold = speed_threshold_px_per_s
        self.min_pause_samples = min_pause_samples
        
        # Calculer la vitesse
        vx = np.gradient(self.trial["X"], self.t)
        vy = np.gradient(self.trial["Y"], self.t)
        self.speed = np.sqrt(vx**2 + vy**2)
        
        # Détecter les pauses
        self.pause_mask, self.pauses = detect_pauses(
            self.t, self.speed, self.trial["NormalPressure"],
            self.speed_threshold, self.min_pause_samples
        )
        
        # Calculate aspect ratio based on actual writing area
        writing_data = self.trial[self.pressure_mask]
        if len(writing_data) > 0:
            x_range = writing_data["X"].max() - writing_data["X"].min()
            y_range = writing_data["Y"].max() - writing_data["Y"].min()
            self.aspect_ratio = x_range / y_range if y_range > 0 else 1.0
        else:
            self.aspect_ratio = 224 / 140
        
    def find_nearest_point(self, x_click, y_click):
        """Trouve le point le plus proche du clic (parmi les points avec pression)"""
        writing_data = self.trial[self.pressure_mask]
        if len(writing_data) == 0:
            return 0
        distances = np.sqrt((writing_data["X"] - x_click)**2 + (writing_data["Y"] - y_click)**2)
        idx_in_writing = np.argmin(distances)
        idx = writing_data.index[idx_in_writing]
        return idx
    
    def start_interactive(self):
        """Lance l'interface interactive de segmentation"""
        self.fig = plt.figure(figsize=(14, 10))
        gs = self.fig.add_gridspec(5, 1, height_ratios=[5, 2, 2, 1.5, 0.6], hspace=0.35)
        
        # Plot 1: Trajectoire spatiale (ZOOMABLE)
        self.ax_traj = self.fig.add_subplot(gs[0])
        self.plot_trajectory()
        
        # Plot 2: X/Y temporel
        self.ax_xy = self.fig.add_subplot(gs[1])
        self.plot_xy_temporal()
        
        # Plot 3: Speed & Pressure
        self.ax_speed = self.fig.add_subplot(gs[2])
        self.plot_speed_pressure()
        
        # Plot 4: Instructions
        self.ax_info = self.fig.add_subplot(gs[3])
        self.ax_info.axis('off')
        self.update_instructions()
        
        # Plot 5: Buttons
        self.ax_buttons = self.fig.add_subplot(gs[4])
        self.ax_buttons.axis('off')
        self._create_buttons()
        
        # Connexion des événements
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.suptitle(f"Trial {self.trial_id} - Segmentation Interactive", 
                    fontsize=14, weight="bold")
        plt.show()
        
        # Demander un nom/label pour chaque segment créé
        for seg in self.segments:
            label = input(f"\nDonnez un nom au segment {seg['segment_id']} (durée: {seg['duration_ms']:.1f}ms)\n"
                         f"  (ex: 't-i', 'base', 'suffixe', ou appuyez sur Entrée pour ignorer): ")
            seg['label'] = label.strip() if label.strip() else f"segment_{seg['segment_id']}"
        
        return self.segments
    
    def _create_buttons(self):
        """Crée les boutons d'action"""
        # Bouton "Undo Last Marker"
        ax_undo = plt.axes([0.15, 0.015, 0.15, 0.035])
        self.btn_undo = Button(ax_undo, 'Undo (u)')
        self.btn_undo.on_clicked(lambda e: self.on_key(type('obj', (), {'key': 'u'})))
        
        # Bouton "Create Segment"
        ax_segment = plt.axes([0.4, 0.015, 0.15, 0.035])
        self.btn_segment = Button(ax_segment, 'Make Segment (m)')
        self.btn_segment.on_clicked(lambda e: self.on_key(type('obj', (), {'key': 'm'})))
        
        # Bouton "Reset All"
        ax_reset = plt.axes([0.65, 0.015, 0.15, 0.035])
        self.btn_reset = Button(ax_reset, 'Reset (r)')
        self.btn_reset.on_clicked(lambda e: self.on_key(type('obj', (), {'key': 'r'})))
    
    def plot_trajectory(self):
        """Affiche la trajectoire spatiale"""
        self.ax_traj.clear()

        colors = plt.cm.tab10.colors

        def plot_with_pen_lifts(ax, subset, color, linewidth=2, alpha=0.9, zorder=None):
            x = subset["X"].to_numpy(dtype=float)
            y = subset["Y"].to_numpy(dtype=float)
            pressure = subset["NormalPressure"].to_numpy()
            x[pressure == 0] = np.nan
            y[pressure == 0] = np.nan
            kwargs = dict(color=color, linewidth=linewidth, alpha=alpha)
            if zorder is not None:
                kwargs["zorder"] = zorder
            ax.plot(x, y, **kwargs)

        # ========= MAIN TRAJECTORY =========
        if self.seg is not None:
            # Base layer: full trajectory in gray so nothing is ever invisible
            plot_with_pen_lifts(self.ax_traj, self.trial, color="lightgray", linewidth=2, zorder=1)

            for _, row in self.seg.iterrows():
                if row["Type"] != "Writing":
                    continue
                start_idx = int(row["Start"]) - self.trial_offset
                end_idx = int(row["End"]) - self.trial_offset
                word_id = int(row["WordIndex"])

                if start_idx < 0 or end_idx < 0 or start_idx >= len(self.trial) or end_idx > len(self.trial):
                    continue  # belongs to a different trial, silently skip

                subset = self.trial.iloc[start_idx:end_idx]
                plot_with_pen_lifts(self.ax_traj, subset,
                                    color=colors[word_id % len(colors)],
                                    linewidth=2.5, alpha=0.9, zorder=2)
        else:
            plot_with_pen_lifts(self.ax_traj, self.trial, color="black", linewidth=2, zorder=1)

        # ========= MARKERS =========
        for i, marker in enumerate(self.markers):
            idx = marker["idx"]
            if idx < len(self.trial):
                self.ax_traj.plot(
                    self.trial["X"].iloc[idx],
                    self.trial["Y"].iloc[idx],
                    "ro", markersize=10, zorder=10, label=f"Point {i+1}"
                )
                self.ax_traj.text(
                    self.trial["X"].iloc[idx],
                    self.trial["Y"].iloc[idx],
                    f"  {i+1}", fontsize=12, weight="bold"
                )

        # ========= USER-DEFINED SEGMENTS =========
        for seg_data in self.segments:
            idx1, idx2 = seg_data["idx1"], seg_data["idx2"]

            if idx1 < len(self.trial) and idx2 < len(self.trial):
                # Endpoint line
                self.ax_traj.plot(
                    [self.trial["X"].iloc[idx1], self.trial["X"].iloc[idx2]],
                    [self.trial["Y"].iloc[idx1], self.trial["Y"].iloc[idx2]],
                    "g-", linewidth=3, alpha=0.5, zorder=5
                )

                seg_subset = self.trial.iloc[idx1:idx2 + 1]
                plot_with_pen_lifts(self.ax_traj, seg_subset,
                                    color="lime", linewidth=3, alpha=0.7, zorder=4)

                # Duration label
                mid_x = (self.trial["X"].iloc[idx1] + self.trial["X"].iloc[idx2]) / 2
                mid_y = (self.trial["Y"].iloc[idx1] + self.trial["Y"].iloc[idx2]) / 2
                self.ax_traj.text(
                    mid_x, mid_y,
                    f"{seg_data['duration_ms']:.0f}ms",
                    fontsize=10, weight="bold",
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8)
                )

        # ========= AXIS / STYLE =========
        self.ax_traj.set_aspect(self.aspect_ratio)
        self.ax_traj.set_title(
            "Trajectoire spatiale (cliquez pour placer des marqueurs | molette pour zoom)"
        )
        self.ax_traj.set_xlabel("X (px)")
        self.ax_traj.set_ylabel("Y (px)")
        self.ax_traj.grid(True, alpha=0.3)

        if self.markers:
            self.ax_traj.legend(loc="upper right", fontsize=8)

    def plot_xy_temporal(self):
        """Affiche X et Y temporels"""
        self.ax_xy.clear()
        self.ax_xy.plot(self.t, self.trial["X"], color="steelblue", lw=1.5, label="X")
        self.ax_xy.set_ylabel("X (px)", color="steelblue")
        self.ax_xy.tick_params(axis='y', labelcolor="steelblue")
        
        ax_y = self.ax_xy.twinx()
        ax_y.plot(self.t, self.trial["Y"], color="darkorange", lw=1.5, label="Y")
        ax_y.set_ylabel("Y (px)", color="darkorange")
        ax_y.tick_params(axis='y', labelcolor="darkorange")
        
        # Marquer les points sélectionnés
        for i, marker in enumerate(self.markers):
            if marker['idx'] < len(self.t):
                t_val = self.t.iloc[marker['idx']]
                self.ax_xy.axvline(t_val, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                self.ax_xy.text(t_val, self.ax_xy.get_ylim()[1]*0.95, f"{i+1}",
                            fontsize=10, ha='center', weight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        self.ax_xy.set_xlabel("Time (s)")
        self.ax_xy.set_title("X & Y temporels")
        self.ax_xy.grid(True, alpha=0.3)
    
    def plot_speed_pressure(self):
        """Affiche vitesse et pression"""
        self.ax_speed.clear()
        self.ax_speed.plot(self.t, self.speed, color="purple", lw=1.5, label='Speed')
        self.ax_speed.axhline(self.speed_threshold, color='purple',
                             linestyle='--', alpha=0.5, label=f'Threshold ({self.speed_threshold} px/s)')
        self.ax_speed.set_ylabel("Speed (px/s)", color="purple")
        self.ax_speed.tick_params(axis='y', labelcolor="purple")
        
        ax_p = self.ax_speed.twinx()
        ax_p.plot(self.t, self.trial["NormalPressure"], color="gray", lw=1.5, label='Pressure')
        ax_p.set_ylabel("Pressure", color="gray")
        ax_p.tick_params(axis='y', labelcolor="gray")
        
        # Marquer les pauses
        pause_colors = {'pen_lift': 'red', 'low_speed': 'blue', 'mixed': 'purple'}
        for pause in self.pauses:
            color = pause_colors.get(pause['type'], 'purple')
            self.ax_speed.axvspan(pause['start_time'], pause['end_time'],
                                 color=color, alpha=0.2)
        
        # Marquer les points sélectionnés
        for i, marker in enumerate(self.markers):
            if marker['idx'] < len(self.t):
                t_val = self.t.iloc[marker['idx']]
                self.ax_speed.axvline(t_val, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        self.ax_speed.set_xlabel("Time (s)")
        self.ax_speed.set_title("Speed & Pressure")
        self.ax_speed.grid(True, alpha=0.3)
        self.ax_speed.legend(loc='upper left', fontsize=8)
    
    def update_instructions(self):
        """Met à jour les instructions"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        text = "INSTRUCTIONS:\n"
        text += "• Cliquez sur la trajectoire pour placer des marqueurs\n"
        text += "• Utilisez la MOLETTE ou le ZOOM de Matplotlib pour zoomer sur la trajectoire\n"
        text += "• Appuyez sur 'm' (ou bouton) pour créer un segment entre les 2 derniers points\n"
        text += "• Appuyez sur 'u' (ou bouton) pour annuler le dernier marqueur\n"
        text += "• Appuyez sur 'r' (ou bouton) pour recommencer (supprimer tout)\n"
        text += "• Fermez la fenêtre pour terminer\n\n"
        
        if self.markers:
            text += f"Marqueurs placés: {len(self.markers)}\n"
        if self.segments:
            text += f"Segments créés: {len(self.segments)}\n"
            for i, seg in enumerate(self.segments):
                text += f"  Segment {i+1}: {seg['duration_ms']:.1f}ms (avec {seg['num_pauses']} pauses)\n"
        
        self.ax_info.text(0.05, 0.5, text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    def on_click(self, event):
        """Gère les clics de souris"""
        if event.inaxes != self.ax_traj:
            return
        
        # Trouver le point le plus proche (among writing points only)
        idx = self.find_nearest_point(event.xdata, event.ydata)
        
        if idx >= len(self.trial):
            return
        
        self.markers.append({
            'idx': idx,
            'x': self.trial["X"].iloc[idx],
            'y': self.trial["Y"].iloc[idx],
            't': self.t.iloc[idx]
        })
        
        print(f"✓ Marqueur {len(self.markers)} placé à t={self.t.iloc[idx]:.3f}s, idx={idx}")
        
        self.refresh_plots()
    
    def on_key(self, event):
        """Gère les touches clavier"""
        if event.key == 'm':
            # Créer un segment entre les 2 derniers points
            if len(self.markers) < 2:
                print("❌ Il faut au moins 2 marqueurs pour créer un segment")
                return
            
            m1 = self.markers[-2]
            m2 = self.markers[-1]
            
            idx1, idx2 = min(m1['idx'], m2['idx']), max(m1['idx'], m2['idx'])
            
            if idx1 >= len(self.trial) or idx2 >= len(self.trial):
                print("❌ Indices invalides")
                return
                
            duration = (self.t.iloc[idx2] - self.t.iloc[idx1]) * 1000  # ms
            
            # Calculer les statistiques du segment
            segment_speed = self.speed[idx1:idx2+1]
            segment_pressure = self.trial["NormalPressure"].iloc[idx1:idx2+1]
            
            # Compter les pauses dans le segment
            segment_pauses = [p for p in self.pauses 
                             if idx1 <= p['start_idx'] <= idx2 or idx1 <= p['end_idx'] <= idx2]
            
            seg_data = {
                'trial': self.trial_id,
                'segment_id': len(self.segments) + 1,
                'idx1': idx1,
                'idx2': idx2,
                't1': self.t.iloc[idx1],
                't2': self.t.iloc[idx2],
                'duration_ms': duration,
                'num_pauses': len(segment_pauses),
                'total_pause_duration_ms': sum([p['duration_ms'] for p in segment_pauses]),
                'net_writing_duration_ms': duration - sum([p['duration_ms'] for p in segment_pauses]),
                'mean_speed': segment_speed.mean(),
                'mean_pressure': segment_pressure.mean(),
                'path_length': np.sum(np.sqrt(
                    np.diff(self.trial["X"].iloc[idx1:idx2+1])**2 + 
                    np.diff(self.trial["Y"].iloc[idx1:idx2+1])**2
                )),
                'label': ''  # Sera rempli après la session interactive
            }
            
            self.segments.append(seg_data)
            print(f"✓ Segment {len(self.segments)} créé: {duration:.1f}ms, {len(segment_pauses)} pauses")
            
        elif event.key == 'u':
            # Annuler le dernier marqueur
            if self.markers:
                removed = self.markers.pop()
                print(f"↶ Dernier marqueur annulé (était à idx={removed['idx']})")
            else:
                print("❌ Aucun marqueur à annuler")
            
        elif event.key == 'r':
            # Recommencer
            self.markers = []
            self.segments = []
            print("⟳ Remise à zéro complète")
        
        self.refresh_plots()
    
    def refresh_plots(self):
        """Rafraîchit tous les graphiques"""
        self.plot_trajectory()
        self.plot_xy_temporal()
        self.plot_speed_pressure()
        self.update_instructions()
        self.fig.canvas.draw()