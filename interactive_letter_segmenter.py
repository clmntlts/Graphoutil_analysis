"""
Interactive interface for letter segmentation with visualization and manual correction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from letter_segmenter import LetterSegmenter


class InteractiveLetterSegmenter:
    """
    Interface interactive pour la segmentation des lettres avec :
    - Segmentation automatique initiale
    - Correction manuelle (ajout/suppression/déplacement de frontières)
    - Labellisation des lettres
    """
    
    def __init__(self, trial_data, trial_id):
        self.trial = trial_data.reset_index(drop=True)
        self.trial_id = trial_id
        self.t = (self.trial["PacketTime"] - self.trial["PacketTime"].iloc[0]) / 1000
        
        # Segmentation automatique
        self.segmenter = LetterSegmenter(trial_data)
        self.boundaries = list(self.segmenter.detect_letter_boundaries())
        self.letters = []
        
        # État de l'interface
        self.selected_boundary = None
        
    def start_interactive(self):
        """Lance l'interface interactive"""
        self.fig = plt.figure(figsize=(14, 10))
        gs = self.fig.add_gridspec(5, 2, height_ratios=[4, 2, 2, 1, 0.5], 
                                   width_ratios=[3, 1], hspace=0.3, wspace=0.3)
        
        # Plot principal : Trajectoire
        self.ax_traj = self.fig.add_subplot(gs[0, 0])
        
        # Plot secondaire : Vitesse/Pression
        self.ax_speed = self.fig.add_subplot(gs[1, 0])
        
        # Plot : Courbure
        self.ax_curv = self.fig.add_subplot(gs[2, 0])
        
        # Zone d'instructions
        self.ax_info = self.fig.add_subplot(gs[3, :])
        self.ax_info.axis('off')
        
        # Boutons
        self.ax_buttons = self.fig.add_subplot(gs[4, :])
        self.ax_buttons.axis('off')
        
        # Liste des lettres
        self.ax_letters = self.fig.add_subplot(gs[0:3, 1])
        self.ax_letters.axis('off')
        
        # Créer les boutons
        self._create_buttons()
        
        # Affichage initial
        self.refresh_plots()
        
        # Connexion des événements
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.suptitle(f"Trial {self.trial_id} - Segmentation Automatique des Lettres", 
                    fontsize=14, weight="bold")
        plt.show()
        
        # Labellisation finale
        print("\n" + "="*60)
        print("LABELLISATION DES LETTRES")
        print("="*60)
        self._label_letters()
        
        return self.letters
    
    def _create_buttons(self):
        """Crée les boutons d'action"""
        # Bouton "Auto Segment"
        ax_auto = plt.axes([0.1, 0.02, 0.15, 0.04])
        self.btn_auto = Button(ax_auto, 'Auto Segment')
        self.btn_auto.on_clicked(self.auto_segment)
        
        # Bouton "Validate"
        ax_validate = plt.axes([0.3, 0.02, 0.15, 0.04])
        self.btn_validate = Button(ax_validate, 'Validate & Label')
        self.btn_validate.on_clicked(self.validate_and_label)
        
        # Bouton "Clear All"
        ax_clear = plt.axes([0.5, 0.02, 0.15, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_clear.on_clicked(self.clear_all)
    
    def auto_segment(self, event=None):
        """Re-calcule la segmentation automatique"""
        self.boundaries = list(self.segmenter.detect_letter_boundaries())
        self.refresh_plots()
        print(f"✓ Auto-segmentation: {len(self.boundaries)+1} lettres détectées")
    
    def validate_and_label(self, event=None):
        """Valide et passe à la labellisation"""
        plt.close(self.fig)
    
    def clear_all(self, event=None):
        """Supprime toutes les frontières"""
        self.boundaries = []
        self.refresh_plots()
        print("✓ Toutes les frontières supprimées")
    
    def refresh_plots(self):
        """Rafraîchit tous les graphiques"""
        self._plot_trajectory()
        self._plot_speed_pressure()
        self._plot_curvature()
        self._update_instructions()
        self._update_letter_list()
        self.fig.canvas.draw()
    
    def _plot_trajectory(self):
        """Affiche la trajectoire avec les frontières"""
        self.ax_traj.clear()
        
        # Tracer la trajectoire complète
        self.ax_traj.plot(self.trial["X"], self.trial["Y"], 
                         color="black", linewidth=1.5, alpha=0.3)
        
        # Colorier chaque lettre
        colors = plt.cm.tab20.colors
        all_bounds = [0] + sorted(self.boundaries) + [len(self.trial)-1]
        
        for i in range(len(all_bounds)-1):
            idx1, idx2 = all_bounds[i], all_bounds[i+1]
            subset = self.trial.iloc[idx1:idx2+1]
            self.ax_traj.plot(subset["X"], subset["Y"],
                             color=colors[i % len(colors)],
                             linewidth=2.5, alpha=0.9)
            
            # Numéro de lettre
            mid_x = subset["X"].mean()
            mid_y = subset["Y"].mean()
            self.ax_traj.text(mid_x, mid_y, str(i+1),
                             fontsize=12, weight='bold',
                             bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        # Marquer les frontières
        for b in self.boundaries:
            self.ax_traj.plot(self.trial["X"].iloc[b], self.trial["Y"].iloc[b],
                             'ro', markersize=10, zorder=10)
        
        self.ax_traj.set_aspect(224 / 140)
        self.ax_traj.set_title(f"Trajectoire ({len(all_bounds)-1} lettres)")
        self.ax_traj.axis('off')
    
    def _plot_speed_pressure(self):
        """Affiche vitesse et pression avec frontières"""
        self.ax_speed.clear()
        
        # Vitesse
        self.ax_speed.plot(self.t, self.segmenter.speed_smooth, 
                          color='purple', linewidth=1.5)
        self.ax_speed.set_ylabel('Speed (px/s)', color='purple')
        
        # Pression
        ax_p = self.ax_speed.twinx()
        ax_p.plot(self.t, self.segmenter.pressure, 
                 color='gray', linewidth=1.5, alpha=0.6)
        ax_p.set_ylabel('Pressure', color='gray')
        
        # Marquer les frontières
        for b in self.boundaries:
            t_val = self.t.iloc[b]
            self.ax_speed.axvline(t_val, color='red', linestyle='--', alpha=0.7)
        
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_title('Speed & Pressure')
        self.ax_speed.grid(True, alpha=0.3)
    
    def _plot_curvature(self):
        """Affiche la courbure avec frontières"""
        self.ax_curv.clear()
        self.ax_curv.plot(self.t, self.segmenter.curvature_smooth,
                         color='green', linewidth=1.5)
        self.ax_curv.set_ylabel('Curvature', color='green')
        
        # Marquer les frontières
        for b in self.boundaries:
            t_val = self.t.iloc[b]
            self.ax_curv.axvline(t_val, color='red', linestyle='--', alpha=0.7)
        
        self.ax_curv.set_xlabel('Time (s)')
        self.ax_curv.set_title('Curvature')
        self.ax_curv.grid(True, alpha=0.3)
    
    def _update_instructions(self):
        """Met à jour les instructions"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        text = "INSTRUCTIONS:\n"
        text += "• Cliquez sur la TRAJECTOIRE pour ajouter/déplacer une frontière\n"
        text += "• Cliquez sur une frontière (point rouge) puis 'd' pour la supprimer\n"
        text += "• 'Auto Segment': Re-calculer automatiquement\n"
        text += "• 'Validate & Label': Passer à la labellisation\n"
        text += f"\nFrontières actuelles: {len(self.boundaries)} → {len(self.boundaries)+1} lettres"
        
        self.ax_info.text(0.05, 0.95, text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    def _update_letter_list(self):
        """Affiche la liste des lettres avec leurs propriétés"""
        self.ax_letters.clear()
        self.ax_letters.axis('off')
        
        # Calculer les propriétés de chaque lettre
        all_bounds = [0] + sorted(self.boundaries) + [len(self.trial)-1]
        
        text = "LETTRES DÉTECTÉES\n" + "="*30 + "\n\n"
        for i in range(len(all_bounds)-1):
            idx1, idx2 = all_bounds[i], all_bounds[i+1]
            duration = (self.t.iloc[idx2] - self.t.iloc[idx1]) * 1000
            width = self.trial["X"].iloc[idx1:idx2+1].max() - self.trial["X"].iloc[idx1:idx2+1].min()
            height = self.trial["Y"].iloc[idx1:idx2+1].max() - self.trial["Y"].iloc[idx1:idx2+1].min()
            
            text += f"Lettre {i+1}:\n"
            text += f"  Durée: {duration:.0f}ms\n"
            text += f"  Taille: {width:.0f}×{height:.0f}px\n\n"
        
        self.ax_letters.text(0.05, 0.95, text, transform=self.ax_letters.transAxes,
                            fontsize=9, verticalalignment='top', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    def on_click(self, event):
        """Gère les clics (ajouter/sélectionner frontière)"""
        if event.inaxes != self.ax_traj:
            return
        
        # Trouver le point le plus proche
        distances = np.sqrt((self.trial["X"] - event.xdata)**2 + 
                           (self.trial["Y"] - event.ydata)**2)
        idx = np.argmin(distances)
        
        # Vérifier si on a cliqué près d'une frontière existante
        clicked_boundary = None
        for b in self.boundaries:
            dist_to_boundary = np.sqrt((self.trial["X"].iloc[b] - event.xdata)**2 +
                                      (self.trial["Y"].iloc[b] - event.ydata)**2)
            if dist_to_boundary < 20:  # Tolérance en pixels
                clicked_boundary = b
                break
        
        if clicked_boundary is not None:
            # Sélectionner cette frontière
            self.selected_boundary = clicked_boundary
            print(f"✓ Frontière sélectionnée à idx={clicked_boundary}")
        else:
            # Ajouter une nouvelle frontière
            self.boundaries.append(idx)
            self.boundaries = sorted(self.boundaries)
            print(f"✓ Frontière ajoutée à idx={idx}")
        
        self.refresh_plots()
    
    def on_key(self, event):
        """Gère les touches clavier"""
        if event.key == 'd' and self.selected_boundary is not None:
            # Supprimer la frontière sélectionnée
            self.boundaries.remove(self.selected_boundary)
            print(f"✓ Frontière supprimée à idx={self.selected_boundary}")
            self.selected_boundary = None
            self.refresh_plots()
    
    def _label_letters(self):
        """Labellise chaque lettre"""
        all_bounds = [0] + sorted(self.boundaries) + [len(self.trial)-1]
        
        print(f"\nVous avez {len(all_bounds)-1} lettres à labelliser.")
        print("Pour chaque lettre, entrez la lettre correspondante (ex: 'a', 'b', 't', 'i'...)")
        print("Ou appuyez sur Entrée pour ignorer.\n")
        
        for i in range(len(all_bounds)-1):
            idx1, idx2 = all_bounds[i], all_bounds[i+1]
            letter_data = self.segmenter._compute_letter_features(idx1, idx2, i+1)
            letter_data['trial'] = self.trial_id
            
            label = input(f"Lettre {i+1} (durée: {letter_data['duration_ms']:.0f}ms, "
                         f"taille: {letter_data['width']:.0f}×{letter_data['height']:.0f}px): ")
            letter_data['label'] = label.strip() if label.strip() else f"letter_{i+1}"
            
            self.letters.append(letter_data)
