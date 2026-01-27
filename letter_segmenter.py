"""
Automatic letter segmentation for cursive handwriting.
Includes LetterSegmenter class for detecting letter boundaries.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import DBSCAN


class LetterSegmenter:
    """
    Segmentation automatique des lettres cursives basée sur plusieurs critères :
    1. Minima locaux de vitesse
    2. Changements de direction (courbure)
    3. Pics de pression
    4. Analyse de la hauteur (y) pour détecter les lettres montantes/descendantes
    """
    
    def __init__(self, trial_data, min_letter_duration_ms=100, smoothing_window=5):
        self.trial = trial_data.reset_index(drop=True)
        self.t = (self.trial["PacketTime"] - self.trial["PacketTime"].iloc[0]) / 1000
        self.min_letter_samples = int(min_letter_duration_ms / 5)  # ~5ms par échantillon
        self.smoothing_window = smoothing_window
        
        # Calculer les métriques
        self._compute_metrics()
        
    def _compute_metrics(self):
        """Calcule toutes les métriques nécessaires"""
        # Vitesse
        vx = np.gradient(self.trial["X"], self.t)
        vy = np.gradient(self.trial["Y"], self.t)
        self.speed = np.sqrt(vx**2 + vy**2)
        self.speed_smooth = savgol_filter(self.speed, 
                                          min(self.smoothing_window, len(self.speed)//2*2-1), 
                                          2)
        
        # Accélération
        ax = np.gradient(vx, self.t)
        ay = np.gradient(vy, self.t)
        self.accel = np.sqrt(ax**2 + ay**2)
        
        # Courbure (changement de direction)
        self.curvature = np.abs(np.gradient(np.arctan2(vy, vx)))
        self.curvature_smooth = savgol_filter(self.curvature,
                                             min(self.smoothing_window, len(self.curvature)//2*2-1),
                                             2)
        
        # Dérivée verticale (pour détecter montées/descentes)
        self.dy_dt = np.gradient(self.trial["Y"], self.t)
        
        # Pression
        self.pressure = self.trial["NormalPressure"].values
        
    def detect_letter_boundaries(self, method='multi_criteria'):
        """
        Détecte les frontières de lettres
        
        Methods:
        - 'speed_minima': Basé uniquement sur les minima de vitesse
        - 'curvature_peaks': Basé sur les pics de courbure
        - 'multi_criteria': Combinaison de plusieurs critères (RECOMMANDÉ)
        """
        
        if method == 'speed_minima':
            return self._detect_speed_minima()
        elif method == 'curvature_peaks':
            return self._detect_curvature_peaks()
        else:
            return self._detect_multi_criteria()
    
    def _detect_speed_minima(self):
        """Détecte les minima de vitesse comme frontières"""
        # Trouver les minima locaux de vitesse
        minima, _ = find_peaks(-self.speed_smooth, 
                               distance=self.min_letter_samples,
                               prominence=np.percentile(self.speed_smooth, 30))
        return minima
    
    def _detect_curvature_peaks(self):
        """Détecte les pics de courbure comme frontières"""
        # Trouver les pics de courbure
        peaks, _ = find_peaks(self.curvature_smooth,
                             distance=self.min_letter_samples,
                             prominence=np.percentile(self.curvature_smooth, 70))
        return peaks
    
    def _detect_multi_criteria(self):
        """
        Détection multi-critères (méthode recommandée)
        Combine vitesse, courbure, et changements verticaux
        """
        # 1. Candidats basés sur la vitesse
        speed_minima, _ = find_peaks(-self.speed_smooth,
                                     distance=self.min_letter_samples//2,
                                     prominence=np.percentile(self.speed_smooth, 20))
        
        # 2. Candidats basés sur la courbure
        curv_peaks, _ = find_peaks(self.curvature_smooth,
                                   distance=self.min_letter_samples//2,
                                   prominence=np.percentile(self.curvature_smooth, 60))
        
        # 3. Candidats basés sur les changements de direction verticale
        dy_changes = np.where(np.diff(np.sign(self.dy_dt)) != 0)[0]
        
        # Combiner tous les candidats
        all_candidates = np.unique(np.concatenate([speed_minima, curv_peaks, dy_changes]))
        
        # Scorer chaque candidat
        scores = np.zeros(len(all_candidates))
        for i, idx in enumerate(all_candidates):
            # Score basé sur plusieurs critères (plus c'est haut, mieux c'est)
            score = 0
            
            # Vitesse faible
            if self.speed_smooth[idx] < np.percentile(self.speed_smooth, 30):
                score += 2
            
            # Courbure élevée
            if self.curvature_smooth[idx] > np.percentile(self.curvature_smooth, 70):
                score += 2
            
            # Changement de direction verticale
            if idx in dy_changes:
                score += 1
            
            # Changement de pression (début/fin de lettre)
            if idx > 0 and idx < len(self.pressure) - 1:
                pressure_change = abs(self.pressure[idx] - self.pressure[idx-1])
                if pressure_change > np.percentile(np.abs(np.diff(self.pressure)), 60):
                    score += 1
            
            scores[i] = score
        
        # Garder les meilleurs candidats (score >= 3)
        good_candidates = all_candidates[scores >= 3]
        
        # Filtrer pour avoir une distance minimale entre frontières
        if len(good_candidates) > 0:
            # Clustering pour grouper les candidats proches
            good_candidates_2d = good_candidates.reshape(-1, 1)
            clustering = DBSCAN(eps=self.min_letter_samples//2, min_samples=1).fit(good_candidates_2d)
            
            # Prendre le centre de chaque cluster
            final_boundaries = []
            for label in np.unique(clustering.labels_):
                cluster_points = good_candidates[clustering.labels_ == label]
                # Prendre le point avec le meilleur score dans le cluster
                cluster_indices = np.where(clustering.labels_ == label)[0]
                best_in_cluster = cluster_points[np.argmax(scores[cluster_indices])]
                final_boundaries.append(best_in_cluster)
            
            return np.array(sorted(final_boundaries))
        else:
            return np.array([])
    
    def segment_into_letters(self, boundaries=None):
        """
        Crée des segments de lettres à partir des frontières
        
        Returns:
        - Liste de dictionnaires avec les propriétés de chaque lettre
        """
        if boundaries is None:
            boundaries = self.detect_letter_boundaries()
        
        # Ajouter début et fin
        all_boundaries = np.concatenate([[0], boundaries, [len(self.trial)-1]])
        
        letters = []
        for i in range(len(all_boundaries)-1):
            idx1 = int(all_boundaries[i])
            idx2 = int(all_boundaries[i+1])
            
            # Calculer les propriétés de la lettre
            letter_data = self._compute_letter_features(idx1, idx2, i+1)
            letters.append(letter_data)
        
        return letters
    
    def _compute_letter_features(self, idx1, idx2, letter_id):
        """Calcule les features pour une lettre donnée"""
        duration = (self.t.iloc[idx2] - self.t.iloc[idx1]) * 1000  # ms
        
        # Trajectoire
        x = self.trial["X"].iloc[idx1:idx2+1].values
        y = self.trial["Y"].iloc[idx1:idx2+1].values
        
        # Longueur du tracé
        path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        
        # Distance directe
        straight_line = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        
        # Dimensions
        width = x.max() - x.min()
        height = y.max() - y.min()
        
        # Vitesse et pression moyennes
        mean_speed = self.speed[idx1:idx2+1].mean()
        mean_pressure = self.pressure[idx1:idx2+1].mean()
        
        # Direction dominante (montante/descendante)
        dy_total = y[-1] - y[0]
        direction = 'ascending' if dy_total < -10 else ('descending' if dy_total > 10 else 'horizontal')
        
        # Nombre de pics (peut indiquer des lettres avec boucles)
        y_segment = y - y.mean()
        n_peaks_y, _ = find_peaks(y_segment, prominence=height*0.2)
        n_valleys_y, _ = find_peaks(-y_segment, prominence=height*0.2)
        n_loops = len(n_peaks_y) + len(n_valleys_y)
        
        return {
            'letter_id': letter_id,
            'idx1': idx1,
            'idx2': idx2,
            't1': self.t.iloc[idx1],
            't2': self.t.iloc[idx2],
            'duration_ms': duration,
            'path_length': path_length,
            'straight_line': straight_line,
            'linearity': straight_line / path_length if path_length > 0 else 0,
            'width': width,
            'height': height,
            'aspect_ratio': width / height if height > 0 else 0,
            'mean_speed': mean_speed,
            'mean_pressure': mean_pressure,
            'direction': direction,
            'n_loops': n_loops,
            'label': ''  # À remplir manuellement
        }
