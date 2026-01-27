"""
Main script for handwriting analysis with trial detection, pause analysis, 
and interactive segmentation.

This script orchestrates all the modules for a complete handwriting analysis pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from pathlib import Path

# Import custom modules
from pause_detection import detect_pauses
from letter_segmenter import LetterSegmenter
from interactive_letter_segmenter import InteractiveLetterSegmenter
from interactive_word_segmenter import InteractiveSegmenter


def robust_z(x):
    """Calculate robust z-score using median absolute deviation"""
    med = x.median()
    mad = np.median(np.abs(x - med))
    return (x - med) / (mad + 1e-6)


def run_letter_segmentation(df, trials_to_segment='all'):
    """
    Lance la segmentation des lettres pour les trials spécifiés
    
    Args:
        df: DataFrame avec les données
        trials_to_segment: 'all' ou liste d'IDs de trials
    
    Returns:
        DataFrame avec toutes les lettres détectées
    """
    all_letters = []
    
    available_trials = sorted(df['Trial'].unique())
    
    if trials_to_segment == 'all':
        trials = available_trials
    else:
        trials = [t for t in trials_to_segment if t in available_trials]
    
    for trial_id in trials:
        print(f"\n{'='*60}")
        print(f"SEGMENTATION DES LETTRES - Trial {trial_id}")
        print('='*60)
        
        trial_data = df[df['Trial'] == trial_id]
        segmenter = InteractiveLetterSegmenter(trial_data, trial_id)
        letters = segmenter.start_interactive()
        all_letters.extend(letters)
    
    return pd.DataFrame(all_letters) if all_letters else None


# === 1. PARAMÈTRES GÉNÉRAUX ===
file_path = Path(r"C:\Users\cletesson\Downloads\Hurisah-Erpent-prétest.xlsx")
sampling_rate = 200  # Hz

# Nombre d'essais attendus (mettre None si inconnu)
N_trials = 30  # ex: 60

expected_interval = 1000 / sampling_rate  # ms (5 ms)
jump_threshold = 100  # 15 ms = vraie pause entre trials
pdf_output = file_path.with_suffix(".pdf")
csv_output = file_path.with_name(file_path.stem + "_summary.csv")
pause_csv_output = file_path.with_name(file_path.stem + "_pauses.csv")
segments_csv_output = file_path.with_name(file_path.stem + "_segments.csv")

# Paramètres de détection des pauses (selon Critten 2023)
min_pause_duration_ms = 20  # ms
min_pause_samples = int(np.ceil(min_pause_duration_ms / expected_interval))
speed_threshold_px_per_s = 50  # ~1-2 cm/s (à ajuster selon votre résolution)

# === 2. CHARGEMENT DES DONNÉES PRINCIPALES ===
df = pd.read_excel(file_path)
df.columns = [c.strip() for c in df.columns]
keep_cols = ["PacketTime", "X", "Y", "NormalPressure"]
df = df[[c for c in df.columns if any(k in c for k in keep_cols)]]
df.columns = ["PacketTime", "X", "Y", "NormalPressure"]

# Nettoyage
df = df.dropna(subset=["PacketTime", "X", "Y", "NormalPressure"])
df = df[df["NormalPressure"] >= 0]

# === 2bis. CHARGEMENT DE LA FEUILLE "Segmentation" ===
try:
    seg = pd.read_excel(file_path, sheet_name="Segmentation", header=None)
    seg.columns = ["Type", "Start", "End", "WordIndex"]
    seg["WordIndex"] = seg["WordIndex"].ffill().astype(int)
except Exception as e:
    seg = None
    print("⚠️ No valid 'Segmentation' sheet found:", e)

# === 3. DÉTECTION MULTI-OBJECTIFS DES TRIALS ===

assert N_trials is not None, "N_trials must be specified"

# Variables temporelles
df["DeltaT"] = df["PacketTime"].diff().fillna(expected_interval)

# Variables spatiales
dx = df["X"].diff().fillna(0)
dy = df["Y"].diff().fillna(0)
df["DistJump"] = np.sqrt(dx**2 + dy**2)

# Pression précédente
df["PressurePrev"] = df["NormalPressure"].shift(1).fillna(0)

# --- candidats larges ---
min_gap_ms = 20
candidates = df[df["DeltaT"] > min_gap_ms].copy()

candidates["Score"] = (
    robust_z(candidates["DeltaT"]) * 2.0 +
    robust_z(candidates["DistJump"]) * 1.0 +
    robust_z(-candidates["PressurePrev"]) * 1.5
)

candidates = candidates.sort_values("Score", ascending=False)

min_separation_ms = 500
min_sep_samples = int(min_separation_ms / expected_interval)

# --- fonction d'évaluation multi-objectifs ---
def evaluate(k):
    selected = candidates.head(k).index.sort_values()

    filtered = []
    last = -np.inf
    violations = 0

    for idx in selected:
        if idx - last >= min_sep_samples:
            filtered.append(idx)
            last = idx
        else:
            violations += 1

    n_trials_found = len(filtered) + 1

    score_mean = candidates.loc[selected, "Score"].mean()

    cost = (
        1000 * abs(n_trials_found - N_trials)   # contrainte dure
        - 10 * score_mean                       # qualité
        + 50 * violations                       # séparation
    )

    return cost, filtered, n_trials_found


# --- recherche optimale ---
best_solution = None
best_cost = np.inf

max_k = min(len(candidates), N_trials * 3)

for k in range(N_trials - 1, max_k):
    cost, idxs, n_found = evaluate(k)

    if cost < best_cost:
        best_cost = cost
        best_solution = idxs

    if n_found == N_trials and cost < 0:
        break

if best_solution is None or len(best_solution) + 1 != N_trials:
    raise RuntimeError("❌ Impossible de trouver exactement le nombre d'essais demandé")

boundary_idx = best_solution

# --- construction finale ---
df["NewTrial"] = False
df.loc[boundary_idx, "NewTrial"] = True
df["Trial"] = df["NewTrial"].cumsum()

print(f"📊 Detected {df['Trial'].max()+1} trials (target={N_trials})")
print(f"   Min separation = {min_separation_ms} ms")
print(f"   Candidate boundaries used = {len(boundary_idx)}")


plt.figure(figsize=(12,4))
plt.plot(df["PacketTime"], df["DeltaT"])
plt.scatter(df.loc[boundary_idx, "PacketTime"],
            df.loc[boundary_idx, "DeltaT"], c="red")
plt.yscale("log")
plt.title("Detected trial boundaries")
plt.show()

# === 4. ANALYSES PAR TRIAL ===
trials_summary = []
all_pauses = []
all_segments = []

for trial_id, trial in df.groupby("Trial"):
    trial = trial.copy().reset_index(drop=True)
    t = (trial["PacketTime"] - trial["PacketTime"].iloc[0]) / 1000  # secondes
    p = trial["NormalPressure"].to_numpy()
    
    nonzero_idx = np.where(p > 0)[0]
    if len(nonzero_idx) == 0:
        continue

    onset = nonzero_idx[0]
    offset = nonzero_idx[-1]
    rt = t.iloc[onset]
    writing_duration = t.iloc[offset] - t.iloc[onset]

    # Calcul de vitesse
    vx = np.gradient(trial["X"], t)
    vy = np.gradient(trial["Y"], t)
    speed = np.sqrt(vx**2 + vy**2)
    
    # Détection des pauses
    pause_mask, pauses = detect_pauses(t, speed, trial["NormalPressure"], 
                                       speed_threshold_px_per_s, min_pause_samples)
    
    # Ajouter les pauses à la liste globale
    for pause in pauses:
        pause['trial'] = trial_id
        all_pauses.append(pause)
    
    # Durée totale des pauses
    total_pause_duration = sum([p['duration_ms'] for p in pauses]) / 1000  # secondes
    num_pauses = len(pauses)

    # Indicateurs spatiaux
    path_length = np.sum(np.sqrt(np.diff(trial["X"])**2 + np.diff(trial["Y"])**2))
    straight_line = np.sqrt((trial["X"].iloc[-1] - trial["X"].iloc[0])**2 +
                            (trial["Y"].iloc[-1] - trial["Y"].iloc[0])**2)
    linearity = straight_line / path_length if path_length > 0 else np.nan
    width = trial["X"].max() - trial["X"].min()
    height = trial["Y"].max() - trial["Y"].min()
    area = width * height

    trials_summary.append({
        "Trial": trial_id,
        "RT_s": round(rt, 3),
        "WritingDuration_s": round(writing_duration, 3),
        "TotalPauseDuration_s": round(total_pause_duration, 3),
        "NumPauses": num_pauses,
        "TotalDuration_s": round(t.iloc[-1], 3),
        "MeanSpeed": round(np.nanmean(speed), 2),
        "MaxSpeed": round(np.nanmax(speed), 2),
        "SpeedSD": round(np.nanstd(speed), 2),
        "PathLength": round(path_length, 2),
        "Linearity": round(linearity, 3),
        "Width": round(width, 1),
        "Height": round(height, 1),
        "Area": round(area, 1)
    })

# === 5. SAUVEGARDE CSV ===
summary_df = pd.DataFrame(trials_summary)
summary_df.to_csv(csv_output, index=False)
print(f"✅ Summary saved to: {csv_output}")

# Sauvegarde des pauses détaillées
if all_pauses:
    pauses_df = pd.DataFrame(all_pauses)
    pauses_df.to_csv(pause_csv_output, index=False)
    print(f"✅ Pauses saved to: {pause_csv_output}")
    print(f"   Total pauses detected: {len(all_pauses)}")

# === 5bis. AJOUT DES IDENTIFIANTS DE SEGMENTS AUX DONNÉES BRUTES ===
# Initialiser la colonne segment_id à -1 (pas de segment)
df['segment_id'] = -1
df['segment_label'] = ''

# === 6. SEGMENTATION INTERACTIVE DES MOTS ===
print("\n" + "="*60)
print("SEGMENTATION INTERACTIVE DES MOTS")
print("="*60)
response = input("\nVoulez-vous segmenter des mots interactivement? (o/n): ")

if response.lower() == 'o':
    # Demander quel(s) trial(s) segmenter
    available_trials = sorted(df['Trial'].unique())
    print(f"\nTrials disponibles: {available_trials}")
    trials_input = input("Entrez les numéros de trials à segmenter (séparés par des virgules, ou 'all'): ")
    
    if trials_input.lower() == 'all':
        trials_to_segment = available_trials
    else:
        trials_to_segment = [int(x.strip()) for x in trials_input.split(',')]
    
    for trial_id in trials_to_segment:
        if trial_id not in available_trials:
            print(f"⚠️ Trial {trial_id} non trouvé, ignoré")
            continue
        
        print(f"\n--- Segmentation du Trial {trial_id} ---")
        trial_data = df[df['Trial'] == trial_id]
        segmenter = InteractiveSegmenter(trial_data, trial_id, seg,
                                        speed_threshold_px_per_s, min_pause_samples)
        segments = segmenter.start_interactive()
        all_segments.extend(segments)
    
    # Sauvegarder les segments
    if all_segments:
        segments_df = pd.DataFrame(all_segments)
        # Créer un identifiant unique pour chaque segment
        segments_df['unique_segment_id'] = segments_df.apply(
            lambda row: f"T{row['trial']}_S{row['segment_id']}", axis=1
        )
        segments_df.to_csv(segments_csv_output, index=False)
        print(f"\n✅ Segments saved to: {segments_csv_output}")
        print(f"   Total segments created: {len(all_segments)}")
        
        # Annoter les données brutes avec les segments
        for _, seg_row in segments_df.iterrows():
            trial_id = seg_row['trial']
            idx1, idx2 = int(seg_row['idx1']), int(seg_row['idx2'])
            
            # Trouver les indices dans df original
            trial_mask = df['Trial'] == trial_id
            trial_indices = df[trial_mask].index
            
            if idx1 < len(trial_indices) and idx2 < len(trial_indices):
                segment_indices = trial_indices[idx1:idx2+1]
                df.loc[segment_indices, 'segment_id'] = seg_row['segment_id']
                df.loc[segment_indices, 'segment_label'] = seg_row['label']
                df.loc[segment_indices, 'unique_segment_id'] = seg_row['unique_segment_id']
        
        # Sauvegarder les données brutes annotées
        annotated_csv = file_path.with_name(file_path.stem + "_annotated.csv")
        df.to_csv(annotated_csv, index=False)
        print(f"✅ Annotated raw data saved to: {annotated_csv}")
        
        # Aussi annoter les pauses qui appartiennent à des segments
        if all_pauses:
            pauses_df['segment_id'] = -1
            pauses_df['segment_label'] = ''
            pauses_df['unique_segment_id'] = ''
            
            for _, seg_row in segments_df.iterrows():
                trial_id = seg_row['trial']
                t1, t2 = seg_row['t1'], seg_row['t2']
                
                # Trouver les pauses dans ce segment
                mask = (pauses_df['trial'] == trial_id) & \
                       (pauses_df['start_time'] >= t1) & \
                       (pauses_df['end_time'] <= t2)
                
                pauses_df.loc[mask, 'segment_id'] = seg_row['segment_id']
                pauses_df.loc[mask, 'segment_label'] = seg_row['label']
                pauses_df.loc[mask, 'unique_segment_id'] = seg_row['unique_segment_id']
            
            # Re-sauvegarder les pauses avec les annotations
            pauses_df.to_csv(pause_csv_output, index=False)
            print(f"✅ Pauses updated with segment annotations")

# Chemins de sortie pour les lettres
letters_csv_output = file_path.with_name(file_path.stem + "_letters.csv")
letters_summary_csv = file_path.with_name(file_path.stem + "_letters_summary.csv")

# === 7. SEGMENTATION INTERACTIVE DES LETTRES ===
print("\n" + "="*60)
print("SEGMENTATION INTERACTIVE DES LETTRES")
print("="*60)
response = input("\nVoulez-vous segmenter les lettres automatiquement? (o/n): ")

all_letters = []

if response.lower() == 'o':
    # Demander quel(s) trial(s) segmenter
    available_trials = sorted(df['Trial'].unique())
    print(f"\nTrials disponibles: {available_trials}")
    trials_input = input("Entrez les numéros de trials à segmenter (séparés par des virgules, ou 'all'): ")
    
    if trials_input.lower() == 'all':
        trials_to_segment = available_trials
    else:
        trials_to_segment = [int(x.strip()) for x in trials_input.split(',')]
    
    # Lancer la segmentation
    letters_df = run_letter_segmentation(df, trials_to_segment)
    
    if letters_df is not None and len(letters_df) > 0:
        # Sauvegarder les données détaillées
        letters_df.to_csv(letters_csv_output, index=False)
        print(f"\n✅ Letters saved to: {letters_csv_output}")
        print(f"   Total letters detected: {len(letters_df)}")
        
        # Créer un résumé par trial
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
        letters_summary.to_csv(letters_summary_csv)
        print(f"✅ Letters summary saved to: {letters_summary_csv}")
        
        # Annoter les données brutes avec les letter_id
        df['letter_id'] = -1
        df['letter_label'] = ''
        
        for _, letter_row in letters_df.iterrows():
            trial_id = letter_row['trial']
            idx1, idx2 = int(letter_row['idx1']), int(letter_row['idx2'])
            
            # Trouver les indices dans df original
            trial_mask = df['Trial'] == trial_id
            trial_indices = df[trial_mask].index
            
            if idx1 < len(trial_indices) and idx2 < len(trial_indices):
                letter_indices = trial_indices[idx1:idx2+1]
                df.loc[letter_indices, 'letter_id'] = letter_row['letter_id']
                df.loc[letter_indices, 'letter_label'] = letter_row['label']
        
        # Re-sauvegarder les données annotées
        annotated_csv = file_path.with_name(file_path.stem + "_annotated.csv")
        df.to_csv(annotated_csv, index=False)
        print(f"✅ Annotated data updated with letter_id")
        
        all_letters = letters_df.to_dict('records')

# === 8. GÉNÉRATION DU PDF ===
print("\n" + "="*60)
print("GÉNÉRATION DU PDF (avec lettres)")
print("="*60)

with PdfPages(pdf_output) as pdf:
    for trial_id, trial in df.groupby("Trial"):
        trial = trial.reset_index(drop=True)
        t = (trial["PacketTime"] - trial["PacketTime"].iloc[0]) / 1000
        p = trial["NormalPressure"]
        if p.max() == 0:
            continue

        vx = np.gradient(trial["X"], t)
        vy = np.gradient(trial["Y"], t)
        speed = np.sqrt(vx**2 + vy**2)

        # Détection des pauses
        pause_mask, pauses = detect_pauses(t, speed, p, 
                                          speed_threshold_px_per_s, min_pause_samples)

        # Récupérer les lettres pour ce trial
        trial_letters = [l for l in all_letters if l['trial'] == trial_id] if all_letters else []

        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(6, 1, height_ratios=[4, 0.3, 0.3, 2, 2, 0.5], hspace=0.4)
        
        # === Plot 1 : Trajectoire spatiale avec lettres ===
        ax = fig.add_subplot(gs[0])
        
        # Tracer la trajectoire de base
        ax.plot(trial["X"], trial["Y"], color="lightgray", linewidth=1, alpha=0.5)
        
        # Colorier chaque lettre
        if trial_letters:
            colors_letters = plt.cm.tab20.colors
            for i, letter in enumerate(trial_letters):
                idx1, idx2 = letter['idx1'], letter['idx2']
                subset = trial.iloc[idx1:idx2+1]
                ax.plot(subset["X"], subset["Y"],
                       color=colors_letters[i % len(colors_letters)],
                       linewidth=2.5, alpha=0.9)
                
                # Label de la lettre
                mid_x = subset["X"].mean()
                mid_y = subset["Y"].mean()
                label_text = letter['label'] if letter['label'] else str(i+1)
                ax.text(mid_x, mid_y, label_text,
                       fontsize=11, weight='bold',
                       bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        # Ajouter les segments de mots si présents
        trial_segments = [s for s in all_segments if s['trial'] == trial_id] if all_segments else []
        for i, seg_data in enumerate(trial_segments):
            idx1, idx2 = seg_data['idx1'], seg_data['idx2']
            ax.plot([trial["X"].iloc[idx1], trial["X"].iloc[idx2]],
                   [trial["Y"].iloc[idx1], trial["Y"].iloc[idx2]],
                   'r-', linewidth=3, alpha=0.4)

        ax.set_aspect(224 / 140)
        title_text = f"Trial {trial_id} – Spatial Trajectory"
        if trial_letters:
            title_text += f" ({len(trial_letters)} letters)"
        ax.set_title(title_text, fontsize=11, weight="bold")
        ax.axis("off")

        # === Plot 2 : Barre de pression ===
        ax_pressure = fig.add_subplot(gs[1])
        norm_p = Normalize(vmin=p.min(), vmax=p.max())
        pressure_gradient = p.values.reshape(1, -1)
        ax_pressure.imshow(pressure_gradient, aspect='auto', cmap='YlOrRd', 
                          extent=[t.iloc[0], t.iloc[-1], 0, 1], norm=norm_p)
        ax_pressure.set_ylabel('Pressure', fontsize=8)
        ax_pressure.set_yticks([])
        ax_pressure.set_xticks([])
        
        # === Plot 3 : Barre de vitesse ===
        ax_speed_bar = fig.add_subplot(gs[2])
        norm_s = Normalize(vmin=speed.min(), vmax=speed.max())
        speed_gradient = speed.reshape(1, -1)
        ax_speed_bar.imshow(speed_gradient, aspect='auto', cmap='viridis',
                       extent=[t.iloc[0], t.iloc[-1], 0, 1], norm=norm_s)
        ax_speed_bar.set_ylabel('Speed', fontsize=8)
        ax_speed_bar.set_yticks([])
        ax_speed_bar.set_xlabel('Time (s)', fontsize=8)

        # === Plot 4 : X/Y + Pauses + Lettres ===
        ax2 = fig.add_subplot(gs[3])
        ax2.plot(t, trial["X"], color="steelblue", lw=1, label="X")
        ax2.set_ylabel("X (px)", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")

        ax3 = ax2.twinx()
        ax3.plot(t, trial["Y"], color="darkorange", lw=1, label="Y")
        ax3.set_ylabel("Y (px)", color="darkorange")
        ax3.tick_params(axis="y", labelcolor="darkorange")

        # Marquer les frontières de lettres
        if trial_letters:
            for letter in trial_letters:
                t1, t2 = letter['t1'], letter['t2']
                ax2.axvline(t1, color='green', linestyle=':', alpha=0.5, linewidth=1)
                ax2.axvline(t2, color='green', linestyle=':', alpha=0.5, linewidth=1)

        # Marquer les pauses
        for pause in pauses:
            color = {'pen_lift': 'red', 'low_speed': 'blue', 'mixed': 'purple'}[pause['type']]
            ax2.axvspan(pause['start_time'], pause['end_time'], 
                       color=color, alpha=0.2, zorder=0)

        ax2.set_title(f"X & Y trajectories + Pauses + Letters", 
                     fontsize=10, weight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True, alpha=0.3)

        # === Plot 5 : Speed & Pressure ===
        ax4 = fig.add_subplot(gs[4])
        ax4.plot(t, speed, color="purple", lw=1)
        ax4.axhline(speed_threshold_px_per_s, color='purple', 
                   linestyle='--', alpha=0.5)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Speed (px/s)", color="purple")
        ax4.tick_params(axis="y", labelcolor="purple")

        ax5 = ax4.twinx()
        ax5.plot(t, p, color="gray", lw=1)
        ax5.set_ylabel("Pressure", color="gray")
        ax5.tick_params(axis="y", labelcolor="gray")

        ax4.set_title("Speed & Pressure over time", fontsize=10, weight="bold")
        ax4.grid(True, alpha=0.3)

        # === Plot 6 : Table des lettres ===
        if trial_letters:
            ax_table = fig.add_subplot(gs[5])
            ax_table.axis('off')
            
            # Créer une table avec les propriétés des lettres
            table_data = []
            for letter in trial_letters[:10]:  # Limiter à 10 pour l'espace
                table_data.append([
                    letter['label'],
                    f"{letter['duration_ms']:.0f}",
                    f"{letter['mean_speed']:.0f}",
                    f"{letter['width']:.0f}×{letter['height']:.0f}"
                ])
            
            if table_data:
                table = ax_table.table(cellText=table_data,
                                      colLabels=['Letter', 'Duration (ms)', 'Speed (px/s)', 'Size (px)'],
                                      cellLoc='center',
                                      loc='center',
                                      bbox=[0, 0, 1, 1])
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.5)

        plt.suptitle(f"Trial {trial_id} | {len(pauses)} pauses | "
                    f"{len(trial_segments)} word segments | "
                    f"{len(trial_letters)} letters", 
                    fontsize=12, weight="bold")
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

print(f"✅ PDF exported to: {pdf_output}")

# === 9. RÉSUMÉ FINAL ===
print("\n📝 RÉSUMÉ FINAL:")
print(f"   Trials analysés: {len(trials_summary)}")
print(f"   Pauses détectées: {len(all_pauses)}")
if all_segments:
    print(f"   Segments de mots créés: {len(all_segments)}")
if all_letters:
    print(f"   Lettres segmentées: {len(all_letters)}")
print(f"   Seuil de vitesse: {speed_threshold_px_per_s} px/s")
print(f"   Durée minimale de pause: {min_pause_duration_ms} ms")

print("\n📂 FICHIERS GÉNÉRÉS:")
print(f"   • {csv_output.name} - Statistiques par trial")
print(f"   • {pause_csv_output.name} - Détail des pauses")
if all_segments:
    print(f"   • {segments_csv_output.name} - Segments de mots")
if all_letters:
    print(f"   • {letters_csv_output.name} - Lettres détaillées")
    print(f"   • {letters_summary_csv.name} - Résumé par trial")
    print(f"   • {file_path.stem}_annotated.csv - Données brutes avec letter_id")
print(f"   • {pdf_output.name} - Visualisations")
