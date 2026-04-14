"""
Data loading and preprocessing utilities.
Handles Excel file loading, column mapping, and data cleaning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

_UINT32_RANGE = 2**32


# ---------------------------------------------------------------------------
# Interactive timestamp repair
# ---------------------------------------------------------------------------

class InteractiveTimestampRepair:
    """
    Matplotlib GUI that shows aberrant PacketTime regions and lets the
    user drag-select spans to interpolate.

    Layout
    ------
    Top panel  : Raw PacketTime signal (full recording)
    Middle panel: diff(PacketTime) — shows jumps and anomalies clearly
    Bottom panel: Zoomed view of the currently selected / hovered region

    Controls
    --------
    Click + drag inside the TOP or MIDDLE panel  →  mark a span for repair
    Double-click a marked span                   →  remove it
    "Apply & Continue" button                    →  interpolate all marked
                                                    spans and close
    "Reset" button                               →  clear all marks
    """

    _BTN_H          = 0.040
    _BTN_GAP        = 0.010
    _TOOLBAR_BOTTOM = 0.005

    def __init__(self, df: pd.DataFrame):
        self.df       = df.copy()
        self.spans    = []          # list of (start_idx, end_idx) integer positions
        self._drag_start = None
        self._drag_rect  = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Launch GUI, block until closed, return repaired DataFrame."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.widgets import Button

        self._plt = plt

        t   = self.df["PacketTime"].to_numpy(dtype=np.float64)
        idx = np.arange(len(t))

        fig, axes = plt.subplots(
            3, 1, figsize=(15, 9),
            gridspec_kw={"height_ratios": [2, 2, 1.5]}
        )
        self.fig   = fig
        self.ax_t  = axes[0]   # raw PacketTime
        self.ax_dt = axes[1]   # diff
        self.ax_z  = axes[2]   # zoom

        # ── Button row ────────────────────────────────────────────────
        toolbar_h  = self._BTN_H + self._BTN_GAP + self._TOOLBAR_BOTTOM
        fig.subplots_adjust(
            top=0.93, bottom=toolbar_h + 0.05,
            left=0.07, right=0.97, hspace=0.30
        )

        btn_w, gap = 0.18, 0.02
        total_w    = 2 * btn_w + gap
        left_start = (1.0 - total_w) / 2

        ax_btn_apply = fig.add_axes([
            left_start, self._TOOLBAR_BOTTOM, btn_w, self._BTN_H
        ])
        ax_btn_reset = fig.add_axes([
            left_start + btn_w + gap, self._TOOLBAR_BOTTOM, btn_w, self._BTN_H
        ])

        self.btn_apply = Button(
            ax_btn_apply, "✓  Apply & Continue",
            color="#b6d7a8", hovercolor="#6aa84f"
        )
        self.btn_reset = Button(
            ax_btn_reset, "↺  Reset",
            color="#ffe599", hovercolor="#f1c232"
        )
        self.btn_apply.on_clicked(lambda e: self._do_apply())
        self.btn_reset.on_clicked(lambda e: self._do_reset())

        # ── Store data ────────────────────────────────────────────────
        self._t   = t
        self._idx = idx
        self._dt  = np.diff(t, prepend=t[0])

        self._draw_all()
        self._connect_events()
        self._print_instructions()

        plt.show()
        return self._build_result()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_all(self):
        self._draw_raw()
        self._draw_diff()
        self._draw_zoom()
        self._update_title()
        self.fig.canvas.draw_idle()

    def _draw_raw(self):
        ax = self.ax_t
        ax.clear()
        ax.plot(self._idx, self._t, linewidth=0.6, color="steelblue", alpha=0.8)
        self._shade_spans(ax, y_coords=(self._t.min(), self._t.max()))
        ax.set_ylabel("PacketTime", fontsize=9)
        ax.set_title(
            "Raw PacketTime   —   click + drag to mark a span for interpolation",
            fontsize=9
        )
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelbottom=False)

    def _draw_diff(self):
        ax = self.ax_dt
        ax.clear()
        ax.plot(self._idx, self._dt, linewidth=0.6, color="tomato", alpha=0.85)

        # Mark negative diffs (the primary symptom of non-monotonicity)
        neg_mask = self._dt < 0
        if neg_mask.any():
            ax.scatter(
                self._idx[neg_mask], self._dt[neg_mask],
                c="red", s=30, zorder=5, label="Non-monotonic"
            )
        # Mark large positive jumps
        med   = np.median(np.abs(self._dt[self._dt != 0])) if np.any(self._dt != 0) else 1
        big   = self._dt > med * 50
        if big.any():
            ax.scatter(
                self._idx[big], self._dt[big],
                c="orange", s=25, zorder=4, marker="^", label="Large jump"
            )

        self._shade_spans(ax, y_coords=(self._dt.min(), self._dt.max()))
        ax.set_ylabel("ΔPacketTime", fontsize=9)
        ax.set_title(
            "diff(PacketTime)   —   red dots = non-monotonic,  orange = large jump",
            fontsize=9
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelbottom=False)

    def _draw_zoom(self):
        ax = self.ax_z
        ax.clear()
        if not self.spans:
            ax.text(
                0.5, 0.5, "No spans selected yet.\nDrag on either panel above.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="gray"
            )
            ax.set_title("Zoom view (last selected span)", fontsize=9)
            ax.grid(True, alpha=0.25)
            return

        s, e   = self.spans[-1]
        margin = max(int((e - s) * 0.3), 20)
        lo     = max(0, s - margin)
        hi     = min(len(self._t) - 1, e + margin)

        zoom_idx = self._idx[lo:hi+1]
        zoom_t   = self._t[lo:hi+1]
        zoom_dt  = self._dt[lo:hi+1]

        ax2 = ax.twinx()
        ax.plot(zoom_idx, zoom_t,  color="steelblue", linewidth=1.2,
                alpha=0.8, label="PacketTime")
        ax2.plot(zoom_idx, zoom_dt, color="tomato",   linewidth=1.0,
                alpha=0.7, label="ΔT", linestyle="--")

        ax.axvspan(s, e, color="gold", alpha=0.30, label="Selected")
        ax.set_ylabel("PacketTime", fontsize=8, color="steelblue")
        ax2.set_ylabel("ΔT", fontsize=8, color="tomato")
        ax.set_title(
            f"Zoom: indices {s}–{e}  ({e-s+1} frames)",
            fontsize=9
        )
        ax.grid(True, alpha=0.25)

    def _shade_spans(self, ax, y_coords):
        """Draw all marked spans as semi-transparent gold rectangles."""
        y0, y1 = y_coords
        for s, e in self.spans:
            ax.axvspan(s, e, color="gold", alpha=0.28)
            ax.text(
                (s + e) / 2, y0 + (y1 - y0) * 0.02,
                f"[{s}:{e}]",
                ha="center", va="bottom", fontsize=7, color="darkgoldenrod"
            )

    def _update_title(self):
        n   = len(self.spans)
        neg = int(np.sum(self._dt < 0))
        self.fig.suptitle(
            f"Interactive Timestamp Repair   —   "
            f"{neg} non-monotonic frame(s) detected   |   "
            f"{n} span(s) marked for interpolation",
            fontsize=10, fontweight="bold",
            color="darkred" if neg > 0 else "darkgreen"
        )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _connect_events(self):
        fig = self.fig
        self._cid_press   = fig.canvas.mpl_connect("button_press_event",
                                                    self._on_press)
        self._cid_release = fig.canvas.mpl_connect("button_release_event",
                                                    self._on_release)
        self._cid_motion  = fig.canvas.mpl_connect("motion_notify_event",
                                                    self._on_motion)

    def _on_press(self, event):
        if event.inaxes not in (self.ax_t, self.ax_dt):
            return
        if event.dblclick:
            self._try_remove_span(event)
            return
        self._drag_start = int(round(event.xdata))
        self._drag_rect  = None

    def _on_motion(self, event):
        if self._drag_start is None:
            return
        if event.inaxes not in (self.ax_t, self.ax_dt):
            return
        cur = int(round(event.xdata))
        s, e = sorted([self._drag_start, cur])

        # Live rubber-band rect on the diff panel
        if self._drag_rect is not None:
            self._drag_rect.remove()
        self._drag_rect = self.ax_dt.axvspan(s, e, color="gold", alpha=0.4)
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._drag_start is None:
            return
        if event.inaxes not in (self.ax_t, self.ax_dt):
            self._drag_start = None
            self._drag_rect  = None
            return

        end = int(round(event.xdata))
        s, e = sorted([self._drag_start, end])
        s = max(0, s)
        e = min(len(self._t) - 1, e)

        if e > s:
            self.spans.append((s, e))
            logger.info(f"Marked span [{s} : {e}] for interpolation")
        else:
            logger.debug("Ignored click (no drag range)")

        self._drag_start = None
        self._drag_rect  = None
        self._draw_all()

    def _try_remove_span(self, event):
        """Remove span if double-click falls inside it."""
        x = int(round(event.xdata))
        for i, (s, e) in enumerate(self.spans):
            if s <= x <= e:
                removed = self.spans.pop(i)
                logger.info(f"Removed span [{removed[0]} : {removed[1]}]")
                self._draw_all()
                return
        logger.debug("Double-click outside any marked span — nothing removed")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _do_reset(self):
        self.spans.clear()
        logger.info("Reset all marked spans")
        self._draw_all()

    def _do_apply(self):
        logger.info(
            f"Applying interpolation to {len(self.spans)} span(s) and closing"
        )
        self._plt.close(self.fig)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _build_result(self) -> pd.DataFrame:
        """Interpolate all marked spans and return the repaired DataFrame."""
        if not self.spans:
            logger.warning("No spans selected — returning original data unchanged")
            return self.df

        t_fixed = self._t.copy()

        for s, e in self.spans:
            left  = s - 1
            right = e + 1

            if left < 0 or right >= len(t_fixed):
                # Edge case: hold the nearest clean value
                if left < 0:
                    t_fixed[s:e+1] = t_fixed[right]
                else:
                    t_fixed[s:e+1] = t_fixed[left]
                logger.warning(
                    f"Span [{s}:{e}] is at file edge — held nearest value"
                )
                continue

            interp = np.linspace(t_fixed[left], t_fixed[right], (e - s + 1) + 2)[1:-1]
            t_fixed[s:e+1] = interp
            logger.info(
                f"Interpolated [{s}:{e}]  "
                f"({t_fixed[left]:.0f} → {t_fixed[right]:.0f})"
            )

        out = self.df.copy()
        out["PacketTime"] = t_fixed.astype(np.int64)
        return out

    # ------------------------------------------------------------------
    # Instructions
    # ------------------------------------------------------------------

    @staticmethod
    def _print_instructions():
        w = 64
        print("\n" + "=" * w)
        print("INTERACTIVE TIMESTAMP REPAIR")
        print("=" * w)
        print("The top two panels show the raw PacketTime signal and")
        print("its frame-to-frame diff.  Aberrant regions appear as")
        print("negative dips (red) or large spikes (orange).")
        print()
        print("MARKING SPANS")
        print("  Click + drag   in the top or middle panel to mark a")
        print("  span for linear interpolation (shown in gold).")
        print("  Double-click   inside a marked span to remove it.")
        print()
        print("FINISHING")
        print("  ✓ Apply & Continue   interpolate all marked spans,")
        print("                       then return to the pipeline.")
        print("  ↺ Reset              clear all marks.")
        print()
        print("  If you close the window without clicking Apply,")
        print("  the data is returned UNCHANGED (no interpolation).")
        print("=" * w + "\n")


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
class DataLoader:
    """Handles loading and preprocessing of handwriting data"""

    def __init__(self, config):
        self.config = config

    # ────────────────────────────────────────────────────────────────
    # RAW LOADING (NO CLEANING)
    # ────────────────────────────────────────────────────────────────

    def load_raw_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.config.input_file}")

        df = pd.read_excel(self.config.input_file)
        df.columns = [c.strip() for c in df.columns]

        keep_cols = [self.config.col_time, self.config.col_x,
                     self.config.col_y, self.config.col_pressure]

        try:
            df = df[[c for c in df.columns if any(k in c for k in keep_cols)]]
            df.columns = ["PacketTime", "X", "Y", "NormalPressure"]
        except KeyError as e:
            logger.error(f"Could not find required columns: {e}")
            logger.info(f"Available columns: {df.columns.tolist()}")
            raise

        return df

    # ────────────────────────────────────────────────────────────────
    # CLEANING (UNCHANGED LOGIC)
    # ────────────────────────────────────────────────────────────────

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)

        df = df.dropna(subset=["PacketTime", "X", "Y", "NormalPressure"])
        if len(df) < initial_count:
            logger.warning(
                f"Removed {initial_count - len(df)} rows with missing values"
            )

        negative_pressure = df["NormalPressure"] < 0
        if negative_pressure.any():
            logger.warning(
                f"Removed {negative_pressure.sum()} rows with negative pressure"
            )
            df = df[~negative_pressure]

        df = df.reset_index(drop=True)

        # Apply automatic fixes AFTER manual step
        df = self._fix_packet_time_overflow(df)
        df = self._fix_timestamp_outliers(df)

        return df

    def _fix_packet_time_overflow(self, df: pd.DataFrame) -> pd.DataFrame:
        t = df["PacketTime"].to_numpy(dtype=np.int64)
        diffs = np.diff(t, prepend=t[0])
        rollovers = np.where(diffs < 0)[0]

        if len(rollovers) == 0:
            return df

        logger.warning(f"Detected {len(rollovers)} overflow(s) — correcting")

        correction = np.zeros(len(t), dtype=np.int64)
        cumulative = 0

        for idx in rollovers:
            cumulative += _UINT32_RANGE
            correction[idx:] = cumulative

        df = df.copy()
        df["PacketTime"] = t + correction
        return df

    def _fix_timestamp_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        nominal = 1000.0 / self.config.sampling_rate
        threshold = nominal * 20
        max_interp_span = self.config.min_gap_ms * 0.8

        t = df["PacketTime"].to_numpy(dtype=np.float64)
        diffs = np.diff(t, prepend=t[0])

        suspicious = diffs > threshold
        outgoing = np.zeros(len(t), dtype=bool)
        outgoing[:-1] = suspicious[1:]

        bad = suspicious & outgoing

        if not bad.any():
            return df

        starts = np.where(np.diff(bad.astype(int), prepend=0) == 1)[0]
        ends = np.where(np.diff(bad.astype(int), append=0) == -1)[0]

        logger.warning(f"Auto-fixing {len(starts)} corrupted run(s)")

        t_fixed = t.copy()

        for start, end in zip(starts, ends):
            left = start - 1
            right = end + 1

            if left < 0 or right >= len(t):
                continue

            span_ms = t[right] - t[left]
            if span_ms > max_interp_span:
                continue

            interp = np.linspace(t[left], t[right], (end - start + 3))[1:-1]
            t_fixed[start:end+1] = interp

        df = df.copy()
        df["PacketTime"] = t_fixed.astype(np.int64)

        return df

    # ────────────────────────────────────────────────────────────────
    # VALIDATION (UNCHANGED)
    # ────────────────────────────────────────────────────────────────

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, list]:
        issues = []

        if len(df) == 0:
            issues.append("DataFrame is empty")

        if "PacketTime" in df.columns:
            if (df["PacketTime"].diff() < 0).any():
                issues.append("Time values are not monotonically increasing")

        return len(issues) == 0, issues

    def load_segmentation_sheet(self) -> Optional[pd.DataFrame]:
        try:
            seg = pd.read_excel(self.config.input_file,
                                sheet_name="Segmentation",
                                header=None)
            seg.columns = ["Type", "Start", "End", "WordIndex"]
            seg["WordIndex"] = seg["WordIndex"].ffill().astype(int)
            return seg
        except:
            return None
# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_and_validate(config) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    loader = DataLoader(config)

    # ────────────────────────────────────────────────
    # 1. LOAD RAW DATA
    # ────────────────────────────────────────────────
    df = loader.load_raw_data()

    # ────────────────────────────────────────────────
    # 2. DETECT ANOMALIES (RAW)
    # ────────────────────────────────────────────────
    diffs = df["PacketTime"].diff()

    has_non_monotonic = (diffs < 0).any()

    nominal = 1000.0 / config.sampling_rate
    large_jumps = (diffs > nominal * 20).sum()

    if has_non_monotonic or large_jumps > 0:
        logger.warning(
            f"Raw anomalies detected: "
            f"{int(has_non_monotonic)} non-monotonic, "
            f"{large_jumps} large jumps"
        )

        print(
            "\n⚠ Raw timestamp anomalies detected.\n"
            "Launching interactive repair BEFORE cleaning.\n"
        )

        repairer = InteractiveTimestampRepair(df)
        df = repairer.run()

        print(f"✔ Manual spans selected: {len(repairer.spans)}")

    else:
        logger.info("No raw timestamp anomalies detected")

    # ────────────────────────────────────────────────
    # 3. AUTOMATIC CLEANING (ALWAYS)
    # ────────────────────────────────────────────────
    df = loader._clean_data(df)

    # ────────────────────────────────────────────────
    # 4. FINAL VALIDATION
    # ────────────────────────────────────────────────
    is_valid, issues = loader.validate_data(df)

    if not is_valid:
        raise ValueError(f"Data validation failed after cleaning: {issues}")

    logger.info("Data successfully cleaned and validated")

    # ────────────────────────────────────────────────
    # 5. LOAD SEGMENTATION
    # ────────────────────────────────────────────────
    seg = loader.load_segmentation_sheet()

    return df, seg