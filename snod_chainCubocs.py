from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import to_hex, to_rgb

# --------------------------------------------------------------------------- #
# Configuration dataclasses                                                   #
# --------------------------------------------------------------------------- #

Obstacle = Tuple[float, float, float]  # (start, end, height)
ModuleConnection = Tuple[int, int, float]


class SensingMode(str, Enum):
    LOCAL = "local"
    EXPENSIVE_NO_COMM = "expensive_no_comm"


@dataclass
class GeometryConfig:
    M: int
    cpm: int = 6
    module_width: float = 0.5
    gap: float = 0.1
    goal_x: float = 20.0
    velocity: float = 0.3
    H0: float = 0.5
    obs_regions: Sequence[Obstacle] = field(default_factory=list)

    @property
    def N(self) -> int:
        return self.M * self.cpm


@dataclass
class DynamicsConfig:
    tau_z: float = 0.05
    tau_us: float = 0.1
    d: float = 0.5
    u0: float = 0.1
    Ku: float = 1.0
    Kus: float = 3.0
    a_intra: float = 0.5
    a_inter: float = 1.0
    s_gain: float = 0.3
    extra_module_connections: Optional[List[ModuleConnection]] = None


@dataclass
class SensingConfig:
    mode: SensingMode = SensingMode.LOCAL  # EXPENSIVE_NO_COMM = "camera/lookahead" mode
    sensor_lookahead: float = 2.0          # only used in EXPENSIVE_NO_COMM mode
    contact_margin: float = 0.2
    clear_margin: float = 0.01
    safety_clear: float = 0.05
    K_b_obst: float = 3.0
    K_b_recover: float = 0.5


@dataclass
class TiltConfig:
    alpha_tilt: float = 0.6
    g_tilt_weight: float = 1.0
    g_tilt_wrap: bool = False
    tilt_norm: str = "relative"  # 'relative' or 'absolute'
    normalize_A: bool = False
    normalize_G: bool = True
    u_min: Optional[float] = None
    u_max: Optional[float] = None
    print_diagnostics: bool = False


@dataclass
class RunSettings:
    dt: float = 0.0005
    duration: float = 60.0
    fast_substeps: int = 10
    z_spike_thr: float = 0.25
    # optional animation knobs (used by run_and_visualize)
    animate: bool = False
    animate_show_every: int = 20
    animate_render_every: Optional[int] = None
    animate_save_path: Optional[str] = None
    animate_fps: int = 30


@dataclass
class SimulationConfig:
    geometry: GeometryConfig
    dynamics: DynamicsConfig
    sensing: SensingConfig = field(default_factory=SensingConfig)
    tilt: TiltConfig = field(default_factory=TiltConfig)
    run: RunSettings = field(default_factory=RunSettings)


@dataclass
class SimulationResult:
    time: np.ndarray
    z: np.ndarray
    us: np.ndarray
    s: np.ndarray
    bias: np.ndarray
    h_mod: np.ndarray
    x_back: np.ndarray
    x_mod_b: np.ndarray
    x_mod_f: np.ndarray
    A: np.ndarray
    G_tilt: np.ndarray


# --------------------------------------------------------------------------- #
# Connectivity helpers                                                       #
# --------------------------------------------------------------------------- #

def build_adjacency(M: int, cpm: int, a_intra: float, a_inter: float,
                    extra_connections: Optional[Iterable[ModuleConnection]] = None) -> np.ndarray:
    """
    Chamber-level adjacency (N x N):
      - Intra-module fully connected (off-diagonal) with weight a_intra.
      - Inter-module connects same-index chambers between module pairs.
    """
    N = M * cpm
    A = np.zeros((N, N))

    B = np.full((cpm, cpm), a_intra)
    np.fill_diagonal(B, 0.2)
    for m in range(M):
        sl = slice(m * cpm, (m + 1) * cpm)
        A[sl, sl] = B

    pairs: List[ModuleConnection] = []
    pairs.extend((m, m + 1, a_inter) for m in range(M - 1))
    if extra_connections:
        for tup in extra_connections:
            if len(tup) == 3:
                i, j, w = tup
            else:
                i, j = tup[:2]
                w = a_inter
            if 0 <= i < M and 0 <= j < M and i != j:
                pairs.append((i, j, w))

    for i_mod, j_mod, w_ij in pairs:
        for k in range(cpm):
            i = i_mod * cpm + k
            j = j_mod * cpm + k
            A[i, j] = w_ij
            A[j, i] = w_ij
    return A


def row_normalize(MxM: np.ndarray) -> np.ndarray:
    """Row-normalize; rows that sum to zero are kept at zero."""
    MxM = MxM.copy()
    rs = MxM.sum(axis=1, keepdims=True)
    mask = rs[:, 0] != 0.0
    MxM[mask] = MxM[mask] / rs[mask]
    return MxM


def build_module_chain_graph(M: int, weight: float = 1.0, wrap: bool = False) -> np.ndarray:
    """Simple module-level chain graph for tilt coupling, independent of comms."""
    G = np.zeros((M, M))
    for i in range(M - 1):
        G[i, i + 1] = G[i + 1, i] = weight
    if wrap and M > 2:
        G[0, M - 1] = G[M - 1, 0] = weight
    return G


# --------------------------------------------------------------------------- #
# Sensing helpers                                                            #
# --------------------------------------------------------------------------- #

def in_contact_or_vicinity(seg_back: float, seg_front: float,
                           obs_regions: Sequence[Obstacle], margin: float) -> Tuple[bool, Optional[float]]:
    for start, end, H_obs in obs_regions:
        if seg_front >= (start - margin) and seg_back <= (end + margin):
            return True, H_obs
    return False, None


def detect_overhang_with_lookahead(front: float, lookahead: float,
                                   obs_regions: Sequence[Obstacle], margin: float) -> Tuple[bool, Optional[float]]:
    ahead_limit = front + lookahead
    for start, end, H_obs in obs_regions:
        if start - margin <= ahead_limit <= end + lookahead + margin:
            if front <= end + lookahead + margin:
                return True, H_obs
    return False, None


def compute_module_bias(geometry: GeometryConfig, sensing: SensingConfig,
                        h_mod_prev: np.ndarray, backs: np.ndarray, fronts: np.ndarray,
                        mode: SensingMode) -> np.ndarray:
    """Module-level evidence term based on contact or lookahead sensing."""
    M = geometry.M
    H0 = geometry.H0
    b_mod = np.zeros(M)
    for m in range(M):
        in_vic, H_loc = in_contact_or_vicinity(
            backs[m], fronts[m], geometry.obs_regions, sensing.contact_margin
        )
        if mode == SensingMode.EXPENSIVE_NO_COMM:
            saw_ahead, H_cam = detect_overhang_with_lookahead(
                fronts[m], sensing.sensor_lookahead, geometry.obs_regions, sensing.contact_margin
            )
            if saw_ahead:
                H_loc = H_cam
                in_vic = True

        if in_vic and H_loc is not None:
            target = max(0.0, H_loc - sensing.safety_clear)
            if h_mod_prev[m] > target + sensing.clear_margin:
                delta = h_mod_prev[m] - target
                b_mod[m] = -sensing.K_b_obst * delta
        else:
            if h_mod_prev[m] < H0 - sensing.clear_margin:
                delta = H0 - h_mod_prev[m]
                b_mod[m] = sensing.K_b_recover * delta
    return b_mod


# --------------------------------------------------------------------------- #
# Core simulation                                                            #
# --------------------------------------------------------------------------- #

def simulate_snod(config: SimulationConfig) -> SimulationResult:
    """
    Run the S-NOD chain simulation with clear separation of sensing, tilt, and comms.
    Returns a SimulationResult containing trajectories and connectivity matrices.
    """
    geom, dyn, sensing, tilt, run = (
        config.geometry, config.dynamics, config.sensing, config.tilt, config.run
    )
    dt = run.dt
    time = np.arange(0.0, run.duration, dt)
    Tn = len(time)

    drop_inter = sensing.mode == SensingMode.EXPENSIVE_NO_COMM
    A = build_adjacency(
        geom.M, geom.cpm, dyn.a_intra,
        (0.0 if drop_inter else dyn.a_inter),
        (None if drop_inter else dyn.extra_module_connections)
    )
    if tilt.normalize_A:
        A = row_normalize(A)

    G_tilt = build_module_chain_graph(geom.M, weight=tilt.g_tilt_weight, wrap=tilt.g_tilt_wrap)
    if tilt.normalize_G:
        G_tilt = row_normalize(G_tilt)

    fast_substeps = max(1, int(run.fast_substeps))
    dt_fast = dt / fast_substeps

    z = np.zeros((geom.N, Tn))
    us = np.zeros_like(z)
    s = np.ones_like(z)
    bias = np.zeros_like(z)
    h_mod = np.zeros((geom.M, Tn))
    x_back = np.zeros(Tn)
    x_mod_b = np.zeros((geom.M, Tn))
    x_mod_f = np.zeros((geom.M, Tn))

    def module_footprint(x_back_scalar: float) -> Tuple[np.ndarray, np.ndarray]:
        backs = np.array([x_back_scalar + m * (geom.module_width + geom.gap) for m in range(geom.M)])
        fronts = backs + geom.module_width
        return backs, fronts

    for m in range(geom.M):
        sl = slice(m * geom.cpm, (m + 1) * geom.cpm)
        h_mod[m, 0] = geom.H0 * np.mean(s[sl, 0])

    if tilt.print_diagnostics:
        try:
            lam_A = float(np.max(np.abs(np.linalg.eigvals(A))))
        except Exception:
            lam_A = float(np.linalg.norm(A, ord=2))
        try:
            lam_G = float(np.max(np.abs(np.linalg.eigvals(G_tilt))))
        except Exception:
            lam_G = float(np.linalg.norm(G_tilt, ord=2))
        print(f"[diag] λ_max(A)≈{lam_A:.3f}, λ_max(G_tilt)≈{lam_G:.3f}, alpha_tilt={tilt.alpha_tilt}, mode={sensing.mode}")

    for i in range(1, Tn):
        backs, fronts = module_footprint(x_back[i - 1])
        x_mod_b[:, i] = backs
        x_mod_f[:, i] = fronts

        for m in range(geom.M):
            sl = slice(m * geom.cpm, (m + 1) * geom.cpm)
            h_mod[m, i] = geom.H0 * np.mean(s[sl, i - 1])

        blocked = False
        eps_h = 1e-3
        for m in range(geom.M):
            in_vic, H_loc = in_contact_or_vicinity(backs[m], fronts[m], geom.obs_regions, sensing.contact_margin)
            if in_vic and H_loc is not None and (h_mod[m, i] > H_loc - eps_h):
                blocked = True
                break
        x_back[i] = x_back[i - 1] if blocked else min(x_back[i - 1] + geom.velocity * dt, geom.goal_x)

        b_mod = compute_module_bias(geom, sensing, h_mod[:, i], backs, fronts, sensing.mode)
        for m in range(geom.M):
            sl = slice(m * geom.cpm, (m + 1) * geom.cpm)
            bias[sl, i] = b_mod[m]

        if tilt.tilt_norm == "relative":
            h_nd = (h_mod[:, i] - geom.H0) / max(geom.H0, 1e-9)
        else:
            h_nd = h_mod[:, i].copy()
        deg = G_tilt.sum(axis=1)
        tilt_mod = (G_tilt @ h_nd) - deg * h_nd
        tilt_ch = np.repeat(tilt_mod, geom.cpm)

        z_fast = z[:, i - 1].copy()
        us_fast = us[:, i - 1].copy()
        b_ch = np.repeat(b_mod, geom.cpm)

        for _ in range(fast_substeps):
            ni = A @ z_fast
            u_i = dyn.u0 + dyn.Ku * z_fast**2 - us_fast
            drive = u_i * ni + b_ch + tilt.alpha_tilt * tilt_ch
            z_dot = (-dyn.d * z_fast + np.tanh(drive)) / dyn.tau_z
            us_dot = (-us_fast + dyn.Kus * z_fast**4) / dyn.tau_us
            z_fast = z_fast + dt_fast * z_dot
            us_fast = us_fast + dt_fast * us_dot

        z[:, i] = z_fast
        us[:, i] = us_fast

        active = (np.abs(z[:, i]) > run.z_spike_thr).astype(float)
        delta_s = dyn.s_gain * np.tanh(z[:, i]) * dt * active
        s[:, i] = np.clip(s[:, i - 1] + delta_s, 0.0, 1.0)

        for m in range(geom.M):
            sl = slice(m * geom.cpm, (m + 1) * geom.cpm)
            h_mod[m, i] = geom.H0 * np.mean(s[sl, i])

    return SimulationResult(
        time=time, z=z, us=us, s=s, bias=bias,
        h_mod=h_mod, x_back=x_back, x_mod_b=x_mod_b, x_mod_f=x_mod_f,
        A=A, G_tilt=G_tilt
    )


# --------------------------------------------------------------------------- #
# Analysis helpers                                                           #
# --------------------------------------------------------------------------- #

def compute_clear_time_from_results(results: SimulationResult, geom: GeometryConfig,
                                    margin: Optional[float] = None) -> List[Dict[str, float]]:
    """
    Returns list of dicts with t_start, t_end, and duration per obstacle.
    t_start: first time any module front enters obstacle start - margin
    t_end: first time all module backs pass obstacle end + margin
    """
    time = results.time
    x_fronts = results.x_mod_f
    M, Tn = x_fronts.shape
    w = geom.module_width
    obs_regions = geom.obs_regions
    margin_val = geom.gap if margin is None else margin
    out = []
    for (start, end, _) in obs_regions:
        t_start = None
        t_end = None
        for k in range(Tn):
            fronts = x_fronts[:, k]
            backs = fronts - w
            if t_start is None and np.any(fronts >= start - margin_val):
                t_start = time[k]
            if t_start is not None and np.all(backs > end + margin_val):
                t_end = time[k]
                break
        out.append({
            "start": start, "end": end,
            "t_start": t_start, "t_end": t_end,
            "duration": None if (t_start is None or t_end is None) else (t_end - t_start)
        })
    return out


def measure_clear_time(config: SimulationConfig) -> Optional[float]:
    """Convenience wrapper that runs a simulation and returns the first obstacle clear time."""
    res = simulate_snod(config)
    stats = compute_clear_time_from_results(res, config.geometry)
    if not stats or stats[0]["duration"] is None:
        return None
    return stats[0]["duration"]


# --------------------------------------------------------------------------- #
# Convenience runner (simulate + optional animation)                         #
# --------------------------------------------------------------------------- #

def run_and_visualize(config: SimulationConfig, snapshot_times: Optional[Sequence[float]] = None,
                      overlay_only: bool = True, animation_kwargs: Optional[Dict] = None):
    """
    Run a simulation, optionally animate, and optionally show overlay snapshots.
    Uses RunSettings.animate* fields plus any overrides in animation_kwargs.
    """
    res = simulate_snod(config)

    run = config.run
    anim_kwargs = animation_kwargs or {}
    if run.animate:
        animate_modules(
            res, config.geometry, run.dt,
            show_every=run.animate_show_every,
            render_every=run.animate_render_every,
            save_path=run.animate_save_path,
            fps=run.animate_fps,
            **anim_kwargs
        )

    if snapshot_times is not None:
        plot_opinion_with_overlay_snapshots(
            res, config.geometry, snapshot_times,
            title_top="",
            fontsize=16, ticklabelsize=14,
            wspace=0.01, hspace=0.1, w_pad=0.01, h_pad=0.1,
            auto_fit_pad=0.8,
        )
        plt.show()

    return res


# --------------------------------------------------------------------------- #
# Visualization                                                              #
# --------------------------------------------------------------------------- #

def module_colors(M: int) -> List[str]:
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    if M <= len(base):
        return base[:M]
    reps = int(np.ceil(M / len(base)))
    return (base * reps)[:M]


def animate_modules(results: SimulationResult, geom: GeometryConfig, dt: float,
                    show_every: int = 20, save_path: Optional[str] = None,
                    fps: int = 30, blit: bool = True, render_every: Optional[int] = None,
                    dpi: int = 120, codec: str = "libx264", crf: int = 23,
                    preset: str = "veryfast", downsample_ts: int = 1,
                    z_title: Optional[str] = None):
    """Optimized animated viewer for the chain; same visual as the original code."""
    time = results.time
    z = results.z
    bias = results.bias
    h_mod = results.h_mod
    x_b = results.x_mod_b

    M = geom.M
    cpm = geom.cpm
    H0 = geom.H0
    w = geom.module_width
    goal_x = geom.goal_x
    obs_regions = geom.obs_regions

    def mean_per_module(arrN_T: np.ndarray) -> np.ndarray:
        out = np.zeros((M, arrN_T.shape[1]))
        for m in range(M):
            sl = slice(m * cpm, (m + 1) * cpm)
            out[m] = np.mean(arrN_T[sl, :], axis=0)
        return out

    ts_stride = max(1, int(downsample_ts))
    t_plot = time[::ts_stride]
    z_m = mean_per_module(z)[:, ::ts_stride]
    b_m = mean_per_module(bias)[:, ::ts_stride]

    def idx_ts(i: int) -> int:
        return i // ts_stride

    cols = module_colors(M)

    fig, (ax_z, ax_b, ax_r) = plt.subplots(
        3, 1, figsize=(11, 8), dpi=dpi,
        gridspec_kw={"height_ratios": [2, 1.1, 2]}
    )

    lines_z = []
    zmax = max(1e-6, float(np.max(np.abs(z_m))))
    for m in range(M):
        ln, = ax_z.plot([], [], lw=1.4, color=cols[m], label=f"m{m+1}")
        lines_z.append(ln)
    ax_z.set_ylabel("mean opinion (z)")
    ax_z.set_ylim(-zmax, zmax)
    ax_z.set_xlim(t_plot[0], t_plot[-1])
    ax_z.legend(ncol=min(5, M), fontsize="small")
    ax_z.grid(True, alpha=0.3)
    tm_z = ax_z.axvline(t_plot[0], color="k", lw=0.8)
    if z_title is not None:
        ax_z.set_title(z_title)

    lines_b = []
    bmax = max(1e-6, float(np.max(np.abs(b_m))))
    for m in range(M):
        ln, = ax_b.plot([], [], lw=1.4, color=cols[m], label=f"m{m+1}")
        lines_b.append(ln)
    ax_b.set_ylabel("evidence (b)")
    ax_b.set_ylim(-bmax, bmax)
    ax_b.set_xlim(t_plot[0], t_plot[-1])
    ax_b.legend(ncol=min(5, M), fontsize="small")
    ax_b.grid(True, alpha=0.3)
    tm_b = ax_b.axvline(t_plot[0], color="k", lw=0.8)

    ax_r.set_xlim(0, goal_x + 1.0)
    ax_r.set_ylim(0, H0 * 1.35)
    ax_r.set_xlabel("x")
    ax_r.set_ylabel("height")
    for start, end, H_obs in obs_regions:
        ax_r.plot([start, end], [H_obs, H_obs], lw=4, color="gray", zorder=1)
    ax_r.axhline(H0, linestyle="--", linewidth=1, color="black", zorder=1)

    bars = [plt.Rectangle((x_b[m, 0], 0.0), w, h_mod[m, 0],
                          facecolor=cols[m], alpha=0.65, ec="k", lw=0.6, zorder=2)
            for m in range(M)]
    for rect in bars:
        if blit:
            rect.set_animated(True)
        ax_r.add_patch(rect)

    artists = lines_z + lines_b + [tm_z, tm_b] + bars

    plt.tight_layout()
    if render_every is None:
        render_every = show_every
    frame_indices = list(range(0, len(time), render_every))
    if frame_indices[-1] != len(time) - 1:
        frame_indices.append(len(time) - 1)

    def init():
        for m in range(M):
            lines_z[m].set_data([], [])
            lines_b[m].set_data([], [])
        tm_z.set_xdata((t_plot[0], t_plot[0]))
        tm_b.set_xdata((t_plot[0], t_plot[0]))
        for m in range(M):
            bars[m].set_xy((x_b[m, 0], 0.0))
            bars[m].set_width(w)
            bars[m].set_height(h_mod[m, 0])
        return artists

    def update(i):
        j = min(idx_ts(i), len(t_plot) - 1)
        t = t_plot[j]
        for m in range(M):
            lines_z[m].set_data(t_plot[:j + 1], z_m[m, :j + 1])
            lines_b[m].set_data(t_plot[:j + 1], b_m[m, :j + 1])
        tm_z.set_xdata((t, t))
        tm_b.set_xdata((t, t))
        for m in range(M):
            bars[m].set_xy((x_b[m, i], 0.0))
            bars[m].set_height(h_mod[m, i])
            bars[m].set_width(w)
        return artists

    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=frame_indices,
        interval=dt * 1000 * render_every,
        blit=blit, save_count=len(frame_indices), cache_frame_data=False
    )

    if save_path is not None:
        ext = save_path.lower().rsplit(".", 1)[-1]
        if ext == "mp4":
            Writer = animation.FFMpegWriter
            writer = Writer(
                fps=fps,
                codec=codec,
                bitrate=-1,
                extra_args=["-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
            )
            ani.save(save_path, writer=writer, dpi=dpi)
        elif ext == "gif":
            ani.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            raise ValueError("Unsupported extension. Use .mp4 or .gif")
        print(f"Saved animation to {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return ani


# --- Publication-style snapshots ----------------------------------------- #

def snap_palette_dark(max_panels: int, mod_cols: List[str]) -> List[Tuple[float, float, float]]:
    dark_candidates = list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    drop_hex = {"#aec7e8", "#c7c7c7", "#9edae5", "#c49c94"}
    dark_candidates = [c for c in dark_candidates if to_hex(c) not in drop_hex]
    mod_rgb = np.array([to_rgb(c) for c in mod_cols])
    cand_rgb = np.array([to_rgb(c) for c in dark_candidates])

    def is_distinct(c):
        return np.all(np.linalg.norm(mod_rgb - c, axis=1) > 0.25)

    filtered = [tuple(c) for c in cand_rgb if is_distinct(c)]
    reps = int(np.ceil(max_panels / len(filtered)))
    return (filtered * reps)[:max_panels]


def plot_opinion_with_overlay_snapshots(
    res: SimulationResult, geom: GeometryConfig, snapshot_times: Sequence[float],
    title_top: str = "S-NOD for Obstacle Clearing",
    fontsize: int = 14, ticklabelsize: Optional[int] = None, title_pad: int = 6, label_pad: int = 2,
    wspace: float = 0.0, hspace: float = 0.08, w_pad: float = 0.0, h_pad: float = 0.02,
    auto_fit_pad: float = 0.6, y_max: Optional[float] = None, max_panels: int = 6,
    alpha_latest: float = 0.95, alpha_start: float = 0.55, alpha_decay: float = 0.78,
    past_edge_kw: Dict[str, float] = None, latest_edge_kw: Dict[str, float] = None
):
    """Compact two-row plot: mean opinion vs time + overlayed x–h snapshots."""
    past_edge_kw = past_edge_kw or {"ls": "--", "lw": 0.9, "color": "k"}
    latest_edge_kw = latest_edge_kw or {"ls": "-", "lw": 1.0, "color": "k"}

    time = res.time
    Z = res.z
    Hm = res.h_mod
    Xb = res.x_mod_b
    Xf = res.x_mod_f

    M = geom.M
    cpm = geom.cpm
    H0 = geom.H0
    w = geom.module_width
    obs = geom.obs_regions
    goal = geom.goal_x

    if ticklabelsize is None:
        ticklabelsize = max(8, fontsize - 2)

    mod_cols = module_colors(M)
    snap_cols = snap_palette_dark(max_panels, mod_cols)

    Zm = np.zeros((M, Z.shape[1]))
    for m in range(M):
        sl = slice(m * cpm, (m + 1) * cpm)
        Zm[m] = np.mean(Z[sl, :], axis=0)

    snap_times = list(snapshot_times)[:max_panels]
    snap_idx = [int(np.clip(np.searchsorted(time, t, side="left"), 0, len(time) - 1))
                for t in snap_times]

    def chain_min(i: int) -> float:
        return float(np.min(Xb[:, i])) if Xb.size else 0.0

    def chain_max(i: int) -> float:
        return float(np.max(Xf[:, i])) if Xf.size else goal

    if len(snap_idx) > 0:
        xmin_all = min(chain_min(i) for i in snap_idx) - auto_fit_pad
        xmax_all = max(chain_max(i) for i in snap_idx) + auto_fit_pad
    else:
        xmin_all, xmax_all = 0.0, min(goal + 1.0, 10.0)
    xlo_all, xhi_all = max(0.0, xmin_all), min(goal + 1.0, xmax_all)
    if xhi_all - xlo_all < 1.0:
        pad = 0.5 * (1.0 - (xhi_all - xlo_all))
        xlo_all = max(0.0, xlo_all - pad)
        xhi_all = min(goal + 1.0, xhi_all + pad)

    if y_max is None:
        if len(snap_idx) and Hm.size:
            imax = float(np.max([np.max(Hm[:, i]) for i in snap_idx]))
            ytop = max(H0 * 1.35, imax * 1.10)
        else:
            ytop = max(H0 * 1.35, float(np.max(Hm)) * 1.10 if Hm.size else H0)
    else:
        ytop = y_max

    fig = plt.figure(figsize=(12, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], wspace=wspace)
    fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad, hspace=hspace, wspace=wspace)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])

    zlim = max(1e-6, float(np.max(np.abs(Zm))))
    for m in range(M):
        ax_top.plot(time, Zm[m], lw=1.4, color=mod_cols[m], label=f"m{m+1}")
    ax_top.set_xlim(time[0], time[-1])
    ax_top.set_ylim(-zlim, zlim)
    ax_top.set_ylabel("opinion (z)", fontsize=fontsize, labelpad=label_pad)
    ax_top.set_xlabel("time (s)", fontsize=fontsize, labelpad=label_pad)
    ax_top.set_title(title_top, fontsize=fontsize + 1, pad=title_pad)
    ax_top.tick_params(labelsize=ticklabelsize)

    ax_bot.set_xlim(xlo_all, xhi_all)
    ax_bot.set_ylim(0, ytop)
    ax_bot.set_ylabel("height", fontsize=fontsize, labelpad=label_pad)
    ax_bot.set_xlabel("x-position", fontsize=fontsize, labelpad=label_pad)
    ax_bot.tick_params(labelsize=ticklabelsize)
    for (start, end, H_obs) in obs:
        ax_bot.plot([start, end], [H_obs, H_obs], lw=3.0, color="gray")

    nS = len(snap_idx)
    for k, (t, i) in enumerate(zip(snap_times, snap_idx)):
        is_latest = (k == nS - 1)
        if is_latest:
            face_alpha = alpha_latest
            edge_kw = latest_edge_kw
            zorder = 5
        else:
            age = nS - 1 - k
            face_alpha = alpha_start * (alpha_decay ** age)
            edge_kw = past_edge_kw
            zorder = 3
        for m in range(M):
            rect = plt.Rectangle(
                (Xb[m, i], 0.0), w, Hm[m, i],
                facecolor=mod_cols[m], alpha=face_alpha,
                ec=edge_kw.get("color", "k"),
                lw=edge_kw.get("lw", 0.9),
                ls=edge_kw.get("ls", "--"),
                zorder=zorder
            )
            ax_bot.add_patch(rect)

    return fig, {"top": ax_top, "overlay": ax_bot}


# --------------------------------------------------------------------------- #
# Quick example                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    geom = GeometryConfig(
        M=4, cpm=6, module_width=0.5, gap=0.1, goal_x=15.0, velocity=0.3,
        H0=0.5, obs_regions=[(3.0, 5.0, 0.35),(8.0, 9.5, 0.4)]
    )
    dyn = DynamicsConfig(a_intra=0.5, a_inter=1.0)

    # Toggle between local contact sensing and camera/lookahead mode here:
    #   SensingMode.LOCAL -> cheap contact sensing + comms
    #   SensingMode.EXPENSIVE_NO_COMM -> camera/lookahead, no inter-module comms
    sensing = SensingConfig(mode=SensingMode.LOCAL, sensor_lookahead=1.5)

    tilt = TiltConfig(alpha_tilt=3.0, normalize_G=True, tilt_norm="absolute")
    run = RunSettings(
        dt=0.0005, duration=60.0, fast_substeps=1, z_spike_thr=0.25,
        animate=True, animate_show_every=40, animate_save_path=None
    )
    cfg = SimulationConfig(geometry=geom, dynamics=dyn, sensing=sensing, tilt=tilt, run=run)

    # snapshots = []
    run_and_visualize(cfg, snapshot_times=None, overlay_only=False)
