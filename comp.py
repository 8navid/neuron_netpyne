"""
Computational Neuroscience Lab
Simple Excitatory-Inhibitory Network Demo with NetPyNE + NEURON

Goal:
- Show how a small E/I network can generate collective activity
- Save a plain English text report
- Save plots for presentation
- Optionally save a simple raster animation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import signal
from pathlib import Path

from netpyne import sim, specs
from neuron import h, gui  # NEURON Python interface

# -----------------------------------------------------------------------------
# 1) Simulation settings
# -----------------------------------------------------------------------------
simConfig = specs.SimConfig()
simConfig.duration = 1000          # ms
simConfig.dt = 0.025               # ms
simConfig.recordStep = 0.1         # ms
simConfig.verbose = False
simConfig.filename = "ei_network_demo"

# -----------------------------------------------------------------------------
# 2) Network definition
# -----------------------------------------------------------------------------
netParams = specs.NetParams()
netParams.sizeX = 100
netParams.sizeY = 100
netParams.sizeZ = 50

# Populations
N_E = 40
N_I = 10

netParams.popParams["E"] = {
    "numCells": N_E,
    "cellModel": "HH",
    "cellType": "Excitatory",
    "yRange": [0, 50],
}

netParams.popParams["I"] = {
    "numCells": N_I,
    "cellModel": "HH",
    "cellType": "Inhibitory",
    "yRange": [0, 50],
}

# Simple Hodgkin-Huxley cell
netParams.cellParams["HH_rule"] = {
    "conds": {"cellModel": "HH"},
    "secs": {
        "soma": {
            "geom": {"diam": 18.8, "L": 18.8},
            "mechs": {
                "hh": {
                    "gnabar": 0.12,
                    "gkbar": 0.036,
                    "gl": 0.0003,
                    "el": -65
                }
            }
        }
    }
}

# Synapses
netParams.synMechParams["exc"] = {
    "mod": "Exp2Syn",
    "tau1": 0.5,
    "tau2": 5.0,
    "e": 0
}

netParams.synMechParams["inh"] = {
    "mod": "Exp2Syn",
    "tau1": 1.0,
    "tau2": 10.0,
    "e": -80
}

# Connections
netParams.connParams["E_to_I"] = {
    "preConds": {"pop": "E"},
    "postConds": {"pop": "I"},
    "probability": 0.30,
    "weight": 0.005,
    "delay": 2,
    "synMech": "exc",
}

netParams.connParams["I_to_E"] = {
    "preConds": {"pop": "I"},
    "postConds": {"pop": "E"},
    "probability": 0.50,
    "weight": 0.010,
    "delay": 2,
    "synMech": "inh",
}

netParams.connParams["E_to_E"] = {
    "preConds": {"pop": "E"},
    "postConds": {"pop": "E"},
    "probability": 0.10,
    "weight": 0.002,
    "delay": 2,
    "synMech": "exc",
}

# External background current
netParams.stimSourceParams["background"] = {
    "type": "IClamp",
    "dur": 800,
    "amp": 0.15,     # nA
    "delay": 100
}

netParams.stimTargetParams["bg_all"] = {
    "source": "background",
    "conds": {"pop": ["E", "I"]},
    "sec": "soma",
    "loc": 0.5
}

# -----------------------------------------------------------------------------
# 3) Run simulation
# -----------------------------------------------------------------------------
print("=" * 60)
print("Simple Excitatory-Inhibitory Network Demo")
print("=" * 60)
print("Building the network...")
sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)
print("Simulation finished.")

# -----------------------------------------------------------------------------
# 4) Extract spike data
# -----------------------------------------------------------------------------
spike_times = np.array(sim.allSimData.get("spkt", []), dtype=float)
spike_ids = np.array(sim.allSimData.get("spkid", []), dtype=int)

# Get GIDs by population
e_gids = [cell.gid for cell in sim.net.allCells if cell.tags.get("pop") == "E"]
i_gids = [cell.gid for cell in sim.net.allCells if cell.tags.get("pop") == "I"]

e_mask = np.isin(spike_ids, e_gids)
i_mask = np.isin(spike_ids, i_gids)

total_spikes = len(spike_times)
e_spikes = int(np.sum(e_mask))
i_spikes = int(np.sum(i_mask))
duration_sec = simConfig.duration / 1000.0

e_rate = e_spikes / (len(e_gids) * duration_sec) if len(e_gids) > 0 else 0
i_rate = i_spikes / (len(i_gids) * duration_sec) if len(i_gids) > 0 else 0
avg_rate = total_spikes / ((len(e_gids) + len(i_gids)) * duration_sec)

# -----------------------------------------------------------------------------
# 5) Population activity and spectrum
# -----------------------------------------------------------------------------
bin_size = 5  # ms
bins = np.arange(0, simConfig.duration + bin_size, bin_size)
time_axis = (bins[:-1] + bins[1:]) / 2

e_hist, _ = np.histogram(spike_times[e_mask], bins=bins)
i_hist, _ = np.histogram(spike_times[i_mask], bins=bins)

# Convert to firing rate per bin (Hz)
e_rate_trace = e_hist / (len(e_gids) * (bin_size / 1000.0)) if len(e_gids) > 0 else np.zeros_like(e_hist)
i_rate_trace = i_hist / (len(i_gids) * (bin_size / 1000.0)) if len(i_gids) > 0 else np.zeros_like(i_hist)

net_activity = e_rate_trace - i_rate_trace

if len(net_activity) > 8:
    fs = 1000 / bin_size
    freqs, psd = signal.welch(net_activity, fs=fs, nperseg=min(128, len(net_activity)))
else:
    freqs = np.array([])
    psd = np.array([])

peak_freq = 0.0
peak_power = 0.0
if len(freqs) > 0:
    freq_range = (freqs >= 5) & (freqs <= 40)
    if np.any(freq_range):
        idx = np.argmax(psd[freq_range])
        peak_freq = float(freqs[freq_range][idx])
        peak_power = float(psd[freq_range][idx])

isi_all = np.diff(np.sort(spike_times))
cv_isi = float(np.std(isi_all) / np.mean(isi_all)) if len(isi_all) > 1 and np.mean(isi_all) > 0 else 0.0

# -----------------------------------------------------------------------------
# 6) Write plain-English report
# -----------------------------------------------------------------------------
report_lines = []
report_lines.append("=" * 60)
report_lines.append("SIMULATION REPORT")
report_lines.append("=" * 60)
report_lines.append("Model: Simple Excitatory-Inhibitory Network")
report_lines.append("Purpose: Demonstrate how E and I interaction can generate collective activity")
report_lines.append("")
report_lines.append("Simulation settings")
report_lines.append(f"  Duration: {simConfig.duration:.1f} ms")
report_lines.append(f"  Time step: {simConfig.dt:.4f} ms")
report_lines.append(f"  Bin size for population analysis: {bin_size} ms")
report_lines.append("")
report_lines.append("Network size")
report_lines.append(f"  Excitatory cells (E): {len(e_gids)}")
report_lines.append(f"  Inhibitory cells (I): {len(i_gids)}")
report_lines.append(f"  Total cells: {len(e_gids) + len(i_gids)}")
report_lines.append("")
report_lines.append("Spike summary")
report_lines.append(f"  Total spikes: {total_spikes}")
report_lines.append(f"  E spikes: {e_spikes}")
report_lines.append(f"  I spikes: {i_spikes}")
report_lines.append(f"  Mean firing rate: {avg_rate:.2f} Hz")
report_lines.append(f"  E population firing rate: {e_rate:.2f} Hz")
report_lines.append(f"  I population firing rate: {i_rate:.2f} Hz")
report_lines.append(f"  ISI coefficient of variation: {cv_isi:.3f}")
report_lines.append("")
report_lines.append("Rhythm analysis")
if peak_freq > 0:
    report_lines.append(f"  Dominant frequency: {peak_freq:.2f} Hz")
    report_lines.append(f"  Dominant power: {peak_power:.4f}")
else:
    report_lines.append("  Dominant frequency: not detected")
    report_lines.append("  Dominant power: not detected")
report_lines.append("")
if 8 <= peak_freq <= 30:
    report_lines.append("Interpretation: The network shows a collective rhythm in the alpha/beta range.")
else:
    report_lines.append("Interpretation: The network is active, but a clear alpha/beta rhythm was not detected.")
report_lines.append("=" * 60)

report_text = "\n".join(report_lines)

print(report_text)

report_path = Path("simulation_report.txt")
report_path.write_text(report_text, encoding="utf-8")
print(f"\nSaved text report to: {report_path.resolve()}")

# -----------------------------------------------------------------------------
# 7) Static figure
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 13))
fig.suptitle("Simple E/I Network Demo", fontsize=16, fontweight="bold")

# Raster plot
ax = axes[0]
if total_spikes > 0:
    ax.scatter(spike_times[e_mask], spike_ids[e_mask], s=8, alpha=0.7, label="E")
    ax.scatter(spike_times[i_mask], spike_ids[i_mask], s=12, alpha=0.8, label="I")
ax.set_xlim(0, simConfig.duration)
ax.set_ylabel("Cell ID")
ax.set_title("Spike Raster")
ax.legend()
ax.grid(True, alpha=0.3)

# Population rate
ax = axes[1]
ax.plot(time_axis, e_rate_trace, linewidth=2, label="E rate")
ax.plot(time_axis, i_rate_trace, linewidth=2, label="I rate")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("Population Firing Rates")
ax.legend()
ax.grid(True, alpha=0.3)

# Net activity
ax = axes[2]
ax.plot(time_axis, net_activity, linewidth=2)
ax.set_ylabel("E - I rate (Hz)")
ax.set_title("Net Population Activity")
ax.grid(True, alpha=0.3)

# Power spectrum
ax = axes[3]
if len(freqs) > 0:
    ax.semilogy(freqs, psd, linewidth=2)
    ax.set_xlim(0, 60)
    if peak_freq > 0:
        ax.axvline(peak_freq, linestyle="--")
        ax.text(peak_freq + 0.5, max(psd) * 0.7 if np.max(psd) > 0 else 0.1,
                f"Peak {peak_freq:.1f} Hz")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.set_title("Power Spectrum of Network Activity")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = Path("network_summary.png")
plt.savefig(fig_path, dpi=160, bbox_inches="tight")
plt.show()

print(f"Saved static figure to: {fig_path.resolve()}")

#Blue = excitatory neurons (E) Red = inhibitory neurons (I). Some neurons fire a lot because
#they receive more excitation random connections favor them they are easier to activate Other neurons may:
#receive stronger inhibition get fewer inputs stay mostly quiet In real brains this also happens. Not all neurons fire equally.
# -----------------------------------------------------------------------------
# 8) Optional animation
# -----------------------------------------------------------------------------
make_animation = True

if make_animation:
    try:

        anim_fig, anim_ax = plt.subplots(figsize=(12, 5))

        anim_ax.set_xlim(0, simConfig.duration)
        anim_ax.set_ylim(-1, len(e_gids) + len(i_gids))

        anim_ax.set_xlabel("Time (ms)")
        anim_ax.set_ylabel("Cell ID")

        anim_ax.set_title("Spike Raster Animation")

        # Separate scatter objects for E and I neurons
        scatter_e = anim_ax.scatter(
            [], [],
            s=10,
            c="blue",
            label="Excitatory (E)",
            alpha=0.7
        )

        scatter_i = anim_ax.scatter(
            [], [],
            s=20,
            c="red",
            label="Inhibitory (I)",
            alpha=0.9
        )

        anim_ax.legend()

        # Animation frames
        frame_times = np.linspace(
            0,
            simConfig.duration,
            120
        )

        def update(frame_t):

            # Excitatory spikes
            mask_e = (spike_times <= frame_t) & e_mask

            e_points = (
                np.column_stack(
                    (
                        spike_times[mask_e],
                        spike_ids[mask_e]
                    )
                )
                if np.any(mask_e)
                else np.empty((0, 2))
            )

            # Inhibitory spikes
            mask_i = (spike_times <= frame_t) & i_mask

            i_points = (
                np.column_stack(
                    (
                        spike_times[mask_i],
                        spike_ids[mask_i]
                    )
                )
                if np.any(mask_i)
                else np.empty((0, 2))
            )

            # Update scatter plots
            scatter_e.set_offsets(e_points)
            scatter_i.set_offsets(i_points)

            anim_ax.set_title(
                f"Spike Raster Animation | t = {frame_t:.0f} ms"
            )

            return scatter_e, scatter_i

        animation = FuncAnimation(
            anim_fig,
            update,
            frames=frame_times,
            interval=50,
            blit=True
        )

        gif_path = Path("network_animation.gif")

        animation.save(
            gif_path,
            writer=PillowWriter(fps=20)
        )

        plt.close(anim_fig)

        print(f"Saved animation to: {gif_path.resolve()}")

    except Exception as exc:
        print(f"Animation could not be saved: {exc}")

print("Done.")