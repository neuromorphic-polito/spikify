import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from spikify.encoding.rate import poisson_rate
import os

# Set style for better visualization
plt.style.use("dark_background")
os.makedirs("animation", exist_ok=True)

# Generate a sine wave signal with more points for smoother animation
t = np.linspace(0, 4 * np.pi, 200)  # Longer time axis, more points
signal = np.sin(2 * t) + 0.5 * np.sin(4 * t)  # More complex signal

# Generate spikes
spikes = poisson_rate(signal=signal, interval_length=5)
spike_times = t[spikes]

# Color settings
signal_color = "#00fff5"  # Cyan
spike_color = "#ff0055"  # Pink
highlight_color = "#ffffff"  # White

# Prepare frames for GIF
frames = []
spike_history = []  # Keep track of all previous spikes

for i in range(len(t)):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#000000")

    # Add a gradual fade effect for the signal
    alpha_values = np.linspace(0.2, 1, i + 1)
    for j in range(i + 1):
        axes[0].plot(t[j : j + 2], signal[j : j + 2], color=signal_color, alpha=alpha_values[j], linewidth=2)

    # Current point with glow effect
    axes[0].scatter(t[i], signal[i], color=highlight_color, s=100, zorder=3, alpha=0.8)
    axes[0].scatter(t[i], signal[i], color=highlight_color, s=200, zorder=2, alpha=0.4)
    axes[0].scatter(t[i], signal[i], color=highlight_color, s=300, zorder=1, alpha=0.2)

    axes[0].set_ylabel("Signal", color="white")
    axes[0].set_title("Poisson Rate Encoding", color="white", pad=20, fontsize=14)
    axes[0].grid(True, alpha=0.2)

    # Bottom plot: spikes with persistence
    axes[1].set_ylabel("Spikes", color="white", labelpad=15)

    if spikes[i]:  # If there's a spike at current time
        spike_history.append(t[i])  # Add current spike to history

    # Plot all spikes in history
    if spike_history:
        # Plot vertical lines for all previous spikes
        axes[1].vlines(spike_history, 0.9, 1.1, colors=spike_color, linewidth=2, label="Spikes")

        # Add glow effect to all previous spikes
        for spike_time in spike_history:
            axes[1].scatter(spike_time, 1, color=spike_color, s=50, zorder=4, alpha=0.6)

    # Highlight the current time point
    if spikes[i]:  # If current point is a spike
        # Create pulsing effect
        pulse_alpha = 0.4 + 0.4 * np.sin(i * 0.2)  # Varies between 0.4 and 0.8
        axes[1].vlines(t[i], 0.9, 1.1, colors=highlight_color, linewidth=3, zorder=5, alpha=pulse_alpha)
        axes[1].scatter(t[i], 1, color=highlight_color, s=100, zorder=6, alpha=0.8)
        axes[1].scatter(t[i], 1, color=highlight_color, s=200, zorder=5, alpha=pulse_alpha)

    axes[1].set_ylim(0.8, 1.2)
    axes[1].set_yticks([1.0])
    # axes[1].tick_params(axis='y', pad=15)
    axes[1].set_xlabel("Time", color="white")
    axes[1].grid(True, alpha=0.1)

    # Style adjustments
    for ax in axes:
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#000000")

    plt.tight_layout()

    # Save frame
    filename = f"animation/frame_{i}.png"
    plt.savefig(filename, dpi=150, facecolor="black", edgecolor="none")
    plt.close(fig)
    frames.append(imageio.imread(filename))

# Save as GIF with faster frame rate
imageio.mimsave("animation/spike_encoding.gif", frames, duration=0.05, loop=0)
print("Enhanced GIF saved as spike_encoding.gif")
