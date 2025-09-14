import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_loss_log(file_path):
    """
    Parses a loss log file and returns a list of loss values.
    """
    loss_values = []
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return []

    with open(file_path, 'r') as f:
        for line in f:
            try:
                # Simply try to convert the line to a float, ignoring any whitespace
                loss_values.append(float(line.strip()))
            except ValueError:
                # If a line can't be parsed as a float, just skip it and print a warning
                print(f"Warning: Could not parse loss from line: {line.strip()}")
    return loss_values

def compute_delta_and_slope(y_series, x_series):
    """
    y_series: list or array of values (may contain nan)
    x_series: list or array of x positions (episodes)
    Returns (delta, slope) where:
      - delta = last_valid_y - first_valid_y
      - slope = linear-regression slope (y per unit x)
    Returns None if <2 valid points.
    """
    y = np.array(y_series, dtype=float)
    x = np.array(x_series, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return None
    xv = x[mask]
    yv = y[mask]
    delta = float(yv[-1] - yv[0])
    try:
        slope, _ = np.polyfit(xv, yv, 1)
    except Exception:
        dx = float(xv[-1] - xv[0]) if xv[-1] != xv[0] else 1.0
        slope = float((yv[-1] - yv[0]) / dx)
    return delta, slope

def trend_label_and_color(delta_slope_tuple, y_series):
    """
    Format label like '+2 (slope +0.040/ep)' and choose color & alpha.
    Negative slope -> green, positive -> red, None/zero -> gray.
    Alpha scales with magnitude relative to data range.
    """
    def _smart_round(x):
        if abs(x) >= 1:
            return str(int(np.round(x)))
        else:
            return f"{x:.2f}"

    if delta_slope_tuple is None:
        return 'n/a', 'lightgray', 0.45
    delta, slope = delta_slope_tuple

    y = np.array(y_series, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() >= 2:
        yv = y[mask]
        yrange = float(np.nanmax(yv) - np.nanmin(yv))
    else:
        yrange = 0.0

    score = 0.0
    if yrange > 0:
        score = abs(delta) / yrange
    
    alpha = 0.35 + min(score * 0.6, 0.6)
    
    lbl = f"{_smart_round(delta)} (slope {slope:+.3f}/ep)"

    # Green for decreasing, red for increasing
    if slope < 0:
        return lbl, '#2ca02c', float(np.clip(alpha, 0.15, 0.95))
    elif slope > 0:
        return lbl, '#d62728', float(np.clip(alpha, 0.15, 0.95))
    else:
        return lbl, 'gray', 0.45

def plot_loss(file_path, plot_title, ax, start_step=0):
    """
    Parses a log file, plots the loss, and adds a trend box.
    """
    loss_values = parse_loss_log(file_path)
    print(f"Parsed {len(loss_values)} loss values from {file_path}")
    
    # Slice the data to start from the specified step
    if start_step > 0:
        loss_values = loss_values[start_step - 1:]
    
    if not loss_values:
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', fontsize=12)
        ax.set_title(plot_title)
        return

    steps = list(range(start_step, start_step + len(loss_values)))
    ax.plot(steps, loss_values, label='Loss')

    # Compute and plot trend
    trend_tuple = compute_delta_and_slope(loss_values, steps)
    lbl, color, alpha = trend_label_and_color(trend_tuple, loss_values)
    ax.text(0.98, 0.95, lbl, transform=ax.transAxes, ha='right', va='top',
            fontsize=10, color='white', bbox=dict(facecolor=color, alpha=alpha, boxstyle='round,pad=0.3'))

    ax.set_title(plot_title)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.6)

if __name__ == '__main__':
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    plot_loss('logs/online_system.log', 'Online Network Loss (Full)', axs[0])
    plot_loss('logs/rnd_predictor_system.log', 'RND Predictor Loss', axs[1])
    plot_loss('logs/online_system.log', 'Online Network Loss (Zoomed)', axs[2], start_step=10000)

    plt.tight_layout()
    plt.show()

