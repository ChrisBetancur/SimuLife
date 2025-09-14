import re
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter

def parse_system_log(file_path):
    # (same parser you provided)
    patterns = {
        'episode_start': re.compile(r'\[\s*(?P<timestamp>.*?)\]\s+\[INFO\]\s+[-]+ Episode\s+(?P<num>\d+)\s+of\s+\d+\s+[-]+'),
        'episode_end':   re.compile(r'\[\s*(?P<timestamp>.*?)\]\s+\[DEBUG\]\s+[-]+ End of Episode\s+(?P<num>\d+)\s+[-]+'),
        'boltzmann':     re.compile(r'Boltzmann Policy Q-Values:\s+(?P<v1>[-+]?[0-9]*\.?[0-9]+),\s*(?P<v2>[-+]?[0-9]*\.?[0-9]+),\s*(?P<v3>[-+]?[0-9]*\.?[0-9]+),\s*(?P<v4>[-+]?[0-9]*\.?[0-9]+)'),
        'selected':      re.compile(r'Selected Action:\s+(?P<action>\d+)\s+with Temperature:\s+(?P<temp>[-+]?[0-9]*\.?[0-9]+)'),
        'z_score':       re.compile(r'Z-Score:\s+(?P<z>[-+]?[0-9]*\.?[0-9]+)'),
        'extrinsic':     re.compile(r'Extrinsic Reward:\s+(?P<e>[-+]?[0-9]*\.?[0-9]+)'),
        'total':         re.compile(r'Total Reward:\s+(?P<tr>[-+]?[0-9]*\.?[0-9]+)\s+\(Beta:\s+(?P<beta>[-+]?[0-9]*\.?[0-9]+)\)'),
        'intrinsic':     re.compile(r'Intrinsic Reward \(MSE\):\s*(?P<mse>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'),
        'learning':      re.compile(r'Learning Stage Q-Values:\s+(?P<l1>[-+]?[0-9]*\.?[0-9]+),\s*(?P<l2>[-+]?[0-9]*\.?[0-9]+),\s*(?P<l3>[-+]?[0-9]*\.?[0-9]+),\s*(?P<l4>[-+]?[0-9]*\.?[0-9]+)'),
        'best':          re.compile(r'Best Action Index:\s+(?P<best>\d+),\s*Max Q-Value:\s*(?P<maxq>[-+]?[0-9]*\.?[0-9]+)')
    }

    episodes = []
    current_ep = None
    current_record = {}

    with open(file_path, 'r') as f:
        for line in f:
            m = patterns['episode_start'].search(line)
            if m:
                ep_num = int(m.group('num'))
                current_ep = {'episode': ep_num, 'records': []}
                episodes.append(current_ep)
                continue

            m = patterns['episode_end'].search(line)
            if m:
                if current_record and current_ep:
                    current_ep['records'].append(current_record)
                current_record = {}
                current_ep = None
                continue

            if current_ep is None:
                continue

            for key in ('boltzmann','selected','z_score','extrinsic','intrinsic','total','learning','best'):
                m = patterns[key].search(line)
                if not m:
                    continue

                if key == 'boltzmann':
                    if current_record:
                        current_ep['records'].append(current_record)
                    current_record = {'timestamp': m.group(0).split(']')[0].lstrip('['),
                                      'boltzmann': [float(m.group(g)) for g in ('v1','v2','v3','v4')]}
                elif key == 'selected':
                    current_record['selected_action'] = int(m.group('action'))
                elif key == 'z_score':
                    current_record['z_score'] = float(m.group('z'))
                elif key == 'extrinsic':
                    current_record['extrinsic_reward'] = float(m.group('e'))
                elif key == 'intrinsic':
                    current_record['intrinsic_mse'] = float(m.group('mse'))
                elif key == 'total':
                    current_record['total_reward'] = float(m.group('tr'))
                    current_record['beta'] = float(m.group('beta'))
                elif key == 'learning':
                    current_record['learning_q'] = [float(m.group(g)) for g in ('l1','l2','l3','l4')]
                elif key == 'best':
                    current_record['best_action'] = int(m.group('best'))
                    current_record['max_q_value'] = float(m.group('maxq'))
                break

    if current_record and current_ep:
        current_ep['records'].append(current_record)
    return episodes


def plot_dqn_diagnostics(episodes, ma_window=50):
    # Episode-level aggregates
    episode_nums      = [ep['episode'] for ep in episodes]
    extrinsic_rewards = [sum(r.get('extrinsic_reward', 0) for r in ep['records']) for ep in episodes]
    total_rewards     = [sum(r.get('total_reward',   0) for r in ep['records']) for ep in episodes]
    lengths           = [len(ep['records']) for ep in episodes]
    avg_z             = [np.nan if len(ep['records'])==0 else np.nanmean([r.get('z_score',    np.nan) for r in ep['records']]) for ep in episodes]

    # average intrinsic mse per episode
    avg_intrinsic = []
    for ep in episodes:
        vals = [r['intrinsic_mse'] for r in ep['records'] if 'intrinsic_mse' in r]
        avg_intrinsic.append(float(np.nanmean(vals)) if vals else np.nan)

    # average beta per episode
    avg_beta = []
    for ep in episodes:
        betas = [r['beta'] for r in ep['records'] if 'beta' in r]
        avg_beta.append(float(np.mean(betas)) if betas else np.nan)

    # Avg extrinsic per step and total intrinsic reward
    avg_extrinsic_per_step = [extrinsic_rewards[i] / lengths[i] if lengths[i] > 0 else np.nan for i in range(len(extrinsic_rewards))]
    total_intrinsic_rewards = [sum(r.get('beta', 0) * r.get('z_score', 0) for r in ep['records']) for ep in episodes]

    # moving average for extrinsic (if desired)
    window = min(ma_window, len(extrinsic_rewards))
    if window > 1:
        mov_avg = np.convolve(extrinsic_rewards, np.ones(window)/window, mode='valid')
        mov_x = episode_nums[window-1:]
    else:
        mov_avg = np.array([])
        mov_x = []

    # moving average for lengths (used to determine if lengths increased)
    len_window = min(ma_window, len(lengths))
    if len_window > 1:
        len_mov_avg = np.convolve(lengths, np.ones(len_window)/len_window, mode='valid')
        len_mov_x = episode_nums[len_window-1:]
    else:
        len_mov_avg = np.array([])
        len_mov_x = []

    # ---- Trend helpers: compute both raw delta and slope (per episode) ----
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
        # slope via least-squares linear fit
        try:
            slope, intercept = np.polyfit(xv, yv, 1)
        except Exception:
            # fallback to simple slope, using endpoints
            dx = float(xv[-1] - xv[0]) if xv[-1] != xv[0] else 1.0
            slope = float((yv[-1] - yv[0]) / dx)
        return delta, slope

    def trend_label_and_color(delta_slope_tuple, y_series):
        """
        Format label like '+2 (slope +0.040/ep)' and choose color & alpha.
        Positive slope -> green, negative -> red, None/zero -> gray.
        Alpha scales with magnitude relative to data range (so bigger changes are stronger color).
        """
        if delta_slope_tuple is None:
            return 'n/a', 'lightgray', 0.45
        delta, slope = delta_slope_tuple
        # compute a sensible alpha based on delta magnitude relative to observed range
        y = np.array(y_series, dtype=float)
        mask = np.isfinite(y)
        if mask.sum() >= 2:
            yv = y[mask]
            yrange = float(np.nanmax(yv) - np.nanmin(yv))
        else:
            yrange = 0.0
        # base alpha
        if yrange > 0:
            score = abs(delta) / yrange
        else:
            score = min(abs(delta), abs(slope))
        alpha = 0.35 + min(score * 0.6, 0.6)
        if slope > 0:
            lbl = f"{_smart_round(delta)} (slope {slope:+.3f}/ep)"
            return lbl, '#2ca02c', float(np.clip(alpha, 0.15, 0.95))
        elif slope < 0:
            lbl = f"{_smart_round(delta)} (slope {slope:+.3f}/ep)"
            return lbl, '#d62728', float(np.clip(alpha, 0.15, 0.95))
        else:
            lbl = f"{_smart_round(delta)} (slope {slope:+.3f}/ep)"
            return lbl, 'gray', 0.45

    def _smart_round(x):
        # If magnitude >= 1, show integer; else show one decimal
        if abs(x) >= 1:
            return str(int(np.round(x)))
        else:
            return f"{x:.2f}"

    # Plotting
    fig, axs = plt.subplots(4,2,figsize=(14,16))
    axs = axs.flatten()

    # (0) Extrinsic Reward per Episode
    axs[0].plot(episode_nums, extrinsic_rewards, label='Extrinsic')
    if mov_avg.size:
        axs[0].plot(mov_x, mov_avg, label=f'MA (window={window})')
    axs[0].set(title='Extrinsic Reward per Episode', xlabel='Episode', ylabel='Reward')
    axs[0].legend()
    t0 = compute_delta_and_slope(extrinsic_rewards, episode_nums)
    lbl0, col0, a0 = trend_label_and_color(t0, extrinsic_rewards)
    axs[0].text(0.98, 0.95, lbl0, transform=axs[0].transAxes, ha='right', va='top',
                fontsize=10, color='white', bbox=dict(facecolor=col0, alpha=a0, boxstyle='round,pad=0.3'))

    # (1) Avg Extrinsic Reward per Step
    axs[1].plot(episode_nums, avg_extrinsic_per_step)
    axs[1].set(title='Avg Extrinsic Reward per Step', xlabel='Episode', ylabel='Avg Reward')
    t1 = compute_delta_and_slope(avg_extrinsic_per_step, episode_nums)
    lbl1, col1, a1 = trend_label_and_color(t1, avg_extrinsic_per_step)
    axs[1].text(0.98, 0.95, lbl1, transform=axs[1].transAxes, ha='right', va='top',
                fontsize=10, color='white', bbox=dict(facecolor=col1, alpha=a1, boxstyle='round,pad=0.3'))

    # (2) Total Reward per Episode
    axs[2].plot(episode_nums, total_rewards)
    axs[2].set(title='Total Reward per Episode', xlabel='Episode', ylabel='Reward')
    t2 = compute_delta_and_slope(total_rewards, episode_nums)
    lbl2, col2, a2 = trend_label_and_color(t2, total_rewards)
    axs[2].text(0.98, 0.95, lbl2, transform=axs[2].transAxes, ha='right', va='top',
                fontsize=10, color='white', bbox=dict(facecolor=col2, alpha=a2, boxstyle='round,pad=0.3'))

    # (3) Episode Lengths (now shows raw lengths + moving average + summary)
    axs[3].plot(episode_nums, lengths, label='Length (per-episode)', alpha=0.7)
    if len(len_mov_avg) > 0:
        axs[3].plot(len_mov_x, len_mov_avg, label=f'Length MA (window={len_window})', linewidth=2)
    axs[3].set(title='Episode Lengths', xlabel='Episode', ylabel='Steps')
    # LEGEND REMOVED FOR THIS AXIS TO AVOID BOX OVERLAP
    # axs[3].legend()

    # trend using moving-average when available, else raw lengths
    if len(len_mov_avg) > 1:
        tlen = compute_delta_and_slope(len_mov_avg, len_mov_x)
    else:
        tlen = compute_delta_and_slope(lengths, episode_nums)
    lbllen, collen, a_len = trend_label_and_color(tlen, len_mov_avg if len(len_mov_avg)>0 else lengths)
    axs[3].text(0.98, 0.95, lbllen, transform=axs[3].transAxes, ha='right', va='top',
                fontsize=10, color='white', bbox=dict(facecolor=collen, alpha=a_len, boxstyle='round,pad=0.3'))

    # numeric summary box: overall avg, first-window avg, last-window avg, percent change
    overall_avg_len = float(np.nanmean(lengths)) if len(lengths) > 0 else np.nan
    if len_window > 0:
        first_window_avg = float(np.nanmean(lengths[:len_window])) if len(lengths[:len_window]) > 0 else np.nan
        last_window_avg  = float(np.nanmean(lengths[-len_window:])) if len(lengths[-len_window:]) > 0 else np.nan
    else:
        first_window_avg = last_window_avg = np.nan

    pct_change = None
    if np.isfinite(first_window_avg) and first_window_avg != 0 and np.isfinite(last_window_avg):
        pct_change = 100.0 * (last_window_avg - first_window_avg) / abs(first_window_avg)
    # build info string
    info_lines = [
        f"avg len: {_smart_round(overall_avg_len) if np.isfinite(overall_avg_len) else 'n/a'}",
        f"first {len_window}: {_smart_round(first_window_avg) if np.isfinite(first_window_avg) else 'n/a'}",
        f"last {len_window}: {_smart_round(last_window_avg) if np.isfinite(last_window_avg) else 'n/a'}",
        f"Δ%: {f'{pct_change:+.1f}%' if pct_change is not None else 'n/a'}"
    ]
    info_text = "  |  ".join(info_lines)
    # place it under the title (top-left inside axis) with higher zorder so it's not obscured
    axs[3].text(0.02, 0.95, info_text, transform=axs[3].transAxes, ha='left', va='top',
                fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.3'),
                zorder=10)

    # (4) Avg Beta per Episode
    # go through avg_beta and print if there is any nan or negative numbers
    if any(np.isnan(avg_beta)) or any(b < 0 for b in avg_beta if np.isfinite(b)):
        print("Warning: avg_beta contains NaN or negative values.")
    axs[4].plot(episode_nums, avg_beta, marker='o', linestyle='-')
    axs[4].set(title='Avg Beta per Episode', xlabel='Episode', ylabel='Beta')
    t4 = compute_delta_and_slope(avg_beta, episode_nums)
    lbl4, col4, a4 = trend_label_and_color(t4, avg_beta)
    axs[4].text(0.98, 0.95, lbl4, transform=axs[4].transAxes, ha='right', va='top',
                fontsize=10, color='white', bbox=dict(facecolor=col4, alpha=a4, boxstyle='round,pad=0.3'))

    # (5) Avg Z-Score per Episode
    axs[5].plot(episode_nums, avg_z)
    axs[5].set(title='Avg Z-Score per Episode', xlabel='Episode', ylabel='Z-Score')
    t5 = compute_delta_and_slope(avg_z, episode_nums)
    lbl5, col5, a5 = trend_label_and_color(t5, avg_z)
    axs[5].text(0.98, 0.95, lbl5, transform=axs[5].transAxes, ha='right', va='top',
                fontsize=10, color='white', bbox=dict(facecolor=col5, alpha=a5, boxstyle='round,pad=0.3'))

    # (6) Total Intrinsic Reward per Episode
    axs[6].plot(episode_nums, total_intrinsic_rewards)
    axs[6].set(title='Total Intrinsic Reward per Episode', xlabel='Episode', ylabel='Reward')
    t6 = compute_delta_and_slope(total_intrinsic_rewards, episode_nums)
    lbl6, col6, a6 = trend_label_and_color(t6, total_intrinsic_rewards)
    axs[6].text(0.98, 0.95, lbl6, transform=axs[6].transAxes, ha='right', va='top',
                fontsize=10, color='white', bbox=dict(facecolor=col6, alpha=a6, boxstyle='round,pad=0.3'))

    # (7) Avg Intrinsic MSE per Episode (with smart transform if needed)
    vals = np.array(avg_intrinsic, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        print("Intrinsic MSE: no finite data found.")
    else:
        print("Intrinsic MSE stats:")
        print("  count:", finite.size)
        print("  min:  {:g}".format(np.nanmin(finite)))
        print("  median:{:g}".format(np.nanmedian(finite)))
        print("  mean: {:g}".format(np.nanmean(finite)))
        print("  max:  {:g}".format(np.nanmax(finite)))

    axs[7].set(title='Avg Intrinsic MSE per Episode', xlabel='Episode')

    # compute trend on raw mse values (not transformed)
    t7 = compute_delta_and_slope(vals, episode_nums)
    lbl7, col7, a7 = trend_label_and_color(t7, vals)
    axs[7].text(0.98, 0.95, lbl7, transform=axs[7].transAxes, ha='right', va='top',
                fontsize=10, color='white', bbox=dict(facecolor=col7, alpha=a7, boxstyle='round,pad=0.3'))

    if finite.size == 0:
        axs[7].text(0.5, 0.5, 'no intrinsic data', ha='center', va='center', fontsize=12)
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        use_log = False
        if (vmin <= 0 and vmax > 0 and (vmax - vmin) > 1e3) or (vmax / max(vmin, 1e-12) > 1000) or (vmax > 100):
            use_log = True

        if use_log:
            plotted = np.log10(1.0 + np.clip(vals, 0, None))
            axs[7].plot(episode_nums, plotted, marker='o', linestyle='-')
            axs[7].set_ylabel('log10(1 + MSE)')
            def tick_formatter(y, _):
                orig = (10 ** y) - 1.0
                if orig >= 1000:
                    return "{:.0e}".format(orig)
                else:
                    return "{:.3g}".format(orig)
            axs[7].yaxis.set_major_formatter(FuncFormatter(tick_formatter))
            axs[7].yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
        else:
            axs[7].plot(episode_nums, vals, marker='o', linestyle='-')
            axs[7].set_ylabel('MSE')
            axs[7].yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
            sf = ScalarFormatter()
            sf.set_scientific(False)
            axs[7].yaxis.set_major_formatter(sf)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    episodes = parse_system_log('system.log')
    plot_dqn_diagnostics(episodes)
