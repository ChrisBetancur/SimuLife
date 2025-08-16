import re
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter


def parse_system_log(file_path):
    # Compile all patterns once
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
            # 1) Episode start?
            m = patterns['episode_start'].search(line)
            if m:
                ep_num = int(m.group('num'))
                current_ep = {
                    'episode': ep_num,
                    'records': []
                }
                episodes.append(current_ep)
                continue

            # 2) Episode end?
            m = patterns['episode_end'].search(line)
            if m:
                # Save any remaining record for the episode
                if current_record and current_ep:
                    current_ep['records'].append(current_record)
                current_record = {}
                current_ep = None
                continue

            # 3) Only parse if in an episode
            if current_ep is None:
                continue

            # Try each debug pattern (order matters for grouping)
            for key in ('boltzmann','selected','z_score','extrinsic','intrinsic','total','learning','best'):
                m = patterns[key].search(line)
                if not m:
                    continue

                if key == 'boltzmann':
                    # If this is not the first step of the episode, save the previous record.
                    if current_record:
                        current_ep['records'].append(current_record)
                    # Start a new record with the boltzmann data.
                    current_record = {'timestamp': m.group(0).split(']')[0].lstrip('['),
                                      'boltzmann': [float(m.group(g)) for g in ('v1','v2','v3','v4')]}
                elif key == 'selected':
                    current_record['selected_action'] = int(m.group('action'))
                    current_record['temperature']     = float(m.group('temp'))
                elif key == 'z_score':
                    current_record['z_score'] = float(m.group('z'))
                elif key == 'extrinsic':
                    current_record['extrinsic_reward'] = float(m.group('e'))
                elif key == 'intrinsic':
                    current_record['intrinsic_mse'] = float(m.group('mse'))
                elif key == 'total':
                    current_record['total_reward'] = float(m.group('tr'))
                    current_record['beta']         = float(m.group('beta'))
                elif key == 'learning':
                    current_record['learning_q'] = [float(m.group(g)) for g in ('l1','l2','l3','l4')]
                elif key == 'best':
                    current_record['best_action'] = int(m.group('best'))
                    current_record['max_q_value'] = float(m.group('maxq'))
                
                # break from inner loop if we found a match
                break
    
    # After the loop, save any final record that wasn't saved by a 'boltzmann' match.
    if current_record and current_ep:
        current_ep['records'].append(current_record)
        
    return episodes

def plot_dqn_diagnostics(episodes, ma_window=50):
    # Episode-level aggregates
    episode_nums      = [ep['episode'] for ep in episodes]
    extrinsic_rewards = [sum(r.get('extrinsic_reward', 0) for r in ep['records']) for ep in episodes]
    total_rewards     = [sum(r.get('total_reward',   0) for r in ep['records']) for ep in episodes]
    lengths           = [len(ep['records']) for ep in episodes]
    avg_temps         = [np.nan if len(ep['records'])==0 else np.nanmean([r.get('temperature', np.nan) for r in ep['records']]) for ep in episodes]
    avg_z             = [np.nan if len(ep['records'])==0 else np.nanmean([r.get('z_score',    np.nan) for r in ep['records']]) for ep in episodes]

    # average intrinsic mse per episode
    avg_intrinsic = []
    for ep in episodes:
        vals = [r['intrinsic_mse'] for r in ep['records'] if 'intrinsic_mse' in r]
        if len(vals) == 0:
            avg_intrinsic.append(np.nan)
        else:
            avg_intrinsic.append(float(np.nanmean(vals)))

    # New: average beta per episode (use nan when none)
    avg_beta = []
    for ep in episodes:
        betas = [r['beta'] for r in ep['records'] if 'beta' in r]
        if len(betas) == 0:
            avg_beta.append(np.nan)
        else:
            avg_beta.append(float(np.mean(betas)))

    # Ensure window <= #episodes
    window = min(ma_window, len(extrinsic_rewards))
    if window > 1:
        mov_avg = np.convolve(extrinsic_rewards, np.ones(window)/window, mode='valid')
        mov_x = episode_nums[window-1:]
    else:
        mov_avg = np.array([])   # make it an ndarray for safe .size usage
        mov_x   = []

    # Collect TD errors
    td_errors = []
    for ep in episodes:
        for r in ep['records']:
            if 'learning_q' in r and 'boltzmann' in r:
                td_errors.append(abs(max(r['learning_q']) - max(r['boltzmann'])))

    # Plot: 4 rows x 2 columns to add beta + intrinsic subplot
    fig, axs = plt.subplots(4,2,figsize=(14,14))
    axs = axs.flatten()

    # Extrinsic
    axs[0].plot(episode_nums, extrinsic_rewards, label='Extrinsic')
    if mov_avg.size:
        axs[0].plot(mov_x, mov_avg, label=f'MA (window={window})')
    axs[0].set(title='Extrinsic Reward per Episode', xlabel='Episode', ylabel='Reward')
    axs[0].legend()

    # Total
    axs[1].plot(episode_nums, total_rewards)
    axs[1].set(title='Total Reward per Episode', xlabel='Episode', ylabel='Reward')

    # Length
    axs[2].plot(episode_nums, lengths)
    axs[2].set(title='Episode Lengths', xlabel='Episode', ylabel='Steps')

    # Temp
    axs[3].plot(episode_nums, avg_temps)
    axs[3].set(title='Avg Temperature per Episode', xlabel='Episode', ylabel='Temp')

    # Z-Score
    axs[4].plot(episode_nums, avg_z)
    axs[4].set(title='Avg Z-Score per Episode', xlabel='Episode', ylabel='Z-Score')

    # TD Error
    axs[5].plot(td_errors)
    axs[5].set(title='Per-Step TD Error', xlabel='Step', ylabel='|ΔQ|')

    # Beta (new)
    axs[6].plot(episode_nums, avg_beta, marker='o', linestyle='-')
    axs[6].set(title='Avg Beta per Episode', xlabel='Episode', ylabel='Beta')

    # Intrinsic MSE (new)
    vals = np.array(avg_intrinsic, dtype=float)

    # Print numeric summary (so you see the actual numbers)
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

    # Choose plotting strategy depending on dynamic range:
    if finite.size == 0:
        axs[7].text(0.5, 0.5, 'no intrinsic data', ha='center', va='center', fontsize=12)
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        # If dynamic range is large or values are huge, use log10(1 + x) transform for plot
        use_log = False
        if (vmin <= 0 and vmax > 0 and (vmax - vmin) > 1e3) or (vmax / max(vmin, 1e-12) > 1000) or (vmax > 100):
            use_log = True

        if use_log:
            # plot log10(1 + x) so zeros are handled and dynamic range compressed
            plotted = np.log10(1.0 + np.clip(vals, 0, None))
            axs[7].plot(episode_nums, plotted, marker='o', linestyle='-')
            axs[7].set_ylabel('log10(1 + MSE)')
            # convert back tick labels to show original scale approx
            def tick_formatter(y, _):
                orig = (10 ** y) - 1.0
                if orig >= 1000:
                    return "{:.0e}".format(orig)
                else:
                    return "{:.3g}".format(orig)
            axs[7].yaxis.set_major_formatter(FuncFormatter(tick_formatter))
            axs[7].yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
        else:
            # linear plot, force friendly ticks
            axs[7].plot(episode_nums, vals, marker='o', linestyle='-')
            axs[7].set_ylabel('MSE')
            axs[7].yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
            # format tick labels as plain numbers
            sf = ScalarFormatter()
            sf.set_scientific(False)
            axs[7].yaxis.set_major_formatter(sf)

    # final layout
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    episodes = parse_system_log('system.log')
    plot_dqn_diagnostics(episodes)