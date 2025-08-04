import re
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np


def parse_system_log(file_path):
    # Compile all patterns once
    patterns = {
        'episode_start': re.compile(r'\[\s*(?P<timestamp>.*?)\]\s+\[INFO\]\s+[-]+ Episode\s+(?P<num>\d+)\s+of\s+\d+\s+[-]+'),
        'episode_end':   re.compile(r'\[\s*(?P<timestamp>.*?)\]\s+\[DEBUG\]\s+[-]+ End of Episode\s+(?P<num>\d+)\s+[-]+'),
        'boltzmann':     re.compile(r'Boltzmann Policy Q-Values:\s+(?P<v1>[\d\.]+),\s*(?P<v2>[\d\.]+),\s*(?P<v3>[\d\.]+),\s*(?P<v4>[\d\.]+)'),
        'selected':      re.compile(r'Selected Action:\s+(?P<action>\d+)\s+with Temperature:\s+(?P<temp>[\d\.]+)'),
        'z_score':       re.compile(r'Z-Score:\s+(?P<z>[\d\.]+)'),
        'extrinsic':     re.compile(r'Extrinsic Reward:\s+(?P<e>[\d\.]+)'),
        'total':         re.compile(r'Total Reward:\s+(?P<tr>[\d\.]+)\s+\(Beta:\s+(?P<beta>[\d\.]+)\)'),
        'learning':      re.compile(r'Learning Stage Q-Values:\s+(?P<l1>[\d\.]+),\s*(?P<l2>[\d\.]+),\s*(?P<l3>[\d\.]+),\s*(?P<l4>[\d\.]+)'),
        'best':          re.compile(r'Best Action Index:\s+(?P<best>\d+),\s*Max Q-Value:\s*(?P<maxq>[\d\.]+)')
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
                current_record = {}
                current_ep = None
                continue

            # 3) Only parse if in an episode
            if current_ep is None:
                continue

            # Try each debug pattern
            for key in ('boltzmann','selected','z_score','extrinsic','total','learning','best'):
                m = patterns[key].search(line)
                if not m:
                    continue

                if key == 'boltzmann':
                    current_record = {'timestamp': m.group(0).split(']')[0].lstrip('['),
                                      'boltzmann': [float(m.group(g)) for g in ('v1','v2','v3','v4')]}  
                elif key == 'selected':
                    current_record['selected_action'] = int(m.group('action'))
                    current_record['temperature']     = float(m.group('temp'))
                elif key == 'z_score':
                    current_record['z_score'] = float(m.group('z'))
                elif key == 'extrinsic':
                    current_record['extrinsic_reward'] = float(m.group('e'))
                elif key == 'total':
                    current_record['total_reward'] = float(m.group('tr'))
                    current_record['beta']         = float(m.group('beta'))
                elif key == 'learning':
                    current_record['learning_q'] = [float(m.group(g)) for g in ('l1','l2','l3','l4')]
                elif key == 'best':
                    current_record['best_action'] = int(m.group('best'))
                    current_record['max_q_value'] = float(m.group('maxq'))
                    current_ep['records'].append(current_record)
                    current_record = {}

                break
    return episodes


def plot_dqn_diagnostics(episodes, ma_window=50):
    # Dynamically adjust window
    episode_nums      = [ep['episode'] for ep in episodes]
    extrinsic_rewards = [sum(r.get('extrinsic_reward', 0) for r in ep['records']) for ep in episodes]
    total_rewards     = [sum(r.get('total_reward',   0) for r in ep['records']) for ep in episodes]
    lengths           = [len(ep['records']) for ep in episodes]
    avg_temps         = [np.mean([r.get('temperature',0) for r in ep['records']]) for ep in episodes]
    avg_z             = [np.mean([r.get('z_score',    0) for r in ep['records']]) for ep in episodes]

    # Ensure window <= #episodes
    window = min(ma_window, len(extrinsic_rewards))
    if window > 1:
        mov_avg = np.convolve(extrinsic_rewards, np.ones(window)/window, mode='valid')
        # valid mode yields len = N-window+1 -> corresponds to episodes window..N
        mov_x = episode_nums[window-1:]
    else:
        mov_avg = []
        mov_x   = []

    # Collect TD errors
    td_errors = []
    for ep in episodes:
        for r in ep['records']:
            if 'learning_q' in r and 'boltzmann' in r:
                td_errors.append(abs(max(r['learning_q']) - max(r['boltzmann'])))

    # Plot
    fig, axs = plt.subplots(3,2,figsize=(14,10))
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

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    episodes = parse_system_log('system.log')
    plot_dqn_diagnostics(episodes)
