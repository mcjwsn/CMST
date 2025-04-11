# Gifer
# Mostly implemented by AI
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx  
from matplotlib.gridspec import GridSpec
import random
from taksk1 import probability,distance,arbitrary_swap

def generate_annealing_gif(nodes, temperature, max_iterations=40000, modification=arbitrary_swap, 
                          cooling_param=0.999, sample_freq=50, fps=10):
    
    def simulated_annealing_with_history(old, modification, temperature, max_iterations, cooling_param, sample_freq):
        temperatures = list()
        energies = list()
        states = [old.copy()]  
        
        old_energy = sum(map(lambda two: distance(*two), zip(old, old[1:])))
        best_energy = old_energy
        default_temp = 1.0
        
        for iteration in range(max_iterations):
            temp = temperature(default_temp, cooling_param, iteration)
            
            temperatures.append(temp)
            energies.append(old_energy)
            
            new_state = modification(old)
            new_energy = sum(map(lambda two: distance(*two), zip(new_state, new_state[1:])))
            
            prob = probability(old_energy, new_energy, temp)
            
            rand = random.uniform(0, 1)
            if (new_energy < old_energy or rand < prob):
                old = new_state
                best_energy = min(best_energy, new_energy)
                old_energy = new_energy
            
            if iteration % sample_freq == 0:
                states.append(old.copy())
                
        return old, temperatures, energies, states
    
    n_nodes = len(nodes)
    temp_type = temperature.__name__
    mod_type = "arbitrary" if modification.__name__ == "arbitrary_swap" else "consecutive"
    filename = f"annealing_animation_n{n_nodes}_{mod_type}_{temp_type}_iter{max_iterations}.gif"

    path = list(map(lambda i: nodes[i], range(n_nodes)))
    
    final_path, all_temps, all_energies, all_paths = simulated_annealing_with_history(
        path, modification, temperature, max_iterations, cooling_param, sample_freq
    )
    
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig)
    ax_path = fig.add_subplot(gs[0, :])
    ax_temp = fig.add_subplot(gs[1, 0])
    ax_energy = fig.add_subplot(gs[1, 1])
    
    ax_temp.set_title('Temperature', fontsize=9)
    ax_temp.tick_params(labelsize=7)
    ax_temp.grid(color='lightgray')
    ax_temp.set_xlim(0, max_iterations)
    ax_temp.set_ylim(0, max(all_temps) * 1.1)
    
    ax_energy.set_title('Energy', fontsize=9)
    ax_energy.tick_params(labelsize=7)
    ax_energy.grid(color='lightgray')
    ax_energy.set_xlim(0, max_iterations)
    energy_max = max(all_energies)
    energy_min = min(all_energies)
    ax_energy.set_ylim(energy_min * 0.95, energy_max * 1.05)

    temp_line, = ax_temp.plot([], [], linewidth=1, color='orange')
    energy_line, = ax_energy.plot([], [], linewidth=0.5, color='green')
    
    G = nx.Graph()
    
    def init():
        ax_path.clear()
        ax_path.set_title('Path Evolution', fontsize=10)
        ax_path.axis('off')
        temp_line.set_data([], [])
        energy_line.set_data([], [])
        return temp_line, energy_line
    
    def update(frame):
        # Clear the path plot
        ax_path.clear()
        ax_path.set_title(f'Path Evolution (Iteration: {frame * sample_freq})', fontsize=10)
        ax_path.axis('off')
        
        # Update the path
        path = all_paths[frame]
        G.clear()
        P = list(path)
        for j, edge in enumerate(zip(P, P[1:])):
            G.add_edge(*edge, order=j)
        pos = {(x, y): [x, y] for (x, y) in path}
        edge_color = [order for _, _, order in G.edges(data='order')]
        nx.draw_networkx_nodes(G, pos, node_size=5, node_color='black', ax=ax_path)
        nx.draw_networkx_edges(G, pos, width=1, edge_color=edge_color, edge_cmap=plt.cm.winter, ax=ax_path)
        
        plot_idx = min(frame * sample_freq, len(all_temps) - 1)
        temp_line.set_data(range(plot_idx + 1), all_temps[:plot_idx + 1])
        energy_line.set_data(range(plot_idx + 1), all_energies[:plot_idx + 1])
        
        return temp_line, energy_line
    
    ani = animation.FuncAnimation(fig, update, frames=len(all_paths),
                                 init_func=init, blit=False, repeat=False)

    try:
        print("Attempting to save with pillow writer...")
        ani.save(filename, writer='pillow', fps=fps, dpi=100)
    except Exception as e:
        print(f"Could not save with default writer: {e}")
        try:
            print("Attempting to save with ffmpeg writer...")
            ani.save(filename, writer='ffmpeg', fps=fps, dpi=100)
        except Exception as e2:
            print(f"Could not save with ffmpeg either: {e2}")
            print("Saving as mp4 instead...")
            mp4_filename = filename.replace('.gif', '.mp4')
            ani.save(mp4_filename, writer='ffmpeg', fps=fps, dpi=100)
            print(f"Saved as: {mp4_filename}")
            return mp4_filename
    
    plt.close(fig)
    print(f"GIF saved as: {filename}")
    return filename

def generate_multiple_solution_gif(nodes, temperatures, modifications, max_iterations=20000, 
                                 cooling_param=0.999, sample_freq=50, fps=10):
    def simulated_annealing_with_history(old, modification, temperature, max_iterations, cooling_param, sample_freq):
        temperatures = list()
        energies = list()
        states = [old.copy()]
        
        old_energy = sum(map(lambda two: distance(*two), zip(old, old[1:])))
        best_energy = old_energy
        default_temp = 1.0
        
        for iteration in range(max_iterations):
            temp = temperature(default_temp, cooling_param, iteration)
            
            temperatures.append(temp)
            energies.append(old_energy)
            
            new_state = modification(old)
            new_energy = sum(map(lambda two: distance(*two), zip(new_state, new_state[1:])))
            
            prob = probability(old_energy, new_energy, temp)
            
            rand = random.uniform(0, 1)
            if (new_energy < old_energy or rand < prob):
                old = new_state
                best_energy = min(best_energy, new_energy)
                old_energy = new_energy
            
            if iteration % sample_freq == 0:
                states.append(old.copy())
                
        return old, temperatures, energies, states
    
    n_runs = len(temperatures) * len(modifications)
    
    n_nodes = len(nodes)
    filename = f"comparison_annealing_n{n_nodes}_iter{max_iterations}.gif"
    
    combinations = []
    for temp in temperatures:
        for mod in modifications:
            temp_name = temp.__name__
            mod_name = mod.__name__
            combinations.append((temp, mod, f"{mod_name} + {temp_name}"))
    
    all_results = []
    for temp_func, mod_func, label in combinations:
        path = list(map(lambda i: nodes[i], range(n_nodes)))
        
        final_path, temps, energies, paths = simulated_annealing_with_history(
            path, mod_func, temp_func, max_iterations, cooling_param, sample_freq
        )
        
        all_results.append((paths, temps, energies, label))
    
    if n_runs <= 2:
        rows, cols = 1, n_runs
    else:
        rows, cols = 2, (n_runs + 1) // 2
    
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), dpi=100)
    
    if n_runs == 1:
        axs = np.array([[axs]])
    elif n_runs == 2 and rows == 1:
        axs = np.array([axs])
    
    graphs = []
    temp_lines = []
    energy_lines = []
    
    for run_idx, (paths, temps, energies, label) in enumerate(all_results):
        row = run_idx // cols
        col = run_idx % cols
        
        ax_path = axs[row, col]
        ax_path.set_title(f'{label}', fontsize=9)
        ax_path.axis('off')
        
        ax_energy = ax_path.twinx()
        ax_energy.axis('off')
        
        G = nx.Graph()
        graphs.append(G)
        
        temp_line, = ax_energy.plot([], [], 'r-', linewidth=0.5, alpha=0.3)
        energy_line, = ax_energy.plot([], [], 'g-', linewidth=0.5, alpha=0.3)
        
        temp_lines.append(temp_line)
        energy_lines.append(energy_line)
    
    def init():
        for ax in axs.flatten():
            ax.clear()
            ax.axis('off')
        
        for G in graphs:
            G.clear()
        
        for temp_line, energy_line in zip(temp_lines, energy_lines):
            temp_line.set_data([], [])
            energy_line.set_data([], [])
        
        return temp_lines + energy_lines
    
    def update(frame):
        for run_idx, (paths, temps, energies, label) in enumerate(all_results):
            if frame >= len(paths):
                continue
                
            row = run_idx // cols
            col = run_idx % cols
            ax_path = axs[row, col]

            ax_path.clear()
            ax_path.set_title(f'{label} (Iter: {frame * sample_freq})', fontsize=9)
            ax_path.axis('off')
            
            path = paths[frame]
            G = graphs[run_idx]
            G.clear()
            P = list(path)
            for j, edge in enumerate(zip(P, P[1:])):
                G.add_edge(*edge, order=j)
            pos = {(x, y): [x, y] for (x, y) in path}
            edge_color = [order for _, _, order in G.edges(data='order')]
            nx.draw_networkx_nodes(G, pos, node_size=4, node_color='black', ax=ax_path)
            nx.draw_networkx_edges(G, pos, width=0.8, edge_color=edge_color, edge_cmap=plt.cm.winter, ax=ax_path)
            
            plot_idx = min(frame * sample_freq, len(temps) - 1)
            temp_lines[run_idx].set_data(range(plot_idx + 1), temps[:plot_idx + 1])
            energy_lines[run_idx].set_data(range(plot_idx + 1), energies[:plot_idx + 1])
        
        return temp_lines + energy_lines
    
    num_frames = min(len(result[0]) for result in all_results)
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  init_func=init, blit=False, repeat=False)
    
    try:
        print("Attempting to save with pillow writer...")
        ani.save(filename, writer='pillow', fps=fps, dpi=100)
    except Exception as e:
        print(f"Could not save with default writer: {e}")
        try:
            print("Attempting to save with ffmpeg writer...")
            ani.save(filename, writer='ffmpeg', fps=fps, dpi=100)
        except Exception as e2:
            print(f"Could not save with ffmpeg either: {e2}")
            print("Saving as mp4 instead...")
            mp4_filename = filename.replace('.gif', '.mp4')
            ani.save(mp4_filename, writer='ffmpeg', fps=fps, dpi=100)
            print(f"Saved as: {mp4_filename}")
            return mp4_filename
    
    plt.close(fig)
    print(f"Comparison GIF saved as: {filename}")
    return filename
