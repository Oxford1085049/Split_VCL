import torch

# open all the pkl files in the directory to_plot
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

if not os.path.exists('to_plot'):
    raise Exception('No directory to_plot found')

files = os.listdir('to_plot')
files = [f for f in files if f.endswith('.pkl')]

plt.figure(figsize=(10, 7))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'lavender', 'tan', 'maroon', 'navy', 'aquamarine', 'gold', 'coral', 'indigo', 'ivory', 'azure', 'mint', 'rose', 'lemon', 'cerulean', 'apricot', 'lavender', 'peridot', 'russet', 'cobalt', 'mauve', 'pear', 'sapphire', 'celadon', 'vermilion', 'chartreuse', 'amber', 'topaz', 'plum', 'emerald']
for i, f in enumerate(files):
    with open(os.path.join('to_plot', f), 'rb') as file:
        overall,_, _ = pickle.load(file)
    name = f.split('_')[1][4:]
    final_name = 'beta = ' + name[0]+'.'+name[1:]
    
    # plot the data
    plt.plot(overall, label=f'{final_name}', color=colors[i])
plt.xlabel('Task')
plt.ylabel('Overall Accuracy')
plt.legend()
plt.savefig('Beta_influence_simul2_epochs10_zoom.pdf')