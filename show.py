import matplotlib.pyplot as plt
from   matplotlib import colors

import sys
import json
import os
from pathlib import Path
from glob import glob

base_path='/kaggle/input/arc-prize-2024/'

# Loading JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

training_challenges   = load_json(base_path +'arc-agi_training_challenges.json')
training_solutions    = load_json(base_path +'arc-agi_training_solutions.json')

evaluation_challenges = load_json(base_path +'arc-agi_evaluation_challenges.json')
evaluation_solutions  = load_json(base_path +'arc-agi_evaluation_solutions.json')

test_challenges       = load_json(base_path +'arc-agi_test_challenges.json')

def plot_task(task, t):
    """    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app    """

    num_train = len(task['train'])
    num_test  = len(task['test'])

    w=num_train+num_test
    fig, axs  = plt.subplots(2, w, figsize=(3*w ,3*2))
    plt.suptitle(f'Test {t}:', fontsize=20, fontweight='bold', y=1)

    for j in range(num_train):
        plot_one(axs[0, j], j,'train', 'input')
        plot_one(axs[1, j], j,'train', 'output')

    plot_one(axs[0, j+1], 0, 'test', 'input')

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')

    plt.tight_layout()

    plt.show()

def plot_one(ax, i, train_or_test, input_or_output):
    cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)

    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])

    ax.set_title(train_or_test + ' ' + input_or_output)

for t in sys.argv[2:]:
    if sys.argv[1] == "training":
        task=training_challenges[t]
    elif sys.argv[1] == "evaluation":
        task=evaluation_challenges[t]
    else:
        task=test_challenges[t]
    plot_task(task, t)
