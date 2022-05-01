'''
    This file contains several functions
'''
import matplotlib.pyplot as plt


def draw_quality(quality, filepath, dataset):
    """

    Args:
        quality (_type_): _description_
    """
    x = [i for i in range(len(quality))]
    plt.plot(x, quality, color='#4169E1', alpha=0.8, linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('quality')
    plt.title(dataset)
    plt.savefig(filepath + dataset +'.png', bbox_inches='tight')