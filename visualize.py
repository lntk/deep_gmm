import numpy as np
import matplotlib.pyplot as plt
from util import general_utils

def see_activations(activations, names=None, normalize=False, save=None):
    num_activation = len(activations)    
    
    fig, axs = plt.subplots(1, num_activation)
    
    for i in range(num_activation):
        see_activation(activations[i], normalize=normalize, axs=axs, ax_idx=i, name=names[i], show=save is None, save=save)            
    
    if save is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save)
        

def see_activation(x, normalize=False, axs=None, ax_idx=0, name=None, show=True, save=None):
    if axs is None:
        fig, axs = plt.subplots(1, 1)
        
    x = x.cpu().detach().numpy()
    if normalize:
        x = 1 / (1 + np.exp(-x))
    
    x = (x * 255).astype("uint8")
    
    axs[ax_idx].imshow(x, cmap="gray")
    axs[ax_idx].set_xticks([])
    axs[ax_idx].set_yticks([])
    
    if name:
        axs[ax_idx].set_title(name)        
    
    if show:
        plt.show()