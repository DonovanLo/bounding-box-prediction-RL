import numpy as np
import os
from pylab import subplot
import matplotlib.pyplot as plt

def print_outputs_9(orig_img, arr_regionImg,actions,index):
    """
    This method creates plots and saves them as png
    
    Args:
        orig_img (numpy array): original image
        arr_regionImg (list): list of cropped region 
        actions (list): list of actions for each cropped region
        index (int): index for each car processed
    Returns:
        None
    """
    if not os.path.exists('./output'):
        os.makedirs('./output')
    d = {'1':'U','2':'UR','3':'R','4':'DR','5':'D', '6':'DL', '7':'L', '8':'UL', '9':'T'}

    plt.gcf().clear()
    ax1 = subplot(5,5,1)
    ax1.set_title('Original', fontsize=8)
    ax1.axis('off')
    ax1.autoscale_view(False)
    ax1.imshow(orig_img)

    numActions = np.size(arr_regionImg,0)
    for x in range(0,numActions):
        img = arr_regionImg[x]
        ax1 = subplot(5,5,x+2)
        ax1.set_title('a%d - %s' % (x,d[str(actions[x])]), fontsize=8)
        ax1.axis('off')
        ax1.autoscale_view(False)
        ax1.imshow(img)
    fig = ax1.get_figure()
    fig.savefig('./output/test_%d.png' % index)
    plt.gcf().clear()