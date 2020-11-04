from PIL import Image
import numpy as np
import matplotlib.pylab as plt
from torchvision.datasets import MNIST
plt.ion()

#not using any vectorized numpy operations

GAMMA = 50
gamma = 15

#img = np.array(Image.open('lsp.jpg').convert('L'))
ds = MNIST(root='./mnist', download=True)

img = ds.data[16].numpy()

def get_center_filter(sig1, sig2):
    R = 3
    S = 2*R + 1 #filter dimensions
    
    f = np.zeros((S,S))

    for i in range(S):
        for j in range(S): #can use symmetry of f but no optims here
            d = (i-R)**2 + (j-R)**2

            m1 = 1/(2 * np.pi * sig1**2)
            m2 = 1/(2 * np.pi * sig2**2)

            f[i][j] = m1 * np.exp(-d/(2*sig1**2)) - m2 * np.exp(-d/(2*sig2**2))

    #f = 10 * f

    return f

def plot_filters(on_center_filter, off_center_filters):
    plt.figure()
    colors = plt.pcolor(on_center_filter)
    plt.colorbar(colors)

    plt.figure()
    colors = plt.pcolor(off_center_filter)
    plt.colorbar(colors)

'''
def convolve(img, f):
    H, W = img.shape
    S1, S2 = f.shape #symmetric but keeping it general

    res = np.zeros((H, W)) #add padding on both sides
    
    for i in range(H):
        for j in range(W):
            
            for f1 in range(S1):
                for f2 in range(S2):
                    if 0 <= i+f1-S1 < H and 0 <= j+f2-S2 < W:
                        res[i][j] += img[i + f1 - S1][j + f2 - S2] * f[f1][f2]

    return res
'''

def create_spike_train_simple(img):
    H, W = img.squeeze().shape

    T = 10

    delays = np.zeros(shape=(H, W))

    for i in range(H):
        for j in range(W):
            if img[i][j] > GAMMA:
                delays[i][j] = 1000./img[i][j]
            else:
                delays[i][j] = -1

    delays = np.floor(delays).astype(int)

    #result = np.zeros(shape=(H,W,T))
    result = []
    for t in range(T):
        #temp = delays.copy()
        #temp[temp!=t] = 0
        #temp[temp==t] = 1
        temp = np.zeros_like(delays)

        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if delays[i][j]==t:
                    temp[i][j] = 1
                    
        #result[:,:,t] = temp
        result.append(temp)

    return result, delays

def create_spike_train(img):
    H, W, _ = img.shape

    #thresholds = [np.inf] + [1000./i for i in range(1, 11)] + [1000./20]
    thresholds = [1000./i for i in range(1, 11)] + [1000./20]
    #max act is ~982 (positive filter elements * 255)

    img_list = []
    for i in range(len(thresholds)-1):
        curr_t = thresholds[i]
        next_t = thresholds[i+1]

        #print(curr_t, next_t)

        #very memory-inefficient way of doing this
        img_t = img.copy() 
        img_t[(img_t >= curr_t) | (img_t < next_t)] = 0
        img_t[img_t > 50] = 1

        #print(img_t.sum())
        img_list.append(img_t)

    return img_list

def plot_spike_train(img_list):
    plt.figure()
    for i in range(len(img_list)):
        img = np.where(img_list[i].flatten()==1)[0]
        try:
            if len(img) > 0:
                plt.plot([i]*len(img), img, 'p', c='b')
        except:
            print(img)

on_center_filter = get_center_filter(1, 2)
off_center_filter = get_center_filter(2, 1)

def initialize_filters_l1_to_l2(size=5, channels=2, n=30):
    return np.random.normal(0.8, 0.01, size=(size, size, channels, n))

def convolve(img, f):
    if len(img.shape)==2: #1 channel
        img = np.expand_dims(img, axis=[0,3])
    
    if len(f.shape)==2: #1 channel, 1 filter
        f = np.expand_dims(f, axis=[2,3])

    T, H, W, C = img.shape #(height, width, channels)
    #channels -> (RGB) -> 3
    #channels -> 0/1 -> off DoG/on DoG

    S1, S2, D, N = f.shape #(height, width, depth = channels, number of filters)

    assert(C==D)

    res = np.zeros((H, W, N))
    
    for n in range(N): #loop over filters
        
        #loop over image dimensions
        for i in range(H):
            for j in range(W):

                #loop over filter elements
                for c in range(C):
                    for f1 in range(S1):
                        for f2 in range(S2):
                            #res[i][j][n] += img[][][] * f[][][][n] abstract structure
                            if 0 <= i+f1-S1 < H and 0 <= j+f2-S2 < W:
                                res[i][j][n] += img[i + f1 - S1][j + f2 - S2][c] * f[f1][f2][c][n]

    return res


def seq(img):
    #get on-center, off-center filters
    on_center_filter = get_center_filter(1, 2)
    on_center_filter -= np.mean(on_center_filter)
    on_center_filter /= np.max(on_center_filter)

    #off_center_filter = get_center_filter(2, 1)
    off_center_filter = -on_center_filter

    #convolution with on/off filters
    img_on = convolve(img, on_center_filter)
    img_off = convolve(img, off_center_filter)

    #rescale before computing spike trains
    #img_on = rescale(img_on.squeeze())
    #img_off = rescale(img_off.squeeze())

    #create spike trains
    spike_train_on = create_spike_train(img_on)
    spike_train_off = create_spike_train(img_off)

    #spike trains should be of equal length (in the paper, this is done by binning time)
    assert(len(spike_train_on)==len(spike_train_off))

    #create stacked on/off images for each time - should remove any time that has no spikes
    spike_train_stacked = np.array([np.stack([on,off], axis=2) for (on,off) in zip(spike_train_on, spike_train_off)]).squeeze()

    #initialize filters randomly for L1 -> L2
    w = initialize_filters_l1_to_l2(size=5, channels=2, n=30)

    #convolve filters with every time-step - can be parallelized trivially
    l2_act = np.array([convolve(s.squeeze(), w) for s in spike_train_stacked])

    l2_act_cumsum = np.cumsum(l2_act, axis=0)

    l2_act_last = (l2_act_cumsum[-1] > gamma).astype(int)

    #lateral inhibition


    return l2_act, l2_act_cumsum, l2_act_last


    #l2_act = rescale(l2_act)

    #cumulative sums to get V_L2

    #compare V_L2 to threshold gamma

    #figure 14

    #lateral inhibition

    #figure 15