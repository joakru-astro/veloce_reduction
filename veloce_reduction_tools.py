from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from csaps import csaps
from cv2 import connectedComponentsWithStats, merge

def remove_overscan_bias(frame, overscan_range=32):
###     overscan_range - size of gap to remove

    ylen, xlen = frame.shape
    xdiv, ydiv = int(xlen/2), int(ylen/2)

    overscan_mask = np.zeros_like(frame)

    # top left
    q00 = frame[overscan_range:ydiv-overscan_range, overscan_range:xdiv-overscan_range].astype(np.float64)
    q00_overscan_mask = overscan_mask.copy()
    # middle
    q00_overscan_mask[:ydiv,xdiv-overscan_range:xdiv] = 1
    q00_overscan_mask[ydiv-overscan_range:ydiv,:xdiv] = 1
    # edge
    q00_overscan_mask[:ydiv,:overscan_range] = 1
    q00_overscan_mask[:overscan_range,:xdiv] = 1
    q00 -= np.median(frame[q00_overscan_mask == 1])

    # bottom left
    q10 = frame[ydiv+overscan_range:ylen-overscan_range,overscan_range:xdiv-overscan_range].astype(np.float64) 
    q10_overscan_mask = overscan_mask.copy()
    # middle
    q10_overscan_mask[ydiv:,xdiv-overscan_range:xdiv] = 1
    q10_overscan_mask[ydiv:ydiv+overscan_range,:xdiv] = 1
    # edge
    q10_overscan_mask[ydiv:,:overscan_range] = 1
    q10_overscan_mask[ylen-overscan_range:,:xdiv] = 1
    q10 -= np.median(frame[q10_overscan_mask == 1])

    # top right
    q01 = frame[overscan_range:ydiv-overscan_range,xdiv+overscan_range:xlen-overscan_range].astype(np.float64)
    q01_overscan_mask = overscan_mask.copy()
    # middle
    q01_overscan_mask[:ydiv,xdiv:xdiv+overscan_range] = 1
    q01_overscan_mask[ydiv-overscan_range:ydiv,xdiv:] = 1
    # edge
    q01_overscan_mask[:ydiv,xlen-overscan_range:] = 1
    q01_overscan_mask[:overscan_range,xdiv:] = 1
    q01 -= np.median(frame[q01_overscan_mask == 1])

    # bottom right
    q11 = frame[ydiv+overscan_range:ylen-overscan_range,xdiv+overscan_range:xlen-overscan_range].astype(np.float64)
    q11_overscan_mask = overscan_mask.copy()
    # middle
    q11_overscan_mask[ydiv:,xdiv:xdiv+overscan_range] = 1
    q11_overscan_mask[ydiv:ydiv+overscan_range,xdiv:] = 1
    # edge
    q11_overscan_mask[ydiv:,xlen-overscan_range:] = 1
    q11_overscan_mask[ylen-overscan_range:,xdiv:] = 1
    q11 -= np.median(frame[q11_overscan_mask == 1])

    image_substracted_bias = np.concatenate(
        (np.concatenate((q00, q10), axis=0), 
         np.concatenate((q01, q11), axis=0)),
        axis=1)
    image_substracted_bias[image_substracted_bias <= 0] = 0
    
    return image_substracted_bias

def plot_cross_section(frame, n, axis='x'):
    xlen, ylen = frame.shape
    if axis=='x':
        row = n
        x = np.arange(len(frame[row,:]))
        threshold = median_filter(frame[row,:],501) + 1
        plt.plot(x, threshold)
        plt.step(x, frame[row,:])
        
    elif axis=='y':
        col = n
        x = np.arange(len(frame[:,col]))
        threshold = median_filter(frame[:,col],501) + 1
        plt.plot(x, threshold)
        plt.step(x, frame[:col])

def get_binary_mask(frame, axis='x'):
    binary_mask = np.zeros(frame.shape)
    if axis=='x':
        for i in range(len(frame[:,0])):
            threshold = median_filter(frame[i,:],501) + 1
            mask = frame[i,:]>threshold
            binary_mask[i,:][mask] = 1       
    elif axis=='y':
        for i in range(len(frame[0,:])):
            threshold = median_filter(frame[:,i],501) + 1
            mask = frame[:,i]>threshold
            binary_mask[:,i][mask] = 1
    return binary_mask

### old version 
# def get_binary_mask(frame, axis='x'):
#     if axis=='x':
#         binary_mask = np.zeros(frame.shape)
#         for i in range(len(frame[:,0])):
#             median_mean_diff = abs(np.mean(frame[i,:])-np.median(frame[i,:]))
#             noise_std = np.std(frame[i,:][frame[i,:]<np.median(frame[i,:])])
#             if median_mean_diff > 5*noise_std:
#                 threshold = median_filter(frame[i,:],501) + 1
#                 mask = frame[i,:]>threshold
#                 binary_mask[i,:][mask] = 1       
#     elif axis=='y':
#         binary_mask = np.zeros(frame.shape)
#         for i in range(len(frame[0,:])):
#             median_mean_diff = abs(np.mean(frame[:,i])-np.median(frame[:,i]))
#             noise_std = np.std(frame[:,i][frame[:,i]<np.median(frame[:,i])])
#             if median_mean_diff > 5*noise_std:
#                 threshold = median_filter(frame[:,i],501) + 1
#                 mask = frame[:,i]>threshold
#                 binary_mask[:,i][mask] = 1
#     return binary_mask

def get_orders_masks(binarized):
    bin_uint8 = (binarized * 255).astype(np.uint8)
    nb_blobs, im_with_separated_blobs, stats, _ = connectedComponentsWithStats(bin_uint8) # from cv2                                                                           
    sizes = stats[:, -1]
    sizes = sizes[1:] # skip background
    nb_blobs -= 1
    # minimum size of orders to keep (number of pixels).
    sorted_sizes = sorted(sizes)
    min_size = sorted_sizes[np.argmax(np.diff(sorted_sizes))+1] 
    orders = []
    # keep blobs above min_size - hopefuly results in orders
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            order = np.zeros_like(binarized)
            order[im_with_separated_blobs == blob + 1] = 1
            orders.append(order)
    orders = np.array(orders, dtype=np.uint16)
    return orders

def get_traces(frame, orders):
    traces = []
    ylen, xlen = frame.shape
    y = range(ylen)
    for order in orders:
        img = frame.copy()
        img[order == 0] = 0
        fit_x = []
        fit_y = []
        for i in y:
            if np.sum(img[i,:]) != 0:
                fit_y.append(i)
                weighted_average = np.average(np.arange(len(img[i,:])),weights=img[i,:])
                fit_x.append(weighted_average)
        f = np.polyfit(fit_y,fit_x,2)
        f = np.polyval(f,y)
        fit_x = []
        fit_y = []
        fit_width = 20 # 35 catches calibration fibers
        for i in y:
            row = frame[i,:].copy()
            xmin = max(0,int(f[i])-fit_width)
            xmax = max(0,min(int(f[i])+fit_width,xlen))
            row[:xmin] = 0
            row[xmax:] = 0
            if np.sum(row) != 0 and f[i]-fit_width>0 and f[i]+fit_width<xlen:
                weighted_average = np.average(np.arange(len(row)),weights=row)
                fit_x.append(weighted_average)
                fit_y.append(i)
        f = np.polyfit(fit_y,fit_x,5)
        f = np.polyval(f,y)
        traces.append(f)
    traces = np.array(sorted(traces, key=lambda x: x[2000]))
    return traces

def find_summing_range(frame, traces):
    # TO DO, asymetric version
    ylen, xlen = frame.shape
    summing_ranges = []
    y = np.arange(ylen)
    max_width = 30
    for trace in traces:
        summing_range = []
        for i in y:
            row = frame[i,:].copy()
            xmin = max(0,int(trace[i])-max_width)
            xmax = max(0, min(int(trace[i])+max_width,xlen))
            if xmin != xmax:
                row[:xmin] = 0
                row[xmax:] = 0
                mask = row > 0
        #         threshold = np.median(row[(row<np.median(row[mask])) & (row > 0)]) + 3 * np.std(row[(row<np.median(row[mask])) & (row > 0)])
        #         threshold = 3 * np.std(row[(row<np.median(row[mask]))])
                threshold = np.std(row[mask])
                if sum(row > threshold) < 1.8*max_width and sum(row > threshold) > max_width/2:
                    summing_range.append(sum(row > threshold))
        summing_range = int(np.median(summing_range)/2)
        summing_ranges.append(summing_range)
    summing_ranges = np.array(summing_ranges)
    return summing_ranges

def remove_order_background(order, n_pix=5):
    ymax, xmax = order.shape
    y = np.arange(ymax)
    left = np.median(order[:,:n_pix], axis=1)
    right = np.median(order[:,-n_pix:], axis=1)
    sm_left = csaps(y, median_filter(left, 10), y, smooth=1e-8)
    sm_right = csaps(y, median_filter(right, 10), y, smooth=1e-8)

    sm_surface = []
    x= np.arange(xmax)
    for l, r in zip(sm_left, sm_right):
        p = np.polyfit([0,xmax-1], [l, r], 1)
        sm_surface.append(np.polyval(p, x))
    sm_surface = np.array(sm_surface)

    # xs_data = [y, x]
    # i, j = np.meshgrid(*xs_data, indexing='ij')

    # fig = plt.figure(figsize=(7, 4.5))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_facecolor('none')
    # ax.plot_surface(j, i, sm_surface)
    # # ax.plot_surface(j, i, order)
    # ax.view_init(elev=20., azim=50)
    # ax.plot(np.zeros_like(y), y, sm_left)
    # ax.plot(np.ones_like(y)*xmax-1, y, sm_right)
    # plt.show()
    # plt.close()
    
    if np.min(sm_surface) < 0: sm_surface -= np.min(sm_surface)

    order -= sm_surface
    order[order < 0] = 0
    return order

def extract_orders_with_trace(frame, traces, summing_ranges, remove_background=False):
    extracted_orders = []
    extracted_order_imgs = []
    y = range(len(frame[:,0]))
    # xmax = len(frame[0,:])
    if isinstance(summing_ranges[0], np.int64):
        lower_limit = min(traces[0])-summing_ranges[0]-1
    elif len(summing_ranges[0]) == 2:
        lower_limit = min(traces[0])-summing_ranges[0,0]-1
    
    if lower_limit < 0:
        offset = int(-1*np.floor(lower_limit))
        buffer = np.ones((len(y), offset))
        frame = np.concatenate((buffer, frame), axis=1)
    else:
        offset = 0
    for trace, summing_range in zip(traces, summing_ranges):
        if isinstance(summing_range, np.int64):
            lower_range, upper_range = summing_range, summing_range
        elif len(summing_range) == 2:
            lower_range, upper_range = summing_range
        extracted_order_img = np.array([frame[yval, int(xval-lower_range+offset):int(xval+upper_range+1+offset)] for yval, xval in zip(y, trace)])
        if remove_background: extracted_order_img = remove_order_background(extracted_order_img, n_pix=5)
        extracted_order_imgs.append(extracted_order_img)
        extracted_orders.append(np.sum(extracted_order_img, axis=1))
    extracted_orders = np.array(extracted_orders, dtype=np.float64)
    return extracted_orders, extracted_order_imgs

### old version
# def extract_orders(frame, traces, summing_ranges):
#     ylen, xlen = frame.shape
#     y = np.arange(ylen)
#     extracted_orders = []
#     for trace, summing_range in zip(traces, summing_ranges):
#         if isinstance(summing_range, np.int64):
#             lower_range, upper_range = summing_range, summing_range
#         elif len(summing_range) == 2:
#             lower_range, upper_range = summing_range
#         extracted_order = []
#         for i in y:
#             row = frame[i,:].copy()
#             xmin = max(0,int(trace[i])-lower_range)
#             xmax = max(0, min(int(trace[i])+upper_range,xlen))
#             if xmin != xmax:
#                 row[:xmin] = 0
#                 row[xmax:] = 0
#                 extracted_order.append(np.sum(row))
#             else:
#                 extracted_order.append(0)    
#         extracted_orders.append(extracted_order)    
#     extracted_orders = np.array(extracted_orders, dtype=np.uint32)
#     return extracted_orders

def plot_order_cross_section(frame, traces, summing_range, order):
    if isinstance(summing_range, int):
        lower_range, upper_range = summing_range, summing_range
    elif len(summing_range) == 2:
        lower_range, upper_range = summing_range
    
    ylen, xlen = frame.shape

    trace = traces[order].copy()
    y = np.arange(ylen)
    lower_limit = min(trace)-lower_range-5
    upper_limit = max(trace)+upper_range+1

    if upper_limit > xlen:
        offset = int(np.floor(upper_limit))
        buffer = np.ones((len(y), offset))
        frame = np.concatenate((frame, buffer), axis=1)
    if lower_limit < 0:
        offset = int(-1*np.floor(lower_limit))
        # trace += offset
        buffer = np.ones((len(y), offset))
        frame = np.concatenate((buffer, frame), axis=1)
    else:
        offset = 0

    extracted_order = np.array([frame[yval, int(np.round(xval-lower_range-5, 0)+offset):int(np.round(xval+upper_range+6, 0)+offset)] 
                                            for yval, xval in zip(y, trace)])

    x = np.arange(-lower_range-5,upper_range+6)
    plt.step(x,np.median(extracted_order, axis=0), where='mid')
    plt.axline([-lower_range+5,0], [-lower_range+5,1], ls=':',c='lightgray')
    plt.axline([upper_range-5,0], [upper_range-5,1], ls=':',c='lightgray')
    plt.axline([-lower_range-1,0], [-lower_range-1,1], ls=':',c='gray')
    plt.axline([upper_range+1,0], [upper_range+1,1], ls=':',c='gray')

    return extracted_order

# if arm == 'red':
#     trace_data = np.load('veloce_extracted_trace_red.npz')
# elif arm == 'green':
#     trace_data = np.load('veloce_extracted_trace_green.npz')
# else:
#     raise ValueError('Unknown arm. Must be "red" or "green"')
# traces, summing_ranges, blazes = trace_data['traces'], trace_data['summing_ranges'], trace_data['extracted_orders']

# ### calibration fibers traces? they don't match well to collected data 

# ### Order ymin ymax
# ### Coeff0 Coeff1 Coeff2 … Coeffn
# ### X1 X2 X3 … Xi
# ### Y1 Y1 Y3 … Yi

# filename = 'rosso-lc-m65-104-all.trace'

# with open(filename, 'r') as file:
#     lines = [line[:-1] for line in file]
    
# head_list = np.array([line.split() for line in lines[::4]])
# Order, Ymin, Ymax = head_list[:,0], head_list[:,1], head_list[:,2]
# coef_list = np.array([line.split() for line in lines[1::4]], dtype=np.float64)
# X = [np.array(line.split(), dtype=np.float64) for line in lines[2::4]]
# Y = [np.array(line.split(), dtype=np.float64) for line in lines[3::4]]

def load_prefitted_ThAr(arm='red'):

    if arm == 'red':
        filename = '/Users/joachimkruger/Desktop/ModeID/HD20203/Veloce/Wave_guide/vdarc_rosso_230919.dat'
    elif arm == 'green':
        filename = '/Users/joachimkruger/Desktop/ModeID/HD20203/Veloce/Wave_guide/vdarc_verde_230920.dat'
    else:
        raise ValueError('Unknown arm. Must be "red" or "green"')

    with open(filename, 'r') as file:
        lines = [line[:-1] for line in file]
        
    ORDER = np.array([line.split()[-1] for line in lines[::9]], dtype=np.uint8)
    COEFFS = [np.array(line.split()[1:], dtype=np.float64) for line in lines[3::9]]
    MATCH_LAM = [np.array(line.split()[1:], dtype=np.float64) for line in lines[4::9]]
    MATCH_PIX = [np.array(line.split()[1:], dtype=np.float64) for line in lines[5::9]]
    MATCH_LRES = [np.array(line.split()[1:], dtype=np.float64) for line in lines[6::9]]
    GUESS_LAM = [np.array(line.split()[1:], dtype=np.float64) for line in lines[7::9]]
    Y0 = np.array([line.split()[-1] for line in lines[1::9]], dtype=np.uint16)

    return ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0

def calibrate_orders_to_wave(orders, Y0, coefficients):
    x_arr = np.arange(len(orders[0]), dtype=np.float64)-Y0
    WAVE = []
    for i in range(len(orders)):
        wave = np.zeros_like(x_arr, dtype=np.float64)
        for j, coeff in enumerate(coefficients[i]):
            wave += (x_arr**j)*coeff
        WAVE.append(wave)
    WAVE = np.array(WAVE)
    return WAVE

if __name__ == '__main__':
    filename = '24aug30010.fits' # use the flat because of visibility
    spectrum_filename =  os.path.join(os.getcwd(), 'Data', filename)

    image_data = fits.getdata(spectrum_filename)
    # plt.imshow(image_data, cmap='gray', norm="log")
    image_substracted_bias = remove_overscan_bias(image_data, overscan_range=32)
    blured = median_filter(image_substracted_bias, (10,10))
