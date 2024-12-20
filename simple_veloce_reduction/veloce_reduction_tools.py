from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, label, find_objects
from csaps import csaps
# from cv2 import connectedComponentsWithStats, merge

from . import veloce_path

def remove_overscan_bias(frame, hdr, overscan_range=32, amplifier_mode=4):
    """
    Removes the overscan bias from an image frame by subtracting the median of the overscan regions.

    The function divides the image into four quadrants (top left, bottom left, top right, bottom right),
    calculates the median value of the overscan regions for each quadrant, and subtracts this value from the quadrant.
    The overscan regions are defined by the `overscan_range` parameter and are located at the edges and middle of the image.
    After processing, the quadrants are recombined, and any negative values are set to zero.

    Parameters:
    - frame (numpy.ndarray): The input image frame as a 2D numpy array.
    - overscan_range (int, optional): The width of the overscan region to be removed. Defaults to 32 pixels.

    Returns:
    - numpy.ndarray: The image frame with overscan and bias removed, with negative values set to zero.

    Note:
    - The function assumes the image is divided into quadrants symmetrically.
    - The input frame is modified in place for each quadrant before recombination.
    """
    ylen, xlen = frame.shape
    xdiv, ydiv = int(xlen/2), int(ylen/2)

    # overscan_mask = np.zeros_like(frame)
    if amplifier_mode == 4:
        # top left - Q1
        q1 = frame[overscan_range:ydiv-overscan_range, overscan_range:xdiv-overscan_range].copy().astype(np.float64)
        q1_overscan_mask = np.zeros_like(frame)
        # middle
        q1_overscan_mask[:ydiv,xdiv-overscan_range:xdiv] = 1
        q1_overscan_mask[ydiv-overscan_range:ydiv,:xdiv] = 1
        # edge
        q1_overscan_mask[:ydiv,:overscan_range] = 1
        q1_overscan_mask[:overscan_range,:xdiv] = 1
        q1 -= np.median(frame[q1_overscan_mask == 1])
        q1[q1 < 0] = 0
        q1_gain = float(hdr['DETA1GN'])
        print(f'Gain for quadrant 1: {q1_gain}')
        q1 /= q1_gain

        # bottom left - Q2
        q2 = frame[ydiv+overscan_range:ylen-overscan_range,overscan_range:xdiv-overscan_range].copy().astype(np.float64) 
        q2_overscan_mask = np.zeros_like(frame)
        # middle
        q2_overscan_mask[ydiv:,xdiv-overscan_range:xdiv] = 1
        q2_overscan_mask[ydiv:ydiv+overscan_range,:xdiv] = 1
        # edge
        q2_overscan_mask[ydiv:,:overscan_range] = 1
        q2_overscan_mask[ylen-overscan_range:,:xdiv] = 1
        q2 -= np.median(frame[q2_overscan_mask == 1])
        q2[q2 < 0] = 0
        q2_gain = float(hdr['DETA2GN'])
        print(f'Gain for quadrant 2: {q2_gain}')
        q2 /= q2_gain

        # bottom right - Q3
        q3 = frame[ydiv+overscan_range:ylen-overscan_range,xdiv+overscan_range:xlen-overscan_range].copy().astype(np.float64)
        q3_overscan_mask = np.zeros_like(frame)
        # middle
        q3_overscan_mask[ydiv:,xdiv:xdiv+overscan_range] = 1
        q3_overscan_mask[ydiv:ydiv+overscan_range,xdiv:] = 1
        # edge
        q3_overscan_mask[ydiv:,xlen-overscan_range:] = 1
        q3_overscan_mask[ylen-overscan_range:,xdiv:] = 1
        q3 -= np.median(frame[q3_overscan_mask == 1])
        q3[q3 < 0] = 0
        q3_gain = float(hdr['DETA3GN'])
        print(f'Gain for quadrant 3: {q3_gain}')
        q3 /= q3_gain

        # top right - Q4
        q4 = frame[overscan_range:ydiv-overscan_range,xdiv+overscan_range:xlen-overscan_range].copy().astype(np.float64)
        q4_overscan_mask = np.zeros_like(frame)
        # middle
        q4_overscan_mask[:ydiv,xdiv:xdiv+overscan_range] = 1
        q4_overscan_mask[ydiv-overscan_range:ydiv,xdiv:] = 1
        # edge
        q4_overscan_mask[:ydiv,xlen-overscan_range:] = 1
        q4_overscan_mask[:overscan_range,xdiv:] = 1
        q4 -= np.median(frame[q4_overscan_mask == 1])
        q4[q4 < 0] = 0
        q4_gain = float(hdr['DETA4GN'])
        print(f'Gain for quadrant 4: {q4_gain}')
        q4 /= q4_gain

        image_substracted_bias = np.concatenate(
            (np.concatenate((q1, q2), axis=0), 
            np.concatenate((q4, q3), axis=0)),
            axis=1)
    elif amplifier_mode == 2:
        gain_scale = 0.964 # right/left gain ratio
        # left - H1
        h1 = frame[overscan_range:ylen-overscan_range, overscan_range:xdiv-overscan_range].copy().astype(np.float64)
        h1_overscan_mask = np.zeros_like(frame)
        # middle
        h1_overscan_mask[:,xdiv-overscan_range:xdiv] = 1
        # edge
        h1_overscan_mask[:,:overscan_range] = 1
        h1_overscan_mask[:overscan_range,:xdiv] = 1
        h1_overscan_mask[ylen-overscan_range:,:xdiv] = 1
        h1 -= np.median(frame[h1_overscan_mask == 1])
        h1[h1 < 0] = 0
        h1_gain = float(hdr['DETA1GN'])
        print(f'Gain for half 1: {h1_gain}')
        h1 /= h1_gain

        # right - H2
        h2 = frame[overscan_range:ylen-overscan_range,xdiv+overscan_range:xlen-overscan_range].copy().astype(np.float64)
        h2_overscan_mask = np.zeros_like(frame)
        # middle
        h2_overscan_mask[:,xdiv:xdiv+overscan_range] = 1
        # edge
        h2_overscan_mask[:,:overscan_range] = 1
        h2_overscan_mask[ylen-overscan_range:,xdiv:] = 1
        h2_overscan_mask[ylen-overscan_range:,xdiv:] = 1
        h2 -= np.median(frame[h2_overscan_mask == 1])
        h2[h2 < 0] = 0
        # h2_gain = float(hdr['DETA2GN'])
        h2_gain = h1_gain * gain_scale
        print(f'Gain for half 2: {h2_gain}')
        h2 /= h2_gain

        image_substracted_bias = np.concatenate((h1, h2), axis=1)
    else:
        raise ValueError("Invalid amplifier mode. Amplifier mode must be 2 or 4.")
    
    return image_substracted_bias

def plot_cross_section(frame, n, axis='x'):
    """
    Plots a cross-section of an image frame along a specified axis at a given position.

    This function extracts a row or column from the image frame, applies a median filter to it,
    and then plots both the filtered signal (as a threshold line) and the original signal.
    The median filter is used to find the threshold for signal.

    Parameters:
    - frame (numpy.ndarray): The input image frame as a 2D numpy array.
    - n (int): The position (row or column index) at which to extract the cross-section.
    - axis (str, optional): The axis along which to extract the cross-section. 'x' for a row,
      'y' for a column. Defaults to 'x'.

    Note:
    - The median filter size is hardcoded to 501, which may not be suitable for all image sizes.
      Adjust accordingly.
    """
    fig, ax = plt.subplots()
    if axis=='x':
        row = n
        x = np.arange(len(frame[row,:]))
        threshold = median_filter(frame[row,:],501,mode='nearest') + 1
        ax.plot(x, threshold)
        ax.step(x, frame[row,:])
        ax.set_title(f'Cross-section at row {n}')
        
    elif axis=='y':
        col = n
        x = np.arange(len(frame[:,col]))
        threshold = median_filter(frame[:,col],501,mode='nearest') + 1
        ax.plot(x, threshold)
        ax.step(x, frame[:,col])
        ax.set_title(f'Cross-section at column {n}')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Counts')
    ax.set_xlim(0, len(frame[0,:]))
    ax.set_ylim(0,)
    return fig, ax

def get_binary_mask(frame, arm, axis='x'):
    """
    Generates a binary mask for an image frame based on a threshold determined by a median filter.

    This function applies a median filter along the specified axis of the image frame and adds a constant value
    of 1 to the result to determine a threshold. For each pixel, if its value is greater than this threshold,
    it is marked as 1 (True) in the binary mask; otherwise, it is marked as 0 (False). This process is repeated
    for each row (if axis='x') or each column (if axis='y') of the frame.

    Parameters:
    - frame (numpy.ndarray): The input image frame as a 2D numpy array.
    - axis (str, optional): The axis along which the median filter is applied. 'x' for rows, 'y' for columns.
      Defaults to 'x'.

    Returns:
    - numpy.ndarray: A binary mask of the same shape as `frame`, where each pixel is marked as 1 if its value
      is greater than the threshold determined by the median filter and the constant value, and 0 otherwise.

    Note:
    - The median filter size is set to 501, which may need adjustment based on the image size and the desired
      level of detail in the binary mask.
    """
    if arm == 'red':
      if axis == 'x':
          # Apply median filter along rows
          filtered_frame = median_filter(frame, size=(1, 501), mode='nearest')
      elif axis == 'y':
          # Apply median filter along columns
          filtered_frame = median_filter(frame, size=(501, 1), mode='nearest')
      else:
          raise ValueError("Axis must be 'x' or 'y'.")
    elif arm == 'green':
      if axis == 'x':
          # Apply median filter along rows
          filtered_frame = median_filter(frame, size=(1, 501), mode='nearest')
      elif axis == 'y':
          # Apply median filter along columns
          filtered_frame = median_filter(frame, size=(501, 1), mode='nearest')
      else:
          raise ValueError("Axis must be 'x' or 'y'.")
    elif arm == 'blue':
      if axis == 'x':
          # Apply median filter along rows
          filtered_frame = median_filter(frame, size=(1, 151), mode='nearest')
      elif axis == 'y':
          # Apply median filter along columns
          filtered_frame = median_filter(frame, size=(151, 1), mode='nearest')
      else:
          raise ValueError("Axis must be 'x' or 'y'.")
    else:
      raise ValueError("Invalid arm. Arm must be 'red', 'green', or 'blue'.")

    # Create binary mask
    binary_mask = frame > (filtered_frame + 1)
    binary_mask[:100,:] = 0
    binary_mask[-100:,:] = 0
    
    return binary_mask.astype(np.uint8)

### old version 
# def get_binary_mask(frame, axis='x'):
#     """
#     Generates a binary mask for an image frame based on a threshold determined by a median filter.

#     This function applies a median filter along the specified axis of the image frame and adds a constant value
#     of 1 to the result to determine a threshold. For each pixel, if its value is greater than this threshold,
#     it is marked as 1 (True) in the binary mask; otherwise, it is marked as 0 (False). This process is repeated
#     for each row (if axis='x') or each column (if axis='y') of the frame.

#     Parameters:
#     - frame (numpy.ndarray): The input image frame as a 2D numpy array.
#     - axis (str, optional): The axis along which the median filter is applied. 'x' for rows, 'y' for columns.
#       Defaults to 'x'.

#     Returns:
#     - numpy.ndarray: A binary mask of the same shape as `frame`, where each pixel is marked as 1 if its value
#       is greater than the threshold determined by the median filter and the constant value, and 0 otherwise.

#     Note:
#     - The median filter size is set to 501, which may need adjustment based on the image size and the desired
#       level of detail in the binary mask.
#     """
#     binary_mask = np.zeros(frame.shape)
#     if axis=='x':
#         for i in range(len(frame[:,0])):
#             threshold = median_filter(frame[i,:],501) + 1
#             mask = frame[i,:]>threshold
#             binary_mask[i,:][mask] = 1       
#     elif axis=='y':
#         for i in range(len(frame[0,:])):
#             threshold = median_filter(frame[:,i],501) + 1
#             mask = frame[:,i]>threshold
#             binary_mask[:,i][mask] = 1
#     return binary_mask

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
    """
    Identifies and extracts spectral orders from a binarized image of a spectrum.

    This function processes a binarized image to identify connected components (blobs) representing potential
    spectral orders. It uses connected component analysis to separate and label different blobs, and then filters
    these blobs based on a size threshold to isolate the spectral orders. The size threshold is dynamically determined
    by finding the largest gap in the sorted sizes of the blobs, assuming this gap differentiates between noise and
    actual orders.

    Parameters:
    - binarized (numpy.ndarray): A binary image (2D numpy array) where pixels belonging to spectral orders
      are marked as 1 (True) and others as 0 (False).

    Returns:
    - numpy.ndarray: A 3D numpy array where each 2D slice represents a binary mask for an individual spectral
      order. The dtype is np.uint16 to accommodate potentially large images while saving memory compared to
      the default integer size.

    Note:
    - The function assumes that the largest size gap in the sorted list of blob sizes effectively separates
      noise from actual spectral orders. This heuristic may need adjustment for images with different
      characteristics or noise levels.
    """
    # Label connected components
    labeled_array, num_features = label(binarized)
    
    # Find objects (bounding boxes) for each labeled component
    objects = find_objects(labeled_array)
    
    # Calculate sizes of each component
    sizes = [np.sum(labeled_array[obj] > 0) for obj in objects]
    
    # Determine size threshold
    sorted_sizes = sorted(sizes)
    min_size = sorted_sizes[np.argmax(np.diff(sorted_sizes)) + 1]
    
    # Filter components based on size threshold
    orders = []
    for i, obj in enumerate(objects):
        if sizes[i] >= min_size:
            mask = np.zeros_like(binarized, dtype=np.uint16)
            mask[obj] = labeled_array[obj] == (i + 1)
            orders.append(mask)
    
    orders = np.array(orders, dtype=np.uint16)
    return orders

### old version
# def get_orders_masks(binarized):
#     """
#     Identifies and extracts spectral orders from a binarized image of a spectrum.

#     This function processes a binarized image to identify connected components (blobs) representing potential
#     spectral orders. It converts the binary mask to an 8-bit format, uses connected component analysis to
#     separate and label different blobs, and then filters these blobs based on a size threshold to isolate
#     the spectral orders. The size threshold is dynamically determined by finding the largest gap in the sorted
#     sizes of the blobs, assuming this gap differentiates between noise and actual orders.

#     Parameters:
#     - binarized (numpy.ndarray): A binary image (2D numpy array) where pixels belonging to spectral orders
#       are marked as 1 (True) and others as 0 (False).

#     Returns:
#     - numpy.ndarray: A 3D numpy array where each 2D slice represents a binary mask for an individual spectral
#       order. The dtype is np.uint16 to accommodate potentially large images while saving memory compared to
#       the default integer size.

#     Note:
#     - The function assumes that the largest size gap in the sorted list of blob sizes effectively separates
#       noise from actual spectral orders. This heuristic may need adjustment for images with different
#       characteristics or noise levels.
#     """
    
#     bin_uint8 = (binarized * 255).astype(np.uint8)

#     nb_blobs, im_with_separated_blobs, stats, _ = connectedComponentsWithStats(bin_uint8) # from cv2                                                                           
#     sizes = stats[:, -1]
#     sizes = sizes[1:] # skip background
#     nb_blobs -= 1
    
#     # minimum size of orders to keep (number of pixels).
#     sorted_sizes = sorted(sizes)
#     min_size = sorted_sizes[np.argmax(np.diff(sorted_sizes))+1] 
#     orders = []
    
#     # keep blobs above min_size - hopefuly results in orders
#     for blob in range(nb_blobs):
#         if sizes[blob] >= min_size:
#             order = np.zeros_like(binarized)
#             order[im_with_separated_blobs == blob + 1] = 1
#             orders.append(order)
#     orders = np.array(orders, dtype=np.uint16)

#     return orders

def get_traces(frame, orders):
    """
    Extracts the polynomial fits for spectral order traces from an astronomical image.

    This function processes an input image frame and a set of binary masks corresponding to spectral orders
    to extract and fit polynomial curves that trace the center of each spectral order across the image.
    The process involves two main steps for each order:
    1. Initial rough centerline extraction and quadratic fitting.
    2. Refined extraction with a fixed width around the initial fit and polynomial fitting of degree 5.

    Parameters:
    - frame (numpy.ndarray): The input image frame as a 2D numpy array of shape (ylen, xlen), where ylen and
      xlen are the dimensions of the image.
    - orders (list of numpy.ndarray): A list of binary masks for each spectral order. Each mask is a 2D numpy
      array of the same shape as `frame`, where pixels belonging to the order are marked as 1.

    Returns:
    - numpy.ndarray: An array of polynomial coefficients for each order. Each element in the array is a
      1D numpy array of polynomial coefficients that describe the trace of a spectral order across the image.
      The traces are sorted based on their position at a fixed y-coordinate (2000 in this implementation).

    Note:
    - The function assumes that the input frame and orders are preprocessed and that the orders are correctly
      isolated in the binary masks.
    - The fit_width parameter (set to 30) determines the horizontal range considered around the initial fit
      for the refined fitting process. This width may need adjustment based on the specific characteristics
      of the image and the spectral orders.
    """
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
        fit_width = 35 # 35 catches calibration fibers
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

def refit_traces(frame, trace_x, trace_y):
    """

    """
    traces = []
    ylen, xlen = frame.shape
    full_y = range(ylen)
    for order in range(len(trace_x)):
        fit_x, fit_y = [], []
        fit_width = 50 # 35 catches calibration fibers
        for y, x in zip(trace_y[order], trace_x[order]):
            row = frame[int(y),:].copy()
            xmin = max(0,int(x)-fit_width)
            xmax = max(0,min(int(x)+fit_width,xlen))
            row[:xmin] = 0
            row[xmax:] = 0
            if np.sum(row) != 0 and x-fit_width>0 and x+fit_width<xlen:
                weighted_average = np.average(np.arange(len(row)),weights=row)
                fit_x.append(weighted_average)
                fit_y.append(y)
        f = np.polyfit(fit_y,fit_x,2)
        f = np.polyval(f,full_y)
        for y in full_y:
            row = frame[y,:].copy()
            xmin = max(0,int(f[y]-fit_width))
            xmax = max(0,min(int(f[y])+fit_width,xlen))
            row[:xmin] = 0
            row[xmax:] = 0
            if np.sum(row) != 0 and f[y]-fit_width>0 and f[y]+fit_width<xlen:
                weighted_average = np.average(np.arange(len(row)),weights=row)
                fit_x.append(weighted_average)
                fit_y.append(y)
        f = np.polyfit(fit_y,fit_x,5)
        f = np.polyval(f,full_y)
        traces.append(f)
    traces = np.array(sorted(traces, key=lambda x: x[2000]))
    return traces

# def find_summing_range(frame, traces):
#     # TO DO, asymetric version
#     """
#     Calculates the optimal summing range for each spectral trace.

#     This function iterates over each trace in a given set of spectral traces, determining the optimal
#     summing range for each. The summing range is defined as the width around the trace where the signal
#     is significantly above the background noise, which is determined by a threshold based on the standard
#     deviation of the signal within a maximum width from the trace center. The function aims to dynamically
#     adjust the summing range for each trace based on the local signal characteristics.

#     Parameters:
#     - frame (numpy.ndarray): The input image frame as a 2D numpy array of shape (ylen, xlen), where ylen and
#       xlen are the dimensions of the image.
#     - traces (numpy.ndarray): An array of polynomial coefficients for each spectral order. Each element in
#       the array is a 1D numpy array of polynomial coefficients that describe the trace of a spectral order
#       across the image.

#     Returns:
#     - numpy.ndarray: An array of integers, each representing the calculated summing range for the corresponding
#       spectral trace in `traces`.

#     Note:
#     - The function currently implements a symmetric summing range calculation, with a TODO note for developing
#       an asymmetric version in the future.
#     - The maximum width considered for summing is hardcoded to 30 pixels, but this could be adjusted based on
#       the specific characteristics of the image and the spectral orders.
#     - The threshold for determining significant signal is currently set as the standard deviation of the signal
#       within the maximum width from the trace center, but alternative thresholding methods are commented out.
#     """


#     ylen, xlen = frame.shape
#     summing_ranges = []
#     y = np.arange(ylen)
#     max_width = 30
#     for trace in traces:
#         summing_range = []
#         for i in y:
#             row = frame[i,:].copy()
#             xmin = max(0,int(trace[i])-max_width)
#             xmax = max(0, min(int(trace[i])+max_width,xlen))
#             if xmin != xmax:
#                 row[:xmin] = 0
#                 row[xmax:] = 0
#                 mask = row > 0
#         #         threshold = np.median(row[(row<np.median(row[mask])) & (row > 0)]) + 3 * np.std(row[(row<np.median(row[mask])) & (row > 0)])
#         #         threshold = 3 * np.std(row[(row<np.median(row[mask]))])
#                 threshold = np.std(row[mask])
#                 if sum(row > threshold) < 1.8*max_width and sum(row > threshold) > max_width/2:
#                     summing_range.append(sum(row > threshold))
#         summing_range = int(np.median(summing_range)/2)
#         summing_ranges.append(summing_range)
#     summing_ranges = np.array(summing_ranges)
#     return summing_ranges

def remove_order_background(order, n_pix=5):
    """
    Removes the background from a spectral order by fitting and subtracting a smooth surface.

    This function processes a 2D array representing a spectral order and removes the background by fitting
    a smooth surface to the median values at the left and right edges of the order. The fitted surface is
    then subtracted from the order, and any negative values are set to zero.

    Parameters:
    - order (numpy.ndarray): The input 2D array representing the spectral order.
    - n_pix (int, optional): The number of pixels at the edges to use for fitting the background. Defaults to 5.

    Returns:
    - numpy.ndarray: The background-subtracted spectral order, with negative values set to zero.

    Note:
    - The function uses cubic smoothing splines (csaps) to fit the median values at the edges.
    - A linear polynomial is fitted between the smoothed left and right edge values to create the background surface.
    - The function assumes that the input order is a 2D array with the shape (ymax, xmax).
    """
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
    """
    Extracts spectral orders from an astronomical image frame based on provided trace positions and summing ranges.

    This function processes an input image frame alongside trace positions and summing ranges for each spectral order
    to extract the spectral data. Optionally, it can also remove the background signal from each extracted spectral
    order. The function supports both symmetric and asymmetric summing ranges and can adjust for cases where the
    extraction would otherwise extend beyond the image boundaries by adding a buffer.

    Parameters:
    - frame (numpy.ndarray): The input image frame as a 2D numpy array of shape (ylen, xlen), where ylen and
      xlen are the dimensions of the image.
    - traces (list of numpy.ndarray): A list of 1D numpy arrays, each representing the x-coordinate positions
      of a spectral order trace across the y-axis of the frame.
    - summing_ranges (list of int or tuple): A list where each element specifies the summing range for the
      corresponding trace in `traces`. An element can be an integer (for symmetric summing) or a tuple of two
      integers (for asymmetric summing).
    - remove_background (bool or int): If False, background removal is not performed. If an integer, it specifies
      the number of pixels to use for background estimation and removal.

    Returns:
    - tuple: A tuple containing two elements:
        - numpy.ndarray: An array of extracted spectral orders, where each row corresponds to a spectral order
          and columns represent the summed signal across the specified summing range.
        - list of numpy.ndarray: A list of 2D numpy arrays, each representing the extracted image of a spectral
          order before summing. Useful for diagnostics or further processing.

    Note:
    - If the calculated lower limit for extraction is negative (i.e., would extend beyond the image boundary),
      a buffer is added to the frame to allow for extraction without losing data.
    - The function dynamically adjusts the summing range if `remove_background` is enabled, to ensure that
      background estimation does not interfere with the spectral signal.
    """
    extracted_orders = []
    extracted_order_imgs = []
    ylen, xlen = frame.shape
    y = np.arange(ylen)
    if isinstance(summing_ranges[0], np.int64):
        lower_limit = min(traces[0])-summing_ranges[0]-1
        upper_limit = max(traces[-1])+summing_ranges[1]+1
    elif len(summing_ranges[0]) == 2:
        lower_limit = min(traces[0])-summing_ranges[0,0]-1
        upper_limit = max(traces[-1])+summing_ranges[-1,1]+1
    if remove_background: lower_limit-=remove_background
    
    if upper_limit > xlen:
        offset = int(np.ceil(upper_limit))
        buffer = np.zeros((len(y), offset))
        frame = np.concatenate((frame, buffer), axis=1)
    if lower_limit < 0:
        offset = int(-1*np.floor(lower_limit))
        buffer = np.zeros((len(y), offset))
        frame = np.concatenate((buffer, frame), axis=1)
    else:
        offset = 0
    for trace, summing_range in zip(traces, summing_ranges):
        if isinstance(summing_range, np.int64):
            lower_range, upper_range = summing_range, summing_range
        elif len(summing_range) == 2:
            lower_range, upper_range = summing_range
        if remove_background:
            lower_range += remove_background
            upper_range += remove_background
        extracted_order_img = np.array([frame[yval, int(xval-lower_range+offset):int(xval+upper_range+1+offset)] for yval, xval in zip(y, trace)])
        if remove_background: extracted_order_img = remove_order_background(extracted_order_img, n_pix=remove_background)
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

def plot_order_cross_section(frame, traces, summing_range, order, plot_type='median', margin=[10,10]):
    """
    Plots a cross-section of a spectral order from a 2D frame.

    This function extracts a cross-section of a specified spectral order from an echelle
    and plots it using a specified method (median, mean, sum, or a specific row). The cross-section
    is defined by a summing range around the trace of the order, adjusted by a margin for plotting.

    Parameters:
    - frame (numpy.ndarray): A 2D numpy array representing the CCD frame from which to extract the order.
    - traces (list of numpy.ndarray): A list of 1D numpy arrays, each representing the y-coordinate of the trace
      of a spectral order across the x-axis of the frame.
    - summing_range (int or list of int): The range around the trace to sum over. If an int, the same range is used
      on both sides of the trace. If a list of two ints, the first and second values are used as the lower and
      upper ranges, respectively.
    - order (int): The index of the order to plot, corresponding to the position in the `traces` list.
    - plot_type (str or int): The method to use for plotting the cross-section. Can be 'median', 'mean', 'sum',
      or an int specifying a specific row to plot.
    - margin (int, optional): An additional margin to add to both sides of the summing range. Defaults to 10.

    Returns:
    - numpy.ndarray: A 2D numpy array representing the extracted cross-section of the order.

    Raises:
    - TypeError: If `summing_range` is neither an int nor a list of two ints.
    - ValueError: If `plot_type` is not one of 'median', 'mean', 'sum', or an int.

    Note:
    - The function adjusts the frame if the calculated limits of the cross-section exceed the frame's boundaries,
      by padding the frame with zeros. This may affect the plotted values near the edges.
    - The function plots the cross-section using matplotlib's `step` function, with vertical lines indicating
      the boundaries of the summing range.
    """
    
    lower_margin, upper_margin = margin[0], margin[1]
    if isinstance(summing_range, int):
        lower_range, upper_range = summing_range, summing_range+1
    elif len(summing_range) == 2:
        lower_range, upper_range = summing_range
        # upper_range += 1
    else:
        raise TypeError("Incorrect summing range, must be int or 2-element list.")
    
    ylen, xlen = frame.shape

    trace = traces[order].copy()
    y = np.arange(ylen)
    lower_limit = min(trace)-lower_range-lower_margin
    upper_limit = max(trace)+upper_range+upper_margin

    if upper_limit > xlen:
        offset = int(np.ceil(upper_limit))
        buffer = np.zeros((len(y), offset))
        frame = np.concatenate((frame, buffer), axis=1)
    if lower_limit < 0:
        offset = int(-1*np.floor(lower_limit))
        # trace += offset
        buffer = np.zeros((len(y), offset))
        frame = np.concatenate((buffer, frame), axis=1)
    else:
        offset = 0

    extracted_order_img = np.array([frame[yval, int(np.round(xval-lower_range-lower_margin, 0)+offset):int(np.round(xval+upper_range+upper_margin, 0)+offset)] 
                                            for yval, xval in zip(y, trace)])

    # x = np.arange(-lower_range-lower_margin,upper_range+upper_margin)
    x = np.arange(-lower_range-lower_margin,upper_range+upper_margin)
    if plot_type == 'median':
        plt.step(x,np.median(extracted_order_img, axis=0), where='mid')
    elif plot_type == 'mean':
        plt.step(x,np.mean(extracted_order_img, axis=0), where='mid')
    elif plot_type == 'sum':
        plt.step(x,np.sum(extracted_order_img, axis=0), where='mid')
    elif isinstance(plot_type, int) and plot_type < ylen:
        plt.step(x,extracted_order_img[plot_type,:], where='mid')
    else:
        raise ValueError(f"Incorrect plot type, must be 'median', 'mean', 'sum' or row number [0,{ylen}].")
    plt.axvline(x=-lower_range-1, ls=':',c='gray')
    plt.axvline(x=upper_range, ls=':',c='gray')

    if isinstance(plot_type, int):
        plt.title(f"Order {order}, row {plot_type}")
    else:
        plt.title(f"Order {order}, {plot_type}")
    plt.xlabel("Pixel")
    plt.ylabel("Counts")

    return extracted_order_img

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

def load_prefitted_wavecalib_trace(arm='red', calib_type='Th', trace_path=None, filename=None):
  """
  Load pre-fitted wavelength calibration trace data.
  
  Parameters:
  - arm (str): The arm of the spectrograph ('red', 'green', or 'blue'). Default is 'red'.
  - calib_type (str): The type of calibration ('Th' for ThAr or 'LC' for Laser Comb). Default is 'Th'.
  - trace_path (str, optional): The path to the directory containing the trace files. If None, a default path is used.
  - filename (str, optional): The specific filename to load. If None, a default filename based on the arm and calib_type is used.
  Returns:
  - tuple: A tuple containing:
    - ORDER (np.ndarray): Array of order numbers.
    - COEFFS (list of np.ndarray): List of coefficient arrays for each order.
    - Y (list of np.ndarray): List of Y-coordinate arrays for each order.
    - X (list of np.ndarray): List of X-coordinate arrays for each order.
  Raises:
  - FileNotFoundError: If the specified file is not found.
  - ValueError: If 'LC' calibration is requested for the 'blue' arm, which is not available.
  - ValueError: If unsupported arm name is requested.
  """

  if trace_path is None:
      veloce_paths = veloce_path.VelocePaths()
      trace_path = veloce_paths.trace_dir
  if filename is not None:
      filename = os.path.join(trace_path, filename)
          
  elif arm == 'red':
      if calib_type == 'Th':
          filename = os.path.join(trace_path, 'rosso-th-m65-104-all.trace')
      elif calib_type == 'LC':
          filename = os.path.join(trace_path, 'rosso-lc-m65-104-all.trace')
  elif arm == 'green':
      if calib_type == 'Th':
          filename = os.path.join(trace_path, 'verde-th-m104-139-all.trace')
      elif calib_type == 'LC':
          filename = os.path.join(trace_path, 'verde-lc-m104-135-all.trace')
  elif arm == 'blue':
      if calib_type == 'Th':
          filename = os.path.join(trace_path, 'azzurro-th-m138-167-all.trace')
      elif calib_type == 'LC':
          raise ValueError('No LC calibration for blue arm.')
  else:
      raise ValueError('Unknown arm. Must be "red", "green" or "blue".')

  try:
      with open(filename, 'r') as file:
          lines = [line[:-1] for line in file]
          ORDER = np.array([line.split()[0] for line in lines[::4]], dtype=np.uint8)
          COEFFS = [np.array(line.split()[1:], dtype=np.float64) for line in lines[1::4]]
          Y = [np.array(line.split(), dtype=np.float64) for line in lines[2::4]]
          X = [np.array(line.split(), dtype=np.float64) for line in lines[3::4]]
  except FileNotFoundError:
      raise FileNotFoundError(f"File {filename} not found in {trace_path}.")
  return ORDER, COEFFS, Y, X

def load_prefitted_wave(arm='red', wave_path=None, filename=None):
    """
    Loads pre-fitted thorium-argon (ThAr) calibration data for a specified spectrograph arm.

    This function reads from a data file containing pre-fitted calibration information for either the red or green arm
    of a spectrograph.

    Parameters:
    - arm (str): The spectrograph arm for which to load the calibration data. Valid values are 'red' and 'green'.
                 Defaults to 'red'.

    Returns:
    - tuple: A tuple containing the following elements:
        - ORDER (numpy.ndarray): An array of spectral order numbers.
        - COEFFS (list of numpy.ndarray): A list of arrays, each containing polynomial coefficients for wavelength
          calibration of a spectral order.
        - MATCH_LAM (list of numpy.ndarray): A list of arrays, each containing matched wavelengths for a spectral order.
        - MATCH_PIX (list of numpy.ndarray): A list of arrays, each containing matched pixel positions for a spectral order.
        - MATCH_LRES (list of numpy.ndarray): A list of arrays, each containing matched line resolutions for a spectral order.
        - GUESS_LAM (list of numpy.ndarray): A list of arrays, each containing guessed wavelengths for a spectral order.
        - Y0 (numpy.ndarray): An array of starting y-coordinates for each spectral order.

    Raises:
    - ValueError: If the `arm` parameter is not 'red' or 'green'.

    Note:
    - The data files are expected to be located in a relative path from the script
      directory. The file names are hardcoded based on the arm parameter.
    - The function assumes a specific format for the data files, where each spectral order's data is spread across
      nine lines, and specific lines within this block contain the different types of data (e.g., polynomial coefficients,
      matched wavelengths, etc.).
    """
    if wave_path is None:
        veloce_paths = veloce_path.VelocePaths()
        wave_path = veloce_paths.wave_dir

    if filename is not None:
        try:
            filename = os.path.join(wave_path, filename)
            with open(filename, 'r') as file:
              lines = [line[:-1] for line in file]
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} not found in {wave_path}")
    elif arm == 'red':
        filename = os.path.join(wave_path, 'vdarc_rosso_230919.dat')
    elif arm == 'green':
        filename = os.path.join(wave_path, 'vdarc_verde_240719-correct-vac-wavel.dat')
    elif arm == 'blue':
        filename = os.path.join(wave_path, 'vdarc_azzurro_240820-correct-vac-wavel.dat')
    else:
        raise ValueError('Unknown arm. Must be "red", "green" or "blue".')

    with open(filename, 'r') as file:
        lines = [line[:-1] for line in file]

    ORDER = np.array([line.split()[-1] for line in lines[::9]], dtype=np.uint8)
    COEFFS = [np.array(line.split()[1:], dtype=np.float64) for line in lines[3::9]]
    MATCH_LAM = [np.array(line.split()[1:], dtype=np.float64) for line in lines[4::9]]
    MATCH_PIX = [np.array(line.split()[1:], dtype=np.float64) for line in lines[5::9]]
    MATCH_LRES = [np.array(line.split()[1:], dtype=np.float64) for line in lines[6::9]]
    GUESS_LAM = [np.array(line.split()[1:], dtype=np.float64) for line in lines[7::9]]
    Y0 = np.array([line.split()[-1] for line in lines[1::9]], dtype=np.uint16)

    if arm == 'red':
      return ORDER[1:], COEFFS[1:], MATCH_LAM[1:], MATCH_PIX[1:], MATCH_LRES[1:], GUESS_LAM[1:], Y0[1:]
    elif arm == 'green':
      return ORDER[2:], COEFFS[2:], MATCH_LAM[2:], MATCH_PIX[2:], MATCH_LRES[2:], GUESS_LAM[2:], Y0[2:]
    else:
      return ORDER, COEFFS, MATCH_LAM, MATCH_PIX, MATCH_LRES, GUESS_LAM, Y0

def calibrate_orders_to_wave(orders, Y0, coefficients):
    """
    Converts pixel positions to wavelengths for each spectral order using polynomial coefficients.

    This function applies a polynomial transformation to convert pixel positions along the spectral orders
    into wavelengths. For each order, it uses a set of polynomial coefficients to calculate the wavelength
    at each pixel position. The polynomial degree and the coefficients vary per order and are provided as input.
    The transformation accounts for a starting y-coordinate (Y0) offset before applying the polynomial coefficients.

    Parameters:
    - orders (list of numpy.ndarray): A list of 2D numpy arrays, each representing a spectral order. Only the
      length of the first order is used to generate an array of pixel positions.
    - Y0 (int or float): The starting y-coordinate for the spectral orders, used to adjust the pixel positions
      before applying the wavelength calibration.
    - coefficients (list of list of floats): A list where each element is a list of polynomial coefficients
      for converting pixel positions to wavelengths for a corresponding spectral order.

    Returns:
    - numpy.ndarray: A 2D numpy array where each row corresponds to a spectral order and contains the wavelengths
      for each pixel position in that order.

    Note:
    - The function assumes that all spectral orders have the same length as the first order in the `orders` list.
    - The polynomial is applied as wave = sum(coeff[j] * (x_arr**j)) for each coefficient j in an order, where
      x_arr is the array of adjusted pixel positions.
    """
    x_arr = np.arange(len(orders[0]), dtype=np.float64)-Y0
    WAVE = []
    for i in range(len(orders)):
        wave = np.zeros_like(x_arr, dtype=np.float64)
        for j, coeff in enumerate(coefficients[i]):
            wave += (x_arr**j)*coeff
        WAVE.append(wave)
    WAVE = np.array(WAVE)
    return WAVE

def get_master(obs_list, master_type, data_path, run, date, arm):
    """
    Generates a master frame by median combining individual frames for a given observation type and date.

    This function reads FITS files specified in an observation list for a particular observation type and date,
    combines these frames by stacking them along a new axis, and then calculates the median of these frames to
    produce a master frame. The function is designed to work with astronomical data, specifically for the Veloce
    spectrograph which has different CCDs (charge-coupled devices) for red, green, and blue spectral arms.

    Parameters:
    - obs_list (dict): A nested dictionary where the first key is the master type (e.g., 'bias', 'flat'), the
      second key is the date, and the value is a list of file names for that observation type and date.
    - master_type (str): The type of master frame to generate (e.g., 'bias', 'flat').
    - data_path (str): The base path to the directory containing the observation data.
    - run (str): The observing run identifier, used to further specify the location of the data.
    - date (str): The date of the observation, used to select the correct set of files from the observation list.
    - arm (str): The spectral arm ('red', 'green', 'blue') of the data to process, which determines the CCD to use.

    Returns:
    - numpy.ndarray: A 2D numpy array representing the median-combined master frame for the specified observation
      type, date, and spectral arm.

    Raises:
    - KeyError: If the specified `arm` is not one of 'red', 'green', or 'blue'.
    - FileNotFoundError: If any of the FITS files specified in the observation list cannot be found at the
      constructed file path.
    """
    data_sub_dirs = {'red': 'ccd_3', 'green': 'ccd_2', 'blue': 'ccd_1'}
    if obs_list[master_type][date] != []:
        for file_name in obs_list[master_type][date]:
            fits_image_filename = os.path.join(data_path, run, date, data_sub_dirs[arm], file_name)
            with fits.open(fits_image_filename) as hdul:
                try:
                    frames = np.dstack((frames, hdul[0].data))
                except:
                    frames = np.array(hdul[0].data)

    return np.median(frames, axis=2)

def save_image_fits(filename, output_path, image, hdr):
    """
    Saves a 2D image array to a FITS file with a specified header.

    This function saves a 2D image array to a FITS file with a specified header. The header can include metadata
    information such as the observation date, exposure time, and telescope information. The function uses the
    Astropy package to create a FITS HDU (Header-Data Unit) with the image data and header information and then
    writes this HDU to a FITS file.

    Parameters:
    - filename (str): The name of the FITS file to save the image to.
    - output_path (str): The path to the directory where the FITS file will be saved.
    - image (numpy.ndarray): A 2D numpy array representing the image data to save.
    - hdr (astropy.io.fits.header.Header): An Astropy header object containing metadata information for the image.

    Returns:
    - str: The full file path to the saved FITS file.

    Note:
    - The function assumes that the Astropy package is installed and that the FITS file will be saved in the
      specified output directory.
    """
    hdu = fits.PrimaryHDU(image, header=hdr)
    hdul = fits.HDUList([hdu])
    output_filename = os.path.join(output_path, filename)
    hdul.writeto(output_filename, overwrite=True)
    return output_filename 

def save_extracted_spectrum_fits(filename, output_path, wave, flux, hdr):
    """
    Saves a 2D spectrum array to a FITS file with a specified header.

    This function saves a 2D spectrum array to a FITS file with a specified header. The spectrum array contains
    both the wavelength values and the corresponding flux values. The header can include metadata information
    such as the observation date, exposure time, and telescope information. The function uses the Astropy package
    to create a FITS HDU (Header-Data Unit) with the spectrum data and header information and then writes this HDU
    to a FITS file.

    Parameters:
    - filename (str): The name of the FITS file to save the spectrum to.
    - output_path (str): The path to the directory where the FITS file will be saved.
    - wave (numpy.ndarray): A 2D numpy array representing the wavelength values of the spectrum.
    - flux (numpy.ndarray): A 2D numpy array representing the flux values of the spectrum.
    - hdr (astropy.io.fits.header.Header): An Astropy header object containing metadata information for the spectrum.

    Returns:
    - str: The full file path to the saved FITS file.

    Note:
    - The function assumes that the Astropy package is installed and that the FITS file will be saved in the
      specified output directory.
    """
    hdu_wave = fits.ImageHDU(wave, name='WAVE')
    hdu_flux = fits.ImageHDU(flux, name='FLUX')
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu_wave, hdu_flux])
    hdul[0].header = hdr
    output_filename = os.path.join(output_path, filename)
    hdul.writeto(output_filename, overwrite=True)
    return output_filename

def load_extracted_spectrum_fits(filename):
    """
    Loads a 2D spectrum array from a FITS file.

    This function reads a FITS file containing a 2D spectrum array with wavelength and flux values and returns
    the wavelength and flux arrays separately. The function uses the Astropy package to read the FITS file and
    extract the wavelength and flux arrays from the HDU (Header-Data Unit) list.

    Parameters:
    - filename (str): The name of the FITS file to load the spectrum from.

    Returns:
    - tuple: A tuple containing the following elements:
        - numpy.ndarray: A 2D numpy array representing the wavelength values of the spectrum.
        - numpy.ndarray: A 2D numpy array representing the flux values of the spectrum.
        - astropy.io.fits.header.Header: An Astropy header object containing metadata information for the spectrum.

    Note:
    - The function assumes that the Astropy package is installed and that the FITS file is correctly formatted
      with wavelength and flux arrays stored in separate HDUs.
    """
    with fits.open(filename) as hdul:
        wave = hdul['WAVE'].data
        flux = hdul['FLUX'].data
        hdr = hdul[0].header
    return wave, flux, hdr

if __name__ == '__main__':
    filename = '24aug30010.fits' # use the flat because of visibility
    spectrum_filename =  os.path.join(os.getcwd(), 'Data', filename)

    image_data = fits.getdata(spectrum_filename)
    # plt.imshow(image_data, cmap='gray', norm="log")
    image_substracted_bias = remove_overscan_bias(image_data, overscan_range=32)
    blured = median_filter(image_substracted_bias, (10,10))
