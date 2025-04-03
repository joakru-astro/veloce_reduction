from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt

from . import veloce_reduction_tools, veloce_config

def plot_order_cross_section(frame, traces, order, filename, veloce_paths, plot_type='median', margin=[10,10]):
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
    # if isinstance(summing_range, int):
    lower_range, upper_range = traces.summing_ranges_lower[order], traces.summing_ranges_upper[order]+1
    # elif len(summing_range) == 2:
    #     lower_range, upper_range = summing_range
        # upper_range += 1
    # else:
    #     raise TypeError("Incorrect summing range, must be int or 2-element list.")
    
    ylen, xlen = frame.shape

    # trace_y, trace_x = traces[order]
    # y = np.arange(ylen)
    lower_limit = min(traces.x[order])-lower_range-lower_margin
    upper_limit = max(traces.x[order])+upper_range+upper_margin

    if upper_limit > xlen:
        offset = int(np.ceil(upper_limit))
        buffer = np.zeros((ylen, offset))
        frame = np.concatenate((frame, buffer), axis=1)
    if lower_limit < 0:
        offset = int(-1*np.floor(lower_limit))
        # trace += offset
        buffer = np.zeros((ylen, offset))
        frame = np.concatenate((buffer, frame), axis=1)
    else:
        offset = 0

    extracted_order_img = np.array([frame[int(yval), int(np.round(xval-lower_range-lower_margin, 0)+offset):int(np.round(xval+upper_range+upper_margin, 0)+offset)] 
                                            for yval, xval in zip(*traces.traces[order])])

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
        output_file = os.path.join(veloce_paths.plot_dir,
                               f'Cross_section_order_{order}_row_{plot_type}_{filename}.png')
    else:
        plt.title(f"Order {order}, {plot_type}")
        output_file = os.path.join(veloce_paths.plot_dir,
                               f'Cross_section_order_{order}_{plot_type}_{filename}.png')
    plt.xlabel("Pixel")
    plt.ylabel("Counts")
    
    plt.savefig(output_file)
    plt.close()

    return output_file

def plot_extracted_2D_order(extracted_order_imgs, order, traces, filename, veloce_paths, flatfielded=False, flatfield=None):
    lower_range, upper_range = float(traces.summing_ranges_lower[order]), float(traces.summing_ranges_upper[order])
    
    xticks = np.arange(lower_range % 10, lower_range + upper_range + 1, 10)
    xtick_labels = np.arange(-lower_range + (lower_range % 10), upper_range + 1, 10)
    # xtick_labels = np.arange(-lower_range, upper_range + 1, 10)
    
    if flatfielded:
        _, extracted_orders_flat = veloce_reduction_tools.extract_orders_with_trace(flatfield, traces)
        fig, (ax1, ax2) = plt.subplots(1,2)
        im1 = ax.imshow(extracted_order_imgs[order],
                        extent=[0, lower_range + upper_range, 0, extracted_order_imgs[order].shape[0]],
                        aspect='auto', origin='lower', cmap='viridis', norm='log')
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xtick_labels)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title(f"Extracted order {order}")
        ax1.set_xlabel("Pixel")
        ax1.set_ylabel("Row")
        im2 = ax.imshow(extracted_orders_flat[order],
                        extent=[0, lower_range + upper_range, 0, extracted_order_imgs[order].shape[0]],
                        aspect='auto', origin='lower', cmap='viridis', norm='log')
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xtick_labels)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title(f"Flatfield order {order}")
        ax2.set_xlabel("Pixel")
        ax2.set_ylabel("Row")
    else:
        fig, ax = plt.subplots()
        im = ax.imshow(extracted_order_imgs[order],
                        extent=[0, lower_range + upper_range, 0, extracted_order_imgs[order].shape[0]],
                        aspect='auto', origin='lower', cmap='viridis', norm='log')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Extracted order {order}")
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Row")

    output_file = os.path.join(veloce_paths.plot_dir,
                               f'Extracted_order_{order}_{filename}.png')

    plt.savefig(output_file)
    plt.close()
    return output_file

def plot_scattered_light(frame, background, corrected_frame, veloce_paths, filename, traces):
    ### TODO add statistics inside trace
    head = 'Background statistics:\n---'
    median_str = f'median = {np.median(background)}'
    max_str = f'max = {np.max(background)}'
    std_str = f'stdev = {np.std(background)}'
    background_message = '\n'.join([head, median_str, max_str, std_str])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    im1 = ax1.imshow(frame, origin='lower', cmap='viridis', norm='log')
    plt.colorbar(im1,ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title('Original Image')

    im2 = ax2.imshow(background, origin='lower', cmap='viridis')
    plt.colorbar(im2,ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title('Fitted background')

    im3 = ax3.imshow(corrected_frame, origin='lower', cmap='viridis', norm='log')
    plt.colorbar(im3,ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title('Image - Fitted background')

    # fig.colorbar(ax, ax=fig.get_axes())
    plt.tight_layout()
    output_file = os.path.join(veloce_paths.plot_dir,
                               f'Fitted_scattered_light_{filename}.png')
    plt.savefig(output_file)

    return background_message, output_file