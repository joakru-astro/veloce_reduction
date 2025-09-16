import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.constants import c
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D

from . import veloce_reduction_tools

def plot_order_cross_section(frame, traces, order, filename, veloce_paths, plot_type='median', margin=[10,10], show=False):
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
                               f'Cross_section_order_{order}_row_{plot_type}_{filename.split('.')[0]}.png')
    else:
        plt.title(f"Order {order}, {plot_type}")
        output_file = os.path.join(veloce_paths.plot_dir,
                               f'Cross_section_order_{order}_{plot_type}_{filename.split('.')[0]}.png')
    plt.xlabel("Pixel")
    plt.ylabel("Counts")
    
    plt.savefig(output_file)
    if show:
        plt.show()
    else:
        plt.close()

    return output_file

def plot_extracted_2D_order(extracted_order_imgs, order, traces, filename, veloce_paths, flatfielded=False, flatfield=None, show=False):
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

    if filename is not None:
        output_file = os.path.join(veloce_paths.plot_dir,
                               f'Extracted_order_{order}_{filename.split('.')[0]}.png')
    else:
        output_file = None
    plt.savefig(output_file)
    if show:
        plt.show()
    else:
        plt.close()

    return output_file

def plot_scattered_light(frame, background, corrected_frame, veloce_paths, filename=None, show=False):
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

    if filename is not None:
        output_file = os.path.join(veloce_paths.plot_dir,
                                   f'Fitted_scattered_light_{filename.split('.')[0]}.png')
        plt.savefig(output_file)
    else:
        output_file = None
    if show:
        plt.show()
    else:
        plt.close()

    return background_message, output_file

def plot_ccf(PIX, CCF, order, chunk, fit_lc_peak, general_gaussian, veloce_paths=None, filename=None, show=False):
    # fitting_limit = np.ceil(np.mean(np.diff(find_peaks(CCF[order-1][chunk])[0])))/2 + 1
    shift, popt, fit_lim = fit_lc_peak(PIX[order-1][chunk], CCF[order-1][chunk])
    plt.figure(figsize=(10, 6))
    plt.title('Cross-Correlation Function')
    pixel = PIX[order-1][chunk]
    ccf = CCF[order-1][chunk]
    ccf_mask = np.isfinite(ccf)
    pixel = pixel[ccf_mask]#[fit_slice]
    ccf = ccf[ccf_mask]#[fit_slice]
    plt.plot(pixel, ccf, label=f'Order {order}')
    print(f"Amplitude: {popt[0]}\n Shift: {popt[1]}\n Sigma: {popt[2]}\n Beta: {popt[3]}\n Baseline: {popt[4]}")
    subpixel = np.arange(np.min(pixel), np.max(pixel), 0.01)
    plt.plot(subpixel, general_gaussian(subpixel, *popt), label='Gaussian Fit', linestyle='--')
    plt.axvline(shift, color='r', linestyle='--', label='Peak Position')
    plt.title(f'Cross-Correlation Function for Order {order}, Shift = {popt[1]:.2f}')
    plt.xlabel('Pixel Shift')
    plt.ylabel('Cross-Correlation')
    plt.xlim(shift-fit_lim, shift+fit_lim)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if filename is not None:
        output_file = os.path.join(veloce_paths.plot_dir,
                                   f'LC_peak_fit_order_{order}_{filename.split('.')[0]}.png')
        plt.savefig(output_file)
    else:
        output_file = None
    if show:
        plt.show()
    else:
        plt.close()

    return output_file

def plot_offset_map(dispersion_position, orders_position, offset_array, veloce_paths=None, filename=None, show=False):
    """
    Plot the offset map in 3D.
    """    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    points = ax.scatter(dispersion_position.flatten(), orders_position.flatten(), offset_array.flatten(), c=offset_array.flatten(), cmap='viridis', marker='o')
    ax.set_title('Offset Map')
    ax.set_xlabel('Dispersion Position')
    ax.set_ylabel('Orders')
    ax.set_zlabel('Offset')
    fig.colorbar(points, shrink=0.5, aspect=10)
    plt.tight_layout()

    if filename is not None:
        output_file = os.path.join(veloce_paths.plot_dir,
                                   f'LC_offset_map_{filename.split('.')[0]}.png')
        plt.savefig(output_file)
    else:
        output_file = None
    if show:
        plt.show()
    else:
        plt.close()

    return output_file

def plot_surface(ref_orders, extracted_pixels, surface_points, filtered_points, veloce_paths=None, filename=None, show=False):
    """
    Plot the offset map in 3D.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    max_pixel = max([np.nanmax(order) for order in extracted_pixels])+1
    min_pixel = min([np.nanmin(order) for order in extracted_pixels])
    X, Y = np.meshgrid(np.arange(min_pixel, max_pixel, 1), ref_orders)
    surf = ax.plot_surface(X, Y, surface_points, vmin=np.min(filtered_points[:,2]), vmax=np.max(filtered_points[:,2]), cmap='viridis', edgecolor='none', alpha=0.5)
    points = ax.scatter(filtered_points[:,0], filtered_points[:,1], filtered_points[:,2], c=filtered_points[:,2], cmap='viridis', marker='o')
    ax.set_title('Offset Map')
    ax.set_xlabel('Echelle Dispersion Position')
    ax.set_ylabel('Orders')
    ax.set_zlabel('Offset')
    fig.colorbar(points, shrink=0.5, aspect=10)
    plt.tight_layout()

    if filename is not None:
        output_file = os.path.join(veloce_paths.plot_dir,
                                f'LC_fitted_surface_{filename.split('.')[0]}.png')
        plt.savefig(output_file)
    else:
        output_file = None
    if show:
        plt.show()
    else:
        plt.close()

    return output_file

def plot_ArcTh_points_positions(pixel_positions, order_positions, mask, veloce_paths, filename=None, show=False):
    plt.close('all')

    fig, ax = plt.subplots()

    if np.sum(mask)/len(mask)<=0.5:
        ax.scatter(order_positions[~mask], pixel_positions[~mask], color='r', marker='x', label='rejected points')
        ax.scatter(order_positions[mask], pixel_positions[mask], color='k', marker='o', label='fitted points')
    else:
        ax.scatter(order_positions[mask], pixel_positions[mask], color='k', marker='o', label='fitted points')
        ax.scatter(order_positions[~mask], pixel_positions[~mask], color='r', marker='x', label='rejected points')
    
    # Label number of used points above point with highest pixel per order
    for order_val in np.unique(order_positions):
        in_order = (order_positions == order_val)
        in_order_mask = in_order & mask
        if np.any(in_order):
            # Find the point with the highest pixel in this order (regardless of mask)
            max_pixel = np.max(pixel_positions[in_order])
            idx = np.where(in_order & (pixel_positions == max_pixel))[0]
            if len(idx) > 0:
                idx = idx[0]
                n_used = np.sum(in_order_mask)
                ax.text(order_positions[idx], pixel_positions[idx]+20, f"{n_used}",
                        ha='center', va='bottom', fontsize=9, color='blue')

    ax.set_xlabel('Order number')
    ax.set_ylabel('Echelle Dispersion [pixel]')
    ax.set_title(f'Points used for fitting (X,Y): kept {np.sum(mask)} out of {len(mask)}')
    # Force x-ticks to be integers only
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Shrink the box and move legend outside the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend()
    plt.tight_layout()
    if filename is not None:
        output_file = os.path.join(veloce_paths.plot_dir,
                               f'ArcTh_line_positions_used_per_order_{filename.split('.')[0]}.png')
        plt.savefig(output_file)
    else:
        output_file = None
    if show:
        plt.show()
    else:
        plt.close()

    return output_file

def plot_ArcTh_surface(Z, pixel_positions, order_positions, wave_positions, full_pixels, veloce_paths, filename=None, show=False):
    plt.close('all')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=245)

    # Plot fitted surface
    X, Y = np.meshgrid(full_pixels[0], np.unique(order_positions))
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=False)

    # Plot input data points
    ax.scatter(pixel_positions, order_positions, order_positions*wave_positions, c='k', s=10, label='Input data')

    ax.set_xlabel('Echelle Dispersion [pixel]')
    ax.set_ylabel('Order')
    ax.set_zlabel(r'Order $\times$ $\lambda$ [nm]')
    ax.set_title('Input ArcTh lines and Fitted Surface')
    ax.legend()
    if filename is not None:
        output_file = os.path.join(veloce_paths.plot_dir,
                               f'Th_fitted_surface_{filename.split('.')[0]}.png')
        plt.savefig(output_file)
    else:
        output_file = None
    if show:
        plt.show()
    else:
        plt.close()

    return output_file

def plot_ArcTh_residuals(residuals, order_positions, pixel_positions, wave_positions, mask, veloce_paths, filename=None, plot_type='velocity', show=False):
    per_order_residual_mean = []
    for order in np.unique(order_positions):
        in_order_mask = (order_positions == order)
        in_order_mask *= mask
        if plot_type == 'velocity':
            in_order_residual_mean = np.mean(residuals[in_order_mask]/order_positions[in_order_mask]/wave_positions[in_order_mask]*c.value)
        elif plot_type == 'wavelength':
            in_order_residual_mean = np.mean(residuals[in_order_mask]/order_positions[in_order_mask])
        else:
            raise ValueError("plot_type must be 'velocity' or 'wavelength'")
        per_order_residual_mean.append(in_order_residual_mean)
    per_order_residual_mean = np.array(per_order_residual_mean)
    
    dispersion_binned_mean = []
    dispersion_bin_pos = []
    bin_size = 100
    for dispersion_bin in np.arange(np.min(pixel_positions[mask]), np.max(pixel_positions[mask]), bin_size):
        dispersion_bin_pos.append(dispersion_bin+bin_size/2)
        bin_mask = (pixel_positions>dispersion_bin) & (pixel_positions<dispersion_bin+bin_size)
        bin_mask *= mask
        if plot_type == 'velocity':
            in_bin_residual_mean = np.mean(residuals[bin_mask]/order_positions[bin_mask]/wave_positions[bin_mask]*c.value)
        elif plot_type == 'wavelength':
            in_bin_residual_mean = np.mean(residuals[bin_mask]/order_positions[bin_mask])
        else:
            raise ValueError("plot_type must be 'velocity' or 'wavelength'")
        dispersion_binned_mean.append(in_bin_residual_mean)
    dispersion_binned_mean = np.array(dispersion_binned_mean)
    dispersion_bin_pos = np.array(dispersion_bin_pos)

    if plot_type == 'velocity':
        _residuals = residuals/order_positions/wave_positions*c.value
        rmse = np.sqrt(np.mean(_residuals[mask]**2))
        # std_vel = np.std(_residuals[mask])
        # sme = std_vel/np.sqrt(np.sum(mask))
    elif plot_type == 'wavelength':
        _residuals = residuals/order_positions
        rmse = np.sqrt(np.mean(_residuals[mask]**2))
        # std_vel = np.std(_residuals[mask])
        # sme = std_vel/np.sqrt(np.sum(mask))
    else:
        raise ValueError("plot_type must be 'velocity' or 'wavelength'")

    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), gridspec_kw={'hspace': 0, 'wspace': 0})

    axes[0,0].scatter(pixel_positions[~mask], _residuals[~mask], c='r', s=10, marker='x', alpha=0.7, label='Residuals of rejected lines')
    axes[0,0].scatter(pixel_positions[mask], _residuals[mask], c='k', s=10, label='Residuals of fitted lines')
    axes[0,0].legend()
    axes[1,0].scatter(pixel_positions[mask], _residuals[mask], c='k', s=10, label='Residuals')
    axes[1,0].plot(dispersion_bin_pos, dispersion_binned_mean, c='r', label='Residual mean (binned)')
    axes[1,0].axhline(rmse, color='blue', ls='--')
    axes[1,0].axhline(-1*rmse, color='blue', ls='--')
    axes[1,0].legend()
    ylim = axes[1,0].get_ylim()
    xlim = axes[1,0].get_xlim()
    x_pos = xlim[1] - 0.02 * (xlim[1] - xlim[0])
    y_pos = rmse + 0.02 * (ylim[1] - ylim[0])
    axes[1,0].text(x_pos, y_pos, 'RMSE', color='blue', ha='right', va='bottom', fontsize=9)

    axes[0,1].scatter(order_positions[~mask], _residuals[~mask], c='r', s=10, marker='x', alpha=0.7, label='Residuals of rejected lines')
    axes[0,1].scatter(order_positions[mask], _residuals[mask], c='k', s=10, label='Residuals of fitted lines')
    axes[0,1].legend()
    axes[1,1].scatter(order_positions[mask], _residuals[mask], c='k', s=10, label='Residuals')
    axes[1,1].plot(np.unique(order_positions), per_order_residual_mean, c='r', label='Residual mean per order')
    axes[1,1].axhline(rmse, color='blue', ls='--')
    axes[1,1].axhline(-1*rmse, color='blue', ls='--')
    axes[1,1].legend()
    ylim = axes[1,1].get_ylim()
    xlim = axes[1,1].get_xlim()
    x_pos = xlim[1] - 0.02 * (xlim[1] - xlim[0])
    y_pos = rmse + 0.02 * (ylim[1] - ylim[0])
    axes[1,1].text(x_pos, y_pos, 'RMSE', color='blue', ha='right', va='bottom', fontsize=9)
    
    for ax in axes.flat:
        ax.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)
    
    axes[0,0].tick_params(labelbottom=False)
    axes[0,1].tick_params(labelbottom=False, labelleft=False)
    axes[1,1].tick_params(labelleft=False)

    axes[1,1].set_xlabel('Order number')
    axes[1,0].set_xlabel('Echelle dispersion [pixel]')
    if plot_type == 'velocity':
        plt.suptitle('Residuals of Fitted Surface, RMSE: {:.1e} [m/s]'.format(rmse))
        axes[0,0].set_ylabel('Velocity [m/s]')
        axes[1,0].set_ylabel('Velocity [m/s]')
    elif plot_type == 'wavelength':
        plt.suptitle('Residuals of Fitted Surface, RMSE: {:.1e} [nm]'.format(rmse))
        axes[0,0].set_ylabel(r'$\lambda$ [nm]')
        axes[1,0].set_ylabel(r'$\lambda$ [nm]')
    
    # ax.legend()
    if filename is not None:
        output_file = os.path.join(veloce_paths.plot_dir,
                               f'Th_residuals_{filename.split('.')[0]}.png')
        plt.savefig(output_file)
    else:
        output_file = None
    
    if show:
        plt.show()
    else:
        plt.close()

    return output_file