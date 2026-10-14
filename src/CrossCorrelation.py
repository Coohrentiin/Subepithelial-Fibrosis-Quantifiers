import os
import sys
from matplotlib import pyplot as plt 
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.signal import savgol_filter

def get_central_signal(image, window_central):
    position = image.shape[1]//2
    central_column = image[:,position-window_central:position+window_central]
    central_column = np.mean(central_column, axis=1)
    return central_column

def calculate_offsets(X, X_ref, window_lat=0, subpixel=False, linear_attenuation=False):
    """
    Calculate the shift factor (delta) for each column in X relative to a reference signal X_ref.
    
    Parameters:
        X (numpy.ndarray): A 2D array where each column represents a signal at time t.
        X_ref (numpy.ndarray): A 1D array representing the reference signal.
    
    Returns:
        numpy.ndarray: An array of offset factors (delta) for each column in X.
    """
    num_columns = X.shape[1]
    ref_length = len(X_ref)
    deltas = np.zeros(num_columns)
    cross_correlation = np.zeros((X.shape[0]*2-1,num_columns))
    X_ref_Norm = (X_ref - np.mean(X_ref))/np.std(X_ref)
    if linear_attenuation:
        linear_attenuation_factor = np.linspace(1,0.0,ref_length)
        X_ref_Norm = X_ref_Norm * linear_attenuation_factor

    for t in range(num_columns):
        # Compute cross-correlation between X[:, t] and X_ref
        if window_lat == 0:
            X_t = X[:,t]
        else:
            if t < window_lat:
                X_t = X[:,:t+window_lat]
            elif t > X.shape[1]-window_lat:
                X_t = X[:,t-window_lat:]
            else:
                X_t = X[:,t-window_lat:t+window_lat]
            X_t = np.mean(X_t, axis=1)

        X_t_Norm = (X_t - np.mean(X_t))/np.std(X_t)
        if linear_attenuation:
            X_t_Norm = X_t_Norm * linear_attenuation_factor
        cross_corr = np.correlate(X_t_Norm, X_ref_Norm, mode='full')
        # Find the lag that maximizes the cross-correlation
        max_idx = np.argmax(cross_corr)
        delta = max_idx - (ref_length - 1)
        if subpixel:
            # Refine the offset estimate using parabolic interpolation
            if delta > 0 and delta < len(cross_corr) - 1:
                c_1 = cross_corr[max_idx - 1]
                c_2 = cross_corr[max_idx]
                c_3 = cross_corr[max_idx + 1]
                delta += 0.5 * (c_1 - c_3) / (c_1 - 2 * c_2 + c_3)
        cross_correlation[:,t] = cross_corr
        deltas[t] = -delta
    
    return deltas,cross_correlation


def apply_offsets(X, deltas,subpixel=False):
    """
    Apply column-wise offsets to X and crop rows to handle invalid entries.
    
    Parameters:
        X (numpy.ndarray): A 2D array where each column represents a signal at time t.
        deltas (numpy.ndarray): An array of offsets for each column of X.
    
    Returns:
        numpy.ndarray: A 2D array Y with shifted columns and cropped rows.
    """
    num_rows, num_columns = X.shape
    max_shift = int(np.ceil(np.max(deltas)))
    min_shift = int(np.floor(np.min(deltas)))
    
    # Calculate the valid range of rows after applying shifts
    start = max_shift
    end = num_rows + min_shift  # Adjust for negative shifts
    valid_length = end - start

    # Calculate the valid range of rows after applying shifts
    Y = np.zeros((end - start, num_columns))

    
    # print(f'Y shapes {Y.shape}')
    for t in range(num_columns):
        delta = deltas[t]
        # Check if delta is an integer:
        if not subpixel:
            delta = int(delta)
            Y[:, t] = X[start - delta:end - delta, t]
        else: #Subpixel realignment
            # Interpolate the signal to the new length
            integer_shift = int(np.floor(delta))
            fractional_shift = delta - integer_shift
            
            # Apply integer shift by slicing
            valid_start = start - integer_shift
            valid_end = end - integer_shift
            shifted_column = X[valid_start:valid_end , t]
            
            # Apply fractional shift using linear interpolation
            interp_indices = np.arange(0, valid_length) + fractional_shift
            interpolated_column = np.interp(
                np.arange(valid_length),  # Target indices
                interp_indices[:-1],      # Source indices
                shifted_column[:-1]       # Source values
            )
            
            # Store the result
            Y[:, t] = interpolated_column
    return Y



def plot_NCC_res(cross_correlation, deltas, image, image_realigned):
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.imshow(cross_correlation,cmap='jet')
    plt.title('Cross correlation')
    plt.subplot(2,3,4)
    plt.plot(deltas)
    plt.title('Deltas ')
    plt.subplot(2,3,2)
    plt.imshow(image,cmap='gray')
    plt.title('OCT')
    plt.subplot(2,3,5)
    plt.imshow(image_realigned,cmap='gray')
    plt.title('OCT realigned')
    plt.subplot(2,3,3)
    plt.plot(np.mean(image, axis=1))
    plt.title('Mean OCT')
    plt.subplot(2,3,6)
    plt.plot(np.mean(image_realigned, axis=1))
    plt.title('Mean OCT realigned')

    plt.show()

def realign_image(image, window_central, window_lat,plot=False, subpixel=False, smooth=True,linear_attenuation=True):
    X_ref = get_central_signal(image, window_central)
    deltas,cross_correlation = calculate_offsets(image, X_ref, window_lat=window_lat,subpixel=subpixel,linear_attenuation=linear_attenuation)
    if smooth:
        # plt.figure()
        # plt.plot(deltas,label='Deltas')
        deltas = savgol_filter(deltas, 5, 3)
        # # deltas = np.convolve(deltas, np.ones(5)/5, mode='same')
        # plt.plot(deltas,label='Deltas smoothed')
        # plt.legend()
        # plt.show()

    image_realigned = apply_offsets(image, deltas, subpixel=subpixel)
    if plot:
        plot_NCC_res(cross_correlation, deltas, image, image_realigned)
    return image_realigned

#If main: 
if __name__ == '__main__':
    image = Image.open('Dataset/Healthy/81225_GOLDIM_OS_flat-post-ttm.tiff')
    image = np.array(image)

    window_central = 10
    window_lat = 1

    image_realigned = realign_image(image, window_central, window_lat,plot=True,subpixel=True)