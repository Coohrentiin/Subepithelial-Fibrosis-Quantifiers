# -*- coding: utf-8 -*-
"""
Code developped by Maelle Vilbert & Corentin Soubeiran
"""

import os
import tkinter as tk
import tkinter.messagebox
from copy import deepcopy
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as sgn
import scipy.ndimage as img


from matplotlib.widgets import Cursor
from detecta import detect_peaks # À MODIFIER
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

def histogram_shift(OCT, show=True): 
    """
    Adjusts the histogram of an OCT image to correct overexposure. (specifically designed for RTVue-XR OCT images.)
    Parameters:
    OCT (numpy.ndarray): The input OCT image as a 2D numpy array.
    show (bool, optional): If True, displays the original and adjusted images along with their histograms. Default is True.
    Returns:
    numpy.ndarray: The exposure-adjusted OCT image if overexposed, otherwise returns the original image.
    """
    hist_moy = np.mean(OCT.ravel())
    
    if hist_moy > 25:                           # if the image is overexposed        
        hist_adjust = np.round(hist_moy - 18)
        OCT_optim = OCT - hist_adjust      # histogram sliding
        OCT_optim[OCT_optim < 0] = 0
        if show:
            fig, axs = plt.subplots(2,2)
            fig.suptitle('Overexposed image: exposure correction')
            axs[0,0].imshow(OCT, cmap="gray")
            axs[0,0].set_title('Raw image', size = 15)
            axs[0,0].axis('off')
            axs[1,0].hist(OCT.ravel(),256,[0,256], density=True)
            axs[1,0].set_title('Histogram before', size = 15)
            axs[1,0].set(xlabel = '8-bit graylevel', ylabel = 'Density')
            axs[0,1].imshow(OCT_optim, cmap = 'gray')
            axs[0,1].set_title('Exposure-adjusted image', size = 15)
            axs[0,1].axis('off')
            axs[1,1].hist(OCT_optim.ravel(),256,[0,256], density=True)
            axs[1,1].set_title('Histogram after', size = 15)
            axs[1,1].set(xlabel = '8-bit graylevel')  
            plt.show()            
        return OCT_optim
    
    else:
        return OCT
    
def SNR(signal_1D_or_2D, gauss_sigma=1):
    """
    Compute signal-to-noise ratio (SNR) of the input signal, which can be a 1D or 2D array-like structure.

    Parameters:
        signal_1D_or_2D (array-like): The signal of interest, either a 1D or 2D numpy array.
        gauss_sigma (float, optional): The standard deviation for the Gaussian kernel used in noise estimation. Default is 1.
    Returns:
        float: The computed SNR value, rounded to one decimal place.      
    """
    if len(np.shape(signal_1D_or_2D)) == 2:
        signal = np.sum(np.square(signal_1D_or_2D))
        noise = np.sum(np.square(signal_1D_or_2D-img.gaussian_filter(signal_1D_or_2D, sigma=gauss_sigma)))
        SNR_2D = np.around(10*np.log10(signal/noise),1)
        
        return SNR_2D

def saturation_artifact_cut(OCT_brut, pixel_size_x, auto=True, gauss_sigma=1, 
                            sat_derivative1_threshold=0.67, sat_x_margin_um=55, 
                            show=True):        
    """
    Detects and removes saturation artifacts automatically or manualy from OCT images. 
    Parameter 'auto' allows automatic detection and removal of saturation artifacts. 
    If 'auto' is set to False, the user can manually select the boundaries of the saturation artifact.

    Parameters:
        OCT_brut (numpy.ndarray): Raw OCT image data.
        pixel_size_x (float): Pixel size in the x-dimension (µm).
        auto (bool, optional): If True, automatically detects and removes saturation artifacts. If False, allows manual selection. Default is True.
        gauss_sigma (float, optional): Sigma value for Gaussian smoothing. Default is 1.
        sat_derivative1_threshold (float, optional): Threshold for the first derivative to detect saturation artifacts. Default is 0.67.
        sat_x_margin_um (float, optional): Margin around detected saturation artifact in micrometers. Default is 55.
        show (bool, optional): If True, displays plots for visualization. Default is True.
    Returns:
        tuple: A tuple containing:
            - coord_centre (int): x-coordinate of the corneal apex.
            - xmin_cut (int): x-coordinate of the left boundary of the saturation artifact.
            - xmax_cut (int): x-coordinate of the right boundary of the saturation artifact.
            - saturation_cut_width (int): Width of the saturation artifact in micrometers.
            - derivative1_smooth (numpy.ndarray): Smoothed first derivative of the mean x-intensity signal.
    """
    OCT_smooth = img.gaussian_filter(OCT_brut, sigma = gauss_sigma)
    mean_signal = np.mean(OCT_smooth, axis = 0)

    # Savitsky-Golay smoothing of the mean OCT x-intensity
    savgol_window = 15
    savgol_order = 2
    signal_smooth = sgn.savgol_filter(mean_signal, savgol_window, savgol_order)
    
    # First derivative computation of the smoothed mean OCT x-intensity
    derivative1 = np.diff(signal_smooth)
    derivative1_smooth = sgn.savgol_filter(derivative1, savgol_window, savgol_order)
    
    #Plot
    if show:
        plt.figure(2)
        plt.imshow(OCT_brut,cmap = 'gray')
        plt_offset = np.ceil(np.amax(mean_signal)) + 5
        lineN, = plt.plot(plt_offset + -signal_smooth, color='tab:purple')
        lineO, = plt.plot(plt_offset + -10*derivative1, color='tab:green')
        plt.show()
    
    # Corneal apex x-coordinate
    x_dim_raw = OCT_brut.shape[1]
    coord_centre = np.where(signal_smooth == np.amax(signal_smooth[int(0.2*x_dim_raw):x_dim_raw-int(0.2*x_dim_raw)]))[0][0]
    
    # First derivative of mean x-signal computation
    sat_x_margin = int(np.round(sat_x_margin_um/pixel_size_x)) # [px] ==========================================> marge => sat_x_margin_um
    saturation_cut_width=0
    if auto:
        if np.any(abs(derivative1[20:len(derivative1)-20]) > sat_derivative1_threshold):    # Saturation artifact detected
            xmin_cut = np.where(derivative1_smooth == np.sort(derivative1_smooth[20:-20])[-1])[0][0] # maximum of derivative1_smooth
            xmax_cut = np.where(derivative1_smooth == np.sort(derivative1_smooth[20:-20])[0])[0][0] # minimum of derivative1_smooth
            xmin_cut = xmin_cut - sat_x_margin
            xmax_cut = xmax_cut + sat_x_margin
            if show:
                plt.axvline(xmin_cut, ymin=0, ymax=OCT_smooth.shape[0]-1, color='tab:green', linestyle='dashed', linewidth=1)
                plt.axvline(xmax_cut, ymin=0, ymax=OCT_smooth.shape[0]-1, color='tab:green', linestyle='dashed', linewidth=1)
                lineP, = plt.plot(np.ones([len(OCT_smooth),1]), color='tab:green', linestyle = 'dashed', linewidth=1)
                saturation_cut_width = (xmax_cut-xmin_cut)*pixel_size_x
                plt.title("Artifact removal (auto) : width = %i µm" %saturation_cut_width)
                plt.legend((lineN, lineO, lineP),('Smoothed mean x-signal',
                           'First derivative of x-signal', 'Saturation artifact boundaries'), fontsize=11, loc = 'lower right')   
                lineP.set_visible(False) 
                plt.show()
        else:
            # No saturation artifact
            xmin_cut = 0
            xmax_cut = 0
            saturation_cut_width = 0
            if show:
                lineP, = plt.plot(np.ones([len(OCT_smooth),1]), color='tab:green', linestyle = 'dashed', linewidth=1)
                plt.title('Artifact removal (auto) : width = 0 µm')
                plt.legend((lineN, lineO, lineP),('Smoothed mean x-signal', 'First derivative of x-signal', 'Apex lateral coordinate: x = %i' %coord_centre), fontsize=11, loc = 'lower right') 
                plt.legend((lineN, lineO),('Smoothed mean x-signal', 'First derivative of x-signal'), fontsize=11, loc = 'lower right') 
    else:
        # Manual cut (w/ cursor)
        fig = plt.figure(2)
        plt.clf()
        ax = fig.add_subplot(111, facecolor='w')
        plt.imshow(OCT_smooth, cmap="gray")
        offset_aff = np.ceil(np.amax(mean_signal)) + 5
        lineN, = plt.plot(offset_aff + -signal_smooth, color='tab:purple')
        lineO, = plt.plot(offset_aff + -10*derivative1, color='tab:green')
        ind = np.zeros((1,2))[0].astype(int)
        ind[0] = np.where(derivative1_smooth == np.sort(derivative1_smooth[20:-20])[-1])[0][0] # maximum of derivative1_smooth
        ind[1] = np.where(derivative1_smooth == np.sort(derivative1_smooth[20:-20])[0])[0][0] # minimum of derivative1_smooth
        ind[0] = ind[0] - sat_x_margin
        ind[1] = ind[1] + sat_x_margin            
        plt.vlines(ind[0], 0, OCT_smooth.shape[0]-1, color='tab:green', linestyle='dashed', linewidth=0.5)
        plt.vlines(ind[-1], 0, OCT_smooth.shape[0]-1, color='tab:green', linestyle='dashed', linewidth=0.5)    
        plt.title('Select the lateral boundaries of the central saturation artifact')
        
        fig.cursor = Cursor(ax, horizOn=False, vertOn=True, useblit=True, color='g', linewidth=1)
        R = plt.ginput(2)

        if len(R) == 0:
            xmin_cut = 0
            xmax_cut = 0
            saturation_cut_width = 0
        xmin_cut = int(min(round(R[0][0]), round(R[1][0])))
        xmax_cut = int(max(round(R[0][0]), round(R[1][0])))

        mean_signal = np.mean(OCT_smooth, axis = 0)
        if xmin_cut!=xmax_cut:
            coord_centre = xmin_cut + np.where(mean_signal[xmin_cut:xmax_cut] == np.amax(mean_signal[xmin_cut:xmax_cut]))[0][0]
        else: 
            coord_centre = xmin_cut
        saturation_cut_width = int(round((xmax_cut-xmin_cut)*pixel_size_x))
        time.sleep(0.5)
        plt.close()
    return coord_centre, xmin_cut, xmax_cut, saturation_cut_width, derivative1_smooth

def lateral_borders_cut(OCT_brut, pixel_size_x, auto=True,show=True,left_right_crop=0):
    """
    Cuts automatically or manually the lateral borders of an OCT image either automatically or manually.
    Parameter 'auto' allows automatic detection and removal of saturation artifacts. 
    If 'auto' is set to False, the user can manually select the boundaries of the saturation artifact.

    Parameters:
        OCT_brut (numpy.ndarray): The raw OCT image.
        pixel_size_x (float): The size of a pixel in the x-direction.
        auto (bool, optional): If True, the cut is done automatically. If False, the user selects the cut boundaries manually. Default is True.
        show (bool, optional): If True, the image with the selected boundaries is shown. Default is True.
        left_right_crop (int, optional): Number of pixels to crop from the left and right sides. Default is 0.

    Returns:
        tuple: A tuple containing the x-coordinate of the left cut (xmin_cut) and the x-coordinate of the right cut (xmax_cut).
    """
    if auto:
        xmin_cut = 0 + left_right_crop
        xmax_cut = OCT_brut.shape[1] - left_right_crop
    else:
        # Manual cut (w/ cursor)
        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111, facecolor='w')
        plt.imshow(OCT_brut, cmap="gray")
        plt.title('Select the lateral boundaries')
        if left_right_crop != 0:
            plt.axvline(left_right_crop, color='tab:green', linestyle='dashed', linewidth=3)
            plt.axvline(OCT_brut.shape[1]-left_right_crop, color='tab:green', linestyle='dashed', linewidth=3)
        fig.cursor = Cursor(ax, horizOn=False, vertOn=True, useblit=True, color='g', linewidth=1)
        R = plt.ginput(2)
        # If not input : 
        if len(R) == 0:
            xmin_cut = 0 + left_right_crop
            xmax_cut = OCT_brut.shape[1] - left_right_crop
        else: 
            xmin_cut = int(min(round(R[0][0]), round(R[1][0])))
            xmax_cut = int(max(round(R[0][0]), round(R[1][0])))
            if xmin_cut==xmax_cut:
                xmin_cut = 0
                xmax_cut = OCT_brut.shape[1]
        # saturation_cut_width = int(round((xmax_cut-xmin_cut)*pixel_size_x))
        time.sleep(0.5)
        plt.close()
    return xmin_cut, xmax_cut

def anterior_surface_detection_and_flattening(OCT_cut, saturation_cut_width, xmin_cut, 
                               xmax_cut, derivative1_smooth, Path_eye, 
                               method, max_window_size=10, gauss_sigma=1, median_filter_width=15, 
                               w_filter=101, sgolay_order=2, show=True):
    """
    Detects and flattens the anterior surface of an OCT image.
    For the initial image flattening, a SNR-based intensity threshold is used for OCT image binarization.
    The anterior surface detection is based on a local z-maxima detection near the first corneal interface.

    Parameters:
        OCT_cut (ndarray): The input OCT image w/o saturation artifact
        saturation_cut_width (int): Width of the saturation cut.
        xmin_cut (int): Minimum x-coordinate for the cut.
        xmax_cut (int): Maximum x-coordinate for the cut.
        derivative1_smooth (ndarray): Smoothed first derivative of the image.
        Path_eye (str): Path of the eye image.
        method (str): Method for surface detection ('threshold' or 'z_max_in_window').
        max_window_size (int, optional): Maximum window size for 'z_max_in_window' method. Default is 10.
        gauss_sigma (float, optional): Sigma for Gaussian smoothing. Default is 1.
        median_filter_width (int, optional): Width of the median filter. Default is 15.
        w_filter (int, optional): Window size for Savitzky-Golay filter. Default is 101.
        sgolay_order (int, optional): Order of the Savitzky-Golay filter. Default is 2.
        show (bool, optional): Whether to show the plot of the detected surface. Default is True.
    Returns:
        tuple: A tuple containing:
            - anterior_surface_z_coordinate (ndarray): The z-coordinates of the detected anterior surface.
            - FlattenedImage (ndarray): The flattened OCT image.
            - seuil (float or nan): The threshold value used for detection (only for 'threshold' method).
    """
    SNR_2D = SNR(OCT_cut)
    OCT_smooth = img.gaussian_filter(OCT_cut, sigma = gauss_sigma)
    x_dim = np.shape(OCT_smooth)[1]
    
    if method == 'threshold':
        # SNR-based threshold computation
        seuil = int(-2*SNR_2D + 65) # (+ 65) #====================================> nom: anterior_surface_detection_threshold
        if saturation_cut_width != 0:
            if Path_eye == "PachyWide":
                seuil  = seuil - 10  #(-10)    
            elif np.any(derivative1_smooth[50:xmin_cut] > 0.25) or np.any(derivative1_smooth[xmax_cut:-50] > 0.25):
                if SNR_2D < 16:
                    seuil  = seuil
                else:                
                    seuil = seuil + 10
            else:
                seuil = seuil + 15     
        else:
            if np.any(OCT_smooth[0:50,] > 1.5*seuil):
                if SNR_2D < 15.5:
                    seuil  = seuil + 15                    
                else:
                    seuil = seuil + 25
        
        # Image thresholding
        image_seuil = OCT_smooth > seuil #========================================> nom: OCT_binary
        
        # In-depth maxima detection
        maxima = np.zeros(x_dim)
        for i in range(x_dim):
            ones_positions = np.where(image_seuil[:, i])[0]
            if ones_positions.shape == (0,):
                maxima[i] = 0
            else:
                maxima[i] = ones_positions[0]
        maxima_null_full = np.where(maxima==0)[0] #===============================> nom: empty_detection_x_coordinate
        if len(maxima_null_full) != 0:
            x_delta = 50 # =======================================================> nom: surface_linear_interpolation_window
            delim_bordure = 150 # ================================================> nom: edge_x_margin
            # if missing maxima near the image edges: linear interpolation
            if maxima_null_full[0] < delim_bordure: # left edge
                    maxima_null = maxima_null_full[maxima_null_full < delim_bordure]
                    x1 = maxima_null[-1]+1
                    y1 = maxima[x1]
                    x2 = x1 + x_delta
                    y2 = maxima[x2]
                    a = (y2-y1)/(x2-x1)
                    b = y1 - a*x1
                    temp = np.linspace(a*maxima_null[0]+b,y1,maxima_null.shape[0])
                    temp = temp.astype(int)
                    maxima[maxima_null] = temp
            if maxima_null_full[-1] > x_dim - delim_bordure: # right edge   
                    maxima_null = maxima_null_full[maxima_null_full > x_dim - delim_bordure]
                    x2 = maxima_null[0]-1
                    y2 = maxima[x2]
                    x1 = x2 - x_delta
                    y1 = maxima[x1]
                    a = (y2-y1)/(x2-x1)
                    b = y1 - a*x1
                    temp = np.linspace(y2,a*maxima_null[-1]+b,maxima_null.shape[0])
                    temp = temp.astype(int)
                    maxima[maxima_null] = temp
                    
    elif method == 'z_max_in_window':
        seuil = np.nan
        maxima = np.zeros(x_dim)
        for i in range(x_dim):
            colonne_i = OCT_cut[0:max_window_size, i]
            maxima[i] = np.argmax(colonne_i[:])

    # Interface smoothing to avoid peaks due to image noise
    maxima_smooth = sgn.medfilt(maxima, median_filter_width) # remove peaks
    anterior_surface_z_coordinate = sgn.savgol_filter(maxima_smooth, w_filter, sgolay_order) # smooth the surface
    z_shift = anterior_surface_z_coordinate - min(anterior_surface_z_coordinate)
    
    # Image flattening
    FlattenedImage = np.zeros(OCT_cut.shape)
    for j in range(x_dim):
        if int(round(z_shift[j])) > 0:
            FlattenedImage[:-int(round(z_shift[j])), j] = OCT_cut[int(round(z_shift[j])):, j]
        else:
            FlattenedImage[:, j] = OCT_cut[:, j]
    
    # Plot
    if (show) and (method == 'threshold'):
        plt.figure(3)
        plt.imshow(OCT_cut, cmap = 'gray')
        plt.title('Anterior surface detection')
        plt.plot(maxima, color='orange', label = 'Local maxima')
        plt.plot(maxima_smooth, color='yellow', label = 'Anterior surface (median filter)')
        plt.plot(anterior_surface_z_coordinate, '--', color='c', label = 'Anterior surface (median + Savitsky-Golay filter)')
        plt.legend(fontsize=11, loc = 'upper right')
        plt.show()

    return anterior_surface_z_coordinate, FlattenedImage, seuil

def stromal_z_segmentation(flattened_image, pixel_size_z, margin_postBowman, margin_preDescemet, min_peak_height=0, min_peak_distance=8, peak_threshold=0.01, peak_edge='both', display_peaks=False):
    """
    Detect peaks associated with stromal layers (epithelium, EBM, endothelium) in a corneal OCT image.
    Parameters:
    flattened_image (2D np-array): Corneal OCT image without saturation artifact, flattened.
    pixel_size_z (float): Pixel size in the z-direction.
    margin_postBowman (float): Margin after Bowman's layer in micrometers.
    margin_preDescemet (float): Margin before Descemet's membrane in micrometers.
    min_peak_height (int, optional): Minimum peak height. Default is 0.
    min_peak_distance (int, optional): Minimum peak distance in pixels. Default is 8 pixels (~35 µm).
    peak_threshold (float, optional): Minimal height difference with the closest neighbor. Default is 0.01.
    peak_edge (str, optional): Edge type for peak detection. Default is 'both'.
    display_peaks (bool, optional): If True, display the detected peaks. Default is False.
    Returns:
    tuple: 
        z_coordinates_stroma (tuple): Coordinates of the stromal peaks.
        z_coordinates_cornea (tuple): Coordinates of the corneal peaks.
        ep_cornee_um (int): Thickness of the cornea in micrometers.
        ep_stroma_tot_um (int): Total thickness of the stroma in micrometers.
        stroma_z_anterior_bound (int): Anterior boundary of the stroma in pixels.
        profondeur_max (int): Posterior boundary of the stroma in pixels.
        ep_stroma_analyse_um (int): Depth of the region of interest (ROI) in the stroma in micrometers.
    """
    # Segmentation parameters
    kpsh = False
    valley = False
    
    # Peak detection
    Mean_I_large = np.mean(flattened_image, axis=1) #=============================> nom: z_mean_intensity
    idx = np.where(Mean_I_large > 1.5*np.mean(Mean_I_large))        
    ind = idx[0][0] + detect_peaks(Mean_I_large[idx], min_peak_height, min_peak_distance, peak_threshold, peak_edge, kpsh, valley, show=False)
    
    try:
        if len(ind) > 3:
            if display_peaks:
                fig = plt.figure()
                ax = fig.add_subplot(111, facecolor='w')
                ax.plot(Mean_I_large[:2*ind[-1]], linewidth=1)
                ax.plot(ind, Mean_I_large[(ind),], '+', mfc=None, mec='r', mew=2, ms=8)
                plt.title('Stromal peaks detection: control')
                plt.xlabel('Depth [px]', fontsize = 15)
                plt.ylabel('Mean OCT intensity [a.u.]', fontsize = 15)
                plt.show()
        z_coordinates_stroma = (ind[1], ind[-1])
        z_coordinates_cornea = (ind[0], ind[-1])
    except:
        fig = plt.figure()
        ax = fig.add_subplot(111, facecolor='w')
        ax.plot(Mean_I_large[:2*ind[-1]], linewidth=1)
        ax.plot(ind, Mean_I_large[(ind),], '+', mfc=None, mec='r', mew=2, ms=8)
        plt.title('Stromal peaks detection: control')
        plt.xlabel('Depth [px]', fontsize = 15)
        plt.ylabel('Mean OCT intensity [a.u.]', fontsize = 15)

        plt.show()
        detect_peaks(Mean_I_large, min_peak_height, min_peak_distance, peak_threshold, peak_edge, kpsh, valley, show=True)
        
    # Coordinates
    ep_cornee_um = int(abs(z_coordinates_cornea[0]*pixel_size_z-z_coordinates_cornea[1]*pixel_size_z)) #================> nom: thickness_cornea_um
    ep_stroma_tot_um = int(abs(z_coordinates_stroma[1]*pixel_size_z-z_coordinates_stroma[0]*pixel_size_z)) #============> nom: thickness_stroma_um
    
    # In-depth (z-) boudaries of the ROI
    profondeur_min_um = z_coordinates_stroma[0]*pixel_size_z + margin_postBowman #=============> nom: roi_z_top_um
    profondeur_max_um = z_coordinates_stroma[1]*pixel_size_z - margin_preDescemet #============> nom: roi_z_bottom_um
    ep_stroma_analyse_um = int(abs(round(profondeur_max_um-profondeur_min_um))) #==============> nom: roi_depth_um
    
    stroma_z_anterior_bound = int(round(profondeur_min_um/pixel_size_z)) #================================> nom: roi_z_top_px
    profondeur_max = int(round(profondeur_max_um/pixel_size_z)) #==============================> nom: roi_z_bottom_px  
    
    return z_coordinates_stroma, z_coordinates_cornea, ep_cornee_um, ep_stroma_tot_um, stroma_z_anterior_bound, profondeur_max, ep_stroma_analyse_um

def stromal_x_boundaries(flattened_image, stroma_z_anterior_bound, stroma_z_posterior_bound, z_coordinates_cornea, apex_x_coordinate_refined, FOV_x_mm, saturation_cut_width_px, pixel_size_x, pixel_size_z, margin_postBowman, margin_preDescemet, pathname, MsgBox=np.nan, roi_width_mm=np.nan, mode='auto', save='False'):
    """
    Return coordinates of lateral (x-) stromal boundaries.
    Parameters:
        flattened_image (2D np-array): Corneal OCT image without saturation artifact, flattened.
        stroma_z_anterior_bound (int): Z-coordinate (px) of the anterior boundary of the stroma on flattened images.
        stroma_z_posterior_bound (int): Z-coordinate (px) of the posterior boundary of the stroma on flattened images.
        z_coordinates_cornea (tuple (int, int)): Z-coordinates (px) of the corneal boundaries on flattened images.
        apex_x_coordinate_refined (int): X-coordinate of the corneal apex.
        FOV_x_mm (float): Lateral field-of-view in mm (depends on the device acquisition mode).
        saturation_cut_width_px (int): Width (px) of the apex-centered saturation artifact cut.
        pixel_size_x (float): Lateral pixel size (µm).
        pixel_size_z (float): Axial pixel size (µm).
        margin_postBowman (int): Post-Bowman layer z-margin (µm).
        margin_preDescemet (int): Pre-Descemet membrane z-margin (µm).
        pathname (str): Path of the current folder.
        MsgBox (str, optional): Answer of the user approval canvas [default NaN].
        roi_width_mm (int, optional): Width of the stromal ROI, if specified by user [default NaN].
        mode (str): Chosen mode for ROI lateral delimitation, either 'auto' or 'manual' [default 'auto'].
        save (bool, optional): Approval or not for saving intermediate figures [default False].
    Returns:
    tuple: Coordinates of the left and right stromal boundaries (stroma_x_left, stroma_x_right).
    """
    
    if mode == 'auto':
        x_dim = flattened_image.shape[1]
        # Median smoothing to avoid noise detection
        idx_null_min = np.where(sgn.medfilt(img.gaussian_filter(flattened_image, sigma=0.5)[stroma_z_anterior_bound,],5)==0)
        idx_null_max = np.where(sgn.medfilt(img.gaussian_filter(flattened_image, sigma=0.5)[stroma_z_posterior_bound,],5)==0)
        # Missing signal localization
        stroma_x_left_min = idx_null_min[0][idx_null_min[0] < int(x_dim/2)]
        stroma_x_left_max = idx_null_max[0][idx_null_max[0] < int(x_dim/2)]
        stroma_x_right_min = idx_null_min[0][idx_null_min[0] > int(x_dim/2)]  
        stroma_x_right_max = idx_null_max[0][idx_null_max[0] > int(x_dim/2)]
        if len(stroma_x_left_max) != 0 and len(stroma_x_left_min) != 0:
            stroma_x_left = np.amax([np.amax(stroma_x_left_min),np.amax(stroma_x_left_max)]) +2
        elif len(stroma_x_left_max) != 0:
            stroma_x_left = np.amax(stroma_x_left_max) +2
        else:
            stroma_x_left = 0            
        if len(stroma_x_right_max) != 0 and len(stroma_x_right_min) != 0:
            stroma_x_right = np.amin([np.amin(stroma_x_right_min),np.amin(stroma_x_right_max)]) -2  # (indexation via tuple)
        elif len(stroma_x_right_max) != 0:
            stroma_x_right = np.amin(stroma_x_right_max) -2
        else:
            stroma_x_right = x_dim
            
        # if reduced ROI_x vs. FOV_x
        if roi_width_mm < stroma_x_right-stroma_x_left:
            x_dim_raw = flattened_image.shape[1]+saturation_cut_width_px
            radius = int(np.round(0.5*x_dim_raw*roi_width_mm/FOV_x_mm,0))
            radius_cut = int(np.round(saturation_cut_width_px/2,0))
            if (apex_x_coordinate_refined-radius > stroma_x_left) and (stroma_x_right-apex_x_coordinate_refined > radius):
                stroma_x_left = int(apex_x_coordinate_refined - radius + radius_cut)
                stroma_x_right = int(apex_x_coordinate_refined + radius - radius_cut)
            else:
                if apex_x_coordinate_refined-radius < stroma_x_left:
                    stroma_x_right = int(stroma_x_left + 2*radius - 2*radius_cut)
                elif stroma_x_right-apex_x_coordinate_refined < radius:
                    stroma_x_left = int(stroma_x_right - 2*radius + 2*radius_cut) 
            
    elif mode == 'manual':
        # cursor
        fig = plt.figure(6)
        plt.clf()
        ax = fig.add_subplot(111, facecolor='w')
        ax.imshow(flattened_image, cmap="gray")
        plt.show()
        y2, y1 = ax.get_ylim()
        roi_x_px = np.round((roi_width_mm*1000-saturation_cut_width_px*pixel_size_x)/pixel_size_x,1)
        plt.title('Select the lateral boundaries of the stroma. Dots: %.1f mm field of view.' %np.round(roi_width_mm,1))
        if (apex_x_coordinate_refined-int(roi_x_px/2)>=0) and (apex_x_coordinate_refined+int(roi_x_px/2)<=flattened_image.shape[1]):
            plt.vlines(apex_x_coordinate_refined-int(roi_x_px/2), ymin=y1, ymax=y2, color='r', linestyle='dashed', linewidth=1)
            plt.vlines(apex_x_coordinate_refined+int(roi_x_px/2), ymin=y1, ymax=y2, color='r', linestyle='dashed', linewidth=1)
        else:            
            if apex_x_coordinate_refined-int(roi_x_px/2)<0:
                plt.vlines(0.05, ymin=y1, ymax=y2, color='r', linestyle='dashed', linewidth=1)
                plt.vlines(int(roi_x_px)+0.05, ymin=y1, ymax=y2, color='r', linestyle='dashed', linewidth=1)
            elif apex_x_coordinate_refined+int(roi_x_px/2)>flattened_image.shape[1]:
                plt.vlines(flattened_image.shape[1]-int(roi_x_px), ymin=y1, ymax=y2, color='r', linestyle='dashed', linewidth=1)
                plt.vlines(flattened_image.shape[1], ymin=y1, ymax=y2, color='r', linestyle='dashed', linewidth=1)
                
        fig.cursor = Cursor(ax, horizOn=False, vertOn=True, useblit=True, color='r', linewidth=1)
        R1 = plt.ginput(2)    
        stroma_x_left1 = int(min(round(R1[0][0]), round(R1[1][0])))
        stroma_x_right1 = int(max(round(R1[0][0]), round(R1[1][0])))
        plt.close()
        
        if MsgBox == "no": # on the flattened OCT image
            fig = plt.figure(6)
            plt.clf()
            ax = fig.add_subplot(111, facecolor='w')
            plt.imshow(flattened_image, cmap="gray")
            plt.axvline(stroma_x_left1, color='r', linewidth=1, linestyle='dashed')
            plt.axvline(stroma_x_right1, color='r', linewidth=1, linestyle='dashed')     
            plt.title('Select the in-depth boundaries of stroma (Bowman\'s layer/endothelium)')
            plt.xlabel('Depth [px]', fontsize=15)
            plt.ylabel('Mean OCT intensity', fontsize=15)        
            fig.cursor = Cursor(ax, horizOn=True, vertOn=False, useblit=True, color='r', linewidth=1)
            R2 = plt.ginput(2)
            coord_stroma1 = (int(min(round(R2[0][1]), round(R2[1][1]))),int(max(round(R2[0][1]), round(R2[1][1]))))
            stroma_z_anterior_bound1 = int(coord_stroma1[0] + margin_postBowman/pixel_size_z)
            stroma_z_posterior_bound1 = int(coord_stroma1[1] - margin_preDescemet/pixel_size_z)
            stroma_z_anterior_bound1_um = int(stroma_z_anterior_bound1*pixel_size_z + margin_postBowman)
            stroma_z_posterior_bound1_um = int(stroma_z_posterior_bound1*pixel_size_z - margin_preDescemet)
            
            
        else:# on the averaged in-depth intensity profile
            profondeur_inf1 = 0
            profondeur_sup1 = z_coordinates_cornea[1] + 100
            CutImage = flattened_image[profondeur_inf1:profondeur_sup1, stroma_x_left1:stroma_x_right1]
            Mean_I = np.mean(CutImage, axis=1)
            z = np.arange(0, np.shape(Mean_I)[0]*pixel_size_z, pixel_size_z)
            fig = plt.figure(6)
            ax = fig.add_subplot(111, facecolor='w')
            plt.plot(Mean_I, linewidth=2)
            plt.title('Select the two peaks corresponding to the Bowman`s layer and the endothelium:')
            plt.show()
            fig.cursor = Cursor(ax, horizOn=False, vertOn=True, useblit=True, color='r', linewidth=1)
            R2 = plt.ginput(2)
            coord_stroma1 = (int(min(round(R2[0][0]), round(R2[1][0]))),int(max(round(R2[0][0]), round(R2[1][0]))))
            stroma_z_anterior_bound1 = int(coord_stroma1[0] + margin_postBowman/pixel_size_z)
            stroma_z_posterior_bound1 = int(coord_stroma1[1] - margin_preDescemet/pixel_size_z)
            stroma_z_anterior_bound1_um = int(stroma_z_anterior_bound1*pixel_size_z + margin_postBowman)
            stroma_z_posterior_bound1_um = int(stroma_z_posterior_bound1*pixel_size_z - margin_preDescemet)
        ep_cornee_um = (coord_stroma1[1]-z_coordinates_cornea[0])*pixel_size_z
        ep_stroma_tot = round((stroma_z_posterior_bound-stroma_z_anterior_bound)*pixel_size_z)
        
        # Display #############################################################  
        CutImage = flattened_image[:coord_stroma1[1]+100, stroma_x_left1:stroma_x_right1]
        Mean_I_large_adjust = np.mean(CutImage, axis=1)
        SD_I_large_adjust = np.std(CutImage, axis=1)
        
        #% Average in-depth intensity profile
        N_pix_depth = np.shape(Mean_I_large_adjust)[0]
        z = np.arange(0, N_pix_depth, 1)
        z = pixel_size_z*z
        
        fig = plt.figure(5)
        fig.clf()
        ax = fig.add_subplot(211, facecolor='w')
        ax.plot(z,Mean_I_large_adjust, linewidth=2)
        ax.plot(tuple(pixel_size_z*np.array(coord_stroma1)), Mean_I_large_adjust[(coord_stroma1),], '+', mfc=None, mec='r', mew=2, ms=8)
        plt.fill_between(z, Mean_I_large_adjust-SD_I_large_adjust, Mean_I_large_adjust+SD_I_large_adjust,
        alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=0)
        plt.axvline(stroma_z_anterior_bound1_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(stroma_z_posterior_bound1_um, ymin=0, ymax=1, color='r', linestyle='dashed', linewidth=1)  
        plt.title('In-depth stromal segmentation')
        
        text_str = 'Corneal thickness = %i µm\n Full stromal thickness = %i µm (%.1f%%)\n Analyzed FOV = %.1f mm'%(ep_cornee_um,ep_stroma_tot,100*round(ep_stroma_tot*pixel_size_z/ep_cornee_um,3),np.round((stroma_x_right1-stroma_x_left1+pixel_size_x+saturation_cut_width_px)/1000,1))
        (ymin,ymax) = ax.get_ylim()
        plt.text(np.amax(z)*1.04,0.7*ymax,text_str,horizontalalignment='right',fontsize=9)
        plt.title('Stromal segmentation')
        plt.xlabel('Depth [µm]', fontsize=15)
            
        ax = fig.add_subplot(212)
        plt.imshow(flattened_image[:stroma_z_posterior_bound1+100,], cmap="gray")
        plt.hlines(stroma_z_anterior_bound1, xmin=stroma_x_left1, xmax=stroma_x_right1, color='r', linestyle='dashed', linewidth=1)
        plt.hlines(stroma_z_posterior_bound1, xmin=stroma_x_left1, xmax=stroma_x_right1, color='r', linestyle='dashed', linewidth=1)
        plt.vlines(stroma_x_left1, ymin=stroma_z_anterior_bound1, ymax=stroma_z_posterior_bound1, color='r', linestyle='dashed', linewidth=1)
        plt.vlines(stroma_x_right1, ymin=stroma_z_anterior_bound1, ymax=stroma_z_posterior_bound1, color='r', linestyle='dashed', linewidth=1)
        plt.axis('off')
        plt.show()
        if save:
            plt.savefig(pathname + '/2_segmentation_stroma_champ_%.1fmm_std.png'%((stroma_x_right1-stroma_x_left1)*pixel_size_x/1000), dpi=200, bbox_inches='tight')
        
        if MsgBox == "no":
            fig = plt.figure(5)
            plt.show()
            
        stroma_x_left = stroma_x_left1
        stroma_x_right = stroma_x_right1
        
    else:
        print('ERROR: The \'mode\' variable received a wrong argument (\''+mode+'\' != \'auto\' OR \'manual\').')
    
    return stroma_x_left, stroma_x_right


def stromal_sublayers(N_layers, FlattenedImage2, w_filterLayers, coord_stroma, colonne_inf, colonne_sup, corr='True', save='False', show='True'):
    """
    Return the coordinates and mean signal of N stromal sub-layers.
    Parameters:
        N_layers (int): Number of stromal ROI sub-layers.
        FlattenedImage2 (2D np.array): Corneal OCT image without saturation artifact, flattened.
        w_filterLayers (int, optional): Length (px) of the Savitzky-Golay filter window (must be less than OCT_cut.shape[1]). Default is 101.
        coord_stroma (tuple of int): z-coordinates (px) of the stromal boundaries on flattened images.
        colonne_inf (int): x-coordinate (px) of the left boundary of the ROI.
        colonne_sup (int): x-coordinate (px) of the right boundary of the ROI.
        corr (str, optional): Correction flag. Default is 'True'.
        save (str, optional): Save flag. Default is 'False'.
        show (str, optional): Show flag. Default is 'True'.
    Returns:
        tuple: A tuple containing:
            - coord_layers (np.array): Coordinates of the stromal sub-layers.
            - Mean_I_layers (np.array): Mean signal of each stromal sub-layer.
            - colors (list): List of colors used for plotting.
    """  
    profondeur_Bowman = coord_stroma[0]
    profondeur_endoth = coord_stroma[1]

    # Layer coordinates
    colors = [plt.cm.jet(i) for i in np.linspace(0, 1, N_layers+1)]   # color gradient
    coord_layers = np.around(np.linspace(profondeur_Bowman, profondeur_endoth, N_layers+1))
    coord_layers = coord_layers.astype(int)
        # Signal averaging to each layer
    Mean_I_layers = np.zeros((len(coord_layers)-1,colonne_sup-colonne_inf))
    for i in range(0,len(coord_layers)-1):
        if w_filterLayers != None:      # Smoothing
            Mean_I_layers[i,:] = sgn.savgol_filter(np.mean(FlattenedImage2[coord_layers[i]:coord_layers[i+1],colonne_inf:colonne_sup], axis=0), w_filterLayers, 2)
        else:                           # No smoothing
            Mean_I_layers[i,:] = np.mean(FlattenedImage2[coord_layers[i]:coord_layers[i+1],colonne_inf:colonne_sup], axis=0)
            
    # Mean signals display
    if show and not save:
        plt.figure(8)
        ax = plt.axes()
        ax.set_prop_cycle('color', colors)
        lineM = plt.plot(Mean_I_layers.T, linewidth = 1)
        plt.legend(lineM, range(1,len(coord_layers)), title='n° layer', fontsize = 'xx-small', loc='center left', bbox_to_anchor=(0.95, 0.5))
        plt.title('Averaged and smoothed x-signals for N=%i stromal layers' %N_layers)
        plt.xlabel('Lateral (x-) coordinate [px]', fontsize=15)
        plt.ylabel('Raw mean OCT intensity', fontsize=15)
        plt.show()
    
        # Stromal slicing display
    if show:
        fig = plt.figure(9)
        ax = fig.add_subplot(111, facecolor='#FFFFCC')
        plt.imshow(FlattenedImage2, cmap="gray")
        plt.title('Stromal in-depth slicing')
        plt.axis('off')
        
        plt.figure(9)
        ln_width = 0.45
        for i in range(1,len(coord_layers)-2):
            plt.hlines(coord_layers[i], xmin=colonne_inf, xmax=colonne_sup, color=colors[i], linestyle='dashed', linewidth=ln_width, label='_nolegend_')
            plt.hlines(coord_layers[i+1], xmin=colonne_inf, xmax=colonne_sup, color=colors[i], linestyle='dashed', linewidth=ln_width, label='_nolegend_')
            plt.vlines(colonne_inf, ymin=coord_layers[i], ymax=coord_layers[i+1], color=colors[i], linestyle='dashed', linewidth=ln_width, label='_nolegend_')
            plt.vlines(colonne_sup, ymin=coord_layers[i], ymax=coord_layers[i+1], color=colors[i], linestyle='dashed', linewidth=ln_width, label='_nolegend_')
        plt.axis('off')
        lineL, = plt.plot(np.ones([len(FlattenedImage2),1]), color='r', linestyle='dashed', linewidth=1, label='Segmentation totale du stroma')
        lineL.set_visible(False)
        plt.show()
            
    return coord_layers, Mean_I_layers, colors
            
def posterior_stromal_artifact_localization(Mean_I_layers, colors, mph1, mpd1, threshold1, marge_centre, pathname, show=False, save=False):
    """
    Return the left and right (x-) localization of the posterior stromal artifact.

    Parameters:
        - Mean_I_layers (list of 1D np-arrays): In-depth averaged OCT intensity of the flattened corneal image.
        - colors (list): List of colors for sub-layers plot.
        - mph1 (float, optional): Minimum peak height, between 0 and 1. Default is None.
        - mpd1 (int, optional): Minimum peak distance in pixels. Default is 50.
        - threshold1 (float, optional): Minimum difference with neighbors, between 0 and 1. Default is 0.
        - marge_centre (int): Margin in pixels around the center of the standard deviation peak.
        - pathname (str): Path of the current folder.
        - show (bool, optional): If True, display the plot. Default is False.
        - save (bool, optional): If True, save the plot. Default is False.
    Returns:
        - coord_art_min (int): Left (x-) localization of the posterior stromal artifact.
        - coord_art_max (int): Right (x-) localization of the posterior stromal artifact.
    """
    # PCA_1
    X = StandardScaler(with_std=False).fit_transform(Mean_I_layers)    # standardized data (centered + scaled)
    success = False
    initial_rate = .98
    while not success:
        try:
            pca1 = PCA(initial_rate)                                                    # PCs are kept until 98% of variance is explained
            x_new = pca1.fit_transform(X)
            #print('explained variance [%]: PCA_1')
            #print(100*pca1.explained_variance_ratio_)
            
            nPC_1 = np.shape(pca1.explained_variance_ratio_)[0]
            coeff1 = np.transpose(pca1.components_[0:nPC_1, :])      # eigenvectors
            score1 = x_new[:,0:nPC_1]                                # eigenvalues
            # print('score1 shape:', np.shape(score1)," exmpl var ratio: ", pca1.explained_variance_ratio_," nPC_1:",nPC_1)
            # The second principal component (PC2) used to define the lateral boundaries of the artifact
            PC2 = np.outer(score1[:,1],coeff1.T[1,:])
            success = True
            # print('PCA_1 succeeded with initial_rate = %.2f'%initial_rate)
        except:
            print('PCA_1 failed with initial_rate = %.2f'%initial_rate)
            initial_rate = initial_rate + 0.1*(1-initial_rate)
    std_PC2_smooth = sgn.savgol_filter(np.std(PC2,axis=0),75,2)
    centre_std2 = np.where(np.abs(PC2[-1,]) == np.amax(np.abs(PC2[-1,])))[0][0]    
    ind = detect_peaks(std_PC2_smooth, mph1, mpd1, threshold1, edge='falling', kpsh=False, valley=True, show=show, ax=None, title=True)
    temp = ind[ind > centre_std2 + marge_centre]
    if len(temp) != 0:
        coord_art_max = np.amin(temp)
    else:
        coord_art_max = len(std_PC2_smooth)
    temp = ind[ind < centre_std2 - marge_centre]
    if len(temp) != 0:
        coord_art_min = np.amax(temp)
    else:
        coord_art_min = 0
    
    # Plot
    if show:
        plt.figure(11)
        ax = plt.axes()
        ax.set_prop_cycle('color', colors)
        plt.plot(PC2.T, linewidth = 1)
        xmin, xmax, ymin, ymax = plt.axis()
        plt.axvline(coord_art_min, ymin, ymax, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(coord_art_max, ymin, ymax, color='r', linestyle='dashed', linewidth=1)
        plt.title('Posterior stromal artifact boundaries')
        plt.xlabel('Lateral (x-) coordinate [px]', fontsize=15)
        plt.ylabel('Centered reconstruction of the 2nd PC of PCA_1 [a.u.]', fontsize=15)
        plt.show()
        if save:
                plt.savefig(pathname + '\\3_corr_delim_artefact.png', dpi=150, bbox_inches='tight')
            
    return coord_art_min, coord_art_max #========================================> nom: xmin_posterior_artifact, xmax_posterior_artifact

def posterior_stromal_artifact_mask(Mean_I_layers, coord_stroma, coord_art_min, coord_art_max, coord_layers, FlattenedImage2, colonne_inf, colonne_sup, w_filterLayers, colors, profondeur_min, profondeur_max, pas, pathname, show=True, save=False):
    """
    Compute a correction mask based on the Principal Component Analysis of the in-depth mean lateral (x-) intensity.

    Parameters:
        Mean_I_layers (1D np-array): In-depth averaged OCT intensity of the flattened corneal image.
        coord_stroma (tuple (int, int)): Z-coordinates (px) of the stromal boundaries on flattened images.
        coord_art_min (int): Left (x-) coordinate of the posterior artifact boundary.
        coord_art_max (int): Right (x-) coordinate of the posterior artifact boundary.
        coord_layers (1D np-array): Vector of stromal sub-layers (z-) coordinates.
        FlattenedImage2 (2D np-array): Corneal OCT image without saturation artifact, flattened.
        colonne_inf (int): X-coordinate (px) of the left boundary of the ROI.
        colonne_sup (int): X-coordinate (px) of the right boundary of the ROI.
        w_filterLayers (positive odd int, optional): Length (px) of the Savitzky-Golay filter window (must be less than OCT_cut.shape[1]) [default 101].
        colors (list): List of colors for sub-layers plot.
        profondeur_min (int): Z-coordinate (px) of the anterior boundary of the stroma on flattened images.
        profondeur_max (int): Z-coordinate (px) of the posterior boundary of the stroma on flattened images.
        pas (float): Axial pixel size (µm).
        pathname (str): Path of the current folder.
        show (bool, optional): Whether to display the plots [default True].
        save (bool, optional): Whether to save the plots [default False].
    Returns:
        mask (2D np-array): Computed correction mask.
        mask_offset (float): Offset value used in the mask computation.
        CorrImage (2D np-array): Corrected corneal OCT image.
    """
    N_layers = Mean_I_layers.shape[0]
    profondeur_Bowman = coord_stroma[0]
    profondeur_endoth = coord_stroma[1]

    # Data centering in the artifact area
    Mean_I_artefact = Mean_I_layers[:,coord_art_min:coord_art_max]
    sz_artefact = np.shape(Mean_I_artefact)
    Mean_I_artefact_offset = np.zeros(np.shape(Mean_I_artefact))
    if np.std(Mean_I_artefact[1:20,:]) < np.std(Mean_I_artefact[:,sz_artefact[1]-20:sz_artefact[1]]):
        for i in range(0,N_layers):
            Mean_I_artefact_offset[i,:] = Mean_I_artefact[i,:] - min(Mean_I_artefact[i,1:20])
    else:
        for i in range(0,N_layers):
            Mean_I_artefact_offset[i,:] = Mean_I_artefact[i,:] - min(Mean_I_artefact[i,sz_artefact[1]-20:sz_artefact[1]])

    # PCA_2
    X = StandardScaler(with_std=False).fit_transform(Mean_I_artefact_offset)    # standardized data (centered + scaled)
    pca2 = PCA(0.995)                                                           # PCs are kept until 99.5% of variance is explained
    x_new = pca2.fit_transform(X)

    nPC_2 = np.shape(pca2.explained_variance_ratio_)[0]
    coeff2 = np.transpose(pca2.components_[0:nPC_2, :])                         # eigenvectors
    score2 = x_new[:,0:nPC_2]                                                   # eigenvalues
    score2_PC1 = score2[:,0]
    score2_PC1_smooth = sgn.savgol_filter(score2_PC1, int(2*np.around(score2_PC1.shape[0]/2)-1), 2)
    xp = coord_layers[:-1] + 3
    score2_PC1_interp = np.interp(range(profondeur_Bowman, profondeur_endoth), xp, score2_PC1_smooth, left=None, right=None, period=None)

    # First principal component (PC1) reconstruction
    PC1 = np.outer(score2_PC1_interp,coeff2.T[0,:])    

    # Mask computation
    PC1_base = PC1 - PC1[1,:]                                                   # First sublayer serves as reference
    PC1_base[PC1_base<0] = 0                                                    # Intensity cannot increase with depth

    mask_offset = 1.1*np.max(PC1_base)
    mask = mask_offset*np.ones((profondeur_endoth-profondeur_Bowman,colonne_sup-colonne_inf))
    mask[:,coord_art_min:coord_art_max] = mask[:,coord_art_min:coord_art_max] - PC1_base
        
    # Correction
    ROI = deepcopy(FlattenedImage2[profondeur_Bowman:profondeur_endoth,colonne_inf:colonne_sup])
    newROI = deepcopy(ROI) + mask - mask_offset

    if show:
        plt.figure(12)
        plt.subplot(311)
        plt.imshow(mask-mask_offset,cmap = 'gray')
        plt.title('Mask')
        plt.axis('on')
        plt.colorbar()
        plt.subplot(312)
        plt.imshow(ROI,cmap = 'gray')
        plt.title('Initial ROI')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(313)
        plt.imshow(newROI,cmap = 'gray')
        plt.title('Corrected ROI')
        plt.axis('off')
        plt.colorbar()
        plt.show()
        if save:
                plt.savefig(pathname + '\\4_corr_masque.png', dpi=150, bbox_inches='tight')

    # Control on images
    if show:
        Mean_I_layers_ctrl = np.zeros(np.shape(Mean_I_layers))
        for i in range(0,len(coord_layers)-1):
            if w_filterLayers != None:      # Lissage
                Mean_I_layers_ctrl[i,:] = sgn.savgol_filter(np.mean(newROI[coord_layers[i]-coord_layers[0]:coord_layers[i+1]-coord_layers[0],:], axis=0), w_filterLayers, 2)
            else:                           # Pas de lissage
                Mean_I_layers_ctrl[i,:] = np.mean(newROI[coord_layers[i]-coord_layers[0]:coord_layers[i+1]-coord_layers[0],:], axis=0)
                
        plt.figure(8)    
        plt.clf()
        plt.subplot(121)
        for i in range(N_layers):
            label_str = "%i" %i
            plt.plot(Mean_I_layers[i,:].T, linewidth=1, c=colors[i], label=label_str)
        xmin, xmax, ymin_2, ymax_2 = plt.axis()    
        plt.axvline(coord_art_min, 0, ymax_2, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(coord_art_max, 0, ymax_2, color='r', linestyle='dashed', linewidth=1)
        plt.ylim((ymin_2, ymax_2))
        plt.title('Before correction')
        plt.xlabel('Lateral (x-) coordinate [px]', fontsize=15)
        plt.ylabel('Mean OCT intensity [8-bit grayscale]', fontsize=15)
        plt.show()
        
        ax = plt.subplot(122)
        for i in range(N_layers):
            label_str = "%i" %i
            plt.plot(Mean_I_layers_ctrl[i,:].T, linewidth=1, c=colors[i], label=label_str)
        plt.legend(title='layer', fontsize = 'x-small', loc='center left', bbox_to_anchor=(0.95, 0.5))
        plt.axvline(coord_art_min, 0, ymax_2, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(coord_art_max, 0, ymax_2, color='r', linestyle='dashed', linewidth=1)
        ax.set_ylim((ymin_2,ymax_2))
        plt.title('After correction')
        plt.xlabel('Lateral (x-) coordinate [px]', fontsize=15)
        plt.ylabel('Mean OCT intensity [8-bit grayscale]', fontsize=15)
        plt.show()
        if save:
                plt.savefig(pathname + '\\4_corr_comparaison_%i_layers.png' %N_layers, dpi=150, bbox_inches='tight')
        
        # Control on (z-) intensity profile
        depth_ROI = np.shape(np.mean(ROI, axis=1))[0]
        width_ROI = np.shape(np.mean(ROI, axis=0))[0]
        x = np.arange(pas/2, depth_ROI*pas, pas)
        yerr1 = np.std(ROI, axis=1)/np.sqrt(width_ROI-1) # standard error of mean
        yerr2 = np.std(newROI, axis=1)/np.sqrt(width_ROI-1)
        
        plt.figure(13)
        plt.clf()
        plt.errorbar(x, np.mean(ROI, axis=1), yerr1, label='before mask correction (±SEM)', color='k', linewidth =1, capsize = 2)
        plt.errorbar(x, np.mean(newROI, axis=1), yerr2, label='after mask correction (±SEM)',color='r', linewidth = 1, capsize = 2)
        plt.xlim(0, depth_ROI*pas)
        ymin1, ymax1 = plt.ylim()    
        plt.ylim(ymin1,ymax1)
        lineW, = plt.plot(np.ones([depth_ROI,1]), color='r', linestyle='dashed', linewidth=1, label='ROI (z-) boundaries')
        plt.title('In-depth intensity profile comparison')
        plt.xlabel('Stromal depth [µm]', fontsize=15)
        plt.ylabel('Mean OCT intensity [8-bit grayscale]', fontsize=15)
        plt.legend(loc='upper right', fontsize='x-small')
        plt.vlines((profondeur_min-profondeur_Bowman)*pas, ymin=ymin1, ymax=ymax1, color='r', linestyle='dashed', linewidth=1)
        plt.vlines((profondeur_max-profondeur_Bowman)*pas, ymin=ymin1, ymax=ymax1, color='r', linestyle='dashed', linewidth=1)
        plt.show()
        lineW.set_visible(False)

        if save:
            plt.savefig(pathname + '\\4_corr_comparaison_profils.png', dpi=150, bbox_inches='tight')
        
    CorrImage = deepcopy(FlattenedImage2)
    CorrImage[profondeur_Bowman:profondeur_endoth,colonne_inf:colonne_sup] = deepcopy(newROI)  
    return mask, mask_offset, CorrImage
            
def geometrical_normalization(input_image, coord_stroma, corr, coord_centre, pas_lat, xmin_cut, xmax_cut, profondeur_min, profondeur_max, colonne_inf, colonne_sup, pathname, w_filterNorm=115, sgolay_orderNorm=2, save=False):
    """
    Perform geometrical normalization on a given OCT image.
    Parameters:
        input_image (2D np-array): Flattened OCT image, corrected or not.
        coord_stroma (tuple (int, int)): Z-coordinates (px) of the stromal boundaries on flattened images.
        corr: (Not used in the function)
        coord_centre (int): Lateral (x-) coordinate of the corneal apex (px).
        pas_lat (float): Lateral pixel size (µm).
        xmin_cut (int): X-coordinate (px) of the left boundary of the ROI.
        xmax_cut (int): X-coordinate (px) of the right boundary of the ROI.
        profondeur_min (int): Z-coordinate (px) of the anterior boundary of the stroma on flattened images.
        profondeur_max (int): Z-coordinate (px) of the posterior boundary of the stroma on flattened images.
        colonne_inf (int): (Not used in the function)
        colonne_sup (int): (Not used in the function)
        pathname (str): Path of the current folder.
        w_filterNorm (int, optional): Length (px) of the Savitzky-Golay filter window (must be less than OCT_cut.shape[1]). Default is 115.
        sgolay_orderNorm (int, optional): Order of the polynomial used to fit the samples (must be less than w_filter). Default is 2.
        save (bool, optional): If True, saves the processed image. Default is False.
    Returns:
        tuple: Contains the following elements:
            - VectNorm (np-array): Normalized vector.
            - ProcessedImage (np-array): Processed image after normalization.
            - epith_sat_um (int): Epithelial saturation in micrometers.
            - quant (float): Quantile value used for saturation calculation.
    """

    M = input_image[:int(coord_stroma[1]*1.25), :]
    VectMax = M.max(axis=0)
    VectNorm = sgn.savgol_filter(VectMax, w_filterNorm, sgolay_orderNorm) # median smoothing is too noisy
    VectNorm[VectNorm<=75] = 75
    ProcessedImage = deepcopy(input_image/VectNorm)
    VectNorm_sat = np.where(VectMax==255)[0]
    quant = 0.05
    if VectNorm_sat.shape[0] != 0:        
        epith_sat_min = int(np.quantile(VectNorm_sat,quant))
        epith_sat_max = int(np.quantile(VectNorm_sat,1-quant))
    else:
        epith_sat_min = coord_centre
        epith_sat_max = coord_centre
    epith_sat_um = int((epith_sat_max-epith_sat_min)*pas_lat)
    del(M) 
    if save:
            plt.savefig(pathname + '\\6_resume_OCT.png', dpi=150, bbox_inches='tight')
    return VectNorm, ProcessedImage, epith_sat_um, quant
   

def preprocessing_OCT(filename, pathname, field_analyse_mm, margin_postBowman, margin_preDescemet, save, corr, user_validation=False, plots=False, left_right_crop=0):
    """Run the preprocessing of RTVue-XR SD-OCT corneal images.
    
    Keyword arguments:
        filename (str)
        pathname (str)
        field_analyse_mm (float)
        margin_postBowman (int, in µm)
        margin_preDescemet (int, in µm)
        save (bool)
        corr (bool)
        user_validation (bool)
    """
    if plots:
        # Graphical parameters (plt)
        plt.figure(1)
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['figure.figsize'] = 14.5, 8                # figure size [inch]
        fontsize_title = 20                                     # title font size
        fontsize_label = 15                                     # axis label font size
        plt.rcParams.update({'font.size': fontsize_title})
        plt.rcParams.update({'xtick.labelsize': 'xx-small'})    # x-tick font size
        plt.rcParams.update({'ytick.labelsize': 'xx-small'})    # y-tick font size
        plt.rcParams.update({'axes.labelsize': 'small'})
        fig_dpi = 150                                           # resolution of exported figures

    # Clock
    t0_time = time.time()
    
    #%% AUTOMATED PREPROCESSING ALGORITHM
    #%% (A.1) Initialization
    plt.close('all')

    # Default dictionary: values chosen for RTVue-XR Avanti SD-OCT images.
    dict = {}
    dict.setdefault('max_window_size', 10) #==========================================> nom: anterior_surface_z_window [px]
    dict.setdefault('profondeur_min', None) #=========================================> nom: stroma_z_top
    dict.setdefault('profondeur_max', None) #=========================================> nom: stroma_z_bottom
    dict.setdefault('median_filter_width',15) #=======================================> nom: median_filter_kernel_size
    dict.setdefault('w_filter',101) #=================================================> nom: savgol_filter_window
    dict.setdefault('sgolay_order',2)
    dict.setdefault('w_filterNorm',115) #=============================================> nom: savgol_filter_window_norm
    dict.setdefault('nLayers', 20) #==================================================> nom: n_layers
    dict.setdefault('w_filterLayers',275) # int ou None ==============================> nom: savgol_filter_window_layers
    
    # Metadata import

    # Raw OCT image import
    X = mpimg.imread(os.path.join(pathname,filename))
    extension = '.' + filename.split('.')[-1]
    z0 = 10                                 # image top-marging width [px]
    if extension == '.jpg':
        OCT_brut = X[z0:, 2:, 0]*1.0                # conversion into float64 #=========> OCT
    elif extension == '.jpeg':
        OCT_brut = X[z0:, 2:]*1.0                   # conversion into float64 #=========> OCT
    # #print(X.shape)
    
    target=""
    if X.shape[1] == 1020:  # Test image size to define pixel size
        if np.mean(X[0:70, X.shape[1]-70:X.shape[1]]) - np.mean(X[0:40, 0:40]) > 4:
            target = 'Pachy'
        else:
            target = 'Line'
    elif X.shape[1] == 940:
        target='Cross Line'
    elif X.shape[1]==1536:
        target='Pachymetry Wide'
    Path_eye=target

    if 'Line' in Path_eye: #===========================================================> ? en faire une fonction ?
        dict.setdefault('champ_acquisition_mm', 8)          # FOV = 8 mm ==============> nom: FOV_x_mm
        pas = 4.333                                         # [µm] axial pixel size ===> nom: pixel_size_z
        if 'Cross' in Path_eye:
            Path_eye = 'Cross'
        else:
            Path_eye = 'Line'
    elif 'Pachy' in Path_eye:
        if 'Wide' in Path_eye:
            Path_eye = 'PachyWide'
            dict.setdefault('champ_acquisition_mm', 9)      # FOV = 9 mm
            pas = 4.322                                     
        else:
            Path_eye = 'Pachy'
            dict.setdefault('champ_acquisition_mm', 6)      # FOV = 6 mm
            pas = 4.333
    
    pas_lat = round(1000*dict['champ_acquisition_mm']/OCT_brut.shape[1],2) # [µm] lateral pixel size ===> nom: pixel_size_x
          
    # Deletion of the "orientation arrow" on raw images
    sz_arrow = 70 #=================================================================> nv nom = ARROW_SIZE
    OCT_brut[0:sz_arrow,np.shape(OCT_brut)[1]-sz_arrow:np.shape(OCT_brut)[1]] = np.zeros((sz_arrow,sz_arrow)) # remove the scan orientation arrow
    
    #%% (A.2) Exposure correction of overexposed images
    OCT = OCT_brut
    OCT_brut = histogram_shift(OCT_brut, show=plots)
    
    #%% (B) Saturation artifact detection
    coord_centre = None #============================================================> nom: apex_x_coordinate
    if 'field_analyse_mm' not in locals():
        field_analyse_mm = dict['champ_acquisition_mm']
    auto = True
    coord_centre, xmin_cut, xmax_cut, SpecularCut, derivative1_smooth = saturation_artifact_cut( #========> nom: SpecularCut -> saturation_cut_width
            OCT_brut, pas_lat, auto, show=plots) #====================================================================> nom: xmin_cut => xmin_saturation_cut, idem pour xmax_cut
    
    ###############################################################################
    ############################### User approval #################################
    ###############################################################################
    
    if user_validation:    
        global result
        result = None
        temp = None
        temp_exit = None

        result_auto = saturation_artifact_cut(OCT_brut, pas_lat, auto=True,show=False)
        result = saturation_artifact_cut(OCT_brut, pas_lat, auto=False,show=False)
        if result is None:
            result = result_auto
        xmin_cut = result[1]
        xmax_cut = result[2]

        if result is not None:
            xmin_cut = result[1]
            xmax_cut = result[2]
            coord_centre = result[0]
            SpecularCut = result[3] #=======================================================> nom: saturation_cut_width
            derivative1_smooth = result[4]
        
        if temp is not None:
            SpecularCut = 0 #==============================================================> nom: saturation_cut_width
            xmin_cut = 0
            xmax_cut = 0 

        del temp, temp_exit
        
    ###############################################################################      
    # OCT image cutting

    if 'xmin_cut' in locals():
        OCT_cut_brut = np.delete(OCT_brut, range(xmin_cut, xmax_cut), axis=1) #========> nom: OCT_cut (ATTENTION : renommer d'abord OCT_cut en OCT_cut_smooth !!)
    
    ###########
    if left_right_crop != 0:
        OCT_cut_brut = OCT_cut_brut[:, left_right_crop:-left_right_crop]
    if user_validation:
        x_min_crop, x_max_cut = lateral_borders_cut(OCT_cut_brut,pixel_size_x=4.333,auto=False,show=True,left_right_crop=left_right_crop)
        OCT_cut_brut = OCT_cut_brut[:, x_min_crop:x_max_cut]
    # ############    
    result = None

    #%% (C) Anterior surface detection and (D.1) corneal flattening
    sgolay_order = dict['sgolay_order']                 # order of the polynomial used to fit the samples (must be less than w_filter) [default 2].
    
    anterior_surface_z_coordinate, FlattenedImage, seuil = anterior_surface_detection_and_flattening(OCT_cut_brut, SpecularCut, xmin_cut,
                                   xmax_cut, derivative1_smooth, Path_eye, 
                                   method='threshold', max_window_size=10, gauss_sigma=1, median_filter_width=15, 
                                   w_filter=101, sgolay_order=2, show=plots) #====================================> nom: FlattenedImage => flattened_OCT
    
    columns = np.shape(OCT_cut_brut)[1] #========================================================================> nom: x_dim
    depth = np.shape(OCT_cut_brut)[0] # =========================================================================> nom: z_dim
    
    coord_centre2 = int(np.where(anterior_surface_z_coordinate == np.amin(anterior_surface_z_coordinate[100:columns-100]))[0][0]) # ==========> nom: apex_x_coordinate_refined
    
    # Refined flattening
    anterior_surface_z_coordinate = anterior_surface_z_coordinate - min(anterior_surface_z_coordinate)
    offset = int(round(min(anterior_surface_z_coordinate))) # global z-offset before corneal apex
    
    anterior_surface_z_coordinate_refined, FlattenedImage2, _ = anterior_surface_detection_and_flattening(FlattenedImage, SpecularCut, xmin_cut, 
                                                                                                       xmax_cut, derivative1_smooth, Path_eye,
                                                                                                       method='z_max_in_window', max_window_size=10,
                                                                                                       gauss_sigma=1, median_filter_width=15,
                                                                                                       w_filter=101, sgolay_order=2, show=False) #==========> nom: flattened_OCT_refined
    
    #%% (D.2) Stromal segmentation
    show = False
    # In-depth (z-) segmentation of the cornea and the ROI
    coord_stroma, coord_cornee, ep_cornee_um, ep_stroma_tot_um, profondeur_min, profondeur_max, ep_stroma_analyse_um = stromal_z_segmentation(
            FlattenedImage2, pas, margin_postBowman, margin_preDescemet, min_peak_height=0, min_peak_distance=8, peak_threshold=0.01, peak_edge='both', display_peaks=show)
    saturation_cut_width_px = int(np.round(SpecularCut/pas_lat,0))
    champ_acquisition_mm = dict['champ_acquisition_mm'] #============================> en orienté objet = attribut de classe
    colonne_inf, colonne_sup = stromal_x_boundaries(
            FlattenedImage2, profondeur_min, profondeur_max, coord_cornee, coord_centre2, champ_acquisition_mm, saturation_cut_width_px, pas_lat, pas, margin_postBowman, margin_preDescemet, pathname, roi_width_mm=field_analyse_mm, mode='auto')
    CutImage = deepcopy(FlattenedImage2[:int(coord_cornee[1]*1.25), colonne_inf:colonne_sup])
    Mean_I_large_adjust = np.mean(CutImage, axis=1)  #==========================================> nom: z_mean_intensity  
    N_pix_depth = np.shape(Mean_I_large_adjust)[0]  #===========================================> nom: z_dim (existe déjà : le même ?)
    
    
    
    #%% (E.1) Definition of stromal sub-layers
    N_layers = dict['nLayers']
    w_filterLayers = dict['w_filterLayers']
    coord_layers, Mean_I_layers, colors = stromal_sublayers(N_layers, FlattenedImage2, w_filterLayers, coord_stroma, colonne_inf, colonne_sup, corr='True', save='False', show=plots)

    #%% (E.2) Localization of posterior stromal artifact
    mph1 = None                    # minimum peak height [0;1]
    mpd1 = 50                      # [px] minimum peak distance 
    threshold1 = 0                 # min difference with neighbours [0;1]
    marge_centre = 60              # [px] margin around the center of the standard deviation peak.
    coord_art_min, coord_art_max = posterior_stromal_artifact_localization(Mean_I_layers, colors, mph1, mpd1, threshold1, marge_centre, pathname, save=False)
    
    #%% (F) PCA-based correction mask
    mask, mask_offset, CorrImage = posterior_stromal_artifact_mask(Mean_I_layers, coord_stroma, coord_art_min, coord_art_max, coord_layers, FlattenedImage2, colonne_inf, colonne_sup, w_filterLayers, colors, profondeur_min, profondeur_max, pas, pathname, show=plots, save=False)
    OCT_mask=CorrImage

    #%% (G) Lateral normalization
    w_filterNorm = dict['w_filterNorm']         # length of the Savitzky-Golay filter window
    sgolay_orderNorm = sgolay_order             # order of the polynomial used to fit the samples
    input_image = CorrImage
    VectNorm, ProcessedImage, epith_sat_um, quant = geometrical_normalization(input_image, coord_stroma, corr, coord_centre, pas_lat, xmin_cut, xmax_cut, profondeur_min, profondeur_max, colonne_inf, colonne_sup, pathname, w_filterNorm, sgolay_orderNorm, save=False)
    
    #%% (H) Signal averaging with depth 
    coord_ROI = ((colonne_inf, colonne_sup), (profondeur_min, profondeur_max))
    CutImage = ProcessedImage[profondeur_min:profondeur_max,colonne_inf:colonne_sup]
    
    #%% END
    time_end = time.time()
    time_exe = round(time_end-t0_time,1)
    # #print('\nEND [t=%s s]'%time_exe)
    
    if (0):
        plt.close('all')
        
    return ProcessedImage, mask, coord_ROI, time_exe, [OCT,OCT_brut,OCT_cut_brut,OCT_mask]
        
        

        
        
