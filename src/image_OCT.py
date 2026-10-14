import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import scipy.ndimage as img

import sys
dirname = os.getcwd()
sys.path.append(dirname + '/src')
sys.path.append(dirname + '/..')
print(">>",dirname)
try: 
    from preproc_functions import preprocessing_OCT
except: 
	from preproc_functions import preprocessing_OCT

class image_OCT(object):
	"""
	A class to represent and process OCT (Optical Coherence Tomography) images.
	Attributes
	----------
	Path : str
		Full path to the image file.
	pas : float
		Axial pixel size in micrometers.
	pas_lat : float
		Lateral pixel size in micrometers.
	OCT_brut : ndarray
		Raw OCT image data.
	filename : str
		Name of the image file.
	pathname : str
		Path to the directory containing the image file.
	field_analyse_mm : int
		Width of the analyzed area in millimeters.
	margin_postBowman : int
		Post-Bowman layer z-margin in micrometers.
	margin_preDescemet : int
		Pre-Descemet membrane z-margin in micrometers.
	save : bool
		Flag to save figures and .txt/.csv files.
	corr : bool
		Flag to compute posterior artefact correction.
	user_validation : bool
		Flag for user validation.
	plots : bool
		Flag to display plots.
	left_right_crop : int
		Crop value for left and right sides of the image.
	OCT : ndarray
		Processed OCT image data.
	OCT_cut_brut : ndarray
		Cut raw OCT image data.
	OCT_mask : ndarray
		Mask of the OCT image.
	OCT_flat : ndarray
		Flattened OCT image data.
	Methods
	-------
	AutoTreatment():
		Automatically processes the OCT image.
	set_OCT_flat(OCT_flat):
		Sets the flattened OCT image data.
	ImageSNR(gauss_sigma=1):
		Calculates the Signal-to-Noise Ratio (SNR) of the OCT image.
	PlotImage():
		Plots the different stages of the OCT image processing.
	"""
	def __init__(self, pathname, filename,user_validation=False,plots= False, left_right_crop=0):
		self.Path = os.path.join(pathname,filename)

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

		field_analyse_mm = 6        # [mm] width of analyzed area (comment the line if analyzing the full image width)
		margin_postBowman = 60       # [µm] post-Bowman layer z-margin [default 60 µm]
		margin_preDescemet = 30      # [µm] pre-Descemet membrane z-margin [default 30 µm]
		save = False                # save figures and .txt/.csv files (bool)  
		corr = True                 # compute posterior artefact correction (bool)
		plots= False

		# Raw OCT image import
		X = mpimg.imread(self.Path)
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
		else:
			raise ValueError('Image size in px not recognize')
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
		else:
			raise ValueError('champ_acquisition_mm not set')

		pas_lat = round(1000*dict['champ_acquisition_mm']/OCT_brut.shape[1],2) # [µm] lateral pixel size ===> nom: pixel_size_x
		self.pas=pas
		self.pas_lat=pas_lat
  
		# Deletion of the "orientation arrow" on raw images
		sz_arrow = 70 #=================================================================> nv nom = ARROW_SIZE
		OCT_brut[0:sz_arrow,np.shape(OCT_brut)[1]-sz_arrow:np.shape(OCT_brut)[1]] = np.zeros((sz_arrow,sz_arrow)) # remove the scan orientation arrow
		
		self.OCT_brut=OCT_brut

		self.filename=filename
		self.pathname=pathname
		self.field_analyse_mm=field_analyse_mm
		self.margin_postBowman=margin_postBowman
		self.margin_preDescemet=margin_preDescemet
		self.save=save
		self.corr=corr
		self.user_validation=user_validation
		self.plots=plots
		self.left_right_crop = left_right_crop

	def AutoTreatment(self):
		ProcessedImage, mask, coord_ROI, time_exe, images_ttm = preprocessing_OCT(self.filename, self.pathname, self.field_analyse_mm, self.margin_postBowman, self.margin_preDescemet, self.save, self.corr, self.user_validation,self.plots, self.left_right_crop)
		[OCT,OCT_brut,OCT_cut_brut,OCT_mask]=images_ttm		
		self.OCT = OCT
		self.OCT_brut = OCT_brut
		self.OCT_cut_brut = OCT_cut_brut
		self.OCT_mask = OCT_mask
		self.OCT_flat=ProcessedImage
  
	def set_OCT_flat(self,OCT_flat):
		self.OCT_flat=OCT_flat

	def ImageSNR(self,gauss_sigma=1):
		mImage=self.OCT_flat
		signal = np.sum(np.square(mImage))
		noise = np.sum(np.square(mImage-img.gaussian_filter(mImage,sigma=gauss_sigma)))
		SNR = np.around(10*np.log10(signal/noise),1)
		return(SNR)

	def PlotImage(self):
		try:
			plt.figure(figsize=(15,10))
			plt.subplot(2,3,1)
			plt.imshow(self.OCT,cmap='gray')
			plt.title('OCT')
			plt.subplot(2,3,2)
			plt.imshow(self.OCT_brut,cmap='gray')
			plt.title('OCT brut')
			plt.subplot(2,3,3)
			plt.imshow(self.OCT_cut_brut,cmap='gray')
			plt.title('OCT cut brut')
			plt.subplot(2,3,4)
			plt.imshow(self.OCT_mask,cmap='gray')
			plt.title('OCT mask')
			plt.subplot(2,3,5)
			plt.imshow(self.OCT_flat,cmap='gray')
			plt.title('OCT flat')
			plt.show()
		except:
			print('No image to plot')
			pass
