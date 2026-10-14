import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from os import listdir
from os.path import isfile, join
import pandas as pd

from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from tqdm import tqdm
from PIL import Image

try:
    from image_OCT import *
except:
    from image_OCT import *

def gaus(x,a,x0,sigma):
	"""
	Gaussian function.

	Parameters:
	x (float or array-like): The input value(s) where the Gaussian function is evaluated.
	a (float): The amplitude of the Gaussian function.
	x0 (float): The mean or the center of the Gaussian peak.
	sigma (float): The standard deviation, which controls the width of the Gaussian peak.

	Returns:
	float or array-like: The value of the Gaussian function at x.
	"""
	return a*np.exp(-(x-x0)**2/(2*sigma**2))

def exp_decr(x, a, b, c):
	"""
	Exponential decay function.

	Parameters:
	x (float or array-like): The input value(s) for the function.
	a (float): The initial value coefficient.
	b (float): The decay rate.
	c (float): The offset value.

	Returns:
	float or array-like: The result of the exponential decay function applied to the input value(s).
	"""
	return a * np.exp(-b * x) + c

def create_new_dataset(repository,healthy_value,movingWin=False,eliminate=0,useExpCorr=False):
	"""
	Create a new dataset by quantifying images in the specified repository.

	Parameters:
	repository (str): The path to the directory containing the images to be quantified.
	healthy_value (int or float): The value indicating the health status to be assigned to the quantified images.
	movingWin (bool, optional): Whether to use a moving window for quantification. Default is False.
	eliminate (int, optional): The value to be used for elimination during quantification. Default is 0.
	useExpCorr (bool, optional): Whether to use exponential correction during quantification. Default is False.

	Returns:
	pd.DataFrame: A DataFrame containing the quantification parameters for each successfully processed image.

	Notes:
	- Only images with a '.jpg' extension are processed.
	- The function prints progress and error messages during execution.
	- The function uses the `image_OCT` and `image_Quantification` classes for image processing and quantification.
	- Temporary saving of the flat image is enabled by default.
	"""
	print(">Create dataset from: ",repository)
	files = [f for f in listdir(repository) if isfile(join(repository, f))]
	print("		",str(len(files))," files detected")
	quantification=[]
	i=0
	i_list=[]
	for f in tqdm(files,desc=f"Quantification of {repository}"):
		if(f[-3:]=="jpg"):
			i+=1
			try:
				# im=image_OCT(os.path.join(repository, f))
				im=image_OCT(repository, f)
			except:
				print("Error with: "+f+" image #"+str(i))
			try:
				im.AutoTreatment()
				im_Q=image_Quantification(im,movingWin=movingWin,eliminate=eliminate,useExpCorr=useExpCorr)
				# Temporary save of the flat image: 
				save_im = True
				if save_im:
					im_Q.Plot_All()
					pass
				parameters=im_Q.parameters
				parameters["Healthy"]=healthy_value
				parameters["Origin"]=f[:-4]
				quantification.append(parameters)
				# print(i,":",healthy)
				i_list.append(i)
			except:
				print("Error with: "+f+" image #"+str(i))
	n=len(quantification)
	print(n,"/",len(files)," files succesfully quantified")
	data=pd.DataFrame(quantification,index=range(n)) 
	return(data)

def create_new_dataset_soft(repository,ttm_repository,healthy_value,movingWin=False,eliminate=0,useExpCorr=False,appended_name="",extension="jpg",extension_ttm = "tiff",plot=False,window=100,moving_win_ponderation=False):
	"""
	Creates a new dataset by quantifying images before and after a transformation.
	Parameters:
		repository (str): Path to the directory containing the original images.
		ttm_repository (str): Path to the directory containing the transformed images.
		healthy_value (int): Value indicating the health status of the images.
		movingWin (bool, optional): Whether to use a moving window for quantification. Default is False.
		eliminate (int, optional): Parameter for elimination in quantification. Default is 0.
		useExpCorr (bool, optional): Whether to use exponential correction. Default is False.
		appended_name (str, optional): Name to append to the transformed image files. Default is "".
		extension (str, optional): File extension of the original images. Default is "jpg".
		extension_ttm (str, optional): File extension of the transformed images. Default is "tiff".
		plot (bool, optional): Whether to plot the quantification results. Default is False.
		window (int, optional): Window size for quantification. Default is 100.
		moving_win_ponderation (bool, optional): Whether to use moving window ponderation. Default is False.
	Returns:
		pd.DataFrame: DataFrame containing the quantification results.
	Raises:
		ValueError: If no corresponding transformed image is found for an original image.
	"""
	print(">Create dataset from: ",repository)
	files = [f for f in listdir(repository) if isfile(join(repository, f))]
	files_ttm = [f for f in listdir(ttm_repository) if isfile(join(ttm_repository, f))]
	print("		",str(len(files))," files detected before ttm")
	print("		",str(len(files_ttm))," files detected after ttm")
	quantification=[]
	i=0
	i_list=[]
	for f in tqdm(files,desc=f"Quantification of {repository}"):
		if(f[-len(extension):]==extension):
			i+=1
			try:
				im=image_OCT(repository, f)
			except:
				print("Error with: "+f+" image #"+str(i))
			
			try: 
				#Find corresponding file in files_ttm: 
				if len(appended_name)>0:
					f_ttm_name = f[:-(len(extension)+1)] + f"_{appended_name}" 
				else: 
					f_ttm_name = f[:-(len(extension)+1)]
				f_ttm_name = f_ttm_name + "." + extension_ttm
				f_ttm = [ft for ft in files_ttm if ft == f_ttm_name]
				if len(f_ttm) == 0:
					raise ValueError(f"No corresponding file found for: {f} looked for {f_ttm_name}")
				f_ttm = f_ttm[0]
			except:
				print(f"No corresponding file found for: {f} looked for {f_ttm_name}")
				continue

			try:
				im_ttm = Image.open(join(ttm_repository,f_ttm))
				im_ttm = np.array(im_ttm)
				im.set_OCT_flat(im_ttm)
				im_Q=image_Quantification(im,movingWin=movingWin,eliminate=eliminate,useExpCorr=useExpCorr,window=window,moving_win_ponderation=moving_win_ponderation)
				if plot:
					im_Q.Plot_All()
				parameters=im_Q.parameters
				parameters["Healthy"]=healthy_value
				parameters["Origin"]=f[:-len(extension)]
				quantification.append(parameters)
				i_list.append(i)
			except:
				print("Error with: "+f+" image #"+str(i))
	n=len(quantification)
	print(n,"/",len(files)," files succesfully quantified")
	data=pd.DataFrame(quantification,index=range(n)) 
	return(data)

class image_Quantification(object):
	"""
	A class used to quantify images from OCT (Optical Coherence Tomography) scans.
	Attributes:
	-----------
		im : image_OCT
			An instance of the image_OCT class.
		image : np.ndarray
			The processed OCT image.
		image_init : np.ndarray
			The initial OCT image.
		N : int
			The number of columns in the OCT image minus the window size.
		N_p : int
			The number of columns processed.
		count : int
			The number of successfully processed windows.
		missed : int
			The number of missed windows.
		parameters : dict
			A dictionary containing quantification parameters.
		iProfileCrop : np.ndarray
			The cropped intensity profile.
		iProfileExp : np.ndarray
			The exponential fit of the intensity profile.
		iProfileExpCorrection : np.ndarray
			The corrected intensity profile.
		xmin : int
			The position of the first peak.
		Alpha : float
			The alpha parameter of the exponential fit.
		Beta : float
			The beta parameter of the exponential fit.
		peaks : np.ndarray
			The positions of the peaks in the intensity profile.
		hollows : list
			The positions of the hollows around the second peak.
		iProfileBell : np.ndarray
			The bell-shaped profile.
		iProfileBellGauss : np.ndarray
			The Gaussian fit of the bell-shaped profile.
		MSE : float
			The mean squared error of the Gaussian fit.
		Sigma : float
			The standard deviation of the Gaussian fit.
		Mean : float
			The mean position of the Gaussian fit. Also named depth. 
		PeakWidth : float
			The width of the peak.
		DataCov : float
			The covariance between the fitted data and the initial data.
		Area_ratio : float
			The ratio of the area under the curve to the total area.
		underBowman : bool
			Indicates if fibrosis under Bowman is detected.
	Methods:
	--------
		__init__(image_OCT_element, movingWin=False, eliminate=0, useExpCorr=False, window=100, moving_win_ponderation=False):
			Initializes the image_Quantification object and computes the profile and quantifiers.
		getPeaks(intentityProfile, displayedPeak=3):
			Returns the first `displayedPeak` peak positions on the intensity profile.
		getLowPeak(profile, peaks):
			Finds the first local hollows around the second peak of the intensity profile.
		Profile_quantification(displayedPeak=3, pre_window=2, post_window=100):
			Computes the profile and fits an exponential decay to it.
		Quantification_parameters(useExpCorr=False):
			Computes various quantification parameters from the intensity profile.
		Study_bell_curve(useExpCorr=False):
			Analyzes the bell-shaped curve of the intensity profile.
		Plot_ProfileExp():
			Plots the exponential fit of the intensity profile.
		Plot_ProfileQuantif(useExpCorr=False):
			Plots the quantification results of the intensity profile.
		Plot_All():
			Plots all the relevant profiles and quantification results.
	"""

	def __init__(self, image_OCT_element,movingWin=False,eliminate=0,useExpCorr=False,window=100,moving_win_ponderation = False):
		"""[Initialise the creation of the object image_Quantification by computiong the profile and calculate the quantifiers via Profile_quantification(_moving) and Quantification_parameters(_moving)]
		Args:
			image_OCT_element ([image_OCT]): [object of the class image_OCT]
			plot (bool, optional): [to rather plot fligure automatically or not]. Defaults to False.
			moving (bool, optional): [use the moving widow algorithm (True) or on the entire image (False)]. Defaults to False.

		Raises:
			ValueError: [Error during profile quantification or reation of the parameters]
		"""
		self.im=image_OCT_element
		post_window=200
		if eliminate==0:
			self.image=self.im.OCT_flat
			self.image_init=self.im.OCT_flat
		else:
			width=self.im.OCT_flat.shape[1]
			to_eliminate=int(width*eliminate/100/2)
			self.image=self.im.OCT_flat[:,to_eliminate:(width-to_eliminate)]
			self.image_init=self.im.OCT_flat[:,to_eliminate:(width-to_eliminate)]
		try:
			N=self.im.OCT_flat.shape[1]-window
			self.N=N
			N_p=N
			count=0
			missed=0
			N_sum = 0
   
			PeakWidth=0
			Sigma=0
			DataCov=0
			Mean=0
			MSE=0
			Area_ratio=0
			Alpha=0
			Beta=0
			IntensityPeak=0
			profile=0
			if movingWin:
				PeakWidth=0;Sigma=0;DataCov=0;Mean=0;MSE=0;Area_ratio=0;Alpha=0;Beta=0;IntensityPeak=0
				for i in range(N):
					self.image=self.im.OCT_flat[:,i:(i+window)]
					# print(self.image)
					try:
						self.Profile_quantification()
						self.Quantification_parameters(useExpCorr)
						if moving_win_ponderation:
							coef = (i+window//2)/N
						else:
							coef = 1
						PeakWidth+=coef*self.PeakWidth
						Sigma+=coef*self.Sigma
						DataCov+=coef*self.DataCov
						Mean+=coef*self.Mean
						MSE+=coef*self.MSE
						Area_ratio+=coef*self.Area_ratio
						Alpha+=coef*self.Alpha
						Beta+=coef*self.Beta
						IntensityPeak+=coef*self.IntensityPeak
						if (i==0 or (N_p==N-i)):
							profile=self.iProfileCrop
						else:
							profile+=self.iProfileCrop #self.iProfileCrop[1,:]
						count+=1
						N_sum += coef
					except:
						N_p-=1
						missed+=1
				self.PeakWidth=PeakWidth/N_sum
				self.Sigma=Sigma/N_sum
				self.DataCov=DataCov/N_sum
				self.Mean=Mean/N_sum
				self.MSE=MSE/N_sum
				self.Area_ratio=Area_ratio/N_sum
				self.Alpha=Alpha/N_sum
				self.Beta=Beta/N_sum
				self.IntensityPeak=IntensityPeak/N_sum
				self.parameters={
					"PeakWidth": self.PeakWidth,
					"Sigma": self.Sigma,
					"DataCov": self.DataCov,
					"Mean": self.Mean,
					"MSE": self.MSE,
					"AreaRatio": self.Area_ratio,
					"Alpha": self.Alpha,
					"Beta": self.Beta,
					"IntensityPeak":self.IntensityPeak
				}
				self.iProfileCrop=profile
				self.N_p=N_p
				self.count=count
				self.missed=missed
				# print(str(N_p)+"/"+str(N)+" window computed")
			else:
				self.Profile_quantification()
				self.Quantification_parameters(useExpCorr)
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification init: Image quantification impossible"
			raise ValueError(message)

	def getPeaks(self,intentityProfile,displayedPeak=3):
		"""[Return the #displayedPeak first peak position on the intencity profile in a 1D np-array]

		Args:
			intentityProfile ([1D np-array]): [Mean axial profile]
			displayedPeak (int, optional): [# of peaks to detect]. Defaults to 3.
		"""
		indexMax=np.argmax(intentityProfile)
		peak=argrelextrema(intentityProfile, np.greater)
		npeak1=np.unique(np.where(peak >= indexMax, peak,indexMax)[0])
		return(npeak1[0:displayedPeak])


	def getLowPeak(self,profile,peaks):
		"""[From the position of the second peak of the intensity profile, find the first local hollows in both direction arround the peak.
  			Hollows are detected by looking for local minimum, in case there is no local min:
     		- If it is a constant decrease, then lowpeak is define as the last ascan position available on the side studied
       		A filter is then applied:
         	The two lowpeaks are selected to have the same distance from the central peak as the minimal distance from the central peak to the two possition of lateral hollows]

		Args:
			peak ([1D np-array]): [first peaks position on the axial intencity profile]
			cropedProfile ([type]): [description]
			xmin ([type]): [description]
		"""
		peakLow=argrelextrema(profile, np.less)+self.xmin
		low1=np.unique(np.where(peakLow < peaks[1], peakLow,peaks[1])[0])#[:-1]
		if(low1.shape[0]>1):
				xlow1=low1[-1]
				if xlow1==peaks[1]:
					xlow1=low1[-2]
		else:
			xlow1=low1[0]
		
		low2=np.where(peakLow >= peaks[1], peakLow,peaks[1])[0]
		low2=low2[profile[low2-self.xmin]<0.8*profile[peaks[1]-self.xmin]]
		low2=np.unique(low2)
		if(low2.shape[0]>1):
			xlow2=low2[0]
			if xlow2==peaks[1]:
				xlow2=low2[1]
		elif (low2.shape[0]==1):
			xlow2=low2[0]
			if xlow2==peaks[1]:
				xlow2=profile.shape[0]+self.xmin-1
		else:
			xlow2=profile.shape[0]+self.xmin-1
		self.hollows=[xlow1,xlow2]

	def Profile_quantification(self,displayedPeak=3,pre_window=2,post_window=100):
		"""[summary]
		Args:
			displayedPeak (int, optional): [#of peak to compute]. Defaults to 2.
			window (int, optional): [number of ascan to consider left to first peak and right to second peak for ploting]. Defaults to 10.

		Raises:
			ValueError: [Error if the profile can be averaged]
		"""
		self.pre_window=pre_window
		self.post_window=post_window
		try:
			self.intentityProfile=np.mean(self.image,1)							#Compute the profile
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification Profile quantification: OCT_flat do not exist, run AutoTreatment or Flatenning before Profile_quantification"
			raise ValueError(message)

		peak=self.getPeaks(self.intentityProfile,displayedPeak)
		xdata=np.arange(0,post_window)
		ydata=self.intentityProfile[peak[0]:peak[0]+post_window]
		ydata=(ydata-np.min(ydata))/(np.max(ydata)-np.min(ydata))
		popt, pcov = curve_fit(exp_decr, xdata, ydata)
		ydata_fit=exp_decr(xdata, *popt)
		self.iProfileCrop=ydata
		self.iProfileExp=ydata_fit
		self.iProfileExpCorrection=self.iProfileCrop-self.iProfileExp
		self.xmin=peak[0]
		self.Alpha=popt[0]
		self.Beta=popt[1]
		self.peaks=peak

	def Quantification_parameters(self,useExpCorr=False):
		intensity_peak_bowman=self.iProfileExpCorrection[self.peaks[1]-self.xmin]
		intensity_peak_under_bowman=self.iProfileExpCorrection[self.peaks[2]-self.xmin]
		self.underBowman=False
		if(intensity_peak_bowman<=intensity_peak_under_bowman/0.9):
			self.underBowman=True
			peaks=self.peaks[1:]
		else:
			peaks=self.peaks
		self.IntensityPeak=self.iProfileCrop[peaks[1]-self.xmin]
		if useExpCorr:
			self.getLowPeak(self.iProfileExpCorrection,peaks)
			self.iProfileBell=self.iProfileExpCorrection[(self.hollows[0]-self.xmin):(self.hollows[1]+1-self.xmin)]
		else:
			self.getLowPeak(self.iProfileCrop,peaks)
			self.iProfileBell=self.iProfileCrop[(self.hollows[0]-self.xmin):(self.hollows[1]+1-self.xmin)]
		self.Study_bell_curve(useExpCorr)
		self.parameters={
			"PeakWidth": self.PeakWidth,
			"Sigma": self.Sigma,
			"DataCov": self.DataCov,
			"Mean": self.Mean,
			"MSE": self.MSE,
			"AreaRatio": self.Area_ratio,
			"Alpha": self.Alpha,
			"Beta": self.Beta,
			"IntensityPeak":self.IntensityPeak
		}
		return(0)

	def Study_bell_curve(self,useExpCorr=False):
		PeakWidth=(self.hollows[1]-self.hollows[0])*self.im.pas
		ydata_init=self.iProfileBell-np.min(self.iProfileBell)
		n=ydata_init.shape[0]
		xdata_init=np.arange(0,n)
		mean=np.argmax(ydata_init)
		sigma=0.1
		tck = interpolate.splrep(xdata_init, ydata_init, s=0)
		xdata = np.arange(0, n-1, 1/5) 				#Multiply by 5 points number
		ydata = interpolate.splev(xdata, tck, der=0)
		popt,pcov = curve_fit(gaus,xdata,ydata,p0=[1,mean,sigma])
		## Std of the gaussian function
		sigma=popt[2]*self.im.pas
		## Mean position of gaussian function
		mean=(popt[1]+self.hollows[0]-self.xmin)*self.im.pas
		ydata_fit=gaus(xdata_init,*popt)
		## Correlation between fitted data and initial data
		DataCov=np.cov(ydata_init,ydata_fit)
		DataCov=DataCov[0,1]
		## MSE between fitted data and initial data
		MSE=np.sqrt(np.sum((ydata_init-ydata_fit)**2))
  
		# Area calculation
		if useExpCorr:
			ydata=self.iProfileExpCorrection-np.min(self.iProfileExpCorrection)
		else:
			ydata=self.iProfileCrop-np.min(self.iProfileCrop)
		xdata=np.arange(0,len(ydata))
		x1=self.hollows[0];y1=ydata[self.hollows[0]-self.xmin]
		x2=self.hollows[1];y2=ydata[self.hollows[1]-self.xmin]
		m=(y1-y2)/(x1-x2);p=y1-m*x1
		x_lin_interp=np.arange(x1,x2,1)
		y_lin_interp=m*x_lin_interp+p
		ydata_c=np.concatenate((ydata[:(self.hollows[0]-self.xmin)],y_lin_interp,ydata[(self.hollows[1]-self.xmin):]))
		area_exp = np.trapz(ydata_c, dx=1)
		area_tot = np.trapz(ydata, dx=1)
		area_ratio = (area_tot-area_exp)/area_tot  
  
		self.iProfileBellGauss=ydata_fit
		self.MSE=MSE
		self.Sigma=sigma
		self.Mean=mean
		self.PeakWidth=PeakWidth
		self.DataCov=DataCov
		self.Area_ratio=area_ratio
  
	def Plot_ProfileExp(self):
		try:
			fig,ax=plt.subplots(1,3,figsize=(15,5))
			ax[0].imshow((self.image[self.peaks[0]-self.pre_window:self.peaks[0]+self.post_window][:]).T,aspect="auto",cmap="gray")
			ax[1].plot(self.iProfileCrop,'b')
			ax[1].plot(self.iProfileExp,'r--',label="a={:.2}, b={:.2}".format(self.Alpha,self.Beta))
			ax[1].plot(self.peaks-self.xmin,self.iProfileCrop[self.peaks-self.xmin],"bo",label="Peaks")
			ax[1].legend()
			ax[1].set_xlim([0,self.pre_window+self.post_window])
			ax[1].set_ylim([0,1])
			ax[2].plot(self.iProfileExpCorrection,'orange',label='Corrected profile')
			ax[2].plot(self.peaks-self.xmin,self.iProfileExpCorrection[self.peaks-self.xmin],color="orange",marker="o",linestyle='',label="Peaks")
			# ax[2].plot(savgol_filter(self.iProfileExpCorrection, 5, 2),'--')
			ax[2].legend()
			plt.show()
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification Profile quantification/Plot_ProfileExp: run Profile_quantification and Quantification_parameters before"
			raise ValueError(message)

	def Plot_ProfileQuantif(self,useExpCorr=False):
		try:
			fig,ax=plt.subplots(1,2,figsize=(15,5))
			if useExpCorr:
				ax[0].plot(self.iProfileExpCorrection,'orange',label='Corrected profile')
				ax[0].plot(self.peaks-self.xmin,self.iProfileExpCorrection[self.peaks-self.xmin],color="orange",marker="o",linestyle='',label="Peaks")
				ax[0].plot(self.hollows-self.xmin,self.iProfileExpCorrection[self.hollows-self.xmin],color="green",marker="o",linestyle='',label="Hollows")
			else:
				ax[0].plot(self.iProfileCrop,'orange',label='Corrected profile')
				ax[0].plot(self.peaks-self.xmin,self.iProfileCrop[self.peaks-self.xmin],color="orange",marker="o",linestyle='',label="Peaks")
				ax[0].plot(self.hollows-self.xmin,self.iProfileCrop[self.hollows-self.xmin],color="green",marker="o",linestyle='',label="Hollows")
			ax[0].set_title("Fibrosis under bowman detected:"+str(self.underBowman))
			ax[0].legend()
			ax[1].plot(self.iProfileBell-np.min(self.iProfileBell),label="Extracted bell profile")
			ax[1].plot(self.iProfileBellGauss,'r--',label="N(mean={:.2},sigma={:.2})".format(self.Mean,self.Sigma))
			ax[1].legend()
			plt.show()
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification Profile quantification/Plot_ProfileQuantif: run Profile_quantification before"
			raise ValueError(message)

	def Plot_All(self):
		try:
			fig,ax=plt.subplots(3,2,figsize=(10,15))
			ax[0][0].imshow((self.image[self.peaks[0]-self.pre_window:self.peaks[0]+self.post_window][:]).T,aspect="auto",cmap="gray")
			ax[1][0].plot(self.iProfileCrop,'b')
			ax[1][0].plot(self.iProfileExp,'r--',label="a={:.2}, b={:.2}".format(self.Alpha,self.Beta))
			ax[1][0].plot(self.peaks-self.xmin,self.iProfileCrop[self.peaks-self.xmin],"bo",label="Peaks")
			ax[1][0].legend()
			ax[1][0].set_xlim([0,self.pre_window+self.post_window])
			ax[1][0].set_ylim([0,1])
			ax[2][0].plot(self.iProfileExpCorrection,'orange',label='Corrected profile')
			ax[2][0].plot(self.peaks-self.xmin,self.iProfileExpCorrection[self.peaks-self.xmin],color="orange",marker="o",linestyle='',label="Peaks")
			# ax[2].plot(savgol_filter(self.iProfileExpCorrection, 5, 2),'--')
			ax[2][0].legend()
   
			ax[0][1].plot(self.iProfileExpCorrection,'orange',label='Corrected profile')
			ax[0][1].plot(self.peaks-self.xmin,self.iProfileExpCorrection[self.peaks-self.xmin],color="orange",marker="o",linestyle='',label="Peaks")
			ax[0][1].plot(self.hollows-self.xmin,self.iProfileExpCorrection[self.hollows-self.xmin],color="green",marker="o",linestyle='',label="Hollows")
			ax[0][1].set_title("Fibrosis under bowman detected:"+str(self.underBowman))
			ax[0][1].legend()
			ax[1][1].plot(self.iProfileBell-np.min(self.iProfileBell),label="Extracted bell profile")
			ax[1][1].plot(self.iProfileBellGauss,'r--',label="N(mean={:.2},sigma={:.2})".format(self.Mean,self.Sigma))
			ax[1][1].legend()
			plt.show()
		except:
			message="\n"+"*"*50+"\n"+"Error in image_Quantification Profile quantification/Plot_ProfileQuantif: run Profile_quantification before"
			raise ValueError(message)