#Author: Freddy Alcarazo
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from imutils import paths

class ImageMoments(object):
	def __init__(self, image):
		self.image = image
	def getMean(self, r_filtered, g_filtered, b_filtered):

		#Compute the mean for the 3 channels:
		r_mean = np.mean(r_filtered, axis=(0, 1))
		g_mean = np.mean(g_filtered, axis=(0, 1))
		b_mean = np.mean(b_filtered, axis=(0, 1))
		#r_mean[--, --, mean]
		#g_mean[--, mean, --]
		#b_mean[mean, --, --]
		#Guardar el color medio para las componentes R,G y B
		rgb_mean = []
		rgb_mean = [r_mean[2],g_mean[1],b_mean[0]]
		mean = []
		mean.append(rgb_mean[0])
		mean.append(rgb_mean[1])
		mean.append(rgb_mean[2])
		return (np.array(mean))

	def getStd(self, r_filtered, g_filtered, b_filtered):
		#Calcular la desviación estandar de los 3 channels:
		r_std = np.std(r_filtered, ddof=1)
		g_std = np.std(g_filtered, ddof=1)
		b_std = np.std(b_filtered, ddof=1)
		#Guardar valores de los 3 canales en una array
		rgb_std = []
		rgb_std = [r_std,g_std,b_std]
		#Guardar valores en un segundo array para retornar
		std = []
		std.append(rgb_std[0])
		std.append(rgb_std[1])
		std.append(rgb_std[2])
		return (np.array(std))

	def getSkewness(self, r_filtered, g_filtered, b_filtered):
		#Convert the matrices de los canales en un array 1-D
		#R
		r_filtered = np.array(r_filtered)
		r_filtered = r_filtered.ravel()
		#remover ceros
		r_filtered = r_filtered[r_filtered != 0]
		#G
		g_filtered = np.array(g_filtered)
		g_filtered = g_filtered.ravel()
		#remover ceros
		g_filtered = g_filtered[g_filtered != 0]
		#B
		b_filtered = np.array(b_filtered)
		b_filtered = b_filtered.ravel()
		#remover ceros
		b_filtered = b_filtered[b_filtered != 0]
		#Calcular la Skewness para los 3 channels:
		r_skew = skew(r_filtered)
		g_skew = skew(g_filtered)
		b_skew = skew(b_filtered)
		skewness = []
		skewness.append(r_skew)
		skewness.append(g_skew)
		skewness.append(b_skew)
		return skewness

	def getKurtosis(self, r_filtered, g_filtered, b_filtered):
		#Convert the matrices de los canales en un array 1-D
		#R
		r_filtered = np.array(r_filtered)
		r_filtered = r_filtered.ravel()
		r_filtered = r_filtered[r_filtered != 0] #remover ceros
		#G
		g_filtered = np.array(g_filtered)
		g_filtered = g_filtered.ravel()
		g_filtered = g_filtered[g_filtered != 0] #remover ceros
		#B
		b_filtered = np.array(b_filtered)
		b_filtered = b_filtered.ravel()
		b_filtered = b_filtered[b_filtered != 0] #remover ceros
		#Calcular la Skewness para los 3 channels:
		r_kurt = kurtosis(r_filtered)
		g_kurt = kurtosis(g_filtered)
		b_kurt = kurtosis(b_filtered)
		kurt = []
		kurt.append(r_kurt)
		kurt.append(g_kurt)
		kurt.append(b_kurt)
		return kurt

	def getImageMoments(self):
		r_channel = self.image.copy()
		g_channel = self.image.copy()
		b_channel = self.image.copy()
		#Get red channel
		r_channel[:, :, 0] = 0
		r_channel[:, :, 1] = 0
		#Get gree channel
		g_channel[:, :, 0] = 0
		g_channel[:, :, 2] = 0
		#Get blue channel
		b_channel[:, :, 1] = 0
		b_channel[:, :, 2] = 0
		#Get the channels like arrays
		#R
		r_array = np.asarray(r_channel)
		#G
		g_array = np.asarray(g_channel)
		#B
		b_array = np.asarray(b_channel)
		#Remove zeros that represents black pixels / background
		#R
		r_filtered = np.ma.masked_where(r_array == 0, r_array)
		#G
		g_filtered = np.ma.masked_where(g_array == 0, g_array)
		#B
		b_filtered = np.ma.masked_where(b_array == 0, b_array)

		medias = self.getMean(r_filtered, g_filtered, b_filtered)
		std = self.getStd(r_filtered, g_filtered, b_filtered)
		skewness = self.getSkewness(r_filtered, g_filtered, b_filtered)
		kurt = self.getKurtosis(r_filtered, g_filtered, b_filtered)
		#crear array de las 9 características
		array = np.concatenate([medias, std, skewness, kurt])
		return array.flatten()