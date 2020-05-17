from PIL import Image
import numpy
import scipy
from scipy.ndimage import filters

epsilon = 0.00001
patternth = 4.1
sourceth = 4.1

def localiseonedge(img):
	imgarr = numpy.zeros(img.shape)
	filters.gaussian_filter(imgarr,(3,3),(0,1),imgarr)
	imgarr2 = numpy.zeros(img.shape)
	filters.gaussian_filter(img,(3,3),(1,0),imgarr2)

	Ixy = filters.gaussian_filter(imgarr*imgarr2,3)
	Iyy = filters.gaussian_filter(imgarr*imgarr2,3)
	Ixx = filters.gaussian_filter(imgarr*imgarr2,3)

	det = Ixx*Iyy - Ixy**2
	tr = Ixx + Iyy

	thres = sourceth

	hessian = tr**2/(det + epsilon)
	refined = numpy.where(hessian>thres)
	ind = len(refined[0])

	req = []

	for i in range(ind):
		req.append(refined[0][i],refined[1][i])

	return tuple(req)	

	 
	 