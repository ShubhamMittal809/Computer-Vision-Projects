import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import filters
import cv2
import localization


def plot(img, title=""):
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)   
    


def linearInterpolate(img):
    m , n = img.shape
    upsampled = np.zeros((2*m-1, 2*n-1))
    downsampled = np.zeros((2*m-1, 2*n-1))
    
    for i in range(m):
        for j in range(n):
            downsampled1 = np.zeros((2*m-1, 2*n-1))
            upsampled1 = np.zeros((2*m-1, 2*n-1))
            downsampled1[2*i, 2*j] = img[i, j]
            upsampled[2*i, 2*j] = img[i, j]

    for i in range (1, 2*m-1, 2) :
    	upsampled[i,:] = (upsampled[i+1,:] + upsampled[i-1, :]) / 2
    	downsampled[i,:] = (downsampled[i+1,:] + downsampled[i-1, :]) / 2
    for j in range(1,2*n-1,2):
    	downsampled[:,j] = (downsampled[:, j+1] + downsampled[:,j-1])/2
    	upsampled[:,j] = (upsampled[:, j+1] + upsampled[:,j-1])/2  	        
            
    return upsampled 

def gaussianblur(img, sigma):
	return filters.gaussian_filter(img,sigma)

def Size(img):
    return img.shape


def displayKeypoints(extremas, image):
	#extrema =  np.asarray(list(set(extremas)))
	extremas = list(set(extremas))
	extremas = np.asarray(extremas)
	#image.plot("number of keypoints detected = %d" % (len(extremas)))
	plot(image,"Size: %dpx * %dpx, number of keypoints dete = %d" % (Size(image)[0], Size(image)[1], len(extremas)))
	plt.plot(extremas[:,1], extremas[:,0], 'b.', label = 'Keypoints')
	plt.legend(loc = 'upper left')
	plt.show()

def downsample(img):
	return img[::2, ::2]

k = np.sqrt(2)  
constFactor = np.sqrt(k**2 - 1)
s = 5          
numOctave = 3 
sigma = 1.6


def createScaleSpace(image, display = False ) :
	originalImg = np.array(Image.open(image).convert('L')) # converts PIL image to numpy array

	duplicateImg = originalImg

	duplicateImg = gaussianblur(duplicateImg, 0.5) ;

	upsampledimg = linearInterpolate(duplicateImg)

	listofscale = []
    
    
	for i in range(s):
		newscale= gaussianblur(upsampledimg,sigma*(k**i))
		listofscale.append(newscale)
	
	factor = 4

	for i in range(3):
		for j in range(3):
			listofscale.append(downsample(listofscale[-3]))

		for j in range(factor, factor+2):
			newsigma = (k **j)* constFactor*sigma
			newscale = gaussianblur(listofscale[-1],newsigma)
			listofscale.append(newscale)
		factor += 2
    
    #tp plot octaves
	if display == True:
		arr = np.zeros(numOctave,s)
		for r in range(numOctave):
			fig, arr = plt.subplots(1,s, sharey= True)
			plt.suptitle("ocatve %d" %(row+1))

			for c in range(len(listofscale[:s])) :
				arr[c] = imshow(listofscale[r*s+c] , cmap = 'Greys_r')
				currentsigma = k**(2*r+c)*sigma
				arr[col].set_title('$\sigma = %0.5f$' % currentsigma)

		plt.show()
	return (listofscale,upsampledimg)	


def compute_DOG(listofscale):
	DOG = []
	for octaves in range(numOctave):
		for i in range(s-1):
			DOG.append(listofscale[i+1+octaves*s]-listofscale[i+octaves*s])
	return DOG
	

def extremaDetection(DOG) :
	extremas = []
	for octaves in range(numOctave):
		m, n = DOG[octaves*(s-1)].shape
		for i in range(1,3):
			#neighbours = np.array([DOG[j+octaves*(s-1)][1:i+2,1:i+2] for j in range(1,i+2)]).ravel()
			for a in range(1,m):
				#neighbours = np.array([DOG[j+octaves*(s-1)][a-1:a+2,1:i+2] for j in range(1,i+2)]).ravel()
				for b in range(1,n) :
					#neighbours = np.array([DOG[j+octaves*(s-1)][a-1:a+2,b-1:b+2] for j in range(b-1,b+2)]).ravel()
					Flag = False

					flag = True
                    
					neighbors = np.array([DOG[j+octaves*(s-1)][a-1:a+2,b-1:b+2] for j in range(i-1,i+2)]).ravel()
					neighbors = np.delete(neighbors, len(neighbors)/2)
					sign = np.sign(neighbors[0] - DOG[i+octaves*(s-1)][a,b])
					neighbours = neighbors
					for j in neighbors[1:] :
						if (np.sign(DOG[i+octaves*(s-1)][a,b]- j )!= -sign):
							flag = False
							break
					if flag:
						extremas.append((2**octaves*a,2**octaves*b))
	return extremas




(scales, image) = createScaleSpace('house.jpg', display = False)
DOG  = compute_DOG(scales)
extremas  = extremaDetection(DOG)
#image2 = localization.localiseonedge(image)
displayKeypoints(extremas, image)




         
            
    