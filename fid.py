

# example of calculating the frechet inception distance in Keras for cifar10
import glob

import numpy
import torchvision
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
import os

import cv2

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


def get_img_list(dir, firelist, ext=None):
    newdir = dir
    if os.path.isfile(dir):  # 如果是文件
        if ext is None:
            firelist.append(dir)
        elif ext in dir[-3:]:
            firelist.append(dir)
    elif os.path.isdir(dir):  # 如果是目录
        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            get_img_list(newdir, firelist, ext)

    return firelist

if __name__ == "__main__":

	images1 = []
	# 读取文件夹中图片
	Image_glob = os.path.join('data2/finalDataY', "*.png")
	Image_name_list = []
	Image_name_list.extend(glob.glob(Image_glob))
	for name in Image_name_list:
		image = cv2.imread(name)
		images1.append(image)

	images1 = numpy.array(images1)

	images2 = []
	# 读取文件夹中图片
	Image_glob = os.path.join('output1/output', "*.png")
	Image_name_list = []
	Image_name_list.extend(glob.glob(Image_glob))
	for name in Image_name_list:
		image = cv2.imread(name)
		images2.append(image)

	images2 = numpy.array(images2)
	# prepare the inception v3 model
	model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
	# load cifar10 images
	#(images1, _), (images2, _) = cifar10.load_data()
	#shuffle(images1)
	#images1 = images1[:100]
	#images2 = images2[:100]

	print('Loaded', images1.shape, images2.shape)
	# convert integer to floating point values
	images1 = images1.astype('float32')
	images2 = images2.astype('float32')
	# resize images
	images1 = scale_images(images1, (299,299,3))
	images2 = scale_images(images2, (299,299,3))
	print('Scaled', images1.shape, images2.shape)
	# pre-process images
	images1 = preprocess_input(images1)
	images2 = preprocess_input(images2)
	# calculate fid
	fid = calculate_fid(model, images1, images2)
	print('FID: %.3f' % fid)