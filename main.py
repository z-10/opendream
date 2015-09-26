#!/usr/bin/python
import random

import numpy as np
import scipy.ndimage as nd


from cStringIO import StringIO
import PIL.Image
from google.protobuf import text_format

import os
import sys
import argparse
from cStringIO import StringIO

import caffe
from deepdream import deepdream, net
try:
	import cv2
except ImportError:
	raise Exception ("OpenCV is not available:")

from images2gif import writeGif

###############################################################################
# openCV Preview Window
# ------------
##############################################################################
def show(img, blob):
	cv2.namedWindow('image_process', cv2.WINDOW_AUTOSIZE)
	#cv2.setWindowTitle('image_process', 'Current Blob: '+blob)
	open_cv_image = np.array(img)
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	cv2.imshow('image_process', open_cv_image.view())
	cv2.waitKey(25)

##############################################################################
# Main function
# -------------
# Usage: Usage: $ python main.py -f [source/filename.jpg] -o [output dir]
#                                -s [scale] -i [iterations] -b [all/blobname]
#                                -z [0/1] -p [0/1] -g [0/1]
# Arguments:
# '-f', '--filename'  : Input file
# '-o', '--outputdir': Output directory
# '-s', '--scaleCoef' : Scale Coefficient (default=0.5)
# '-i', '--iterations': Iterations (default=100)
# '-b', '--blob'      : Blob name (default=random)
# '-z', '--zoom'      : Zoom (default=0)
# '-p', '--preview'   : Preview Window (default=0)
# '-g', '--gpu'		  : Enable GPU (default=0, Assumes device ID:0)
##############################################################################

if __name__ == '__main__':
	# get  args if we can.
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--filename', type=str)
	parser.add_argument('-o', '--outputdir', default='out', type=str)
	parser.add_argument('-s', '--scaleCoef', default=0.05, type=float)
	parser.add_argument('-i', '--iterations', default=100, type=int)
	parser.add_argument('-b', '--blob', default=random.choice(net.blobs.keys()), type=str)
	parser.add_argument('-z', '--zoom', default=0, type=int)
	parser.add_argument('-p', '--preview', default=0, type=int)
	parser.add_argument('-g', '--gpu', default=0, type=int)
	parser.add_argument('-a', '--animated',default=0, type=int)
	parser.add_argument('-r', '--rand', default=0, type=int)
	args = parser.parse_args()

	if args.filename == None:
		print 'Error: No source file'
		print 'Usage: $ python main.py -f [source/filename.jpg] -o [output dir] -s [scale] -i [iterations] -b [all/blobname] -z [0/1] -p [0/1] -g [0/1]'
		exit()

	if args.gpu == 1:
		caffe.set_mode_gpu()
		caffe.set_device(0)

  # PIL is stupid, go away PIL

	images = []
	if '.gif' in args.filename:
		gif = PIL.Image.open(args.filename)
		nframes = 0
		while gif:
			images.append(np.float32(gif.convert("RGB")))
			nframes += 1
			try:
				gif.seek( nframes )
			except EOFError:
				break;
	else:
  		f = open(args.filename,"rb")
  		rawImage = f.read()
  		images.append(np.float32(PIL.Image.open(StringIO(rawImage)).convert("RGB")))
  	print 'Loaded', args.filename

  # split file name so we can make a special folder
	fn = args.filename.split('/')[-1].split('.')
	ext = fn[-1]
	fn = fn[0]

  # make sure output path exists
	if not os.path.exists(args.outputdir):
		os.makedirs(args.outputdir)

	allBlobs = []
	for blob in net.blobs.keys():
		if '_split' not in blob:
			allBlobs.append(blob)

	index = 0
	processed = []
	blobs = []
	if args.blob == 'all':
		blobs = allBlobs
	else:
		if args.rand > 0:
			blobs = random.sample(allBlobs,args.rand)
		else:
			blobs = [args.blob]


	for img in images:
		framepath = args.outputdir+'/'+fn
		if len(images) > 1:
			framepath = framepath + "/" + str(index)
 		print "Output: ", framepath
		if not os.path.exists(framepath):
			os.makedirs(framepath)
  	# see ya on the other side
  		frame = img
  		h, w = frame.shape[:2]
  		s = args.scaleCoef # scale coefficient

  	# run all blobs, adopted from script by Cranial_Vault

		PIL.Image.fromarray(np.uint8(frame)).save(framepath+'/source.'+ext)
		j = 0
		for blob in blobs:

			safeblob = blob.replace('/', '-')
			for i in xrange(args.iterations):
          		#Show preview window
				if args.preview == 1:
					show(PIL.Image.fromarray(np.uint8(frame)), safeblob)
				if args.animated == 1:
					processed.append(PIL.Image.fromarray(np.uint8(frame)))

				try:
            		# if we've already generated this image, then don't bother
					frame = deepdream(net, frame, end=blob)
					PIL.Image.fromarray(np.uint8(frame)).save(framepath+'/'+safeblob+'.'+ext)
					print j, str(blob)
					if args.zoom == 1:
						frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
				except ValueError as err:
					print 'ValueError:', str(blob), err
					pass
				except KeyError as err:
					print 'KeyError:', str(blob), err
	if args.animated == 1:
		print "Writing GIF"
		writeGif(args.outputdir+"/animated.gif",processed + list(reversed(processed)), 1.0/6)
