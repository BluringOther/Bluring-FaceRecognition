from deepface.detectors import FaceDetector
import numpy as np
import cv2
import base64
import os
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])
if tf_major_version == 1:
	import keras
	from keras.preprocessing.image import load_img, save_img, img_to_array
	from keras.applications.imagenet_utils import preprocess_input
	from keras.preprocessing import image
elif tf_major_version == 2:
	from tensorflow import keras
	from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
	from tensorflow.keras.applications.imagenet_utils import preprocess_input
	from tensorflow.keras.preprocessing import image


def loadBase64Img(uri):
	encoded_data = uri.split(',')[1]
	nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	return img # 최종 리턴 형식 제대로 알기.

def detect_face(img_b64, detector_backend = 'mtcnn', grayscale = False, enforce_detection = True, align = True):
    img = loadBase64Img(img_b64) #b64->ndarray
    
    img_region = [0, 0, img.shape[0], img.shape[1]]
    if detector_backend == 'mtcnn':
        return img, img_region
    face_detector = FaceDetector.build_model(detector_backend)
    try:
        detected_face, img_region = FaceDetector.detect_face(face_detector, detector_backend, img, align)
    except: #if detected face shape is (0, 0) and alignment cannot be performed, this block will be run
        detected_face = None
        
    if (isinstance(detected_face, np.ndarray)):
        return detected_face, img_region
    else:
        if detected_face == None:
            if enforce_detection != True:
                return img, img_region
            else:
                raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

def load_image(img):
	exact_image = False; base64_img = False; url_img = False

	if type(img).__module__ == np.__name__:
		exact_image = True

	elif len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	elif len(img) > 11 and img.startswith("http"):
		url_img = True

	#---------------------------

	if base64_img == True:
		img = loadBase64Img(img)

	elif url_img:
		img = np.array(Image.open(requests.get(img, stream=True).raw).convert('RGB'))

	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")

		img = cv2.imread(img)

	return img

def makeFileToPredict(base64Img):
   Img_bytes=base64.b64decode(base64Img)
   pil_image = Image.open(BytesIO(Img_bytes)) # bytes-> pill
   file_to_predict = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) # pil->ndarray
   pil_image.close()
   return file_to_predict



def preprocess_face(img_b64, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'mtcnn', return_region = False, align = True):
    #img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    
	img = makeFileToPredict(img_b64)
	base_img = img.copy()

	img, region = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection, align = align)

	#--------------------------

	if img.shape[0] == 0 or img.shape[1] == 0:
		if enforce_detection == True:
			raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
		else: #restore base image
			img = base_img.copy()

	#--------------------------

	#post-processing
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#---------------------------------------------------
	#resize image to expected shape

	# img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

	if img.shape[0] > 0 and img.shape[1] > 0:
		factor_0 = target_size[0] / img.shape[0]
		factor_1 = target_size[1] / img.shape[1]
		factor = min(factor_0, factor_1)

		dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
		img = cv2.resize(img, dsize)

		# Then pad the other side to the target size by adding black pixels
		diff_0 = target_size[0] - img.shape[0]
		diff_1 = target_size[1] - img.shape[1]
		if grayscale == False:
			# Put the base image in the middle of the padded image
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
		else:
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

	#------------------------------------------

	#double check: if target image is not still the same size with target.
	if img.shape[0:2] != target_size:
		img = cv2.resize(img, target_size)

	#---------------------------------------------------

	#normalizing the image pixels

	img_pixels = image.img_to_array(img) #what this line doing? must?
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]

	#---------------------------------------------------

	if return_region == True:
		return img_pixels, region	#return값
	else:
		return img_pixels
