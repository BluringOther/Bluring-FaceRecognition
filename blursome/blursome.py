from tensorflow.keras.preprocessing import image
from io import BytesIO
import base64
from PIL import Image
import cv2
import numpy as np

def loadBase64Img(encoded_data): #base64->ndarray
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def loadBase64Imgs(encodedDatas):
   res=[]
   for encodedData in encodedDatas:
      res.append(loadBase64Img(encodedData)) 
   return res

def makeImgForResult(Img_ndarray,img_extension):
    Img_pill_rgb=Image.fromarray(cv2.cvtColor(Img_ndarray, cv2.COLOR_BGR2RGB)) # ndarray->pil_img
    Img_byteArr_rgb = BytesIO()
    Img_pill_rgb.save(Img_byteArr_rgb,format=img_extension)
    Img_bytes_rgb = Img_byteArr_rgb.getvalue()
    Img_base64_rgb=base64.b64encode(Img_bytes_rgb) # pil_img -> b64
    face_rgb = Img_base64_rgb.decode("utf-8") 
    return face_rgb 

def prepredict(img, target_size=(224, 224), grayscale = False, enforce_detection = True):
	base_img=img.copy()

	if img.shape[0] == 0 or img.shape[1] == 0:
		if enforce_detection == True:
			raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
		else: 
			img = base_img.copy()


	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	if img.shape[0] > 0 and img.shape[1] > 0:
		factor_0 = target_size[0] / img.shape[0]
		factor_1 = target_size[1] / img.shape[1]
		factor = min(factor_0, factor_1)

		dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
		img = cv2.resize(img, dsize)

		diff_0 = target_size[0] - img.shape[0]
		diff_1 = target_size[1] - img.shape[1]
		if grayscale == False:
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
		else:
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

	if img.shape[0:2] != target_size:
		img = cv2.resize(img, target_size)

	img_pixels = image.img_to_array(img)
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]

	return img_pixels