from flask import Flask, request, jsonify
import cv2
import numpy as np
from scipy.spatial.distance import cosine

#from Step import Step
from io import BytesIO
from PIL import Image,ImageFile #nparray->pil_image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import base64

app = Flask(__name__)

#!pip install deepface
from deepface.commons import functions 
from deepface.basemodels import Facenet
from deepface.detectors import FaceDetector
from tensorflow.keras.preprocessing import image

#import matplotlib.pyplot as plt

global model
model = Facenet.loadModel()

thresh = 8.7
DETECTOR_BACKEND='mtcnn'


def loadBase64Img(encoded_data): #base64->ndarray
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def loadBase64Imgs(encodedDatas):
   res=[]
   for encodedData in encodedDatas:
      res.append(loadBase64Img(encodedData)) #mFTP=loadBase64Img
   return res

def makeImgForResult(Img_ndarray,img_extension): # b64로 encode 해주는건 functions에 없어서 이거 사용
    Img_pill_rgb=Image.fromarray(cv2.cvtColor(Img_ndarray, cv2.COLOR_BGR2RGB)) # ndarray->pil_img
    Img_byteArr_rgb = BytesIO()
    Img_pill_rgb.save(Img_byteArr_rgb,format=img_extension)#,format="PNG,JPEG" # img_extension 사용 안함
    Img_bytes_rgb = Img_byteArr_rgb.getvalue()
    Img_base64_rgb=base64.b64encode(Img_bytes_rgb) # pil_img -> b64encode(bytes)
    face_rgb = Img_base64_rgb.decode("utf-8") 
    return face_rgb 

def blursome_prepredict(img, target_size=(224, 224), grayscale = False, enforce_detection = True):
	base_img=img.copy()
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

	return img_pixels

#############################[step 0]#########################################
@app.route('/api2/detectFace',methods=['POST'])
def preRound(): # 블러 처리 제외할 사람 사진 등록시 수행
   data = request.get_json() #b64 이미지 파일 1개 받아옴
   memberImg=loadBase64Img(data['photo']) #b64 이미지를 np.array로 변환하여 저장 #makeFileToPredict 함수를 loadBase64 함수로 변경
   det_croppedface,_=FaceDetector.detect_faces(FaceDetector.build_model(detector_backend=DETECTOR_BACKEND),detector_backend=DETECTOR_BACKEND,img=memberImg, align=True)[0] #[(face,region),(face,region)...]

   face_rgb=makeImgForResult(det_croppedface, "jpeg") #ndarray->base64
   return jsonify({'photo':face_rgb}) 

#############################[step 1]#########################################
@app.route('/api2/blur/step-1', methods=['POST'])# {'originalPhoto':"값", 'exPhotos':["값1","값2",...]}
def predict():

   content_type = request.headers.get('Content-Type')
   if (content_type != 'application/json'):
      return jsonify({"err":"not json"})  

   #####작업사진 전처리#####
   data = request.get_json()
   
   result = {"faceInfo": [],"originalPhoto":data['originalPhoto'],"blurredPhoto":data['originalPhoto']} 

   originImg=loadBase64Img(data['originalPhoto']) 
   origin_croppedfaces_regions=FaceDetector.detect_faces(FaceDetector.build_model(detector_backend=DETECTOR_BACKEND),detector_backend=DETECTOR_BACKEND,img=originImg, align=True)#[(face,region),(face,region)...]
   
   ####등록사진 전처리 #### # 이미 detect하고 crop된 이미지들임.
   member_croppedfaces=loadBase64Imgs(data['exPhotos'])#[]#예측할파일리스트

   for origin_det_croppedface,origin_det_region in origin_croppedfaces_regions:
      blurred=True
      ###### [2] blurred : 원본사진중 인식된 얼굴에 사용자가 블러 처리 제외할 얼굴이 있다면, False
      model_origin_croppedface = blursome_prepredict(origin_det_croppedface, target_size=functions.find_input_shape(model), grayscale = False, enforce_detection = True)# predict전에 preprocess해야 해서
      #predict 값이 이상하게 찍히면 functions.preprocess_face 대신에 resize, reshape 해보기
      face_pred = model.predict(model_origin_croppedface)[0,:]

      for m_croppedImg_ndarray in member_croppedfaces:
         model_member_croppedface = blursome_prepredict(m_croppedImg_ndarray, target_size=functions.find_input_shape(model), grayscale = False, enforce_detection = True)# predict전에 preprocess해야 해서
         m_face_pred = model.predict(model_member_croppedface)[0,:]
         
         distance_vector = np.square(m_face_pred - face_pred)
         #score=cosine(m_face_pred, face_pred)
         #print(distance_vector)
         distance = np.sqrt(distance_vector.sum())
         print(distance)
         
         #log_score('원본'+str(i)+'블러제외'+": "+str(score))
         if distance < thresh:# 블러 처리
            blurred=False # 블러 제외
         print(blurred)
      
      result['faceInfo'].append(
         {'face_crop': makeImgForResult(origin_det_croppedface,"jpeg"), # 원본 사진에서의 인식된 얼굴 #face_rgb->Orgface_rgb
            'face_location': {#원본사진에서의 얼굴 위치
               "bottom": origin_det_region[1], 
               "top": origin_det_region[1]+origin_det_region[3], 
               "left": origin_det_region[0], 
               "right": origin_det_region[0]+origin_det_region[2]
            }, 
            'blurred': blurred
         })
   
   return jsonify(result)


#############################[step 2]#########################################
# input_data(BlurResponse.java); {"originalPhoto":base64_OriginImg,"blurredPhoto":base64_OriginImg} 
@app.route('/api2/blur/step-2', methods=['POST'])
def predict2():

   content_type = request.headers.get('Content-Type')
   if (content_type != 'application/json'):
      return jsonify({"err":"not json"})  
   
   data = request.get_json()
   base64_OriginImg=data['originalPhoto']
   photo=loadBase64Img(base64_OriginImg)
   
   ksize=30 # 블러 처리에 사용할 커널 크기
   for faceInfo in data['faceInfo']: # 얼굴 인식된 갯수만큼 for문 돌기
      if faceInfo['blurred'] == True: # 블러처리
         #블러처리 코드 
         roi=photo[faceInfo['face_location']['bottom']:faceInfo['face_location']['top'], faceInfo['face_location']['left']:faceInfo['face_location']['right']] # 관심영역 지정
         roi=cv2.blur(roi, (ksize, ksize)) # 블러처리
         photo[faceInfo['face_location']['bottom']:faceInfo['face_location']['top'], faceInfo['face_location']['left']:faceInfo['face_location']['right']] = roi # 원본 이미지에 적용
      elif faceInfo['blurred'] == False:
         pass

   result = {"originalPhoto":base64_OriginImg,"blurredPhoto":makeImgForResult(photo,"jpeg")} # face_count는 없어도 됨. faceInfo 길이로 알 수 있음||대신에 단계 나타내는 데이터 추가바람
   return jsonify(result)



if __name__ == "__main__":
   app.debug = True
   app.run()