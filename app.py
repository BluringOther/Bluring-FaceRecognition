from flask import Flask, request, jsonify
import cv2
import numpy as np

from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True


app = Flask(__name__)

from deepface.commons import functions 
from deepface.basemodels import Facenet
from deepface.detectors import FaceDetector
from blursome import blursome

global model
model = Facenet.loadModel()

thresh = 8.7
DETECTOR_BACKEND='mtcnn'


# [step 0]
@app.route('/api2/detectFace',methods=['POST'])
def preRound(): # 블러 처리 제외할 사람 사진 등록시 수행
   data = request.get_json()
   memberImg=blursome.loadBase64Img(data['photo']) 
   det_croppedface,_=FaceDetector.detect_faces(FaceDetector.build_model(detector_backend=DETECTOR_BACKEND),detector_backend=DETECTOR_BACKEND,img=memberImg, align=True)[0] #[(face,region),(face,region)...]

   face_rgb=blursome.makeImgForResult(det_croppedface, "jpeg") #ndarray->base64
   return jsonify({'photo':face_rgb}) 

# [step 1]
@app.route('/api2/blur/step-1', methods=['POST'])# {'originalPhoto':"값", 'exPhotos':["값1","값2",...]}
def predict():

   content_type = request.headers.get('Content-Type')
   if (content_type != 'application/json'):
      return jsonify({"err":"not json"})  

   data = request.get_json()
   
   result = {"faceInfo": [],"originalPhoto":data['originalPhoto'],"blurredPhoto":data['originalPhoto']} 

   originImg=blursome.loadBase64Img(data['originalPhoto']) 
   origin_croppedfaces_regions=FaceDetector.detect_faces(FaceDetector.build_model(detector_backend=DETECTOR_BACKEND),detector_backend=DETECTOR_BACKEND,img=originImg, align=True)#[(face,region),(face,region)...]
   
   # 등록사진 전처리 
   member_croppedfaces=blursome.loadBase64Imgs(data['exPhotos'])# 예측할파일리스트

   for origin_det_croppedface,origin_det_region in origin_croppedfaces_regions:
      blurred=True
      # 원본사진중 인식된 얼굴에 사용자가 블러 처리 제외할 얼굴이 있다면, False
      model_origin_croppedface = blursome.prepredict(origin_det_croppedface, target_size=functions.find_input_shape(model), grayscale = False, enforce_detection = True)# predict전에 preprocess해야 해서
      face_pred = model.predict(model_origin_croppedface)[0,:]

      for m_croppedImg_ndarray in member_croppedfaces:
         model_member_croppedface = blursome.prepredict(m_croppedImg_ndarray, target_size=functions.find_input_shape(model), grayscale = False, enforce_detection = True)# predict전에 preprocess해야 해서
         m_face_pred = model.predict(model_member_croppedface)[0,:]
         
         distance_vector = np.square(m_face_pred - face_pred)
         distance = np.sqrt(distance_vector.sum())
         print(distance)
         
         if distance < thresh:# 블러 처리
            blurred=False # 블러 제외
         print(blurred)
      
      result['faceInfo'].append(
         {'face_crop': blursome.makeImgForResult(origin_det_croppedface,"jpeg"), # 원본 사진에서의 인식된 얼굴
            'face_location': {#원본사진에서의 얼굴 위치
               "bottom": origin_det_region[1], 
               "top": origin_det_region[1]+origin_det_region[3], 
               "left": origin_det_region[0], 
               "right": origin_det_region[0]+origin_det_region[2]
            }, 
            'blurred': blurred
         })
   
   return jsonify(result)


# [step 2]
# input_data(BlurResponse.java); {"originalPhoto":base64_OriginImg,"blurredPhoto":base64_OriginImg} 
@app.route('/api2/blur/step-2', methods=['POST'])
def predict2():

   content_type = request.headers.get('Content-Type')
   if (content_type != 'application/json'):
      return jsonify({"err":"not json"})  
   
   data = request.get_json()
   base64_OriginImg=data['originalPhoto']
   photo=blursome.loadBase64Img(base64_OriginImg)
   
   ksize=30 # 블러 처리에 사용할 커널 크기
   for faceInfo in data['faceInfo']: # 얼굴 인식된 갯수
      if faceInfo['blurred'] == True: # 블러처리
         roi=photo[faceInfo['face_location']['bottom']:faceInfo['face_location']['top'], faceInfo['face_location']['left']:faceInfo['face_location']['right']] # 관심영역 지정
         roi=cv2.blur(roi, (ksize, ksize)) 
         photo[faceInfo['face_location']['bottom']:faceInfo['face_location']['top'], faceInfo['face_location']['left']:faceInfo['face_location']['right']] = roi # 원본 이미지에 적용
      elif faceInfo['blurred'] == False:
         pass

   result = {"originalPhoto":base64_OriginImg,"blurredPhoto":blursome.makeImgForResult(photo,"jpeg")} # face_count는 없어도 됨. faceInfo 길이로 알 수 있음||대신에 단계 나타내는 데이터 추가바람
   return jsonify(result)



if __name__ == "__main__":
   app.debug = True
   app.run()