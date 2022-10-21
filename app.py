from flask import Flask, request, jsonify

from scipy.spatial.distance import cosine
from PIL import Image,ImageFile #nparray->pil_image
from io import BytesIO
import base64
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

#!pip install deepface
from basemodels import Facenet
import functions
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = Facenet.loadModel()
thresh = 0.38
try:
	input_shape = model.layers[0].input_shape[1:3]
except: #issue 470
	input_shape = model.layers[0].input_shape[0][1:3]
#############################[step 0]#########################################
@app.route('/api2/detectFace',methods=['POST'])
def preRound(): # 블러 처리 제외할 사람 사진 등록시 수행
   data = request.get_json()
   base64_MemberImg=data['photo'] # detect_face 안에서 파라미터를 b64로 바꾸고 코드 첫줄에 loadBase64Img 호출하는 것으로 수정
   file_to_predict,dets = functions.detect_face(base64_MemberImg) 
   box_m=dets[0]['box']
   croppedImg_ndarray = file_to_predict[box_m[1]:box_m[1]+box_m[3], box_m[0]:box_m[0]+box_m[2]] 
   face_rgb=makeImgForResult(croppedImg_ndarray) #ndarray->base64
   return jsonify({'photo':face_rgb})

#############################[step 1]#########################################
def makeImgForResult(Img_ndarray,img_extension): # b64로 encode 해주는건 functions에 없어서 이거 사용 #이 함수는 star 눌러서 블러여부 고르는 사진
    Img_pill_rgb=Image.fromarray(cv2.cvtColor(Img_ndarray, cv2.COLOR_BGR2RGB)) # ndarray->pil_img
    Img_byteArr_rgb = BytesIO()
    Img_pill_rgb.save(Img_byteArr_rgb,format=img_extension)#,format="PNG,JPEG" # img_extension 사용 안함
    Img_bytes_rgb = Img_byteArr_rgb.getvalue()
    Img_base64_rgb=base64.b64encode(Img_bytes_rgb) # pil_img -> b64encode(bytes)
    face_rgb = Img_base64_rgb.decode("utf-8") 
    return face_rgb


# input_data(BlurRequest.java); {'originalPhoto':"값", 'exPhotos':["값1","값2",...]}
@app.route('/api2/blur/step-1', methods=['POST'])
def predict():

   content_type = request.headers.get('Content-Type')
   if (content_type != 'application/json'):
      return jsonify({"err":"not json"})  

   #####작업사진 전처리#####
   data = request.get_json()
   
   base64_OriginImg=data['originalPhoto'] # 여기에서 base64_OriginImg의 형식은 base64
   file_to_predict,dets=functions.preprocess_face(base64_OriginImg) # 두개로 나눠받지 말고 한 리스트에 받을까
   #file_to_predict_list=functions.preprocess_face(base64_OriginImg)
   #file_to_predict,dets=functions.detect_face(base64_OriginImg)
   
   ####등록사진 전처리 ####
   base64_MembersImg=data['exPhotos'] # 그룹/사용자 얼굴(list)
   
   #file_to_predict_mem,dets_mem=functions.preprocess_face(base64_MembersImg)

   result = {"faceInfo": [],"originalPhoto":base64_OriginImg,"blurredPhoto":base64_OriginImg,"step":Step.EDIT} 

   for i in range(0, len(dets)):# crop한 결과 
      ###### [1] face_crop : 원본사진중 인식된 얼굴 
      box = dets[i]['box']
      croppedImg_ndarray=file_to_predict[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] 
      face_rgb=makeImgForResult(croppedImg_ndarray,"jpeg")

      ###### [2] blurred : 원본사진중 인식된 얼굴에 사용자가 블러 처리 제외할 얼굴이 있다면, False
      blurred=True
      #사진에서 cropped 얼굴 predict
      face_pred = model.predict(croppedImg_ndarray) 
      
      for m_croppedImg_ndarray in base64_MembersImg:
         m_file_to_predict=functions.preprocess_face(m_croppedImg_ndarray)
         m_face_pred=model.predict(m_file_to_predict)[0,:]
         score=cosine(m_face_pred,face_pred)
         if score<thresh:# 블러 처리
            blurred=False
      
      result['faceInfo'].append(
         {'face_crop': face_rgb, # 원본 사진에서의 인식된 얼굴
            'face_location': {#원본사진에서의 얼굴 위치
               "bottom": box[1], 
               "top": box[1]+box[3], 
               "left": box[0], 
               "right": box[0]+box[2]
            }, 
            'blurred': blurred
         })
   
   return jsonify(result)


#############################[step 2]#########################################
# input_data(BlurResponse.java); {"faceInfo": [],"originalPhoto":base64_OriginImg,"blurredPhoto":base64_OriginImg,"step":Step.DONE} 
@app.route('/api2/blur/step-2', methods=['POST'])
def predict2():

   content_type = request.headers.get('Content-Type')
   if (content_type != 'application/json'):
      return jsonify({"err":"not json"})  
   
   data = request.get_json()
   base64_OriginImg=data['originalPhoto']
   originalImg_bytes=base64.b64decode(base64_OriginImg)
   img_extension="jpeg" #imghdr.what(None, h=originalImg_bytes)
   # print("ORIGINAL: ",img_extension)

   pil_image = Image.open(BytesIO(originalImg_bytes)) 
   file_to_predict = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)#  RGB->BGR 
   pil_image.close()
   # file_to_predict를 변환해서 블러 처리해서 base64_BlurredImg에 넣기
   
   ksize=30 # 블러 처리에 사용할 커널 크기
   for i,faceInfo in enumerate(data['faceInfo']): # 얼굴 인식된 갯수만큼 for문 돌기
      if faceInfo['blurred'] == True: # 블러처리
         #블러처리 코드 
         roi=file_to_predict[faceInfo['face_location']['bottom']:faceInfo['face_location']['top'], faceInfo['face_location']['left']:faceInfo['face_location']['right']] # 관심영역 지정
         roi=cv2.blur(roi, (ksize, ksize)) # 블러처리
         file_to_predict[faceInfo['face_location']['bottom']:faceInfo['face_location']['top'], faceInfo['face_location']['left']:faceInfo['face_location']['right']] = roi # 원본 이미지에 적용
      elif faceInfo['blurred'] == False:
         pass


   
   # 이곳에서 최종결과 얻기
   file_to_predict = cv2.cvtColor(file_to_predict, cv2.COLOR_BGR2RGB) #색 원상복구
   blurredImg_pill=Image.fromarray(file_to_predict) # ndarray->pil_img
      

   # PNG -> originalImg_extension type
   blurredImg_byteArr = BytesIO()
   blurredImg_pill.save(blurredImg_byteArr, format=img_extension)
   blurredImg_bytes = blurredImg_byteArr.getvalue()
      
      
   blurredImg_base64=base64.b64encode(blurredImg_bytes) # pil_img -> b64encode(bytes)
   base64_BlurredImg = blurredImg_base64.decode("utf-8") # b64decode(b64encode(bytes),"utf-8")
   

   result = {"faceInfo": [],"originalPhoto":base64_OriginImg,"blurredPhoto":base64_BlurredImg,"step":Step.DONE} 
   return jsonify(result)



if __name__ == "__main__":
   app.debug = True
   app.run()
