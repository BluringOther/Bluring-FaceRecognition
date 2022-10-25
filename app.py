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
from deepface import DeepFace
from deepface.commons import functions 
from deepface.basemodels import Facenet
from mtcnn.mtcnn import MTCNN
#import matplotlib.pyplot as plt

global model
model = Facenet.loadModel()
net = MTCNN()

thresh = 8.8 
'''
try:
   input_shape = model.layers[0].input_shape[1:3]
except: #issue 470
   input_shape = model.layers[0].input_shape[0][1:3]
'''

#############################[step 0]#########################################
@app.route('/api2/detectFace',methods=['POST'])
def preRound(): # 블러 처리 제외할 사람 사진 등록시 수행
   data = request.get_json() #b64 이미지 파일 1개 받아옴
   base64_MemberImg=data['photo']
   nd_MemberImg=loadBase64Img(base64_MemberImg) #b64 이미지를 np.array로 변환하여 저장 #makeFileToPredict 함수를 loadBase64 함수로 변경
   #file_to_predict,dets = functions.detect_face(nd_MemberImg,'mtcnn') #detect된 얼굴이랑 위치정보 저장. output:ndarray
   dets = net.detect_faces(nd_MemberImg)  # 괄호 안에 file_to_predict
   
   croppedImg_ndarray = nd_MemberImg[dets[1]:dets[1]+dets[3], dets[0]:dets[0]+dets[2]] # detect한 부분만 croppedImg에 저장
   face_rgb=makeImgForResult(croppedImg_ndarray, "jpeg") #ndarray->base64
   return jsonify({'photo':face_rgb}) 

#############################[step 1]#########################################
def loadBase64Img(encoded_data): #makeFileToPredict 대신 사용
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img 

def makeFileToPredict(base64Img):
   Img_bytes=base64.b64decode(base64Img)
   pil_image = Image.open(BytesIO(Img_bytes)) # bytes-> pill
   file_to_predict = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) # pil->ndarray
   pil_image.close()
   return file_to_predict

def makeImgForResult(Img_ndarray,img_extension): # b64로 encode 해주는건 functions에 없어서 이거 사용
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
   nd_OriginImg=loadBase64Img(base64_OriginImg) #base64->ndarray
   o_dets = net.detect_faces(nd_OriginImg)  # 괄호 안에 file_to_predict
   face_count = len(o_dets) # originalPhoto에서 찾은 얼굴
   #detected_face = functions.detect_face(base64_OriginImg, detector_backend = 'mtcnn')
   
   ####등록사진 전처리 #### # 이미 detect하고 crop된 이미지들임.
   base64_MembersImg=data['exPhotos'] # 그룹/사용자 얼굴(list)
   m_file_to_predict_list=[]#예측할파일리스트
   for base64_MemberImg in base64_MembersImg:
      m_file_to_predict_list.append(loadBase64Img(base64_MemberImg)) #mFTP=loadBase64Img
   #file_to_predict_mem,dets_mem=functions.preprocess_face(base64_MembersImg)
   result = {"faceInfo": [],"originalPhoto":base64_OriginImg,"blurredPhoto":base64_OriginImg} 

   for i in range(0, face_count):
      ###### [1] face_crop : 원본사진중 인식된 얼굴 
      box_o = o_dets[i]['box']
      croppedImg_ndarray=nd_OriginImg[box_o[1]:box_o[1]+box_o[3], box_o[0]:box_o[0]+box_o[2]] 
      Orgface_rgb=makeImgForResult(croppedImg_ndarray,"jpeg")
      ###### [2] blurred : 원본사진중 인식된 얼굴에 사용자가 블러 처리 제외할 얼굴이 있다면, False
      blurred=True
      croppedImg_ndarray = functions.preprocess_face(croppedImg_ndarray, target_size=(160, 160), detector_backend='mtcnn')
      #predict 값이 이상하게 찍히면 functions.preprocess_face 대신에 resize, reshape 해보기
      face_pred = model.predict(croppedImg_ndarray)[0,:]
      for m_croppedImg_ndarray in m_file_to_predict_list:
         m_croppedImg_ndarray = functions.preprocess_face(m_croppedImg_ndarray, target_size=(160, 160), detector_backend='mtcnn') 
         m_face_pred = model.predict(m_croppedImg_ndarray)[0,:]
         distance_vector = np.square(m_face_pred - face_pred)
         #score=cosine(m_face_pred, face_pred)
         #print(distance_vector)
         distance = np.sqrt(distance_vector.sum())
         print(distance)
         
         #===============================여기까지 테스트 완료
         #log_score('원본'+str(i)+'블러제외'+": "+str(score))
         if distance < thresh:# 블러 처리
            blurred=False # 블러 제외
         print(blurred)
      
      result['faceInfo'].append(
         {'face_crop': Orgface_rgb, # 원본 사진에서의 인식된 얼굴 #face_rgb->Orgface_rgb
            'face_location': {#원본사진에서의 얼굴 위치
               "bottom": box_o[1], 
               "top": box_o[1]+box_o[3], 
               "left": box_o[0], 
               "right": box_o[0]+box_o[2]
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
   

   result = {"faceInfo": [],"originalPhoto":base64_OriginImg,"blurredPhoto":base64_BlurredImg} # face_count는 없어도 됨. faceInfo 길이로 알 수 있음||대신에 단계 나타내는 데이터 추가바람
   return jsonify(result)



if __name__ == "__main__":
   app.debug = True
   app.run()
