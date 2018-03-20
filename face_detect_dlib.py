"""
HTTU
Dung dlib thuc hien tinh nang face detct
"""
import dlib
import cv2
import imutils

if __name__=="__main__":

	#lay mo hinh face detect
 	face_dectector=dlib.get_frontal_face_detector()
 	#load mo hinh dectect landmark
 	landmark_predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
 	#lay video vao tu webcam
 	vid=cv2.VideoCapture(0)
 	while(True):
 		ret,frame=vid.read()
 		frame=imutils.resize(frame,width=400)
 		#lay anh gray
 		frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 		#thuc hien face detect
 		face_boundaries=face_dectector(frame_gray,0)
 		#lay so luong va toa do cac face.
	 	for(enum,face) in enumerate(face_boundaries):
	 		x=face.left()
	 		y=face.top()
	 		w=face.right()-x
	 		h=face.bottom()-y
	 		#ve hinh chu nhat xung quanh cac face
	 		cv2.rectangle(frame,(x,y),(x+w,y+h),(120,160,230),2)

 		cv2.imshow("test",frame)
 		k=cv2.waitKey(1)&0xff
 		if(k==ord('c')):
 			break
		
vid.release()
cv2.destroyAllWindows()




