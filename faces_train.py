import os
import numpy as np
import cv2
from PIL import Image
import pickle 
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'images')

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}

y_label = []
x_label = []

for root,dirs,files in os.walk(image_dir):
	for file in files:
		if file.endswith('png') or file.endswith('jpg'):
			path = os.path.join(root,file)
			label = os.path.basename(os.path.dirname(path)).replace(" ","-",).lower()
			# print(path,label)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			# print(label_ids)
			
			# y_label.append(label)#some number
			# x_label.append(path)#verify thi image, turn into a numpy array
			pil_image = Image.open(path).convert('L')# first we converted to grayscale
			size = (550,550)
			final_image = pil_image.resize(size,Image.ANTIALIAS)
			image_array = np.array(pil_image,'uint8')
			# print(image_array)# by this code we convert the image to the array of number

			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5,minNeighbors=5)# here we doing the face detection


			for (x,y,w,h) in faces:
				roi = image_array[y:y+h,x:x+w]
				x_label.append(roi)
				y_label.append(id_)

# print(y_label,x_label)

with open('labels.pickle', 'wb') as f:
	pickle.dump(label_ids,f)

recognizer.train(x_label, np.array(y_label))
recognizer.save('trainer.yml')