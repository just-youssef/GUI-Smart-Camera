from imutils import paths
import face_recognition
import pickle
import cv2
import os
 
def is_image_file(file_name):

    image_file_extensions = ('.rgb', '.gif', '.pbm', '.pgm', '.ppm', '.tiff', '.rast' '.xbm',
    	'.jpeg', '.jpg', '.JPG', '.bmp', '.png', '.PNG', '.webp', '.exr')

    return file_name.endswith((image_file_extensions))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'faces')
known_encodings = []
known_names = []

with open('scanned-faces', 'rt') as f:
	scanned = f.readlines()
	scanned = [i.split('\n')[0] for i in scanned]
#print(scanned)

def get_faces_data(mode):
	for root, dirs, files in os.walk(image_dir):
		for file in files:
			path = os.path.join(root, file)
			#print(path)
			if is_image_file(file) and path not in scanned:
				# extract the person name from the image path
				name = os.path.basename(root)
				# load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
				image = cv2.imread(path)
				rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				# Use Face_recognition to locate faces
				boxes = face_recognition.face_locations(rgb, model='hog')
				# compute the facial embedding for the face
				encodings = face_recognition.face_encodings(rgb, boxes)
				# loop over the encodings
				for encoding in encodings:
					known_encodings.append(encoding)
					known_names.append(name)
				#save scanned faces
				with open('scanned-faces', 'at') as f:
					f.write(path+'\n')


	#save encodings along with their names in dictionary faces_data
	faces_data = {"encodings": known_encodings, "names": known_names}
	#print(faces_data['names'])

	#use pickle to save data into a file for later use
	if len(faces_data["encodings"]) != 0:
		with open('faces-data.pickle', mode) as f:
			pickle.dump(faces_data, f)

#get_faces_data("wb")
#get_faces_data("ab")