import face_recognition
import cv2
import numpy as np
import sqlite3
import datetime
import os
import shutil


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
LST_image = face_recognition.load_image_file("LST.jpg")
LST_face_encoding = face_recognition.face_encodings(LST_image)[0]

# Load a second sample picture and learn how to recognize it.
XL_image = face_recognition.load_image_file("XL.jpg")
XL_face_encoding = face_recognition.face_encodings(XL_image)[0]

TLG_image = face_recognition.load_image_file("TLG.jpg")
TLG_face_encoding = face_recognition.face_encodings(TLG_image)[0]

MD_image = face_recognition.load_image_file("MD.jpg")
MD_face_encoding = face_recognition.face_encodings(MD_image)[0]

FXY_image = face_recognition.load_image_file("FXY.jpg")
FXY_face_encoding = face_recognition.face_encodings(FXY_image)[0]

DL_image = face_recognition.load_image_file("DL.jpg")
DL_face_encoding = face_recognition.face_encodings(DL_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    LST_face_encoding,
    XL_face_encoding,
    TLG_face_encoding,
    MD_face_encoding,
    FXY_face_encoding,
    DL_face_encoding
]
known_face_names = [
    "LST",
    "XL",
    "TLG",
    "MD",
    "FXY",
    "DL"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

Id_Dict = {'LST':1,'XL':2,'TLG':3,'MD':4,'FXY':5,'DL':6,'Unknown':7}

Name_list = ["'LST'","'XL'","'TLG'","'MD'","'FXY'","'DL'","'Unknown'"]

conn = sqlite3.connect('Attendence_Checking.db')
c = conn.cursor()

OCR_COUNT = 0

day_info = a = np.datetime64(datetime.datetime.now().strftime('%Y-%m-%d'))

current_path = os.getcwd()

day_index = datetime.datetime.now().strftime('%Y-%m-%d')
    
    
while True:
    
    isExists = os.path.exists(current_path+'/'+datetime.datetime.now().strftime('%Y-%m-%d'))
    
    if not isExists:
        os.mkdir(current_path+'/'+datetime.datetime.now().strftime('%Y-%m-%d'))
        os.mkdir(current_path+'/'+datetime.datetime.now().strftime('%Y-%m-%d')+'/'+'Record')
        os.mkdir(current_path+'/'+datetime.datetime.now().strftime('%Y-%m-%d')+'/'+'Daily_Record')
        
    current_time = np.datetime64(datetime.datetime.now().strftime('%Y-%m-%d'))
    
    if ((current_time - day_info) == np.timedelta64(1,'D')):
        for i in Name_list:
            First_rec = c.execute("SELECT min(rowid), * FROM OCCURENCE WHERE NAME == " + i)
            for row in First_rec:
                Member_Name = row[2]
                if Member_Name == None:
                    First_Occurence_time = 'Absence'
                else:
                    First_Occurence_time = row[3]

            Last_rec = c.execute("SELECT max(rowid), * FROM OCCURENCE WHERE NAME == " + i)

            for row in Last_rec:
                Member_Name = row[2]
                if Member_Name == None:
                    Last_Occurence_time = 'Absence'
                else:
                    Last_Occurence_time = row[3]
            
            str=i.lstrip("'")
            str = str.rstrip("'")
            c.execute('INSERT INTO '+str+'_DAILY_REC VALUES(?,?,?,?,?)',(OCR_COUNT, Member_Name,day_index,First_Occurence_time,Last_Occurence_time))

            img_list = os.listdir(current_path +'/'+day_index+'/'+ 'Record' +'/')
            if First_Occurence_time == 'Absence':
                continue
            else:
                try:
                    shutil.copyfile(current_path +'/'+day_index+'/'+ 'Record' +'/'+First_Occurence_time+Member_Name+'.png',current_path +'/'+day_index+'/'+ 'Daily_Record' +'/'+First_Occurence_time+Member_Name+'.png')
                except:
                    print('No file found')
                
            if Last_Occurence_time == 'Absence':
                continue
            else:
                try:
                    shutil.copyfile(current_path +'/'+day_index+'/'+ 'Record' +'/'+Last_Occurence_time+Member_Name+'.png',current_path +'/'+day_index+'/'+ 'Daily_Record' +'/'+Last_Occurence_time+Member_Name+'.png')
                except:
                    print('No file found')
        
        
        shutil.rmtree(current_path +'/'+day_index+'/'+ 'Record' +'/')


        day_info = a = np.datetime64(datetime.datetime.now().strftime('%Y-%m-%d'))
        day_index = datetime.datetime.now().strftime('%Y-%m-%d')

            #img_list = os.listdir(current_path +'/'+day_index+'/')
        #img_list.sort()
        #for i in img_list:
            
    
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            c.execute("INSERT INTO OCCURENCE VALUES(?,?,?)",(OCR_COUNT, name,nowTime))
            np.datetime64
            OCR_COUNT = OCR_COUNT + 1
            conn.commit()

    process_this_frame = not process_this_frame
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    
        cv2.imwrite(current_path +'/'+datetime.datetime.now().strftime('%Y-%m-%d')+'/'+ 'Record' +'/'+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+ name +'.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
    
    
   
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    

        
        
        
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()