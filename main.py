from deepface import DeepFace
from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt
import sqlite3
import os
import string, random
import glob


# identify faces in picturs function
def identify_face_picture(img):
    img_read = cv2.imread(img)
    capface = RetinaFace.detect_faces(img)
    for key in capface.keys():  # This returns the key value of Retinaface.detect
        detect_face = capface[key]  # This returns all the key values in capface.keys
        facial_area = detect_face["facial_area"]  # This capture the facial area in the dictionary
        cv2.rectangle(img_read, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (0, 255, 0),
                      3)  # Draw a rectangle on the poeples faces

    plt.imshow(img_read[:, :, ::-1])  # This convert the default behavour of Retinal picture in grey to color
    plt.show()


# Extract faces from pictures
def extract_faces(imface):
    exface = RetinaFace.extract_faces(imface)
    for face in exface:
        post_fix = random.choice(string.digits)
        filename = f"img{post_fix}.jpg"
        cv2.imwrite('exfaces/' + filename, face)


# Identify face in a camera function
def extract_face_camera():
    detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(1)  # Capture the vidoe
    # declaring my while loop to continously capture my face on the camera
    while True:

        # Read frame if the the correct value if provided
        ret, frame = cap.read()
        if not ret:
            break

        c2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect.detectMultiScale(c2gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)  # This draws rectangle on the user faces
        cv2.imshow("video Frame", frame)

        # Save the picture of the user to file
        if cv2.waitKey(1) == ord('q'):
            post_fix = random.choice(string.digits)  # This generates 1-9 and append behind name
            filename = f"cap{post_fix}.jpg"
            cv2.imwrite('excampic/' + filename, frame)  # Save the picture to excampic folder

            extract_faces(f'excampic/{filename}')  # Call my extract Picture faces from pictures

            cap.release()
            cv2.destroyAllWindows()


# Save pictures extracted from camera or picture to sqlite_db
def save_sqlite_db():
    # connecting to database and writing my pictures into it
    db_conn = sqlite3.connect("face-detection.db")
    db_cursor = db_conn.cursor()

    # Create a database  and a table if not existed.
    db_cursor.execute("CREATE TABLE IF NOT EXISTS images(image BLOB, name TEXT)")
    db_conn.commit()
    # extract_faces('img/faces.jpg') #Function that extract faces and save them in the exfaces directory

    # save extracted images on pictures to db
    imp_images = glob.glob(f'exfaces/*.jpg')
    for img in imp_images:
        # extract Image names
        cv2.imread(img)  # read image
        read_img_path = os.path.abspath(img)  # Get the absolute path of the picture
        filename, ext = os.path.splitext(os.path.basename(
            read_img_path))  # extract the name of the picture by splitting it into two.. extention and name

        # open images in binary format
        with open(f'{img}', 'rb') as image:  # convert images to raw binary
            db_image = image.read()
            db_cursor.execute("INSERT INTO images (image, name) VALUES(?,?)",
                              (db_image, filename))  # Insert Picture to sqlite db.
            db_conn.commit()

    db_conn.close()
    db_conn.close()

identify_face_picture('img/happy.jpg')
#save_sqlite_db()

