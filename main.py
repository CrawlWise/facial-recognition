import cv2, glob

all_img = glob.glob("img/*.jpg")
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for image in all_img:
    img = cv2.imread(image)
    cvt2Grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(cvt2Grey, 1.7, 5)
    for (x,y,w,h) in faces:
        fn_img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3 )

    cv2.imshow("Face Detector", fn_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()