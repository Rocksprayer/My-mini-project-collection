import cv2 as cv

print(cv.__version__)
RED = (0, 0, 255)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame=cv.flip(frame,1)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)



    face_detector = cv.CascadeClassifier( cv.data.haarcascades +'haarcascade_frontalface_default.xml')

    face_rects = face_detector.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv.CASCADE_SCALE_IMAGE)

    print(f'found {len(face_rects)} face(s)')

    for rect in face_rects:
        cv.rectangle(frame, rect, RED, 2)
        cv.imshow('frame', frame)
    else:
        cv.imshow('frame',frame)
    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()