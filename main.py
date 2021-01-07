import cv2

# face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontal_face.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eyes.xml')

# grab webcam feed
webcam = cv2.VideoCapture(0)  # if you put (0), then it means webcam but you can also put a videofile path in there

# Iterate forever over frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # if there's an error, abort
    if not successful_frame_read:
        break

    # Now we will convert the image to GreyScale as it is easier to recognize face in greyscale
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_detector.detectMultiScale(greyscaled_img, scaleFactor=1.7, minNeighbors=2)

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the sub-frame (usnig numpy N-directional array slicing)
        the_face = frame[y:y+h, x:x+w]
        face_greyscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # here scaleFactor tells how much you want to blur your image, so that you can algorithm can easily identify face
        # minNeighbour=30 is you should have 20 rectangles near the smile to consider it as a smile
        smile = smile_detector.detectMultiScale(face_greyscale, scaleFactor=1.7, minNeighbors=30)

        # detect eyes
        eyes = eye_detector.detectMultiScale(greyscaled_img, scaleFactor=1.7, minNeighbors=25)

        for (x_, y_, w_, h_) in smile:
             cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (0, 0, 255), 2)

        for (x__, y__, w__, h__) in eyes:
            cv2.rectangle(frame, (x__, y__), (x__ + w__, y__ + h__), (255, 255, 255), 2)

        # smile is an array as you can see on line 38, so if there is one or more smiles than this statement will run
        if len(smile)>0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # this will give show the image and give title to the window that is opened
    cv2.imshow('Vendz Smile Detector', frame)

    # here inside 'waitKey(1)' are putting (1) because if we don't put anything then it won't play video in real-time
    # and display each frame after a key is pressed on keyboard, so by putting (1) it will wait 1 millisecond and then
    # display next frame
    key = cv2.waitKey(1)

    # Stop is Q key is pressed
    if key == 81 or key == 113:  # here 113 is for 'q' and 81 is for 'Q'
        break

# cleanup code
webcam.release()
cv2.destroyAllWindows()
