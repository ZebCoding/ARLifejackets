# import the necessary packages
import numpy as np
import cv2
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()
# load the image of the lifejacket
lifejacket = cv2.imread('lifejacket.png', cv2.IMREAD_UNCHANGED)

# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # resize the lifejacket to match the bounding box size
        resized_lifejacket = cv2.resize(lifejacket, (xB - xA, yB - yA))    

                # create a mask of the lifejacket image
        gray_lifejacket = cv2.cvtColor(resized_lifejacket, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_lifejacket, 10, 255, cv2.THRESH_BINARY)

        # combine the mask with the original image
        mask_inv = cv2.bitwise_not(mask)
        person_roi = frame[yA:yB, xA:xB]
        person_masked = cv2.bitwise_and(person_roi, person_roi, mask=mask_inv)
        lifejacket_masked = cv2.bitwise_and(resized_lifejacket, resized_lifejacket, mask=mask)
        combined_roi = cv2.bitwise_or(person_masked, lifejacket_masked)

        # put the combined ROI back into the frame
        frame[yA:yB, xA:xB] = combined_roi

        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)