import cv2
import imutils
from imutils.video import VideoStream
from blur_detector import detect_blur_fft
from skimage.filters import threshold_local
import numpy as np
import argparse
import time
import datetime

# Function to perform the scanning process on a given frame
def perform_scanning(frame):
    # Convert the frame to grayscale, blur it, and find edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # Loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # Apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(frame, screenCnt.reshape(4, 2))

    # Convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255

    # Save the "warped" picture to the "save/" directory with current time as filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"save/{timestamp}.png"
    cv2.imwrite(filename, warped)

    # Display the original and scanned images
    cv2.imshow("Original", frame)
    cv2.imshow("Scanned", warped)
    cv2.waitKey(0)

# Function to order points for four point transform
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Function to perform four point transform
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth = max(int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))), int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))))
    maxHeight = max(int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))), int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--thresh", type=int, default=5, help="threshold for our blur detector to fire")
args = vars(ap.parse_args())

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

# Initialize variables
start_time = None
not_blurry_frames = []

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it
    # to have a maximum width of 500 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Convert the frame to grayscale and detect blur in it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (mean, blurry) = detect_blur_fft(gray, size=60, thresh=args["thresh"], vis=False)

    # If the frame is not blurry
    if not blurry:
        if start_time is None:
            start_time = time.time()

        # Check if not blurry for more than 2 seconds
        if time.time() - start_time >= 2:
            not_blurry_frames.append(frame)

            # Perform scanning on the first frame in the list
            perform_scanning(not_blurry_frames[0])

            # Reset variables
            start_time = None
            not_blurry_frames = []

    else:
        start_time = None
        not_blurry_frames = []

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
