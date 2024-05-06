import cv2
import numpy as np
from picamera2 import Picamera2

# Load the template image
template = cv2.imread('/home/bri/template.jpg', 0)

with Picamera2() as video:
    height = 640
    width = 480
    video.configure(video.create_video_configuration(main={"format": 'RGB888', "size": (width, height)}))
    video.start()
    # Get template dimensions
    w, h = template.shape[::-1]

    # Define a threshold for matching
    threshold = 0.1

    while True:
        # Read the next frame
        frame = video.capture_array()

        # Check if frame is None (end of video)
        if frame is None:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

        # Find locations where the correlation coefficient is above the threshold
        loc = np.where(res >= threshold)

        # Calculate the average coordinates
        if len(loc[0]) > 0:
            avg_x = int(np.mean(loc[1]))
            avg_y = int(np.mean(loc[0]))
            avg_w = w
            avg_h = h

            # Draw the bounding box around the average location
            cv2.rectangle(frame, (avg_x, avg_y), (avg_x + avg_w, avg_y + avg_h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Matching', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video object
    video.release()
    cv2.destroyAllWindows()
