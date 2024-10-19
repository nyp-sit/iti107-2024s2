import cv2

from ultralytics import YOLO
from matplotlib import pyplot as plt

# Load the YOLO model
model = YOLO("best.pt", task="detection")

# Open the video file
video_path = "balloon.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        print(frame.shape)
        plt.imshow(frame)
        plt.show()
        # results = model(frame)

        # # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # # Display the annotated frame
        # cv2.imshow("YOLO Inference", annotated_frame)

        # # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()