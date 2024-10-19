import ultralytics
import cv2
import os 
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import torch

def write_video(video_in_filepath, video_out_filepath, detection_model):
    if not os.path.exists(video_in_filepath):
        print('video filepath not valid')
    
    video_reader = cv2.VideoCapture(video_in_filepath)
    
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    
    video_writer = cv2.VideoWriter(video_out_filepath,
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               fps, 
                               (frame_w, frame_h))

    for (range(nb_frames)):
        ret, image_np = video_reader.read()
        # result = detection_model(image_np)
        print(image_np.shape)
        image_np2 = image_np[...,::-1]
        # image = Image.fromarray(image_np2)
        # image.show()
        results = detection_model(image_np2)
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)
            
    # Release camera and close windows
    video_reader.release()
    video_writer.release() 
    cv2.destroyAllWindows() 
    cv2.waitKey(1)


model = YOLO("best.pt", task="detection")

write_video('balloon.mp4', 'output.mp4', model)