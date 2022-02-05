import cv2 
import os,sys 
from pathlib import Path 
from cv2 import dnn_superres


def superres_image(image, sr):
    # Upscale the image
    result = sr.upsample(image)
    return result

def main(): 
    input_video = "/ps/project/EmotionalFacialAnimation/data/lrs3/extracted/test/27lMmdmySb8/00001.mp4"
    output_video = "test.mp4" 

    # Read image
    image = cv2.imread('./input.png')

    # Read the desired model
    path = Path(__file__).parent / "FSRCNN_x4.pb"# Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(path))

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("fsrcnn", 4)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    outvid = None 
    

    cap = cv2.VideoCapture(input_video)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        
        # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Upscale the image
            super_frame = sr.upsample(frame)
            resized = cv2.resize(super_frame, frame.shape[:2], interpolation = cv2.INTER_CUBIC)
            if outvid is None: 
                outvid = cv2.VideoWriter(output_video, fourcc, 20.0, super_frame.shape[:2])
            outvid.write(super_frame)

        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()
    outvid.release()







if __name__ == "__main__":
    main()
