import os
import logging
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO

from sample_utils.download import download_file

st.set_page_config(
    page_title="Video Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Create temporary folder if doesn't exists
if not os.path.exists('./temp'):
   os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button == True:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

def processVideo(video_file, score_threshold):
    
    # Write the file into disk
    write_bytesio_to_file(temp_file_input, video_file)
    
    videoCapture = cv2.VideoCapture(temp_file_input)

    # Check the video
    if (videoCapture.isOpened() == False):
        st.error('Error opening the video file')
    else:
        _width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        _height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = videoCapture.get(cv2.CAP_PROP_FPS)
        _frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        _duration = _frame_count/_fps
        _duration_minutes = int(_duration/60)
        _duration_seconds = int(_duration%60)
        _duration_strings = str(_duration_minutes) + ":" + str(_duration_seconds)

        st.write("Video Duration :", _duration_strings)
        st.write("Width, Height and FPS :", _width, _height, _fps)

        inferenceBarText = "Performing inference on video, please wait."
        inferenceBar = st.progress(0, text=inferenceBarText)

        imageLocation = st.empty()

        # Issue with opencv-python with pip doesn't support h264 codec due to license, so we cant show the mp4 video on the streamlit in the cloud
        # If you can install the opencv through conda using this command, maybe you can render the video for the streamlit
        # $ conda install -c conda-forge opencv
        # fourcc_mp4 = cv2.VideoWriter_fourcc(*'h264')
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

        # Read until video is completed
        _frame_counter = 0
        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()
            if ret == True:
                
                # Convert color-chanel
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform inference
                _image = np.array(frame)

                image_resized = cv2.resize(_image, (640, 640), interpolation = cv2.INTER_AREA)
                results = net.predict(image_resized, conf=score_threshold)
                
                # Save the results
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    detections = [
                    Detection(
                        class_id=int(_box.cls),
                        label=CLASSES[int(_box.cls)],
                        score=float(_box.conf),
                        box=_box.xyxy[0].astype(int),
                        )
                        for _box in boxes
                    ]

                annotated_frame = results[0].plot()
                _image_pred = cv2.resize(annotated_frame, (_width, _height), interpolation = cv2.INTER_AREA)

                print(_image_pred.shape)
                
                # Write the image to file
                _out_frame = cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR)
                cv2writer.write(_out_frame)
                
                # Display the image
                imageLocation.image(_image_pred)

                _frame_counter = _frame_counter + 1
                inferenceBar.progress(_frame_counter/_frame_count, text=inferenceBarText)
            
            # Break the loop
            else:
                inferenceBar.empty()
                break

        # When everything done, release the video capture object
        videoCapture.release()
        cv2writer.release()

    # Download button for the video
    st.success("Video Processed!")

    col1, col2 = st.columns(2)
    with col1:
        # Also rerun the appplication after download
        with open(temp_file_infer, "rb") as f:
            st.download_button(
                label="Download Prediction Video",
                data=f,
                file_name="RDD_Prediction.mp4",
                mime="video/mp4",
                use_container_width=True
            )
            
    with col2:
        if st.button('Restart Apps', use_container_width=True, type="primary"):
            # Rerun the application
            st.rerun()

st.title("Road Damage Detection - Video")
st.write("Detect the road damage in using Video input. Upload the video and start detecting. This section can be useful for examining and process the recorded videos.")

video_file = st.file_uploader("Upload Video", type=".mp4", disabled=st.session_state.runningInference)
st.caption("There is 1GB limit for video size with .mp4 extension. Resize or cut your video if its bigger than 1GB.")

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=st.session_state.runningInference)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction. You can change the threshold before running the inference.")

if video_file is not None:
    if st.button('Process Video', use_container_width=True, disabled=st.session_state.runningInference, type="secondary", key="processing_button"):
        _warning = "Processing Video " + video_file.name
        st.warning(_warning)
        processVideo(video_file, score_threshold)