import streamlit as st

st.set_page_config(
    page_title="Road Damage Detections Apps",
    page_icon="🛣️",
)

st.image("./resource/banner.png", use_column_width="always")
st.divider()
st.title("Road Damage Detection Application")

st.markdown(
    """
    Introducing our Road Damage Detection Apps, powered by the YOLOv8 deep learning model trained on Crowdsensing-based Road Damage Detection Challenge 2022 Dataset.
    
    This application is designed to enhance road safety and infrastructure maintenance by swiftly identifying and categorizing various forms of road damage, such as potholes and cracks.

    There is four types of damage that this model can detects such as:
    - Longitudinal Crack
    - Transverse Crack
    - Alligator Crack
    - Potholes

    The model trained on YOLOv8 small model on Japan and India CRDDC2022 dataset.

    You can select the apps from the sidebar to try and experiment with any kind of input **(realtime-webcam, video and images)** depends on your use case.

    #### Documentations and Links
    - Github Project Page [Github](https://github.com/oracl4/RoadDamageDetection)
    - You can reach me on it.mahdi.yusuf@gmail.com

    #### License and Citations
    - Road Damage Dataset from Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)
    - All rights reserved on YOLOv8 license permits by [Ultralytics](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io/) framework
"""
)

st.divider()

st.markdown(
    """
    This project is created for the [Road Damage Detection Challenge](https://s.id/RDDHariJalan23) by [Ministry of Public Works and Housing](https://pu.go.id/) for celebrating the "Peringatan Hari Jalan 2023".
    """
    
)

