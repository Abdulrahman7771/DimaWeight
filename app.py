import os
import pandas as pd
import streamlit as st 
from Estimator import AndalibEstimator,Sabry


uploaded_files = st.file_uploader("Choose a images to estimate", accept_multiple_files=True)
option = st.selectbox(
        "How would you like to estimate the weight",
        ("3ndalib2.0", "Sabry1.1"))
for file in uploaded_files:
    if file is not None:
        path = None
        bytes_data = file.read()  # read the content of the file in binary
        with open(os.path.join("tmp", file.name), "wb") as f:
            f.write(bytes_data)  # write this content elsewhere
        path = os.path.join("tmp", file.name)
        
        if(option=="3ndalib2.0"):
            weight,im,gr,thr,CImg,label,area,depth,volume,density,mass = AndalibEstimator(path)
            st.header("Original image")
            st.image(im)
            st.header("Gray Scale image")
            st.image(gr)
            st.header("Thresholded image")
            st.image(thr)
            st.header("Edge detected image")
            st.image(CImg)
            df = pd.DataFrame({
                            "Name":[f.name],
                            "PredictedClass":label,
                            "EstimatedArea":area,
                            "EstimatedDepth":depth,
                            "EstimatedVolume":volume,
                            "ClassDensity":density,
                            "EstimatedMass":mass,
                            "EstimatedWeight":weight,
            })
            st.dataframe(df)
        else:
            weight,im,bl,bi,CImg,label,area,density,mass = Sabry(path)
            st.header("Original image")
            st.image(im)
            st.header("Blurred Scale image")
            st.image(bl)
            st.header("Binary Thresholded image")
            st.image(bi)
            st.header("Edge detected image")
            st.image(CImg)
            df = pd.DataFrame({
                            "Name":[f.name],
                            "PredictedClass":label,
                            "EstimatedArea":area,
                            "ClassDensity":density,
                            "EstimatedMass":mass,
                            "EstimatedWeight":weight,
            })
            st.dataframe(df)




