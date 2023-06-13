import streamlit as st
from transformers import pipeline
from PIL import Image

sample_img = {
    "<None>": None,
    "Right_Normal_1": "img/[0] Normal/Normal_R_1.png",
    "Left_Normal_1": "img/[0] Normal/Normal_L_1.png",
    "Right_Normal_2": "img/[0] Normal/Normal_R_2.png",
    "Left_Normal_2": "img/[0] Normal/Normal_L_2.png",

    "Right_DFU_1": "img/[1] DFU/DFU_R_1.png",
    "Left_DFU_1": "img/[1] DFU/DFU_L_1.png",
    "Right_DFU_2": "img/[1] DFU/DFU_R_2.png",
    "Left_DFU_2": "img/[1] DFU/DFU_L_2.png"
}

def app():
    with st.sidebar:
        st.image("resources/AIB_logo.png")
        st.markdown("<h1 style='text-align: center; color: purple ;'>AI-Builders 3: Arcane Whales</h1>", unsafe_allow_html=True)
        st.write("This project was creted by Picha Jetsadapattarakul under AI-Builders program.")

    st.title("Diabetic Foot Ulcers Classification by using planter foot Thermogram")
    st.info("This is an image classification project that classify thermogram image of Diabtic Foot Ulcers by using Vision Transformer(ViT) model.",icon="ℹ️")
    st.write("Choose sample image for classification.")
    with st.expander("Choose sample image here"):
        sample = st.selectbox(label = "Select image for classification here", options = list(sample_img.keys()), label_visibility="hidden")   
        st.write("Normal Group")
        c1, c2, c3, c4 = st.columns(4)
        st.write("Diabetic Foot Ulcers Group")
        with c1:
            st.image(sample_img["Right_Normal_1"], caption="Right_Normal_1", use_column_width=True)
        with c2:
            st.image(sample_img["Left_Normal_1"], caption="Left_Normal_1", use_column_width=True)
        with c3:
            st.image(sample_img["Right_Normal_2"], caption="Right_Normal_2", use_column_width=True)
        with c4:
            st.image(sample_img["Left_Normal_2"], caption="Left_Normal_2", use_column_width=True)   

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.image(sample_img["Right_DFU_1"], caption="Right_DFU_1", use_column_width=True)
        with c2:
            st.image(sample_img["Left_DFU_1"], caption="Left_DFU_1", use_column_width=True)
        with c3:
            st.image(sample_img["Right_DFU_2"], caption="Right_DFU_2", use_column_width=True)
        with c4:
            st.image(sample_img["Left_DFU_2"], caption="Left_DFU_2", use_column_width=True)

    upload_img = st.file_uploader("Or you can upload your image here", type=["png"])

    col1, col2= st.columns(2)
    col1.header('Input Image')
    col2.header("Output")
    
    if upload_img is not None:
        # try the model
        image = Image.open(upload_img)
        col1.image(image)

        classifier = pipeline("image-classification", model="model")
        col2.success('succeed!', icon="✅")
        col2.balloons()
        preds_result = classifier(image)
        # col2.write(preds_result)

        # change label from 0 to Normal and 1 to Diabetic Foot Ulcers for visualization
        if preds_result[0]['label'] == 0:
            label_0 = "Normal"
        elif preds_result[0]['label'] == 1 :
            label_0 = "Diabetic foot Ulcers"

        if preds_result[1]['label'] == 1:
            label_1 = "Diabetic foot Ulcers"
        elif preds_result[1]['label'] == 0:
            label_1 = "Normal"
        
        col2.write("Prediction Result: ")
        col2.progress(preds_result[0]['score'], text = f"{label_0}: {round(preds_result[0]['score']*100,2)}%")
        col2.progress(preds_result[1]['score'], text = f"{label_1}: {round(preds_result[1]['score']*100,2)}%")
    
    elif sample_img[sample] is None:
        pass

    elif sample:
        # try the model
        image = Image.open(sample_img[sample])
        col1.image(image)

        classifier = pipeline("image-classification", model="model")
        col2.success('succeed!', icon="✅")
        col2.balloons()
        preds_result = classifier(image)

        # change label from 0 to Normal and 1 to Diabetic Foot Ulcers for visualization
        if preds_result[0]['label'] == 0:
            label_0 = "Normal"
        elif preds_result[0]['label'] == 1 :
            label_0 = "Diabetic foot Ulcers"

        if preds_result[1]['label'] == 1:
            label_1 = "Diabetic foot Ulcers"
        elif preds_result[1]['label'] == 0:
            label_1 = "Normal"
        
        
        col2.progress(preds_result[0]['score'], text = f"{label_0}: {round(preds_result[0]['score']*100,2)}%")
        col2.progress(preds_result[1]['score'], text = f"{label_1}: {round(preds_result[1]['score']*100,2)}%")

         
    
if __name__ == "__main__":
    app()