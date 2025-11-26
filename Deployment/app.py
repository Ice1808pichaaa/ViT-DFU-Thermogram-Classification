import streamlit as st
import torch
import numpy as np
import cv2

from transformers import (
    pipeline,
    ViTImageProcessor,
    ViTForImageClassification
)
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_FOLDER_PATH = "final_vit_model"  


sample_img = {
    "<None>": None,
    "Right_Normal_1": "img/[0] Normal/Normal_R_1.png",
    "Left_Normal_1": "img/[0] Normal/Normal_L_1.png",
    "Right_Normal_2": "img/[0] Normal/Normal_R_2.png",
    "Left_Normal_2": "img/[0] Normal/Normal_L_2.png",

    "Right_DFU_1": "img/[1] DFU/DFU_R_1.png",
    "Left_DFU_1": "img/[1] DFU/DFU_L_1.png",
    "Right_DFU_2": "img/[1] DFU/DFU_R_2.png",
    "Left_DFU_2": "img/[1] DFU/DFU_L_2.png",
}

sample_pairs = {
    "<None>": (None, None),

    "Normal Pair 1": (
        "img/[0] Normal/Normal_R_1.png",
        "img/[0] Normal/Normal_L_1.png",
    ),
    "Normal Pair 2": (
        "img/[0] Normal/Normal_R_2.png",
        "img/[0] Normal/Normal_L_2.png",
    ),

    "DFU Pair 1": (
        "img/[1] DFU/DFU_R_1.png",
        "img/[1] DFU/DFU_L_1.png",
    ),
    "DFU Pair 2": (
        "img/[1] DFU/DFU_R_2.png",
        "img/[1] DFU/DFU_L_2.png",
    ),
}

def reshape_transform(tensor):
    """
    For ViT: remove CLS token and reshape sequence (N) to (H, W).
    """
    tensor = tensor[:, 1:, :] 
    B, N, C = tensor.shape
    H = W = int(N ** 0.5)
    tensor = tensor.reshape(B, H, W, C)
    tensor = tensor.permute(0, 3, 1, 2) 
    return tensor

class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


@st.cache_resource
def load_classifier():
    return pipeline(
        task="image-classification",
        model=MODEL_FOLDER_PATH
    )
def load_gradcam():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    processor = ViTImageProcessor.from_pretrained(MODEL_FOLDER_PATH)
    hf_model = ViTForImageClassification.from_pretrained(MODEL_FOLDER_PATH)

    model = HuggingfaceToTensorModelWrapper(hf_model).to(device).eval()
    target_layers = [model.model.vit.encoder.layer[-1].layernorm_before]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    return cam, processor, device

def compute_gradcam_for_pil(pil_img, target_index: int):
    cam, processor, device = load_gradcam()

    img_np = np.array(pil_img).astype(np.float32) / 255.0

    inputs = processor(images=pil_img, return_tensors="pt")
    input_tensor = inputs["pixel_values"].to(device)

    targets = [ClassifierOutputTarget(target_index)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    H, W, _ = img_np.shape
    grayscale_cam_resized = cv2.resize(grayscale_cam, (W, H))

    cam_vis = show_cam_on_image(img_np, grayscale_cam_resized, use_rgb=True)
    return img_np, cam_vis


def pretty_label(raw_label: str) -> str:
    mapping = {
        "0": "Normal",                    
        "1": "Diabetic Foot Ulcers",    
    }
    
    return mapping.get(raw_label, raw_label)


def get_target_index(raw_label: str) -> int:
    pretty = pretty_label(raw_label)

    if pretty == "Normal":
        return 0
    elif pretty == "Diabetic Foot Ulcers":
        return 1

    return 1


def app():
    st.title("Early Detection of Diabetic Foot Ulcers Using Thermal Imaging with Vision Transformer & Grad-CAM")

    mode = st.radio(
        "Choose input mode",
        ["Use sample pair (Right + Left)", "Upload your own Right & Left images"],
        index=0
    )
    
    upload_right = None
    upload_left = None
    if mode == "Upload your own Right & Left images":
        upload_right = st.file_uploader(
            "Upload Right Foot Image",
            type=["png", "jpg", "jpeg"],
            key="right_upl"
        )
        upload_left = st.file_uploader(
            "Upload Left Foot Image",
            type=["png", "jpg", "jpeg"],
            key="left_upl"
        )
    
    right_image = None
    left_image = None
    right_path = left_path = None

    if mode == "Use sample pair (Right + Left)":
        with st.expander("Choose a sample pair and view all sample images", expanded=False):
            pair_name = st.selectbox(
                "Select a sample pair (Right + Left):",
                list(sample_pairs.keys()),
                index=0
            )

            st.markdown("**Normal Group**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.image(sample_img["Right_Normal_1"], caption="Right_Normal_1", width='stretch')
            with c2:
                st.image(sample_img["Left_Normal_1"], caption="Left_Normal_1", width='stretch')
            with c3:
                st.image(sample_img["Right_Normal_2"], caption="Right_Normal_2", width='stretch')
            with c4:
                st.image(sample_img["Left_Normal_2"], caption="Left_Normal_2", width='stretch')

            st.markdown("**Diabetic Foot Ulcers Group**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.image(sample_img["Right_DFU_1"], caption="Right_DFU_1", width='stretch')
            with c2:
                st.image(sample_img["Left_DFU_1"], caption="Left_DFU_1", width='stretch')
            with c3:
                st.image(sample_img["Right_DFU_2"], caption="Right_DFU_2", width='stretch')
            with c4:
                st.image(sample_img["Left_DFU_2"], caption="Left_DFU_2", width='stretch')

        right_path, left_path = sample_pairs[pair_name]

    col_input, col_output = st.columns(2)
    col_input.header("Input Images")
    col_output.header("Predictions")

    right_col_in, left_col_in = col_input.columns(2)
    right_col_in.subheader("Right Foot")
    left_col_in.subheader("Left Foot")

    if mode == "Use sample pair (Right + Left)":
        if right_path is not None:
            right_image = Image.open(right_path).convert("RGB")
            right_col_in.image(right_image, caption="Sample Right Foot", width='stretch')

        if left_path is not None:
            left_image = Image.open(left_path).convert("RGB")
            left_col_in.image(left_image, caption="Sample Left Foot", width='stretch')

    else:
        if upload_right is not None:
            right_image = Image.open(upload_right).convert("RGB")
            right_col_in.image(right_image, caption="Uploaded Right Foot", width='stretch')

        if upload_left is not None:
            left_image = Image.open(upload_left).convert("RGB")
            left_col_in.image(left_image, caption="Uploaded Left Foot", width='stretch')

    run_pred = col_output.button("Run prediction")

    out_right_col, out_left_col = col_output.columns(2)
    out_right_col.subheader("Right Foot Prediction")
    out_left_col.subheader("Left Foot Prediction")

    right_cam_vis = None
    left_cam_vis = None

    if run_pred:
        classifier = load_classifier()
        any_image = False

        if right_image is not None:
            any_image = True
            preds_right = classifier(right_image, top_k=2)

            for pred in preds_right:
                label = pretty_label(pred["label"])
                score = float(pred["score"])
                out_right_col.progress(score, text=f"{label}: {score * 100:.2f}%")

            top_right_raw_label = preds_right[0]["label"]
            right_target_index = get_target_index(top_right_raw_label)
            _, right_cam_vis = compute_gradcam_for_pil(right_image, right_target_index)

        if left_image is not None:
            any_image = True
            preds_left = classifier(left_image, top_k=2)

            for pred in preds_left:
                label = pretty_label(pred["label"])
                score = float(pred["score"])
                out_left_col.progress(score, text=f"{label}: {score * 100:.2f}%")

            top_left_raw_label = preds_left[0]["label"]
            left_target_index = get_target_index(top_left_raw_label)
            _, left_cam_vis = compute_gradcam_for_pil(left_image, left_target_index)

        if any_image:
            col_output.success("Classification finished ✅")
        else:
            col_output.warning("Please provide images before running prediction.")

        if right_cam_vis is not None or left_cam_vis is not None:
            st.markdown("---")
            if right_target_index == 1:
                right_target_index = "DFU"
            else:
                right_target_index = "Normal"
                
            st.subheader(f"Grad-CAM Visualization (Target class: {right_target_index})")

            gcol_r, gcol_l = st.columns(2)

            if right_cam_vis is not None:
                gcol_r.markdown("**Right Foot**")
                gcol_r.image(right_cam_vis, width='stretch')

            if left_cam_vis is not None:
                gcol_l.markdown("**Left Foot**")
                gcol_l.image(left_cam_vis, width='stretch')

if __name__ == "__main__":
    app()