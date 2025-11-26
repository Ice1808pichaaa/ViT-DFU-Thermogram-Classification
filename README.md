# ViT-DFU-Thermogram-Classification

This is an image classification project that classifies thermogram images of Diabetic Foot Ulcers (DFU) by using a Vision Transformer (ViT) model.

## Blog (Thai)

[Medium](https://medium.com/@icepicha/diabetic-foot-ulcers-classification-by-using-planter-foot-thermogram-3de9db31ad54)

## Model Performance

The following table shows the performance metrics (Mean ± Std) across folds for the Vision Transformer (ViT) and Data-efficient Image Transformer (DeiT) models.

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **ViT** | 0.9193 ± 0.0489 | 0.9416 ± 0.0493 | 0.9508 ± 0.0471 | 0.9452 ± 0.0330 |
| **DeiT** | 0.9221 ± 0.0387 | 0.9541 ± 0.0180 | 0.9385 ± 0.0456 | 0.9459 ± 0.0277 |

## Deployment

Try it! : [Huggingface space](https://huggingface.co/spaces/pichaaa1808/Explainable-DFU-ViT)

## Reference

• [Dataset](https://ieee-dataport.org/open-access/plantar-thermogram-database-study-diabetic-foot-complications) 
