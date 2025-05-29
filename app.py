# 1
import gradio as gr
import os
import torch
from model import create_model
from timeit import default_timer as timer
from typing import Tuple, Dict
# setup class names
with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]
# model and transforms
model,model_transforms=create_model(num_classes=10)
# load the save weights
model.load_state_dict(
    torch.load(f="resnet_model.pth",
               map_location=torch.device("cpu"))
)
# predict func
def predict(img):
    try:
        # Check if image is received
        if img is None:
            return "No image uploaded!", 0.0

        # Model loading (assuming effnetb2 is defined in create_model)
        model, model_transforms = create_model(num_classes=10)
        model.load_state_dict(
            torch.load(f="resnet_model.pth", map_location=torch.device("cpu"))
        )

        start_time = timer()
        img = model_transforms(img).unsqueeze(0)  # Add batch dim
        model.eval()
        with torch.inference_mode():
            pred_logit = model(img)
            pred_probs = torch.softmax(pred_logit, dim=1)
            pred_labels_and_probs = {
                class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
            }
            end_time = timer()
            pred_time = round(end_time - start_time, 4)
        return pred_labels_and_probs, pred_time
    except Exception as e:
        return f"An error occurred: {str(e)}", 0.0


# 4gradip app
import gradio as gr
title="CIFAR-10"
description="Classify images into 10 CIFAR-10 classes using a ResNet50 model. Quick, accurate, and a great demonstration of computer vision in action "
# example list
# getting list of list

example_list=[["examples/"+example]for example in os.listdir("examples")]
# craete the gradient demo
demo=gr.Interface(fn=predict, #maps input to output
                  inputs=gr.Image(type="pil"),
                  outputs=[gr.Label(num_top_classes=3,label="Predictions"),
                           gr.Number(label="Prediction time(s)")],
                           examples=example_list,
                           title=title,
                           description=description

                  )
# launch it
demo.launch(
    debug=False

    # preints error locally? like in googlr collab
    # generate link publicaly like share with public
)
