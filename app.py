# import gradio as gr
# import spaces
# import torch

# zero = torch.Tensor([0]).cuda()
# print(zero.device) # <-- 'cpu' 🤔

# @spaces.GPU
# def greet(n):
#     print(zero.device) # <-- 'cuda:0' 🤗
#     return f"Hello {zero + n} Tensor"

# demo = gr.Interface(fn=greet, inputs=gr.Number(), outputs=gr.Text())
# demo.launch()

__all__ = ['is_cat', 'learn', 'classify_image', 'categories', 'image', 'label', 'example', 'intf']

from fastai.vision.all import *
import gradio as gr

def is_cat(x):
    return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Dog','Cat')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories,map(float, probs)))

image = gr.Image(height=256, width=256)
label = gr.Label()

example = [
    'Egyptian_Mau_167.jpg',
    'pug_52.jpg',
    'beagle_104.jpg',
    'Abyssinian_219.jpg'
]

intf = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    examples=example
)

intf.launch(inline=False)