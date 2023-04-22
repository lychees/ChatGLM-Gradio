import re
import os
import json
import gradio as gr
from fastapi import FastAPI
from gradio_client import Client

def process_text(text):
    text = text.encode('raw_unicode_escape').decode('unicode-escape').encode('utf-16_BE','surrogatepass').decode('utf-16_BE')
    text = text.replace('\n', '')  # JSON doesn't accept \n
    return text

def predict(input, history=None):
    if history is None:
        history = []

    client = Client('https://multimodalart-chatglm-6b.hf.space/')
    with open(client.predict(input, fn_index=0)) as f: 
        text = process_text(f.read())
        output = json.loads(text)[0]
        history += [output]
        return history, history

with gr.Blocks() as demo:
    gr.Markdown('''## ChatGLM-6B - unofficial demo
    Unnoficial demo of the [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B/blob/main/README_en.md) model, trained on 1T tokens of English and Chinese
    ''')
    state = gr.State([])
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=400)
    with gr.Row():
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
        with gr.Column(scale=1):
            button = gr.Button("Generate")
    txt.submit(predict, [txt, state], [chatbot, state])
    button.click(predict, [txt, state], [chatbot, state])

CUSTOM_PATH = os.getenv('CUSTOM_PATH')
app = FastAPI()

app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)
