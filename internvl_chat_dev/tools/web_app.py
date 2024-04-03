import json
import os

import gradio as gr
from internvl.train.dataset import TCSLoader
from PIL import Image

tcs_loader = TCSLoader('~/petreloss.conf')

global data_lines
global data_index
global prefix_str


def process_image(image, web_description, model_description, user_edit):
    # 处理逻辑，与之前相同
    return image, web_description, model_description, user_edit


def load_data(prefix, file_path):
    f = open(file_path, 'r')
    global data_lines
    data_lines = f.readlines()
    global data_index
    data_index = 0
    item = json.loads(data_lines[data_index])
    global prefix_str
    prefix_str = prefix
    try:
        caption = item['caption']
    except:
        caption = ''
    try:
        old_caption = item['old_caption']
    except:
        old_caption = caption
    image_path = os.path.join(prefix_str, item['image'])
    print(image_path)
    if image_path.startswith('/'):
        image = Image.open(image_path).convert('RGB')
    else:
        image = tcs_loader(image_path)
    print(image.size)
    return data_index, data_lines[data_index], old_caption, caption, image


# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown('Image Description Annotation Tool')

    with gr.Row():

        with gr.Column():
            image_box = gr.Image(label='Image', type='pil')
            old_caption_box = gr.Textbox(label='Old Caption')
            caption_box = gr.Textbox(label='Caption')
            prev_button = gr.Button('Previous')
            next_button = gr.Button('Next')

        with gr.Column():
            prefix_input = gr.Textbox(label='Enter Prefix to Image File', placeholder='prefix/to/your/image')
            file_path_input = gr.Textbox(label='Enter Path to Annotation File', placeholder='Path/to/your/file')
            load_button = gr.Button('Load')
            index_box = gr.Textbox(label='Index', placeholder='Data Index')
            item_box = gr.Textbox(label='Item', placeholder='Data Item')
            save_button = gr.Button('Save')

    load_button.click(
        fn=load_data,
        inputs=[prefix_input, file_path_input],
        outputs=[index_box, item_box, old_caption_box, caption_box, image_box]
    )

    def load_prev_data():
        global data_index
        if data_index > 0:
            data_index -= 1
        item = json.loads(data_lines[data_index])
        try:
            caption = item['caption']
        except:
            caption = ''
        try:
            old_caption = item['old_caption']
        except:
            old_caption = caption
        global prefix_str
        image_path = os.path.join(prefix_str, item['image'])
        if image_path.startswith('/'):
            image = Image.open(image_path).convert('RGB')
        else:
            image = tcs_loader(image_path)
        return data_index, data_lines[data_index], old_caption, caption, image

    def load_next_data():
        global data_index
        if data_index < len(data_lines) - 1:
            data_index += 1
        item = json.loads(data_lines[data_index])
        try:
            caption = item['caption']
        except:
            caption = ''
        try:
            old_caption = item['old_caption']
        except:
            old_caption = caption
        global prefix_str
        image_path = os.path.join(prefix_str, item['image'])
        if image_path.startswith('/'):
            image = Image.open(image_path).convert('RGB')
        else:
            image = tcs_loader(image_path)
        return data_index, data_lines[data_index], old_caption, caption, image

    prev_button.click(
        fn=load_prev_data,
        outputs=[index_box, item_box, old_caption_box, caption_box, image_box]
    )

    next_button.click(
        fn=load_next_data,
        outputs=[index_box, item_box, old_caption_box, caption_box, image_box]
    )

    save_button.click(
        fn=process_image
    )


demo.launch(server_name='0.0.0.0', server_port=10065)
