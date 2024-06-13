import json
import os
import re

import gradio as gr
from internvl.train.dataset import TCSLoader
from PIL import Image, ImageDraw

tcs_loader = TCSLoader('~/petreloss.conf')


global data_lines
global data_index
global prefix_str


def save_current_conversations(question1, answer1, question2, answer2, question3, answer3, question4, answer4,
                               question5, answer5):
    global data_index
    new_conversations = []
    questions_answers = [(question1, answer1), (question2, answer2), (question3, answer3), (question4, answer4),
                         (question5, answer5)]

    for question, answer in questions_answers:
        if len(question) > 0 and len(answer) > 0:
            new_conversations.append({'from': 'human', 'value': question})
            new_conversations.append({'from': 'gpt', 'value': answer})

    item = json.loads(data_lines[data_index])
    item['conversations'] = new_conversations
    data_lines[data_index] = json.dumps(item, ensure_ascii=False)
    return 'Conversations saved!'


def save_conversations(question1, answer1, question2, answer2, question3, answer3, question4, answer4, question5,
                       answer5, file_path_input):
    save_current_conversations(question1, answer1, question2, answer2, question3, answer3, question4, answer4,
                               question5, answer5)
    with open(file_path_input, 'w') as f:
        f.writelines(data_lines)
        print('Saved to', file_path_input)
    return 'Saved successfully.'


def query_gpt4(question, image_path):
    # 伪代码示例，实际实现需要根据你的GPT-4 API进行
    # 假设返回GPT-4生成的答案
    return 'GPT-4 generated answer for the question.'


def draw_box(image, conversations):
    PATTERN = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')
    image_width, image_height = image.size  # 获取图像的宽度和高度
    for conversation in conversations:
        value = conversation['value']
        bbox = re.findall(PATTERN, value)
        try:
            bbox = (float(bbox[0][0]), float(bbox[0][1]), float(bbox[0][2]), float(bbox[0][3]))
            # 将归一化的坐标转换为实际图像坐标
            bbox = (bbox[0] / 1000 * image_width, bbox[1] / 1000 * image_height,
                    bbox[2] / 1000 * image_width, bbox[3] / 1000 * image_height)
        except:
            bbox = None
        if bbox is not None:
            draw = ImageDraw.Draw(image)
            draw.rectangle(bbox, outline='red')
    return image


def load_data(prefix, file_path):
    f = open(file_path, 'r')
    global data_lines
    data_lines = f.readlines()
    global data_index
    data_index = 0
    item = json.loads(data_lines[data_index])
    global prefix_str
    prefix_str = prefix
    image_path = os.path.join(prefix_str, item['image'])
    if image_path.startswith('/'):
        image = Image.open(image_path).convert('RGB')
    else:
        image = tcs_loader(image_path)
    conversations = item['conversations']
    image = draw_box(image, conversations)
    outputs = [data_index, item['image'], image]
    # 填充对话文本框
    for i in range(10):
        if i < len(conversations):
            conv = conversations[i]
            value = conv['value']
            outputs.append(value)
        else:
            outputs.append('')
    return outputs


def load_prev_data(question1, answer1, question2, answer2, question3, answer3, question4, answer4, question5, answer5):
    global data_index
    save_current_conversations(question1, answer1, question2, answer2, question3, answer3, question4, answer4,
                               question5, answer5)
    if data_index > 0:
        data_index -= 1
    item = json.loads(data_lines[data_index])
    global prefix_str
    image_path = os.path.join(prefix_str, item['image'])
    if image_path.startswith('/'):
        image = Image.open(image_path).convert('RGB')
    else:
        image = tcs_loader(image_path)
    conversations = item['conversations']
    image = draw_box(image, conversations)
    outputs = [data_index, item['image'], image]
    # 填充对话文本框
    for i in range(10):
        if i < len(conversations):
            conv = conversations[i]
            value = conv['value']
            outputs.append(value)
        else:
            outputs.append('')  # 清空文本框
    return outputs


def load_next_data(question1, answer1, question2, answer2, question3, answer3, question4, answer4, question5, answer5):
    global data_index
    save_current_conversations(question1, answer1, question2, answer2, question3, answer3, question4, answer4,
                               question5, answer5)
    if data_index < len(data_lines) - 1:
        data_index += 1
    item = json.loads(data_lines[data_index])
    global prefix_str
    image_path = os.path.join(prefix_str, item['image'])
    if image_path.startswith('/'):
        image = Image.open(image_path).convert('RGB')
    else:
        image = tcs_loader(image_path)
    conversations = item['conversations']
    image = draw_box(image, conversations)
    outputs = [data_index, item['image'], image]
    # 填充对话文本框
    for i in range(10):
        if i < len(conversations):
            conv = conversations[i]
            value = conv['value']
            outputs.append(value)
        else:
            outputs.append('')  # 清空文本框
    return outputs


# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown('Image Description Annotation Tool')

    with gr.Row():
        with gr.Column():
            prefix_input = gr.Textbox(label='Enter Prefix to Image File',
                                      value='/mnt/petrelfs/wangwenhai/workspace/ChatDataPreprocess/chat_history/serve_images')
            file_path_input = gr.Textbox(label='Enter Path to Annotation File',
                                         value='/mnt/petrelfs/wangwenhai/workspace/ChatDataPreprocess/chat_history/conversations_v2.jsonl')
            load_button = gr.Button('Load')
            image_box = gr.Image(label='Image', type='pil')
            prev_button = gr.Button('Previous')
            next_button = gr.Button('Next')

        with gr.Column():
            index_box = gr.Textbox(label='Index', placeholder='Data Index')
            item_box = gr.Textbox(label='Image', value='')

            question1_box = gr.Textbox(label='Question 1', value='', interactive=True)
            answer1_box = gr.Textbox(label='Answer 1', value='', interactive=True)

            question2_box = gr.Textbox(label='Question 2', value='', interactive=True)
            answer2_box = gr.Textbox(label='Answer 2', value='', interactive=True)

            question3_box = gr.Textbox(label='Question 3', value='', interactive=True)
            answer3_box = gr.Textbox(label='Answer 3', value='', interactive=True)

            question4_box = gr.Textbox(label='Question 4', value='', interactive=True)
            answer4_box = gr.Textbox(label='Answer 4', value='', interactive=True)

            question5_box = gr.Textbox(label='Question 5', value='', interactive=True)
            answer5_box = gr.Textbox(label='Answer 5', value='', interactive=True)

            save_button = gr.Button('Save')

    load_button.click(
        fn=load_data,
        inputs=[prefix_input, file_path_input],
        outputs=[index_box, item_box, image_box,
                 question1_box, answer1_box, question2_box, answer2_box, question3_box, answer3_box,
                 question4_box, answer4_box, question5_box, answer5_box]
    )

    prev_button.click(
        fn=load_prev_data,
        inputs=[question1_box, answer1_box, question2_box, answer2_box, question3_box, answer3_box, question4_box,
                answer4_box, question5_box, answer5_box],
        outputs=[index_box, item_box, image_box,
                 question1_box, answer1_box, question2_box, answer2_box, question3_box, answer3_box,
                 question4_box, answer4_box, question5_box, answer5_box]
    )

    next_button.click(
        fn=load_next_data,
        inputs=[question1_box, answer1_box, question2_box, answer2_box, question3_box, answer3_box, question4_box,
                answer4_box, question5_box, answer5_box],
        outputs=[index_box, item_box, image_box,
                 question1_box, answer1_box, question2_box, answer2_box, question3_box, answer3_box,
                 question4_box, answer4_box, question5_box, answer5_box]
    )

    save_button.click(
        inputs=[question1_box, answer1_box, question2_box, answer2_box, question3_box, answer3_box, question4_box,
                answer4_box, question5_box, answer5_box, file_path_input],
        fn=save_conversations
    )

demo.launch(server_name='0.0.0.0', server_port=10027)
