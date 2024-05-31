import json
import os
import random
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

path = '/mnt/petrelfs/wangwenhai/private_data/internvl_hr_data/laion_coco'
out_path = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/metas/merged_laion_coco'
merge_num = 4

prompts = [
    'Please briefly describe the contents of the image.',
    'Describe the key elements in the picture.',
    'What do you see in the image in a few words?',
    'Summarize the image in a short description.',
    "Give a quick overview of the image's content.",
    'Describe the main subject of the photograph.',
    'In a few sentences, explain what the image portrays.',
    'What is the image trying to convey?',
    'Provide a snapshot description of the visual elements.',
    'What are the most striking features of the image?',
    'Briefly discuss the colors and textures in the image.',
    'Describe the lighting and shadows in the image.',
    'How would you describe the perspective of the image?',
    'In a few words, tell us about the background of the image.',
    'What kind of mood or feeling does the image evoke?',
    'Describe the composition of the image.',
    'What is the focal point of the image?',
    "Give a concise summary of the image's theme.",
    'How many objects or people are in the image?',
    'Describe the setting or location of the image.',
    'In a few sentences, explain the relationship between the elements in the image.',
    'Does the image have a clear foreground and background?',
    'What is the most prominent color in the image?',
    'Describe the texture of the main subject in the image.',
    'What is the light source in the image?',
    'Does the image have a sense of depth?',
    'Describe the overall tone or mood of the image.',
    'What is the first thing you notice in the image?',
    'Does the image have a clear center of interest?',
    'What emotions do the elements in the image evoke?',
    'Describe the positioning of the objects in the image.',
    'How would you categorize the image (e.g., landscape, portrait, still life)?',
    'Does the image have a sense of motion or stillness?',
    'What is the relationship between the colors in the image?',
    'Does the image have a horizontal or vertical composition?',
    'Describe the mood set by the lighting in the image.',
    'What is the mood of the subject in the image?',
    'Does the image have a balanced composition?',
    'Describe the shape and form of the main subject in the image.',
    'What is the most important detail in the image?',
    'Does the image have a clear visual hierarchy?',
    'What is the perspective or angle from which the image was taken?',
    'Describe the visual rhythm or pattern in the image.',
    'What is the emotional impact of the image?',
    'Does the image have a dominant color scheme?',
    'Describe the relative sizes of the objects in the image.',
    'What is the subject matter of the image?',
    'How would you describe the brightness of the image?',
    'What is the dominant texture in the image?',
    'Does the image have a clear subject and background?',
    'Describe the mood of the scene in the image.',
    'What is the most eye-catching element in the image?',
    'Does the image have a sense of harmony or contrast?',
    'Describe the relationship between the foreground and background in the image.',
    'What is the dominant color in the image?',
    'Does the image have a sense of scale or proportion?',
    'Describe the emotional tone of the image.',
    'What kind of image is it (e.g., a photo, a painting, a graphic)?',
    'Does the image have a strong visual focus?',
    'What is the dominant shape in the image?',
    'Describe the visual weight of the elements in the image.',
    'How would you describe the style of the image?',
    'What is the most visually arresting part of the image?',
    'Does the image have a sense of balance or asymmetry?',
    "Describe the colors that make up the image's palette.",
    "What is the mood created by the subject's expression in the image?",
    'Does the image have a clear point of view?',
    'Describe the texture of the background in the image.',
    'What is the dominant hue in the image?',
    'Does the image have a sense of spatial depth?',
    'Describe the lighting quality in the image.',
    'What is the most distinctive visual characteristic of the image?',
    'Does the image have a sense of drama or calmness?',
]


def process_file(i):
    try:
        global idx  # ensure we use a global index
        txt_path = os.path.join(path, f'%07d.jsonl' % i)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = json.loads(line)
            new_line = {
                'image': line['image'].replace('hzh:s3://public-dataset/laion-coco/images/', ''),
                'caption': line['caption'],
            }
            new_lines.append(new_line)

        new_lines = [new_lines[i:i + merge_num] for i in range(0, len(new_lines), merge_num)]
        output_path = os.path.join(out_path, f'%07d.jsonl' % i)

        with open(output_path, 'w') as writer:
            for item in new_lines:
                images = [temp['image'] for temp in item]
                captions = [temp['caption'] for temp in item]
                questions = ['<image>\n' + random.choice(prompts) for _ in range(len(images))]
                output = {
                    'id': idx,
                    'image': images,
                    'conversations': []
                }
                for question, answer in zip(questions, captions):
                    output['conversations'].append({'from': 'human', 'value': question})
                    output['conversations'].append({'from': 'gpt', 'value': answer})
                writer.write(json.dumps(output) + '\n')
                idx += 1
    except:
        pass


if __name__ == '__main__':
    idx = 0
    num_processes = cpu_count()
    pool = Pool(processes=num_processes)

    for _ in tqdm(pool.imap_unordered(process_file, range(56753)), total=56753):
        pass

    pool.close()
    pool.join()
