
import os

from tqdm import tqdm

total_ranks = 512
meta =  {
    'root': 'hzh:s3://public-dataset/laion-coco/images/',
    'annotation': 'metas/stage2_v5/merged_laion_coco_59m.jsonl',
    'data_augment': False,
    'repeat_time': 1,
    'length': 59173874
}

basename = os.path.basename(meta['annotation']).replace('.jsonl', '')
data_dir = os.path.join(os.path.dirname(meta['annotation']), f'{basename}_temp')

if not os.path.exists(data_dir):
    try:
        os.makedirs(data_dir)
    except:
        pass
print('Loading data...')
with open(meta['annotation'], 'r') as f:
    raw_data = f.readlines()
print('Splitting data...')
for current_rank in tqdm(range(512)):
    temp_path = os.path.join(data_dir, f'{basename}_{current_rank}_of_{total_ranks}.jsonl')
    total_lines = len(raw_data)
    print(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
    lines_per_rank = total_lines // total_ranks  # 每个rank分得的行数
    start_line = lines_per_rank * current_rank  # 当前rank开始的行数
    end_line = start_line + lines_per_rank  # 当前rank结束的行数
    temp_data = raw_data[start_line:end_line]  # 读取当前rank对应的行
    writer = open(temp_path, 'w')
    writer.writelines(temp_data)
    writer.close()
