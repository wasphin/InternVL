import subprocess
import time


def get_cpu_usage(node_list):
    # 假设的命令来获取CPU使用情况，实际中应替换为正确的命令或API调用
    command = f'swatch -n {node_list} cpu_usage'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if stderr:
        print('Error getting CPU usage:', stderr.decode())
        return None

    cpu_usage = stdout.decode().split('\n')
    cpu_usage = [line.strip() for line in cpu_usage if 'SH-IDC1-10-140' in line or
                 'cpu_used_list' in line or 'cpu_used_count' in line and 'process list' not in line and
                 not line.strip().startswith('SH-IDC1-10')]
    cpu_usage = [line for line in cpu_usage if 'process list' not in line][1:]
    dict = {}
    for line in cpu_usage:
        if '------' in line:
            current_key = line
            dict[current_key] = []
        else:
            dict[current_key].append(line)

    return dict


def monitor_cpu_usage():
    while True:
        # print current time
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print('Checking CPU usage of SH-IDC1-10-140-37-[14,62,82,103,111,115,118,126,128-131,135-136,139,152]...')
        dict = get_cpu_usage('SH-IDC1-10-140-37-[14,62,82,103,111,115,118,126,128-131,135-136,139,152]')
        for k, v in dict.items():
            filter_v = [i for i in v if 'wangwenhai' not in i]
            if len(filter_v) > 0:
                print(k)
                for i in v:
                    print(i)

        time.sleep(10)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print('Checking CPU usage of SH-IDC1-10-140-37-[23,31,33,43,46,52-55,57-58,61,63,68,77,79]...')
        dict = get_cpu_usage('SH-IDC1-10-140-37-[23,31,33,43,46,52-55,57-58,61,63,68,77,79]')
        for k, v in dict.items():
            filter_v = [i for i in v if 'wangwenhai' not in i]
            if len(filter_v) > 0:
                print(k)
                for i in v:
                    print(i)
        time.sleep(10)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print('Checking CPU usage of SH-IDC1-10-140-37-[66-67,69,71-76,78,80,99,104,106,110,154]...')
        dict = get_cpu_usage('SH-IDC1-10-140-37-[66-67,69,71-76,78,80,99,104,106,110,154]')
        for k, v in dict.items():
            filter_v = [i for i in v if 'wangwenhai' not in i]
            if len(filter_v) > 0:
                print(k)
                for i in v:
                    print(i)
        time.sleep(10)


if __name__ == '__main__':
    monitor_cpu_usage()
