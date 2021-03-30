import os

import requests

data = ['a9/2020.csv', 'a9/city.csv', 'a9/test_station.csv', 'a9/trips.csv', 'init_data/clean_data.csv',
        'init_data/test_station.csv', 'init_data/users.csv', 'init_data/workdays2020.csv']

# def get_all_name():
#     data_dirs = ['a9', 'init_data']
#     result = []
#     for dir_item in data_dirs:
#         for root, dirs, files in os.walk(dir_item):
#             for file in files:
#                 result.append('{}/{}'.format(dir_item, file))
#     print(result)


if not os.path.exists('a9'):
    os.mkdir('a9')

if not os.path.exists('init_data'):
    os.mkdir('init_data')

print(os.path.exists(data[0]))

for item in data:
    if not os.path.exists(item):
        print('开始下载 {}'.format(item))
        with open(item, mode='wb') as f:
            f.write(requests.get('http://114.55.125.234:1111/{}'.format(item)).content)
        print('下载成功 {}'.format(item))

print('Success')
# get_all_name()
