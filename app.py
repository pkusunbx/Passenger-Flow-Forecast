import filecmp
import json
import os
import sys
import time
from threading import Thread

from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from export import MyModel

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'secret string')
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://admin:root@114.55.125.234:3306/passenger-flow"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

DB_CACHE = 'db_cache'


def pre_station(date, station):
    pre = MyModel(date=date, text_day=120, station=station, all=True)
    return pre.pre_station()


def pre_route(date, route):
    pre = MyModel(date=date, text_day=90, route=route)
    return pre.pre_route()


def removeSame(name):
    """
    首先移除所有包含name的空文件，然后在剩余文件中选择时间最近的文件
    :param name: 部分文件名
    :return: 无返回值
    """
    path = os.path.join(sys.path[0], DB_CACHE)
    files_list = []
    last_time = '0000_00_00_00_00_00'
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.find(name) != -1:
                if os.path.getsize(os.path.join(root, file)) == 0:
                    os.remove(os.path.join(root, file))
                    continue
                files_list.append(file)
                file_time = file[:19]
                if file_time > last_time:
                    last_time = file_time
    flag = True
    for i in range(len(files_list)):
        for j in range(len(files_list) - 1):
            j = j + 1
            if not filecmp.cmp(os.path.join(sys.path[0], DB_CACHE, files_list[i]),
                               os.path.join(sys.path[0], DB_CACHE, files_list[j])):
                flag = False
                break
        file = files_list[i]
        if file.find(last_time) == -1 and os.path.exists(os.path.join(path, file)):
            os.remove(os.path.join(path, file))
    return flag


def readDB():
    db = SQLAlchemy(app)
    db.reflect()
    all_table = {table_obj.name: table_obj for table_obj in db.get_tables_for_bind()}
    db_tables = list(all_table.keys())
    flag = True
    for name in db_tables:
        now_time = time.strftime("%Y_%m_%d_%H_%M_%S")
        data = [dict(zip(result.keys(), result)) for result in db.session.query(all_table[name]).all()]
        data = json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        print(name)
        with open(os.path.join(sys.path[0], DB_CACHE, '{}_{}_data.json'.format(now_time, name)), mode='w') as f:
            f.write(data)
            flag = removeSame(name)
            if not flag:
                return flag
    return flag


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/<select>/<pre_time>/<name>')
def predict(select, pre_time, name):
    print(pre_time, name)
    if select == 'station':
        data = pre_station(pre_time, name)
        return '{}'.format(int(data))
    elif select == 'route':
        data = pre_route(pre_time, name)
        return '{}'.format(int(data))


class readDBThread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self) -> None:
        while True:
            print(time.ctime(), '开始读取数据库')
            readDB()
            print(time.ctime(), '读取成功')
            time.sleep(10)


readDBThread().start()
