import os

from flask import Flask, render_template

from export import MyModel

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'secret string')
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True


def pre_station(date, station):
    pre = MyModel(date=date, text_day=120, station=station, all=True)
    return pre.pre_station()


def pre_route(date, route):
    pre = MyModel(date=date, text_day=90, route=route)
    return pre.pre_route()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/<select>/<time>/<name>')
def predict(select, time, name):
    print(time, name)
    if select == 'station':
        data = pre_station(time, name)
        return '{}'.format(int(data))
    elif select == 'route':
        data = pre_route(time, name)
        return '{}'.format(int(data))

