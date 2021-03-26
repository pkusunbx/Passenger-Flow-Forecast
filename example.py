from export import MyModel
import time


def pre_station(date, station):
    pre = MyModel(date=date, text_day=120, station=station, all=True)
    return pre.pre_station()


def pre_route(date, route):
    pre = MyModel(date=date, text_day=90, route=route)
    return pre.pre_route()


if __name__ == '__main__':
    be = time.time()
    print(pre_station('2020-8-15', 'Sta126'))
    # print('1运行时间：', int(time.time() - be), 'sec')

    print(pre_route('2020-8-15', '2号线'))
    # print('2运行时间：', int(time.time() - be), 'sec')
