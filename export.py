import datetime
import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import feature_selection
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = 'a9/'  # csv文件所在目录

dic = {
    '晴': 0,
    '多云': 1,
    '阴': 2,
    '小雨': 3,
    '中雨': 4,
    '大雨': 5,
    '暴雨': 6,
    '雷阵雨': 7
}
warnings.filterwarnings('ignore')


class MyModel(object):

    def __init__(self, date, text_day=0, route=None, station=None, all=False):
        self.over = False
        self.data = pd.read_csv(path + 'trips.csv', encoding='gbk')[['进站名称', '进站时间', '出站名称']]
        _date = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
        if _date > datetime.datetime.strptime("2020-7-15", '%Y-%m-%d').strftime('%Y-%m-%d'):
            self.over = True
        if self.over:
            self.end = datetime.datetime.strptime('2020-7-15', '%Y-%m-%d').strftime('%Y-%m-%d')
            self.true_end = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
            self.begin = (datetime.datetime.strptime('2020-7-15', '%Y-%m-%d') - datetime.timedelta(
                days=text_day + 10)).strftime(
                '%Y-%m-%d')
        else:
            self.end = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
            self.true_end = self.end
            self.begin = (
                        datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=text_day + 10)).strftime(
                '%Y-%m-%d')
        self.data.set_index('进站时间', inplace=True, drop=True)
        self.data.index = pd.DatetimeIndex(self.data.index)
        if not all:
            self.data = self.data.loc[self.begin:self.end]
        else:
            self.data = self.data.loc[:self.end]
        self.data.reset_index(inplace=True)
        self.station = station
        self.route = route
        self.text_day = text_day
        self.fea = ['昨日客流量', 'month', '标注', '上周客流量', 'MA5', 'MA10', '最高气温', '最低气温', 'morning', 'afternoon']
        # self.fea = []

    def clean(self):
        station = pd.read_csv(path + 'test_station.csv', encoding='gbk')[['line_name', 'station_name']]
        station.drop(station.loc[~(station['line_name'] == self.route)].index.values, inplace=True)
        self.data.drop(self.data.loc[(self.data['出站名称'] == self.data['进站名称'])].index.values, inplace=True)
        self.data.drop(self.data.loc[~self.data['进站名称'].isin(station['station_name']) & ~self.data['出站名称'].isin(
            station['station_name'])].index.values, inplace=True)
        self.data.rename(columns={'进站时间': '日期'}, inplace=True)
        self.data = self.data.groupby(pd.to_datetime(self.data['日期']).dt.to_period('D')).size().reset_index(
            name='客流量').set_index('日期')
        self.data.fillna(0, inplace=True)
        self.data = self.data['客流量'].to_frame()

    def _clean(self):
        data1 = self.data.copy()
        self.data.drop(self.data.loc[self.data['进站名称'] != self.station].index.values, inplace=True)
        data1.drop(data1.loc[data1['出站名称'] != self.station].index.values, inplace=True)
        self.data = pd.merge(self.data, data1, how='outer')
        self.data.drop(self.data.loc[(self.data['出站名称'] == self.data['进站名称'])].index.values, inplace=True)
        self.data.rename(columns={'进站时间': '日期'}, inplace=True)
        self.data = self.data.groupby(pd.to_datetime(self.data['日期']).dt.to_period('D')).size().reset_index(
            name='客流量').set_index('日期')
        self.data.fillna(0, inplace=True)
        self.data = self.data['客流量'].to_frame()

    def add(self):
        # if self.over:
        weather = pd.read_csv(path + 'city.csv')
        weather['日期'] = weather['日期'].apply(lambda x: datetime.datetime.strptime(x, '%Y年%m月%d日').strftime('%Y-%m-%d'))
        weather['最高气温'] = weather['最高气温'].apply(lambda x: re.findall('\d+', x)[0]).astype(int)
        weather['最低气温'] = weather['最低气温'].apply(lambda x: re.findall('\d+', x)[0]).astype(int)
        weather['morning'] = weather['天气状况'].apply(lambda x: dic[re.findall('(.*) /(.*)', x)[0][0]])
        weather['afternoon'] = weather['天气状况'].apply(lambda x: dic[re.findall('(.*) /(.*)', x)[0][1]])
        weather.drop('天气状况', axis=1, inplace=True)
        weather.set_index('日期', drop=True, inplace=True)
        weather.index = pd.DatetimeIndex(weather.index)

        days = pd.read_csv(path + '2020.csv', encoding='gbk', parse_dates=['日期'], index_col='日期')
        days = pd.concat([weather, days], axis=1)

        days = days.to_period("D")
        self.data = pd.concat([self.data, days], axis=1)
        self.data.reset_index(inplace=True)
        self.data['上周客流量'] = self.data['客流量'].shift(7)
        self.data['昨日客流量'] = self.data['客流量'].shift(1)
        self.data['MA5'] = self.data['客流量'].rolling(5).mean()
        self.data['MA10'] = self.data['客流量'].rolling(10).mean()
        # print(self.data)
        self.data['month'] = self.data['日期'].map(lambda x: x.month)
        self.data.set_index('日期', drop=True, inplace=True)
        if self.over:
            # self.pre_data = self.data.loc[datetime.datetime.strptime('2020-7-6', '%Y-%m-%d').strftime('%Y-%m-%d'):self.true_end]
            self.pre_data = self.data.loc[:self.true_end]
            self.pre_data.reset_index(inplace=True)
        else:
            self.pre_data = None
        self.data.reset_index(inplace=True)
        self.data.dropna(inplace=True)
        # self.data.drop(self.data.loc[(self.data['标注'] == 3)].index.values, inplace=True)
        # exit()
        # print(self.data)

    def test_model(self):
        time.sleep(2)
        self.dat = self.data.tail(1)
        self.data.drop(self.data.tail(1).index.values, inplace=True)
        self.X = self.data[self.fea]

        self.y = self.data['客流量']
        self.X.index = range(self.X.shape[0])
        self.X_length = self.X.shape[0]
        split = int(self.X_length * 0.9)
        self.X_train, self.X_test = self.X[:split], self.X[split:]
        self.y_train, self.y_test = self.y[:split], self.y[split:]
        self.date = self.data['日期'][split:]
        Regressors = [["RandomForest", RandomForestRegressor()]
            , ["DecisionTree", DecisionTreeRegressor()]
            , ["Lasso", Lasso()]
            , ["AdaBoostRegressor", AdaBoostRegressor()]
            , ["GradientBoostingRegressor", GradientBoostingRegressor()]
            , ["XGB", XGBRegressor()]
                      ]
        reg_result = []
        names = []
        prediction = []
        for name, reg in Regressors:
            reg = reg.fit(self.X_train, self.y_train)
            y_pred = reg.predict(self.X_test)
            # 回归评估
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            class_eva = pd.DataFrame([mae, mse, r2])
            reg_result.append(class_eva)
            name = pd.Series(name)
            names.append(name)
            y_pred = pd.Series(y_pred)
            prediction.append(y_pred)
        names = pd.DataFrame(names)
        names = names[0].tolist()
        result = pd.concat(reg_result, axis=1)
        result.columns = names
        result.index = ["mae", "mse", "r2"]
        # print(result.head())
        a = result.loc['r2']
        if max(a.tolist()) < 0:
            print('error')
            return self.data['客流量'].head(30).mean()

        func = getattr(self, "_%s" % a.idxmax())
        return func()

    def feature(self):

        self.X = self.data[self.fea]
        self.y = self.data['客流量']
        # self.data.loc[(self.data['标注'] == 3)]
        fv, pv = feature_selection.f_regression(self.X, self.y)

        # data = [fv, pv]
        df = pd.DataFrame()
        df['feature'] = self.X.columns
        df['fv'] = fv
        df['pv'] = pv
        print(df)
        # self.fea = df.loc[df['fv'] > 0.5]['feature'].tolist()
        self.fea = df.sort_values(by='fv')['feature'].head(7).tolist()
        # print(df)
        print(self.fea)
        # print(df.loc['fv'].reset_index(inplace=True))
        # print(df.loc[df.loc['fv'] > 1])
        # feature = self.data.drop(['客流量'], axis=1)
        # corr = feature.corr()
        # plt.figure(figsize=(15, 6))
        # ax = sns.heatmap(corr, xticklabels=corr.columns,
        #                  yticklabels=corr.columns, linewidths=0.2, cmap="RdYlGn", annot=True)
        # plt.title("Correlation between variables")
        # plt.show()
        # df_onehot = pd.get_dummies(self.data)
        # plt.figure(figsize=(15, 6))
        # df_onehot.corr()['客流量'].sort_values(ascending=False).plot(kind='bar')
        # plt.title('Correlation between 客流量 and variables')
        # plt.show()

    def importance(self, features, reg, name):

        impor = pd.DataFrame([*zip(features, reg.feature_importances_)])
        impor.columns = ['feature', 'importance']
        impor.sort_values(by='importance', inplace=True)
        plt.barh(impor['feature'], height=0.5, width=impor['importance'])
        plt.title("%s选择特征重要性" % name)
        plt.show()

    def pre_route(self):
        # 清洗数据
        self.clean()
        # print('2运行时间：', int(time.time() - be), 'sec')
        # print(self.data)
        # print(self.data['日期'])
        # 增加特征
        self.add()
        # print('3运行时间：', int(time.time() - be), 'sec')
        # 特征工程
        # self.feature()
        # 选择最好模型
        return self.test_model()

    def pre_station(self):
        self._clean()
        # 增加特征
        self.add()
        # 特征工程
        # self.feature()
        # 选择最好模型
        return self.test_model()

    def prin(self, y_pred2, y_test=None):
        print(self.end, '-------------')
        print('预测值：', y_pred2[0])
        if y_test:
            print('真实值：', y_test.tolist()[0])

    def date_over(self, y_pred2, model):
        end = self.end
        while end < self.true_end:
            end = (datetime.datetime.strptime(end, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            self.pre_data.loc[self.pre_data['日期'] == end, '客流量'] = y_pred2
            self.update()
            X_test = self.pre_data.loc[self.pre_data['日期'] == end]
            X_test = X_test[self.fea]
            y_pred2 = model.predict(X_test)

        return y_pred2[0]

    def _Lasso(self):
        # rfc = Lasso()
        # rfc = rfc.fit(self.X_train, self.y_train)
        # y_pred = rfc.predict(self.X_test)

        reg = Lasso()
        reg = reg.fit(self.X, self.y)
        X_test = self.dat[self.fea]
        # y_test = self.dat['客流量']
        y_pred2 = reg.predict(X_test)
        # self.prin(y_pred2, y_test)
        # self.importance(X.columns, reg, 'Lasso')
        # self.draw(y_pred, self.y_test, 'Lasso')
        return self.date_over(y_pred2, reg)

    def _RandomForest(self):
        reg = RandomForestRegressor(n_estimators=30
                                    , random_state=123
                                    , bootstrap=True
                                    , oob_score=True
                                    )
        reg = reg.fit(self.X, self.y)
        X_test = self.dat[self.fea]
        # y_test = self.dat['客流量']
        y_pred2 = reg.predict(X_test)
        # self.prin(y_pred2, y_test)
        return self.date_over(y_pred2, reg)

    def _DecisionTree(self):
        model = DecisionTreeRegressor(max_depth=3, random_state=123)
        model.fit(self.X, self.y)
        X_test = self.dat[self.fea]
        y_pred2 = model.predict(X_test)
        # y_test = self.dat['客流量']
        # self.prin(y_pred2, y_test)
        return self.date_over(y_pred2, model)

    def _AdaBoostRegressor(self):
        model = AdaBoostRegressor(random_state=123)
        # model.fit(self.X_train, self.y_train)
        # y_pred = model.predict(self.X_test)
        model.fit(self.X, self.y)
        X_test = self.dat[self.fea]
        y_pred2 = model.predict(X_test)
        # print(y_test, y_pred2)
        # self.draw(y_pred, self.y_test, 'AdaBoostRegressor')
        return self.date_over(y_pred2, model)

    def _GradientBoostingRegressor(self):
        model = GradientBoostingRegressor(random_state=123)
        # model.fit(self.X_train, self.y_train)
        # y_pred = model.predict(self.X_test)

        model.fit(self.X, self.y)
        X_test = self.dat[self.fea]
        y_pred2 = model.predict(X_test)
        # y_test = self.dat['客流量']
        # self.prin(y_pred2, y_test)
        # self.draw(y_pred, self.y_test, 'GradientBoostingRegressor')
        return self.date_over(y_pred2, model)

    def _XGB(self):
        model = XGBRegressor()
        # model.fit(self.X_train, self.y_train)
        # y_pred = model.predict(self.X_test)
        model.fit(self.X, self.y)
        X_test = self.dat[self.fea]
        y_pred2 = model.predict(X_test)
        # y_test = self.dat['客流量']
        # self.prin(y_pred2, y_test)
        # self.draw(y_pred, self.y_test, 'XGB')
        return self.date_over(y_pred2, model)

    def draw(self, y_pred, y_test, name):
        date = []
        for i in self.date.tolist():
            # print(type(i.to_timestamp().strftime('%Y-%m-%d')), i)
            date.append(i.to_timestamp().strftime('%Y-%m-%d'))
        plt.figure(figsize=(15, 6))
        plt.title(name + '预测结果图')
        plt.plot(date, y_test.ravel(), label='真实值')
        plt.plot(date, y_pred, label='预测值')
        plt.xticks()
        plt.legend()
        plt.show()

    def update(self):
        self.pre_data.loc[:, '上周客流量'] = self.pre_data['客流量'].shift(7)
        self.pre_data.loc[:, '昨日客流量'] = self.pre_data['客流量'].shift(1)
        self.pre_data.loc[:, 'MA5'] = self.pre_data['客流量'].rolling(5).mean()
        self.pre_data.loc[:, 'MA10'] = self.pre_data['客流量'].rolling(10).mean()


if __name__ == '__main__':
    # 控制显示， 生产环境无关
    pd.set_option('display.max_columns', None)  # 显示完整的列
    pd.set_option('display.max_rows', None)  # 显示完整的行
    pd.set_option('display.expand_frame_repr', False)

    pd.set_option('display.max_columns', 1000)  # 对齐
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # path = '../a9/'  # csv文件所在目录
    """
        all = True ==> 所有数据
        text_day 测试数据
        date 预测日
        over 预测日期超过7月15日， 就要True
    """
    be = time.time()
    # pre = MyModel(date='2020-7-15', text_day=90, route='1号线')
    # a = pre.pre_route()
    # print(a)
    # print('1运行时间：', int(time.time()-be), 'sec')
    pre2 = MyModel(date='2020-7-16', text_day=120, station='Sta97', all=True)
    a = pre2.pre_station()
    print(a)
    print('2运行时间：', int(time.time() - be), 'sec')
