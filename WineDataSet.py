import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class WineDataSet:
    red_wine_file = 'winequality/winequality-red.csv'
    white_wine_file = 'winequality/winequality-white.csv'

    red_wine_color = 'red'
    white_wine_color = 'yellow'

    ds_red = None
    ds_white = None
    ds_all = None

    b = 2

    def __init__(self):
        red = pd.read_csv('winequality/winequality-red.csv', sep=";")
        white = pd.read_csv('winequality/winequality-white.csv', sep=";")

        sns.color_palette('dark')
        sns.set(rc={'figure.figsize': (25, 15)})
        # sns.set(font_scale=2)

        self.ds_red = pd.read_csv('winequality/winequality-red.csv', sep=";")
        self.ds_white = pd.read_csv('winequality/winequality-white.csv', sep=";")

        # concat two data sets
        self.ds_red['type'] = 'red'
        self.ds_white['type'] = 'white'

        self.ds_all = pd.concat([red, white], ignore_index=True)

        # calculate additional fields for data analysis
        self.calc_new_fields()

    def calc_new_fields(self):
        self.calc_illegal_so2()

    def calc_illegal_so2(self):
        self.ds_all['above_so2_limit'] = 0

        self.ds_all.loc[
            (self.ds_all['residual sugar'] <= 5) & (self.ds_all['total sulfur dioxide'] > 200),
            'above_so2_limit'
        ] = self.ds_all['total sulfur dioxide'] - 200

        self.ds_all.loc[
            (self.ds_all['residual sugar'] > 5) & (self.ds_all['total sulfur dioxide'] > 250),
            'above_so2_limit'
        ] = self.ds_all['total sulfur dioxide'] - 250

    def chart_illegal_so2(self):
        illegal_wines = self.ds_all[self.ds_all['above_so2_limit'] > 0]

        sns.set(rc={'figure.figsize': (25, 15)})
        sns.displot(self.ds_all, x='total sulfur dioxide', binwidth=20, hue='above_so2_limit')





