import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

class WineDataSet:
    red_wine_file = 'winequality/winequality-red.csv'
    white_wine_file = 'winequality/winequality-white.csv'

    colors = {
        'red': 'red',
        'white': 'yellow',
        'all': '#4500a6'
    }
    col_arr = ['red', 'yellow']

    sw_colors = {
        'dry': '#34dbeb',
        'off-dry': '#34eb43',
        'medium-sweet': '#eb9234'
    }
    sw_col_arr = ['#34dbeb', '#34eb43', '#eb9234']

    ds_red = None
    ds_white = None
    ds = None

    b = 2

    def __init__(self):
        red = pd.read_csv('winequality/winequality-red.csv', sep=";")
        white = pd.read_csv('winequality/winequality-white.csv', sep=";")

        sns.color_palette('dark')
        sns.set_palette('dark')

        # sns.set(font_scale=2)

        self.ds_red = pd.read_csv('winequality/winequality-red.csv', sep=";")
        self.ds_white = pd.read_csv('winequality/winequality-white.csv', sep=";")

        # concat two data sets
        self.ds_red['type'] = 'red'
        self.ds_white['type'] = 'white'

        self.ds = pd.concat([self.ds_red, self.ds_white], ignore_index=True)

        self.ds = self.ds.astype({
            'fixed acidity': 'float',
            'volatile acidity': 'float',
            'citric acid': 'float',
            'residual sugar': 'float',
            'chlorides': 'float',
            'free sulfur dioxide': 'float',
            'total sulfur dioxide': 'float',
            'density': 'float',
            'pH': 'float',
            'sulphates': 'float',
            'alcohol': 'float',
            'quality': 'float'
        })

        self.ds['quality_label'] = self.ds['quality'].apply(
            lambda value: 'low'
            if value <= 5 else 'medium'
            if value <= 7 else 'high'
        )

        self.ds['sweetness'] = self.ds['residual sugar'].apply(
            lambda value: 'dry'
            if value <= 4 else 'off-dry'
            if value <= 11 else 'medium-sweet'
        )

        self.ds['alcohol_label'] = self.ds['alcohol'].apply(
            lambda value: math.floor(value)
        )

        self.ds['cals from alcohol'] = self.ds['alcohol'].apply(
            lambda value: value * 7 * 10
        )

        self.ds['cals from sugar'] = self.ds['residual sugar'].apply(
            lambda value: value * 4
        )

        self.ds['total calories'] = self.ds['cals from sugar'] + self.ds['cals from alcohol']

        self.ds['total acidity'] = self.ds['volatile acidity'] + self.ds['fixed acidity']

        self.ds['quality_label'] = pd.Categorical(self.ds['quality_label'], categories=['low', 'medium', 'high'])
        self.ds['type'] = pd.Categorical(self.ds['type'], categories=['red', 'white'])
        self.ds['sweetness'] = pd.Categorical(
            self.ds['sweetness'],
            categories=['dry', 'off-dry', 'medium-sweet']
        )

        # calculate additional fields for data analysis
        self.calc_new_fields()

    # =============== Calculations ==================


    def sv(self):
        import time

        sns.set(rc={'figure.figsize': (25, 15),
                    "ytick.color": "w",
                    "xtick.color": "w",
                    'legend.labelcolor': 'w',
                    'lines.color': 'w',
                    "text.color": "w",
                    'patch.edgecolor': 'w',
                    'patch.facecolor': 'w',
                    'lines.markerfacecolor': 'w',
                    'boxplot.whiskerprops.color': 'w',
                    'boxplot.medianprops.color': 'w',
                    'boxplot.meanprops.color': 'w',
                    'boxplot.flierprops.markeredgecolor': 'w',
                    'boxplot.flierprops.color': 'w',
                    'boxplot.capprops.color': 'w',
                    "axes.labelcolor": "w",
                    "axes.edgecolor": "w"})
        plt.savefig(
            '/home/user/Desktop/plots/' + str(time.time()) + '.png',
            transparent=True,
            facecolor='black'
        )
    def calc_new_fields(self):
        self.calc_illegal_so2()

    def calc_illegal_so2(self):
        self.ds['above_so2_limit'] = 0

        self.ds.loc[
            (self.ds['residual sugar'] <= 5) & (self.ds['total sulfur dioxide'] > 200),
            'above_so2_limit'
        ] = self.ds['total sulfur dioxide'] - 200

        self.ds.loc[
            (self.ds['residual sugar'] > 5) & (self.ds['total sulfur dioxide'] > 250),
            'above_so2_limit'
        ] = self.ds['total sulfur dioxide'] - 250

    # =============== Charts ==================

    def ds_slice(self, v='all', f='type'):
        return self.ds if v == 'all' else self.ds.loc[self.ds[f] == v]

    def make_autopct(self, values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

        return my_autopct

    def chart_illegal_so2(self, wine_type='all'):

        # are we looking at specific wine type?
        d = self.ds_slice(wine_type, 'type')

        illegal_wine_cnt = len(d[d['above_so2_limit'] > 0])
        legal_wine_cnt = len(d[d['above_so2_limit'] <= 0])
        dt = [illegal_wine_cnt, legal_wine_cnt]
        labels = ['Illegal SO2', 'Legal SO2']
        colors = ['red', self.colors[wine_type]]

        sns.set(rc={'figure.figsize': (25, 15)})
        sns.set(font_scale=2)
        plt.title('Legality of wines SO2 for EU')
        plt.pie(dt, labels=labels, colors=colors, autopct=self.make_autopct(dt))


    def chart_illegal_so2_quality(self):
        fig, axes = plt.subplots(1, 2)

        illegal = self.ds[self.ds["above_so2_limit"] > 0]
        legal = self.ds[self.ds["above_so2_limit"] <= 0]

        axes[0].set(ylim=(2, 10))
        axes[1].set(ylim=(2, 10))

        sns.boxplot(y=legal["quality"], data=legal, color=self.colors['all'], ax=axes[0]).set(title='legal SO2')
        sns.boxplot(y=illegal["quality"], data=illegal, color="red", ax=axes[1]).set(title='Illegal SO2')

    def chart_volatile_acidity(self):
        fig, axes = plt.subplots(1, 2)

        a = self.ds[self.ds["volatile acidity"] >= 0.8]
        b = self.ds[self.ds["volatile acidity"] < 0.8]

        axes[0].set(ylim=(0, 10))
        axes[1].set(ylim=(0, 10))

        sns.violinplot(y=b['quality'], data=b, color="green", ax=axes[0], hue='type').set(title='< 0.8 g/L')
        sns.violinplot(y=a['quality'], data=a, color=self.colors['all'], ax=axes[1], hue='type').set(title='>= 0.8 g/L')

    def chart_sweetness_to_quality(self):
        #fig, axes = plt.subplots(2, 3)

        sns.boxplot(y=self.ds['quality'], x=self.ds['sweetness'], data=self.ds, color="green", hue='sweetness').set(title='quality and sweetness')


    def chart_pair(self, f):
        sns.set(rc={'figure.figsize': (25, 15)})
        sns.set_context('paper', font_scale=1.5)
        #sns.jointplot(x='alcohol', y='quality', data=self.ds, kind='hex', palette='matte')
        sns.pairplot(self.ds, hue=f, corner=True)

    def chart_sweetness_citric_acid(self):
        sns.barplot(x='sweetness', y='citric acid', data=self.ds, estimator=np.mean).set(title='Citric acid and wine sweetness')

    """
        Returns the copy of the data set with only parameters relevant for ML
    """
    def get_ml_copy(self):
        ds_copy = self.ds.copy(deep=True)
        return ds_copy[[
            'residual sugar',
            'chlorides',
            'total sulfur dioxide',
            'pH',
            'sulphates',
            'alcohol'
        ]]






