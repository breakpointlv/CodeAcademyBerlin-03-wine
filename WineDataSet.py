from statistics import mean

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import itertools

from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from ipywidgets import IntProgress
from IPython.display import display
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler

import pickle

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
    price_ds = None

    quality_model = None
    type_model = None

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

        sns.set_context('talk', font_scale=5)
        sns.set_palette('dark')

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

    ml_pipeline = []

    ml_pipeline_algs = [
        'Logistic Regression',
        'SVM',
        'KNN',
        'Decision Tree',
        'Random Forest',
        'Naive Bayes'
    ]
    wine_feature_combinations = []

    ml_evaluation = []

    def gen_pipeline(self):
        self.ml_pipeline = []
        self.ml_pipeline.append(LogisticRegression(solver='liblinear'))
        self.ml_pipeline.append(SVC())
        self.ml_pipeline.append(KNeighborsClassifier())
        self.ml_pipeline.append(DecisionTreeClassifier())
        self.ml_pipeline.append(RandomForestClassifier(
            n_estimators=1000,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=50,
            bootstrap=False
        ))
        self.ml_pipeline.append(GaussianNB())

    """
        Find all possible feature combinations for the wine
    """
    def gen_all_feature_combos(self):

        wine_features = [
            'residual sugar',
            'chlorides',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol'
        ]
        wine_feature_combinations = []

        for L in range(len(wine_features) + 1):
            for subset in itertools.combinations(wine_features, L):
                if len(subset) > 6:
                    wine_feature_combinations.append(subset)

        self.wine_feature_combinations = wine_feature_combinations


    def evaluate_ml(self, smote=False, show_progress=True, random_state=1, wine_type=None):
        if wine_type:
            y = self.ds[self.ds['type'] == wine_type]
            y = y['quality_label']
        else:
            y = self.ds['quality_label']

        # check_features = self.wine_feature_combinations
        check_features = [[
            'residual sugar',
            'chlorides',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol'
        ]]

        if show_progress:
            i = 0
            i_max = len(check_features) * len(self.ml_pipeline)
            f = IntProgress(min=0, max=i_max)  # instantiate the bar
            display(f)  # display the bar

        self.ml_evaluation = []

        for features in check_features:
            for model in self.ml_pipeline:
                x = self.ds.copy(deep=True)

                if wine_type:
                    x = x[x['type'] == wine_type]

                X = x[list(features)]

                if smote:
                    sm = SMOTE(random_state=random_state)
                    X_res, y_res = sm.fit_resample(X, y)
                else:
                    X_res = X
                    y_res = y

                X_train, X_test, y_train, y_test = train_test_split(
                    X_res, y_res,
                    train_size=0.8,
                    random_state=random_state,
                )

                # normalize data
                norm = MinMaxScaler().fit(X_train)
                # transform training data
                X_train = norm.transform(X_train)
                # transform testing data
                X_test = norm.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                cross_val = cross_val_score(model, X_res, y_res)
                self.ml_evaluation.append({
                    'features': features,
                    'algo': model,
                    'cohen kappa score': metrics.cohen_kappa_score(y_test, y_pred),
                    'cross val score': cross_val,
                    'cross val mean': mean(cross_val),
                    'classification report': metrics.classification_report(y_test, y_pred, output_dict=True),
                    'confusion matrix': confusion_matrix(y_test, y_pred)
                })
                if show_progress:
                    f.value += 1
                    i += 1

        return sorted(self.ml_evaluation, key=lambda x: x['cross val mean'])

    def evaluate_ml_for_type(self, smote=False, show_progress=True, random_state=1):

        y = self.ds['type']

        # check_features = self.wine_feature_combinations
        check_features = [[
            'residual sugar',
            'chlorides',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol'
        ]]

        if show_progress:
            i = 0
            i_max = len(check_features) * len(self.ml_pipeline)
            f = IntProgress(min=0, max=i_max)  # instantiate the bar
            display(f)  # display the bar

        self.ml_evaluation = []

        for features in check_features:
            for model in self.ml_pipeline:
                x = self.ds.copy(deep=True)
                X = x[list(features)]

                if smote:
                    sm = SMOTE(random_state=random_state)
                    X_res, y_res = sm.fit_resample(X, y)
                else:
                    X_res = X
                    y_res = y

                X_train, X_test, y_train, y_test = train_test_split(
                    X_res, y_res,
                    train_size=0.8,
                    random_state=random_state,
                )

                # normalize data
                norm = MinMaxScaler().fit(X_train)
                # transform training data
                X_train = norm.transform(X_train)
                # transform testing data
                X_test = norm.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                cross_val = cross_val_score(model, X_res, y_res)
                self.ml_evaluation.append({
                    'features': features,
                    'algo': model,
                    'cohen kappa score': metrics.cohen_kappa_score(y_test, y_pred),
                    'cross val score': cross_val,
                    'cross val mean': mean(cross_val),
                    'classification report': metrics.classification_report(y_test, y_pred, output_dict=True),
                    'confusion matrix': confusion_matrix(y_test, y_pred)
                })
                if show_progress:
                    f.value += 1
                    i += 1

        return sorted(self.ml_evaluation, key=lambda x: x['cross val mean'])

    def get_Xy(self, random_state=1, wine_type=None, smote=True, include_high=True):
        x = self.ds.copy(deep=True)

        if wine_type:
            x = x[x['type'] == wine_type]

        if not include_high:
            x = x[x['quality_label'] != 'high']

        y = x['quality_label']
        X = x[[
            'residual sugar',
            'chlorides',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol'
        ]]

        from sklearn.preprocessing import MinMaxScaler

        if smote:
            sm = SMOTE(random_state=random_state)
            X_res, y_res = sm.fit_resample(X, y)
            return X_res, y_res
        else:
            return X, y


    def get_Xy_type(self, random_state=1, smote=True):
        x = self.ds.copy(deep=True)

        y = x['type']
        X = x[[
            'residual sugar',
            'chlorides',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol'
        ]]

        from sklearn.preprocessing import MinMaxScaler

        if smote:
            sm = SMOTE(random_state=random_state)
            X_res, y_res = sm.fit_resample(X, y)
            return X_res, y_res
        else:
            return X, y

    def split_Xy(self, X, y, random_state=1, normalize=False):
        X_train, X_test, y_train, y_test =  train_test_split(X, y, train_size=0.8, random_state=random_state)

        if normalize:
            # fit scaler on training data
            norm = MinMaxScaler().fit(X_train)

            # transform training data
            X_train = norm.transform(X_train)

            # transform testing data
            X_test = norm.transform(X_test)

        return X_train, X_test, y_train, y_test

    # {'features': ('residual sugar',
    #               'chlorides',
    #               'total sulfur dioxide',
    #               'density',
    #               'pH',
    #               'sulphates',
    #               'alcohol'),
    #  'algo': RandomForestClassifier(),
    #  'accuracy score': 0.8693060876968923,
    #  'cohen kappa score': 0.8039990400303537,
    #  'classification report': '              precision    recall  f1-score   support\n\n        high       0.94      0.99      0.96       753\n         low       0.82      0.87      0.85       796\n      medium       0.85      0.76      0.80       800\n\n    accuracy                           0.87      2349\n   macro avg       0.87      0.87      0.87      2349\nweighted avg       0.87      0.87      0.87      2349\n'}


    def best_rfc_params(self):
        # RandomizedSearchCV
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        from sklearn.model_selection import RandomizedSearchCV

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]  # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=100, cv=3, verbose=2,
            random_state=42, n_jobs=-1
        )  # Fit the random search model

        X, y = self.get_Xy()
        X_train, X_test, y_train, y_test = self.split_Xy(X, y)

        rf_random.fit(X_train, y_train)
        return rf_random.best_params_

    def best_decision_tree_params(self):
        # RandomizedSearchCV
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        from sklearn.model_selection import RandomizedSearchCV

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]  # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=100, cv=3, verbose=2,
            random_state=42, n_jobs=-1
        )  # Fit the random search model

        X, y = self.get_Xy()
        X_train, X_test, y_train, y_test = self.split_Xy(X, y)

        rf_random.fit(X_train, y_train)

        return rf_random.best_params_

    # {'n_estimators': 1000,
    # 'min_samples_split': 2,
    # 'min_samples_leaf': 1,
    # 'max_features': 'auto',
    # 'max_depth': 50,
    # 'bootstrap': False}

    def learn_rfc(self, random_state=1, wine_type=None, smote=True, include_high=True, from_cache=True):
        X, y = self.get_Xy(random_state=random_state, wine_type=wine_type, smote=smote, include_high=include_high)
        X_train, X_test, y_train, y_test = self.split_Xy(X, y, random_state=random_state)
        if from_cache and self.quality_model:
            model = self.quality_model
        else:
            model = RandomForestClassifier(
                n_estimators=1000,
                min_samples_split=2,
                min_samples_leaf=1,
                max_depth=50,
                bootstrap=False
            )

            model.fit(X_train, y_train)
            self.quality_model = model
            pickle.dump(model, open('wine-quality.pkl', 'wb'))

        y_pred = model.predict(X_test)

        cr_val_score = cross_val_score(model, X, y)
        print('===================', wine_type, '===================')
        print('accuracy score', metrics.accuracy_score(y_test, y_pred))
        print('cohen kappa score', metrics.cohen_kappa_score(y_test, y_pred))
        print('cross val score', cr_val_score)
        print('cross val score mean', mean(cr_val_score))
        print('classification report', metrics.classification_report(y_test, y_pred))
        print("Confusion matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        a = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
        display(a)


    def learn_rfc_type(self, random_state=1, smote=True):
        X, y = self.get_Xy_type(random_state=random_state, smote=smote)
        X_train, X_test, y_train, y_test = self.split_Xy(X, y, random_state=random_state)
        model = RandomForestClassifier(
            n_estimators=1000,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=50,
            bootstrap=False
        )

        model.fit(X_train, y_train)

        pickle.dump(model, open('wine-type.pkl', 'wb'))

        y_pred = model.predict(X_test)

        cr_val_score = cross_val_score(model, X, y)
        print('accuracy score', metrics.accuracy_score(y_test, y_pred))
        print('cohen kappa score', metrics.cohen_kappa_score(y_test, y_pred))
        print('cross val score', cr_val_score)
        print('cross val score mean', mean(cr_val_score))
        print('classification report', metrics.classification_report(y_test, y_pred))
        print("Confusion matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        a = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
        display(a)

    def print_confusion_matrix(self, cm):
        ConfusionMatrixDisplay()
        sns.set_context('talk', font_scale=1.5)
        plt = sns.heatmap(cm, annot=True, fmt=".5g", cmap='Blues_r')
        plt.set_xlabel('Predicted values')
        plt.set_ylabel('Actual values')

    def add_price_data(self):
        price_ds = pd.read_csv('winequality/wine_sales_data.csv', sep=",")

        # leave only Portugal's Vinho Verde
        price_ds = price_ds[(price_ds['province'] == 'Vinho Verde') & (price_ds['country'] == 'Portugal')]

        # remove price outliers
        Q1 = price_ds["price"].quantile(0.25)
        Q3 = price_ds["price"].quantile(0.75)
        IQR = Q3 - Q1
        price_ds = price_ds.query('(@Q1 - 1.5 * @IQR) <= price <= (@Q3 + 1.5 * @IQR)')

        display(price_ds)

        # sort by price and leave only that
        price_ds = price_ds.sort_values('price', axis=0)
        price_ds = price_ds['price']


        wine_cnt = len(price_ds)
        low_cnt = round(wine_cnt * 0.37)
        med_cnt = round(wine_cnt * 0.60)
        hi_cnt = wine_cnt - low_cnt - med_cnt


        min_price = min(price_ds)
        max_price = max(price_ds)
        median_price = price_ds.median()
        mean_price = price_ds.mean()
        price_split = max_price - min_price
        price_step = price_split // 3

        med_q_pr = min_price + price_step
        hi_q_pr = max_price - price_step

        print('LOW QUALITY')
        low_q = price_ds.iloc[0:low_cnt]
        print(f"Low cnt: {low_cnt}")
        print(f"min price: {min(low_q)}")
        print(f"max price: {max(low_q)}")
        print(f"mean price: {low_q.mean()}")
        print(f"median price: {low_q.median()}")

        print('-------------------------')

        print('MEDIUM QUALITY')
        med_q = price_ds.iloc[low_cnt:low_cnt+med_cnt]
        print(f"cnt: {med_cnt}")
        print(f"min price: {min(med_q)}")
        print(f"max price: {max(med_q)}")
        print(f"mean price: {med_q.mean()}")
        print(f"median price: {med_q.median()}")

        print('-------------------------')

        print('HIGH QUALITY')
        hi_q = price_ds.iloc[low_cnt + med_cnt:]
        print(f"cnt: {hi_cnt}")
        print(f"min price: {min(hi_q)}")
        print(f"max price: {max(hi_q)}")
        print(f"mean price: {hi_q.mean()}")
        print(f"median price: {hi_q.median()}")


        print(f"Med cnt: {med_cnt}")
        print(f"High cnt: {hi_cnt}")
        print(f"Median price: {median_price}")
        print(f"Median price: {mean_price}")
        print(f"Low quality price: {min_price} - {med_q_pr-1}")
        print(f"Mid quality price: {med_q_pr} - {hi_q_pr}")
        print(f"High quality price: {hi_q_pr+1} - {max_price}")
        sns.set_context('talk', font_scale=1.5)
        sns.histplot(price_ds)
        self.sv()

    def leave_one_out_validate(self):
        from sklearn.model_selection import LeaveOneOut, cross_val_score
        wine_quality_model = pickle.load(open('wine-quality.pkl', 'rb'))
        X, y = self.get_Xy()
        scores = cross_val_score(wine_quality_model, X, y, cv=LeaveOneOut())
        return scores.mean()









