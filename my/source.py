import pandas as pd
import numpy as np
import importlib
import re
import sys
import optuna
from optuna.integration import LightGBMPruningCallback
import time
import lightgbm as lgb
from lightgbm import early_stopping
from collections import Counter, OrderedDict
from scipy.stats import kurtosis, skew, normaltest
from gensim.models import Word2Vec
import string
import jieba
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from inspect import signature
from itertools import chain
from typing import Optional, List
import seaborn as sns
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             r2_score, mean_squared_error,
                             root_mean_squared_error, roc_auc_score, log_loss, mean_absolute_error)
from wordcloud import WordCloud
from PIL import Image
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.options.display.max_columns = 100
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

my = sys.modules[__name__]

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torchtext.vocab import vocab


class ViewDataframe:
    # ViewDataframe用于查看读取的数据框信息
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_dimensions(self):
        # 获取数据的维度
        rows, cols = self.df.shape
        return f"维度: {rows} 行 x {cols} 列"

    def get_column_types(self):
        # 获取数据类型
        return self.df.dtypes.to_frame().rename(columns={0: '数据类型'})

    def count_missing_values(self):
        # 统计缺失值信息
        missing_count = self.df.isnull().sum()
        missing_ratio = missing_count / len(self.df)
        missing_stats = pd.concat([missing_count, missing_ratio.map('{:.2%}'.format)], axis=1)
        missing_stats.columns = ['缺失值统计', '缺失值占比 (%)']
        return missing_stats

    def count_duplicates(self):
        # 统计重复值信息
        dup_counts = self.df.duplicated(keep=False).sum()
        dup_ratio = dup_counts / len(self.df)
        dup_stats = pd.Series({'重复行统计': dup_counts,
                           '重复行占比': '{:.2%}'.format(dup_ratio)})
        return dup_stats

    def preview_data(self):
        # 拼接前五行和后五行数据
        head_data = self.df.head(5)
        tail_data = self.df.tail(5)

        preview_df = pd.concat([head_data, tail_data], ignore_index=False)
        return preview_df

    def display_info(self):
        # 展示前面所有的结果
        print("\n--- 数据维度 ---")
        print(self.get_dimensions())
        print("\n--- 特征类型 ---")
        print(self.get_column_types())
        print("\n--- 缺失值信息 ---")
        print(self.count_missing_values())
        print("\n--- 重复值信息 ---")
        print(self.count_duplicates())
        print("\n--- 数据展示(前五行+后五行) ---")
        print(self.preview_data())


def describe_plus(df: pd.DataFrame):
    # 在原始的pandas的描述性统计结果上加上了峰度、偏度和正态性检验
    df_info = df.describe().T
    kurtosis_values = df.apply(kurtosis)
    skewness_values = df.apply(skew)
    extra_stats = pd.DataFrame({'Skewness': skewness_values, 'Kurtosis': kurtosis_values})
    result = pd.concat([df_info, extra_stats], axis=1)
    # 正态性检验返回一个元组，第一个是统计量，第二个是p值，只取了p值
    normal_test_results = df.apply(lambda col: normaltest(col.dropna())[1])
    result['Normality Test (p-value)'] = normal_test_results
    return result


def try_gpu(i=0):
    # 用于深度学习适用GPU
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def create_vocab(data: list, threshold=10, specials: list = None):
    # 这里的data期待的是一个嵌套的列表[[...],[...]...]
    # 返回一个字典，统计每个词出现的次数
    # Counter接受一个单列表，因此要先把大列表展品
    data_flattened = list(chain.from_iterable(data))
    counter = Counter(data_flattened)
    # counter.items()返回一个dict_items对象，里面是一个大列表
    # 大列表里面是元组，元组的第一个是词，第二个是出现的次数，这里按照次数降序排序
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # OrderedDict也是一个字典，不过可以按照顺序插入元素
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    # 创建一个vocab对象,词频低于阈值的都会被标记成unk,再想查找会报错RuntimeError
    vocabulary = vocab(ordered_dict, min_freq=threshold, specials=specials)
    return vocabulary, sorted_by_freq_tuples


def load_array(data_arrays, batch_size, is_train=True):
    # 创捷pytorch的迭代器
    # data_arrays是一个元组(特征变量的tensor, 响应变量的tensor)
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


class TokenEmbedding:
    # 这个类用于嵌入Glove词向量
    def __init__(self, filepath, special_tokens: list = None):
        self.ID_to_token, self.ID_to_vec = self.load_embedding(filepath, special_tokens)
        self.unknown_ID = 0
        self.token_to_ID = {token: index for index, token in enumerate(self.ID_to_token)}

    def load_embedding(self, filepath, special_tokens: list = None):
        id_to_token, id_to_vec = [], []
        if special_tokens is not None:
            for i in special_tokens:
                id_to_token.append(i)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # rstrip用于去除文本右端的所有空白符，包括但不限于空格、换行符\n、制表符\t等
                elements = line.rstrip().split(' ')
                token, elements = elements[0], [float(element) for element in elements[1:]]
                if len(elements) > 1:
                    id_to_token.append(token)
                    id_to_vec.append(elements)
        # 这一行的作用是在读取预训练的词向量文件后，
        # 为词汇表添加特殊的未知词向量。
        # 这是为了在实际应用中会遇到不在预训练词汇表中的词。
        # 因此通常会引入一个特殊符号 <unk>（未知词标记）
        # 并为其分配一个随机初始化的词向量，或者在这里的情况下，全部赋值为0的向量。
        # [[0] * len(idx_to_vec[0])] 创建了一个全0的向量，其长度与词汇表中第一个词向量相同。
        if special_tokens is not None:
            id_to_vec = [[0] * len(id_to_vec[0])]*len(special_tokens) + id_to_vec
        return id_to_token, torch.tensor(id_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_ID.get(token, self.unknown_ID) for token in tokens]
        vectors = self.ID_to_vec[torch.tensor(indices)]
        return vectors

    def __len__(self):
        return len(self.ID_to_token)


def use_svg_display():
    # 设置清晰的显示样式
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(6, 4)):
    # 设置图形大小
    use_svg_display()
    my.plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    # 设置轴属性
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def custom_tokenizer(text: str, language='english', include_punctuation=False, remove_stopwords=True, character=False):
    """
    自定义分词器，根据参数控制中英文、提取标点和停用词。

    参数:
    - text (str): 输入文本。
    - language (str): 指定语言，可以是 'english' 或 'chinese'。
    - include_punctuation (bool): 是否保留标点符号。
    - remove_stopwords (bool): 是否移除停用词。
    - character (bool):是否进行字符级拆分
    返回:
    - tokens (list of str): 分词后的列表。
    """

    def is_chinese_char(char):
        """判断字符是否为中文字符，如果是返回True，中文里面包含中文标点一样适用"""
        return '\u4e00' <= char <= '\u9fff'

    def remove_punctuation(text):
        """去除文本中的标点符号"""
        # 定义英文和中文的标点符号集合
        punctuation = string.punctuation+r'[，。？！；：‘“”（）【】《》（）\w]+'
        text = re.sub('[%s]' % re.escape(punctuation), "", text)
        return text

    # 去除标点符号
    if include_punctuation:
        text_without_punctuation = remove_punctuation(text)
    else:
        text_without_punctuation = text
    # 区分中英文字符并分别存储
    english_chars = []
    chinese_chars = []
    for char in text_without_punctuation:
        if is_chinese_char(char):
            chinese_chars.append(char)
        else:
            english_chars.append(char)

    # 分别处理中英文文本
    if language == 'english':
        # 英文分词
        tokens = word_tokenize(''.join(english_chars))
        # 移除停用词
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word.lower() not in stop_words]
        if character:
            tokens = [token for line in tokens for token in line]
            return tokens
        else:
            return tokens
    elif language == 'chinese':
        # 中文分词
        tokens = jieba.lcut(''.join(chinese_chars), cut_all=False)
        # 移除停用词
        if remove_stopwords:
            # 这里需要加载中文停用词集
            stop_words = set(stopwords.words('cn_stopwords.txt'))
            tokens = [word for word in tokens if word not in stop_words]
        if character:
            tokens = [token for line in tokens for token in line]
            return tokens
        else:
            return tokens
    else:
        raise ValueError("Unsupported language. Choose 'english' or 'chinese'.")


class DataVisualization:
    # 用于批量绘制图像
    def __init__(self, df, num_var=None, cat_var=None, font_size=12):
        self.df = df
        self.num_var = num_var
        self.cat_var = cat_var
        self.font_size = font_size
        if num_var is not None:
            self.nrows = int(len(self.num_var) / 2) if len(self.num_var) % 2 == 0 else int(len(self.num_var) / 2) + 1

    def histogram_plot(self, fig_type='count', hue=None, bins=50, palette="Set3", color='gold',
                       figsize=(18, 6), jupyter=True, kde=True):
        def plot():
            sns.histplot(self.df, x=self.num_var[i], stat=fig_type, hue=hue, kde=kde, bins=bins,
                              palette=palette if hue is not None else None, color=color)
            plt.xlabel(self.num_var[i], fontsize=self.font_size)
            plt.ylabel('Count', fontsize=self.font_size)
            plt.xticks(fontsize=self.font_size)
            plt.yticks(fontsize=self.font_size)
            plt.tight_layout()
        plt.figure(figsize=figsize)
        for i in range(len(self.num_var)):
            plt.subplot(self.nrows, 2, i + 1)
            if jupyter:
                plot()
            else:
                plot()
                plt.show()

    def join_plot(self, dependent_variable: str, color='yellow', many_points=False):
        if many_points:
            for i in self.num_var:
                sns.jointplot(self.df, x=i, y=dependent_variable, kind='hex', color=color)
                plt.show()
        else:
            for i in self.num_var:
                sns.jointplot(self.df, x=i, y=dependent_variable, kind='reg', color=color)
                plt.show()

    def boxen_plot(self, color='orangered', hue=None, figsize=(6, 4), palette="Set3", jupyter=True, width=0.5):
        def plot():
            sns.boxenplot(self.df, y=self.num_var[i], color=color, hue=hue,
                          palette=palette if hue is not None else None, width=width)
            plt.ylabel(self.num_var[i], fontsize=self.font_size)
            plt.yticks(fontsize=self.font_size)
            plt.tight_layout()
        plt.figure(figsize=figsize)
        for i in range(len(self.num_var)):
            plt.subplot(self.nrows, 2, i + 1)
            if jupyter:
                plot()
            else:
                plot()
                plt.show()

    def box_plot(self, color='orangered', hue=None, figsize=(6, 4), palette="Set3", jupyter=True, width=0.5):
        def plot():
            sns.boxplot(self.df, y=self.num_var[i], color=color, hue=hue,
                        palette=palette if hue is not None else None, width=width)
            plt.ylabel(self.num_var[i], fontsize=self.font_size)
            plt.yticks(fontsize=self.font_size)
            plt.tight_layout()
        plt.figure(figsize=figsize)
        for i in range(len(self.num_var)):
            plt.subplot(self.nrows, 2, i + 1)
            if jupyter:
                plot()
            else:
                plot()
                plt.show()

    def count_plot(self, hue=None, palette="Set3", figsize=(8, 6), max_categories_for_rotation=10, rotation_angle=45,
                   color='g', width=0.5):
        for i in self.cat_var:
            fig, ax = plt.subplots(figsize=figsize)
            # 绘制条形图
            sns.countplot(data=self.df, x=i, hue=hue, palette=palette if hue is not None else None,
                          color=color, width=width)
            # 添加柱子上方的数字标签
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2, height, f'{height:.0f}', ha='center', va='bottom')
            # 获取类别数
            n_categories = len(ax.get_xticklabels())
            # 如果类别数大于指定阈值，旋转 x 轴标签
            if n_categories > max_categories_for_rotation:
                plt.xticks(rotation=rotation_angle)
            plt.show()

    def heatmaps(self, var_list, cmap='Blues', annot=True, vmin=-1,
                 vmax=1, square=True, figsize=(10, 10), fmt='.2f', half=False):
        if half:
            data = self.df[var_list].corr()
            mask = np.triu(np.ones_like(data, dtype=bool))
            plt.figure(figsize=figsize)
            sns.heatmap(data, mask=mask, cmap=cmap, annot=annot, vmin=vmin, vmax=vmax, square=square, fmt=fmt)
            plt.show()
        else:
            plt.figure(figsize=figsize)
            sns.heatmap(self.df[var_list].corr(), cmap=cmap, annot=annot, vmin=vmin, vmax=vmax, square=square, fmt=fmt)
            plt.tight_layout()
            plt.show()


def cut_data(X, y, test_size=0.3, stratified=False, shuffle=True):
    if stratified:
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, stratify=y, test_size=test_size,
                                                                  shuffle=shuffle, random_state=42)
        return X_trainval, X_test, y_trainval, y_test
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size,
                                                                  shuffle=shuffle, random_state=42)
        return X_trainval, X_test, y_trainval, y_test


def kf_choice(n_splits=10, stratified=False, shuffle=True):
    if stratified:
        skf = StratifiedKFold(n_splits=n_splits, random_state=123 if shuffle else None, shuffle=shuffle)
        return skf
    else:
        kf = KFold(n_splits=n_splits, random_state=123 if shuffle else None, shuffle=shuffle)
        return kf


class Timer:
    # 用来计时
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]


def train_regression_baseline(regression_model, X_trainval, y_trainval, n_splits=10,
                              stratified=False, shuffle=True, is_print=False,
                              eval_metrics: Optional[List[callable]] = None):
    kf = my.kf_choice(n_splits=n_splits, stratified=stratified, shuffle=shuffle)
    # 初始化存储自定义评价指标结果的字典
    custom_scores_train = {metric.__name__: [] for metric in eval_metrics} if eval_metrics is not None else {}
    custom_scores_val = {metric.__name__: [] for metric in eval_metrics} if eval_metrics is not None else {}
    train_r2, train_mse, val_r2, val_mse = [], [], [], []
    timer = my.Timer()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval, y_trainval)):
        X_train, X_val = X_trainval.iloc[train_idx, :], X_trainval.iloc[val_idx, :]
        y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

        regression_model.fit(X_train, y_train)
        train_pred = regression_model.predict(X_train)
        val_pred = regression_model.predict(X_val)

        # 计算默认评价指标
        train_r2.append(r2_score(y_train, train_pred))
        val_r2.append(r2_score(y_val, val_pred))
        train_mse.append(mean_squared_error(y_train, train_pred))
        val_mse.append(mean_squared_error(y_val, val_pred))

        # 计算用户指定的自定义评价指标（如果提供）
        if eval_metrics is not None:
            for metric in eval_metrics:
                # 计算自定义指标
                custom_train_score = metric(y_train, train_pred)
                custom_val_score = metric(y_val, val_pred)
                custom_scores_train[metric.__name__].append(custom_train_score)
                custom_scores_val[metric.__name__].append(custom_val_score)

    train_time = timer.stop()
    avg_train_r2_score, avg_train_mse = sum(train_r2)/n_splits, sum(train_mse)/n_splits
    avg_val_r2_score, avg_val_mse = sum(val_r2)/n_splits, sum(val_mse)/n_splits
    result = pd.DataFrame({'R2_score': [avg_train_r2_score, avg_val_r2_score],
                           'MSE': [avg_train_mse, avg_val_mse]}, index=['train', 'val'])

    # 计算自定义评价指标的平均值（如果提供）
    if eval_metrics is not None:
        for metric in eval_metrics:
            avg_train_metric_score = sum(custom_scores_train[metric.__name__]) / n_splits
            avg_val_metric_score = sum(custom_scores_val[metric.__name__]) / n_splits
            result.loc['train', metric.__name__] = avg_train_metric_score
            result.loc['val', metric.__name__] = avg_val_metric_score

    if is_print:
        print(f'运行时间{train_time}秒')
        print('模型结果:\n', result)
    else:
        result_dict = {'avg_train_r2_score': avg_train_r2_score, 'avg_train_mse': avg_train_mse,
                       'avg_val_r2_score': avg_val_r2_score,
                       'avg_val_mse': avg_val_mse}

        # 添加自定义评价指标到结果字典（如果提供）
        if eval_metrics is not None:
            for metric in eval_metrics:
                result_dict[f'avg_train_{metric.__name__}'] = result.loc['train', metric.__name__]
                result_dict[f'avg_val_{metric.__name__}'] = result.loc['val', metric.__name__]

        return result_dict, result


def train_classification_baseline(classification_model, X_trainval,
                                  y_trainval, n_splits=10,
                                  multi_class=False, average_f1='micro', average_auc='macro', multiclass='ovr',
                                  stratified=False, shuffle=True, is_print=False,
                                  eval_metrics: Optional[List[callable]] = None, probably_params: dict = None):
    kf = my.kf_choice(n_splits=n_splits, stratified=stratified, shuffle=shuffle)
    train_f1, train_auc = [], []
    val_f1, val_auc = [], []
    # 初始化存储自定义评价指标结果的字典
    custom_scores_train = {metric.__name__: [] for metric in eval_metrics} if eval_metrics is not None else {}
    custom_scores_val = {metric.__name__: [] for metric in eval_metrics} if eval_metrics is not None else {}
    timer = my.Timer()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval, y_trainval)):
        X_train, X_val = X_trainval.iloc[train_idx, :], X_trainval.iloc[val_idx, :]
        y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

        classification_model.fit(X_train, y_train)
        train_pred = classification_model.predict(X_train)
        val_pred = classification_model.predict(X_val)

        if multi_class:
            train_f1.append(f1_score(y_train, train_pred, average=average_f1))
            val_f1.append(f1_score(y_val, val_pred, average=average_f1))
            train_auc.append(roc_auc_score(y_train, classification_model.predict_proba(X_train),
                                           multi_class=multiclass, average=average_auc))
            val_auc.append(roc_auc_score(y_val, classification_model.predict_proba(X_val),
                                         multi_class=multiclass, average=average_auc))
            # 计算用户指定的自定义评价指标（如果提供）
            if eval_metrics is not None:
                default_kwargs = probably_params
                for metric in eval_metrics:
                    # 过滤出default_kwargs中与当前metric函数参数匹配的部分
                    metric_params = signature(metric).parameters.keys()
                    relevant_kwargs = {} if probably_params is None else \
                        {k: v for k, v in default_kwargs.items() if k in metric_params}
                    # 计算自定义指标
                    custom_train_score = metric(y_train, train_pred, **relevant_kwargs)
                    custom_val_score = metric(y_val, val_pred, **relevant_kwargs)
                    custom_scores_train[metric.__name__].append(custom_train_score)
                    custom_scores_val[metric.__name__].append(custom_val_score)

        else:
            train_f1.append(f1_score(y_train, train_pred))
            val_f1.append(f1_score(y_val, val_pred))
            train_auc.append(roc_auc_score(y_train, classification_model.predict_proba(X_train)[:, 1]))
            val_auc.append(roc_auc_score(y_val, classification_model.predict_proba(X_val)[:, 1]))

            # 计算用户指定的自定义评价指标（如果提供）
            if eval_metrics is not None:
                for metric in eval_metrics:
                    # 计算自定义指标
                    custom_train_score = metric(y_train, train_pred)
                    custom_val_score = metric(y_val, val_pred)
                    custom_scores_train[metric.__name__].append(custom_train_score)
                    custom_scores_val[metric.__name__].append(custom_val_score)

    def average(data: list):
        return sum(data) / n_splits
    train_time = timer.stop()
    avg_train_f1 = average(train_f1)
    avg_train_auc = average(train_auc)
    avg_val_f1 = average(val_f1)
    avg_val_auc = average(val_auc)
    result = pd.DataFrame({'F1_score': [avg_train_f1, avg_val_f1],
                           'AUC': [avg_train_auc, avg_val_auc]},
                          index=['train', 'val'])

    # 计算自定义评价指标的平均值（如果提供）
    if eval_metrics is not None:
        for metric in eval_metrics:
            avg_train_metric_score = sum(custom_scores_train[metric.__name__]) / n_splits
            avg_val_metric_score = sum(custom_scores_val[metric.__name__]) / n_splits
            result.loc['train', metric.__name__] = avg_train_metric_score
            result.loc['val', metric.__name__] = avg_val_metric_score

    if is_print:
        print(f'运行时间{train_time}秒')
        print('模型结果:\n', result)
    else:
        result_dict = {'avg_train_f1': avg_train_f1,
                       'avg_train_auc': avg_train_auc,
                       'avg_val_f1': avg_val_f1, 'avg_val_auc': avg_val_auc}

        # 添加自定义评价指标到结果字典（如果提供）
        if eval_metrics is not None:
            for metric in eval_metrics:
                result_dict[f'avg_train_{metric.__name__}'] = result.loc['train', metric.__name__]
                result_dict[f'avg_val_{metric.__name__}'] = result.loc['val', metric.__name__]

        return result_dict, result


class OptunaForLGB:
    def __init__(self, x_trainval, y_trainval, task_type, stopping_rounds, eval_metric: Optional[callable] = None):
        # LightGBM定义参数时的metric是用于训练时最优化的，使用该指标来度量模型的表现，并据此进行梯度提升树的构建。
        # eval_metric用于交叉验证时验证集评分使用，最终返回cv的平均值作为optimize的输入
        self.x_trainval, self.y_trainval= x_trainval, y_trainval
        self.task_type = task_type
        self.eval_metric = eval_metric
        self.stopping_rounds = stopping_rounds

    def objective(self, trial, custom_params: dict = None, n_splits=10,
                  stratified=True, shuffle=True, multiclass='ovr', average_f1='micro',
                  average_auc='macro', probably_params: dict = None):
        kf = my.kf_choice(n_splits=n_splits, stratified=stratified, shuffle=shuffle)
        cv_score = []
        score = 0
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.x_trainval, self.y_trainval)):
            X_train, X_val = self.x_trainval.iloc[train_idx, :], self.x_trainval.iloc[val_idx, :]
            y_train, y_val = self.y_trainval.iloc[train_idx], self.y_trainval.iloc[val_idx]
            # 基础参数设置
            base_params = {
                'boosting_type': 'gbdt',
                'n_estimators': ['int', 100, 500],
                'num_leaves': ['int', 32, 2**10-1],
                'max_bin': 255,
                'max_depth': ['int', 3, 10],
                'learning_rate': ['float', 0.01, 1],
                'min_child_samples': ['int', 5, 64],
                'subsample': ['float', 0.7, 1.0],
                'colsample_bytree': ['float', 0.75, 1.0],
                'reg_alpha': ['float', 1e-2, 10],
                'reg_lambda': ['float', 1e-2, 10],
                'random_state': 42,
                'verbosity': -1,
                'device': 'gpu',
                'silent': True
            }
            # 添加任务特定参数
            if self.task_type == 'binary':
                base_params.update({
                    'objective': 'binary',
                    'metric': 'auc',
                })
            elif self.task_type == 'multiclass':
                base_params.update({
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                })
            elif self.task_type == 'regression':
                base_params.update({
                    'objective': 'regression',
                    'metric': 'rmse',
                })
            else:
                raise ValueError(
                    f"Unsupported task: {self.task_type}. Choose from 'binary', 'multiclass', or 'regression'.")
            # 可以自行修改参数
            if custom_params is not None:
                base_params.update(custom_params)
            final_params = {}
            for key, value in base_params.items():
                if isinstance(value, list):
                    if value[0] == 'int':
                        step = value[3] if len(value) > 3 else 1
                        final_params[key] = trial.suggest_int(
                            name=key,
                            low=value[1],
                            high=value[2],
                            step=step
                        )
                    elif value[0] == 'float':
                        step = value[3] if len(value) > 3 else None
                        final_params[key] = trial.suggest_float(
                            name=key,
                            low=value[1],
                            high=value[2],
                            step=step
                        )
                    elif value[0] == 'categorical':
                        choices = value[1:]
                        final_params[key] = trial.suggest_categorical(
                            name=key,
                            choices=choices
                        )
                    else:
                        raise ValueError('Unsupported task.'
                                         'It is recommended to use the three data formats of '
                                         'the official Optuna documentation.i.e., int, float, and categorical')
                elif isinstance(value, (str, int, float)):
                    final_params[key] = value
            # 使用早停
            callbacks = [early_stopping(stopping_rounds=self.stopping_rounds),
                         LightGBMPruningCallback(trial, final_params['metric'])]
            # 训练模型
            model = lgb.LGBMRegressor(**final_params) if self.task_type == 'regression' \
                else lgb.LGBMClassifier(**final_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      eval_metric=final_params['metric'], callbacks=callbacks)
            # 评价模型
            if isinstance(model, lgb.LGBMRegressor):
                val_pred = model.predict(X_val)
                score = self.eval_metric(y_val, val_pred) if self.eval_metric else mean_squared_error(y_val, val_pred)
            else:
                val_pred = model.predict(X_val)
                val_pred_proba = model.predict_proba(X_val)
                if self.eval_metric is not None:
                    if self.eval_metric.__name__ == 'roc_auc_score':
                        if self.task_type == 'binary':
                            score = self.eval_metric(y_val, val_pred_proba[:, 1])
                        else:
                            score = self.eval_metric(y_val, val_pred_proba, multi_class=multiclass, average=average_auc)
                    elif self.eval_metric.__name__ == 'f1_score':
                        if self.task_type == 'binary':
                            score = self.eval_metric(y_val, val_pred)
                        else:
                            score = self.eval_metric(y_val, val_pred, average=average_f1)
                    else:
                        default_kwargs = probably_params
                        # 过滤出default_kwargs中与当前eval_metric函数参数匹配的部分
                        metric_params = signature(self.eval_metric).parameters.keys()
                        relevant_kwargs = {} if probably_params is None else \
                            {k: v for k, v in default_kwargs.items() if k in metric_params}
                        if self.task_type == 'binary':
                            score = self.eval_metric(y_val, val_pred, **relevant_kwargs)\
                                if self.eval_metric else accuracy_score(y_val, val_pred)
                        elif self.task_type == 'multiclass':
                            score = self.eval_metric(y_val, val_pred, **relevant_kwargs)\
                                if self.eval_metric else log_loss(y_val, val_pred)
            cv_score.append(score)
        avg_cv_score = sum(cv_score) / n_splits
        return avg_cv_score

    def optimize_lgb(self, n_trials=100, custom_params=None, n_splits=10,
                     stratified=True, shuffle=True, multiclass='ovr', average_f1='micro', average_auc='macro',
                     timeout=600, probably_params: dict = None):
        time = Timer()
        study = optuna.study.create_study(direction='maximize' if self.task_type == 'binary' else 'minimize')
        study.optimize(lambda trial: self.objective(trial, custom_params=custom_params,
                                                    n_splits=n_splits, stratified=stratified,
                                                    shuffle=shuffle, multiclass=multiclass,
                                                    average_f1=average_f1, average_auc=average_auc,
                                                    probably_params=probably_params),
                       n_trials=n_trials, n_jobs=-1, timeout=timeout)
        end_time = time.stop()
        print(f'运行时间: {end_time}秒')
        return study.best_params, study.best_value


class OptunaForGeneralML:
    def __init__(self, x_trainval, y_trainval, task_type: str, ml_method: str, scoring: str, params: dict):
        # ml_method接受一个机器学习模型的名字，比如sklearn.ensemble.RandomForestRegressor
        # scoring期待一个字符串作为评价指标的函数
        # params接收一个字典，使用户要求的参数范围，比如
        # params = {'n_estimators': ['int', 100, 500], 'max_depth': ['int', 1, 6],
        #           'min_samples_split': ['int', 2, 10], 'max_features': int(df.shape[1] / 3),
        #           'random_state': 123, 'n_jobs': -1, 'ccp_alpha': ['float', 0, 10.0]}
        self.x_trainval = x_trainval
        self.y_trainval = y_trainval
        self.ml_method = ml_method
        self.params = params
        self.task_type = task_type
        self.scoring = scoring

    def create_sklearn_model(self):
        father_class, model_name = self.ml_method.rsplit('.', 1)
        try:
            module = importlib.import_module(father_class)
            model_class = getattr(module, model_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Unable to find and import model '{model_name}'. "
                f"Please ensure it is a valid scikit-learn model.") from e
        return model_class

    def objective(self, trial, n_splits=10, stratified=True, shuffle=True,
                  multiclass='ovr', average_f1='micro', average_auc='macro', multi_class=False,
                  eval_metrics: Optional[List[callable]] = None, probably_params: dict = None):
        final_params = {}
        for key, value in self.params.items():
            if isinstance(value, list):
                if value[0] == 'int':
                    step = value[3] if len(value) > 3 else 1
                    final_params[key] = trial.suggest_int(
                        name=key,
                        low=value[1],
                        high=value[2],
                        step=step
                    )
                elif value[0] == 'float':
                    step = value[3] if len(value) > 3 else None
                    final_params[key] = trial.suggest_float(
                        name=key,
                        low=value[1],
                        high=value[2],
                        step=step
                    )
                elif value[0] == 'categorical':
                    choices = value[1:]
                    final_params[key] = trial.suggest_categorical(
                        name=key,
                        choices=choices
                    )
                else:
                    raise ValueError('Unsupported task.'
                                     'It is recommended to use the three data formats of '
                                     'the official Optuna documentation.i.e., int, float, and categorical')
            elif isinstance(value, (str, int, float)):
                final_params[key] = value
        sklearn_model = self.create_sklearn_model()
        model = sklearn_model(**final_params)
        if self.task_type == 'regression':
            result_dict, result = my.train_regression_baseline(model, self.x_trainval, self.y_trainval,
                                                               n_splits=n_splits, stratified=False,
                                                               shuffle=shuffle,
                                                               is_print=False, eval_metrics=eval_metrics)
            for eval_metric in eval_metrics:
                assert self.scoring in ['r2_score', 'mse'] + [eval_metric.__name__]
                metric_name = 'avg_val_' + self.scoring
                return result_dict[metric_name]
        elif self.task_type == 'classification':
            result_dict, result = my.train_classification_baseline(model, self.x_trainval, self.y_trainval,
                                                                   multi_class=multi_class,n_splits=n_splits,
                                                                   stratified=stratified,
                                                                   shuffle=shuffle, is_print=False,
                                                                   multiclass=multiclass, average_f1=average_f1,
                                                                   average_auc=average_auc,
                                                                   eval_metrics=eval_metrics,
                                                                   probably_params=probably_params)
            for eval_metric in eval_metrics:
                assert self.scoring in ['f1', 'auc'] + [eval_metric.__name__]
                metric_name = 'avg_val_' + self.scoring
                return result_dict[metric_name]
        else:
            raise ValueError(
                f"Unsupported task: {self.task_type}. Choose from 'classification' or 'regression'.")

    def optimize(self, n_trials=30, n_splits=10, stratified=True,
                 shuffle=True, multi_class=False, multiclass='ovr', average_f1='micro', average_auc='macro',
                 direction='maximize', eval_metrics: Optional[List[callable]] = None,
                 timeout=600, probably_params: dict = None):
        time = Timer()
        study = optuna.study.create_study(direction=direction)
        study.optimize(lambda trial: self.objective(trial, multi_class=multi_class,
                                                    n_splits=n_splits, stratified=stratified,
                                                    shuffle=shuffle, multiclass=multiclass,
                                                    average_f1=average_f1, average_auc=average_auc,
                                                    eval_metrics=eval_metrics,
                                                    probably_params=probably_params), n_trials=n_trials,
                       n_jobs=-1, timeout=timeout)
        self.best_params = study.best_params
        end_time = time.stop()
        print(f'运行时间: {end_time}秒')
        return study.best_params, study.best_value


class BuildBestModel:
    def __init__(self, x_trainval, y_trainval, x_test, y_test, model_params: dict,
                 task_type: str, ml_method, eval_metrics: Optional[List[callable]] = None,
                 probably_params: dict = None):
        # scoring同LightGBM一样，期待的是sklearn里面metrics的评价指标，my包已经导入了较常用的回归和分类指标
        self.x_test = x_test
        self.y_test = y_test
        self.x_trainval = x_trainval
        self.y_trainval = y_trainval
        self.model_params = model_params
        self.task_type = task_type
        self.eval_metrics = eval_metrics
        self.ml_method = ml_method
        self.probably_params = probably_params
        self.baseline_model = None

    def create_sklearn_model(self):
        father_class, model_name = self.ml_method.rsplit('.', 1)
        try:
            module = importlib.import_module(father_class)
            model_class = getattr(module, model_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Unable to find and import model '{model_name}'. "
                f"Please ensure it is a valid scikit-learn model.") from e
        return model_class

    def train(self, n_splits=10, shuffle=True,
              multiclass='ovr', average_f1='micro', average_auc='macro', multi_class=False):
        sklearn_model = self.create_sklearn_model()
        self.baseline_model = sklearn_model(**self.model_params)
        if self.task_type == 'regression':
            my.train_regression_baseline(self.baseline_model, self.x_trainval, self.y_trainval,
                                         n_splits=n_splits, stratified=False,
                                         shuffle=shuffle, is_print=True, eval_metrics=self.eval_metrics)
        elif self.task_type == 'classification':
            my.train_classification_baseline(self.baseline_model, self.x_trainval, self.y_trainval,
                                             multi_class=multi_class, n_splits=n_splits,
                                             stratified=True,
                                             shuffle=shuffle, is_print=True,
                                             multiclass=multiclass, average_f1=average_f1, average_auc=average_auc,
                                             eval_metrics=self.eval_metrics,
                                             probably_params=self.probably_params)

    def evaluate(self, multi_class=False, multiclass='ovr', average_f1='micro', average_auc='macro'):
        self.pred_test = self.baseline_model.predict(self.x_test)
        self.pred_test_proba = self.baseline_model.predict_proba(self.x_test) if (
                self.task_type == 'classification') else None
        result = {}
        if multi_class:
            auc = roc_auc_score(self.y_test, self.pred_test_proba, multi_class=multiclass, average=average_auc)
            f1 = f1_score(self.y_test, self.pred_test, average=average_f1)
        else:
            auc = roc_auc_score(self.y_test, self.pred_test_proba[:, 1])
            f1 = f1_score(self.y_test, self.pred_test)
        result['auc'] = auc
        result['f1_score'] = f1
        default_kwargs = self.probably_params
        for metric in self.eval_metrics:
            metric_params = signature(metric).parameters.keys()
            relevant_kwargs = {} if self.probably_params is None else \
                {k: v for k, v in default_kwargs.items() if k in metric_params}
            effect = metric(self.y_test, self.pred_test, **relevant_kwargs)
            result[metric.__name__] = effect
        return result

    @property
    def get_test_result(self):
        if self.pred_test_proba is not None:
            return self.pred_test, self.pred_test_proba
        else:
            return self.pred_test


class PlotForML:
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

    def learning_curve(self, model, X_trainval, y_trainval, scoring: str, n_splits=10,
                       stratified=True, shuffle=True, title='Learning Curve'):
        kf = my.kf_choice(n_splits=n_splits, stratified=stratified, shuffle=shuffle)
        skplt.estimators.plot_learning_curve(model, X_trainval, y_trainval,
                                             cv=kf, shuffle=shuffle, scoring=scoring, title=title)
        plt.show()

    def plot_confusion_matrix(self, normalize=False, cmap='Blues'):
        skplt.metrics.plot_confusion_matrix(self.y_true, self.y_pred, normalize=normalize, cmap=cmap)
        plt.show()

    def plot_roc(self, plot_micro=False, plot_macro=False, cmap='Blues'):
        skplt.metrics.plot_roc(self.y_true, self.y_pred_proba,
                               plot_macro=plot_macro, plot_micro=plot_micro, cmap=cmap)
        plt.show()

    def plot_precision_recall(self, cmap='Blues', plot_micro=False):
        skplt.metrics.plot_precision_recall(self.y_true, self.y_pred_proba, plot_micro=plot_micro, cmap=cmap)
        plt.show()


class MakeEmbedding:
    def __init__(self, sentences, window, vector_size, sg, min_count,
                 workers, negative, sample, hs, ns_exponent):
        self.sentences = sentences
        self.window = window
        self.vector_size = vector_size
        self.sg = sg
        self.min_count = min_count
        self.workers = workers
        self.negative = negative
        self.sample = sample
        self.hs = hs
        self.ns_exponent = ns_exponent
        self.model = None

    def train_word2vec(self):
        self.model = Word2Vec(sentences=self.sentences, window=self.window, min_count=self.min_count,
                              workers=self.workers, negative=self.negative, sample=self.sample, hs=self.hs,
                              ns_exponent=self.ns_exponent, vector_size=self.vector_size)
        return self.model

    def test_model(self, text: str, topn):
        sims = self.model.wv.most_similar(text, topn=topn)
        print(self.model.wv[text])
        print(sims)

    def get_embedding_matrix(self):
        # 不含填充符的词向量
        all_words = list(self.model.wv.key_to_index)
        self.embedding_dim = self.model.vector_size
        self.embeds = torch.empty((len(all_words), self.embedding_dim), dtype=torch.float32)
        for i, word in enumerate(all_words):
            self.embeds[i] = torch.tensor(self.model.wv[word], requires_grad=False)
        return self.embeds

    def add_embedding(self, num_specials=None):
        vector_specials = torch.empty(num_specials, self.embedding_dim)
        vector_specials = torch.nn.init.uniform_(vector_specials)
        embedding_metrix = torch.cat((self.embeds, vector_specials), 0)
        return embedding_metrix


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def display_wordcloud(data, background_color="white", filepath=None, figsize=(12, 12),
                      font_path=None, width=400, height=200,
                      stopwords=None, colormap='viridis', max_words=200):
    if filepath:
        mask = np.array(Image.open(filepath))
    else:
        mask = None
    wordcloud = WordCloud(background_color=background_color, mask=mask,
                          font_path=font_path, width=width, height=height,
                          stopwords=stopwords, colormap=colormap, max_words=max_words, collocations=False).generate(data)
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud)
    plt.axis("off")













































