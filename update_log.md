# 更新日志

## 1.0.0 → 1.0.1
更新内容:

### 1.DataVisualization绘图函数更新。
增加类参数font_size，用于调节轴标题和刻度的字体大小，该参数在画布被放大时有用(matplotlib在画布增大时，会自动缩小字体以适应图片分布)。

histogram_plot不再是单图布局，而是变更为2列n行，并且kde变成可调节参数。

1.0.0版本中的box_plot修改为boxen_plot并新添加box_plot。

boxen_plot是一种增强版的箱线图，用于展示数据分布的更多信息，特别是对于含有大量离群点或者数据分布较为复杂的情况。其采用了基于数据点数量的对数规则，
首先根据样本量大小将数据均匀的划分为k层，k=log2(n)-3，然后将层转换为对应的百分数，并采用exponential策略，箱子的宽度与它们代表的百分位范围的
指数相关，意味着越远离中心的箱子（包含更高或更低的百分位数的数据点）宽度会按照指数级减小。这种方式有助于直观地表达数据的分布趋势。

box_plot就是传统的箱线图。

box_plot和boxen_plot均删去了根据classified参数绘制分类属性的功能，建议在需要时自行绘制。

join_plot限定为两种。在多点的情况下为hex模式，否则为reg模式，不再单独保留hue等选项。
### 2.display_wordcloud
新添加固定参数collocations=False，关闭了WordCloud统计搭配词的功能，防止词云图重复。
### 3.OptunaForLGB
optimize_lgb方法种增加参数probably_params，可以采用更多的指标进行调参