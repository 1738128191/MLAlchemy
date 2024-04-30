from my import source as my
import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy

# 读取数据
df = my.pd.read_csv(r'D:\pycharmProject\my_package\my\data\Ebusiness.csv')
viewer = my.ViewDataframe(df)
print(viewer.display_info())

# 去重
df.drop_duplicates(keep='first', inplace=True)
df.reset_index(inplace=True, drop=True)

# 标签编码
dic = {'正面': 0, '负面': 1}
df['label'] = df['label'].map(dic)

# 数据预处理
df['progress_text'] = df['evaluation'].apply(lambda x: my.custom_tokenizer(x, language='chinese'))
df['progress_text'] = df['progress_text'].apply(lambda x: ' '.join(x))
df['text_length'] = df['progress_text'].apply(lambda x: len(x.split(' ')))

# 可视化
# my.plt.figure(figsize=(14, 4))
# my.plt.subplot(1, 2, 1)
# my.sns.countplot(df, x='label', width=0.3, color='yellow')
# my.plt.subplot(1, 2, 2)
# my.sns.histplot(df, x='text_length', hue='label', palette='hot')
# pos = df[df['label'] == 0]
# neg = df[df['label'] == 1]
# pos_text = ' '.join(pos['progress_text'])
# neg_text = ' '.join(neg['progress_text'])
# my.display_wordcloud(pos_text, font_path='simhei.ttf')
# my.plt.title('正面评价')
# my.display_wordcloud(neg_text, font_path='simhei.ttf')
# my.plt.title('负面评价')

# 构建词表
# 先转换成期待的数据格式：大列表嵌套小列表，这种格式在nlp里面很常见
text = df['progress_text'].apply(lambda x: x.split())
data = text.tolist()
print(data[:2])

# 设置词频为2
Vocab, sorted_list = my.create_vocab(data, threshold=2, specials=['<unk>', '<pad>'])
Vocab.set_default_index(0)  # 设置默认索引为0，所有没查到的词都被标记为unk，unk的默认索引是0
print(len(Vocab))

# 拆分数据集
train_data = data[:3203]
val_data = data[3203:]
train_features = [Vocab.lookup_indices(line) for line in train_data]
val_features = [Vocab.lookup_indices(line) for line in val_data]
train_target = df.loc[:3202, 'label'].values  # array
val_target = df.loc[3203:, 'label'].values
print(len(train_data), len(val_data), len(train_features), len(val_features))

# 截断和填充
train_features_pad, val_features_pad = [], []
for line in train_features:
    train_features_pad.append(my.truncate_pad(line, num_steps=30, padding_token=1))

for line in val_features:
    val_features_pad.append(my.truncate_pad(line, num_steps=30, padding_token=1))

# 转换成tensor
train_features_tensor = torch.LongTensor(train_features_pad)
val_features_tensor = torch.LongTensor(val_features_pad)
train_target_tensor = torch.LongTensor(train_target)
val_target_tensor = torch.LongTensor(val_target)
print(train_features_pad[:2])
print(val_features_pad[:2])

# 创造迭代器
train_iter = my.load_array((train_features_tensor, train_target_tensor), batch_size=64)
val_iter = my.load_array((val_features_tensor, val_target_tensor), batch_size=64, is_train=False)
# 测试
for X, Y in train_iter:
    print('X:', X, '\nY:', Y)
    print('X的维度:', X.shape, 'Y的维度:', Y.shape)
    break

# 训练词向量
vector_model = my.MakeEmbedding(sentences=data, window=5, vector_size=50, sg=1, min_count=2, workers=12, negative=5,
                                sample=1e-3, hs=0, ns_exponent=0.75)
word2vec = vector_model.train_word2vec()
vector_model.test_model(text='电视', topn=5)
embeds = vector_model.get_embedding_matrix()
print(embeds.shape)
embedding_matrix = vector_model.add_embedding(num_specials=2)
print(embedding_matrix.shape)

# 构建模型
(rnn_type, vocab_size, embed_size, hidden_size, num_layers,
 output_size, dropout, bidirectional, batch_first) = ('LSTM', len(Vocab),
                                                      50, 100, 2, 2, 0.2, False, True)
net = my.GeneralRNN(rnn_type, vocab_size, embed_size, hidden_size, num_layers, output_size,
                    bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)
device = my.try_gpu()
net.to(device)
print(net)

# 修改权重
net.apply(my.init_weights)
net.embedding.weight.data.copy_(embedding_matrix)
net.embedding.weight.requires_grad = False


# 训练模型
def train(dataloader, model, lr, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


metric = BinaryAccuracy().to(device)


# 测试模型
def test(dataloader, model, device):
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predicted_classes = pred.argmax(dim=1)
            test_loss += loss_fn(pred, y).item()
            acc = metric(predicted_classes, y)
    test_loss /= num_batches
    acc = metric.compute()
    print(f"Test Error: \n Accuracy: {acc}%, Avg loss: {test_loss:>8f} \n")


# 运行
epoch = 100
for t in range(epoch):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_iter, net, lr=1e-4, device=device)
    test(val_iter, net, device=device)
print("Done!")
