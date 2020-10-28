import torch
from torch import nn
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD
from data_helper import get_all_apths, get_corpus, data_generator, get_train_test_dataloader
from data_helper import window, word_size, nb_negative

nb_epoch = 10  # 迭代次数


class Word2VecCBOW(Module):
    def __init__(self, window, id2word, nb_negative, embedding_dim):
        """
            CBOW模型
        :param window:窗口大小
        :param id2word:
        :param nb_negative:负采样数量
        :param embedding_dim:词向量维度
        """
        super(Word2VecCBOW, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(id2word), embedding_dim=embedding_dim)
        self.window = window
        self.id2word = id2word
        self.nb_negative = nb_negative
        self.embedding_dim = embedding_dim

    def forward(self, input_words, negative_samples):
        """

        :param input_words: 上下文单词
        :param negative_samples:中心词和负采样单词
        :return:
        """
        input_vecs = self.embedding(input_words)  # shape=(,window*2,word_size)
        input_vecs_sum = torch.sum(input_vecs, dim=1)  # CBOW模型直接对上下文单词的嵌入进行求和操作 shape=(,word_size)

        negative_sample_vecs = self.embedding(negative_samples)  # shape=(,nb_negative + 1,word_size)

        out = torch.matmul(negative_sample_vecs, torch.unsqueeze(input_vecs_sum, dim=2))
        out = torch.squeeze(out)
        out = torch.softmax(out, dim=-1)
        return out


def train(model, train_dataloader, device, optimizer, crossEntropyLoss):
    model.train()
    train_loss = 0.0
    for i, data in enumerate(train_dataloader):
        x_train, y_train, z_train = data
        x_train, y_train, z_train = x_train.to(torch.long).to(device), y_train.to(torch.long).to(device), z_train.to(torch.long).to(device)
        optimizer.zero_grad()  # 梯度清零
        z_predict = model(x_train, y_train)  # (batch_size,51)
        loss = crossEntropyLoss(z_predict, z_train)
        loss.backward()  # 梯度反向传播
        optimizer.step()  # 梯度更新
        train_loss += loss.item()
        # if i % 10 == 0:
        #     print(loss.item())
    return train_loss / i


def test(model, test_dataloader, device, crossEntropyLoss):
    model.eval()
    test_loss = 0.0
    for i, data in enumerate(test_dataloader):
        x_test, y_test, z_test = data
        x_test, y_test, z_test = x_test.to(torch.long).to(device), y_test.to(torch.long).to(device), z_test.to(torch.long).to(device)
        z_predict = model(x_test, y_test)  # (batch_size,51)
        loss = crossEntropyLoss(z_predict, z_test)
        test_loss += loss.item()

    return test_loss / i


def train_test(epochs, batch_size):
    file_dir = "F:\\data\\machine_learning\\THUCNews\\THUCNews"
    paths = get_all_apths(file_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    words, corpus, id2word, word2id = get_corpus(paths)

    x, y, z = data_generator(corpus, word2id, id2word)
    print(x.shape, y.shape, z.shape)
    train_dataloader, test_dataloader = get_train_test_dataloader(x, y, z, batch_size=batch_size)
    loss_fun = CrossEntropyLoss()
    cbow = Word2VecCBOW(window, id2word, nb_negative, word_size)
    cbow.to(device)
    optimizer = SGD(cbow.parameters(), lr=0.01)

    print("------开始训练------:", device)
    for epoch in range(1, epochs + 1):
        train_loss = train(cbow, train_dataloader, device, optimizer, loss_fun)
        test_loss = test(cbow, test_dataloader, device, loss_fun)
        print("epoch %d, train loss: %.2f, test loss:%.2f" % (epoch, train_loss, test_loss))

    torch.save(cbow, "../models/cbow_w2v.pkl")


if __name__ == '__main__':
    # train_test(nb_epoch, 32) #训练、测试
    cbow = torch.load("../models/cbow_w2v.pkl")  # 加载模型
    print(cbow.embedding.weight.shape)  # 提取训练好的Embedding
