import matplotlib.pyplot as plt
import warnings
from bs4 import BeautifulSoup
import re
import xml.sax.saxutils as saxutils
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from pandas import DataFrame
from random import random
import numpy as np
np.random.seed(1)

warnings.filterwarnings("ignore")

data_folder = "/mnt/f/data/machine_learning/分类数据/多标签文本分类/reuters21578.tar/reuters21578/"
sgml_number_of_files = 22
sgml_file_name_template = "reut2-{}.sgm"

category_files = {
    'to_': ('Topics', 'all-topics-strings.lc.txt'),
    'pl_': ('Places', 'all-places-strings.lc.txt'),
    'pe_': ('People', 'all-people-strings.lc.txt'),
    'or_': ('Organizations', 'all-orgs-strings.lc.txt'),
    'ex_': ('Exchanges', 'all-exchanges-strings.lc.txt')
}

category_data = []
for category_prefix in category_files.keys():
    with open(data_folder + category_files[category_prefix][1], "r") as file:
        for category in file.readlines():
            category_data.append(
                [category_prefix+category.strip().lower(), category_files[category_prefix][0], 0])

news_categories = DataFrame(data=category_data, columns=[
                            "Name", "Type", "NewsLines"])
# print(news_categories.tail())


def update_frequencies(categories):
    for category in categories:
        idx = news_categories[news_categories.Name == category].index[0]
        f = news_categories.loc[idx, "NewsLines"]
        news_categories.at[idx, "NewsLines"] = f + 1


def to_category_vector(categories, target_categories):
    vector = np.zeros(len(target_categories)).astype(np.float32)
    for i in range(len(target_categories)):
        if target_categories[i] in categories:
            vector[i] = 1.0
    return vector


# Those are the top 20 categories we will use for the classification
selected_categories = ['pl_usa', 'to_earn', 'to_acq', 'pl_uk', 'pl_japan', 'pl_canada', 'to_money-fx',
                       'to_crude', 'to_grain', 'pl_west-germany', 'to_trade', 'to_interest',
                       'pl_france', 'or_ec', 'pl_brazil', 'to_wheat', 'to_ship', 'pl_australia',
                       'to_corn', 'pl_china']

# 解析 SGML文件
document_X = []
document_Y = []


def strip_tags(text):
    return re.sub("<[^<]+?>", "", text).strip()


def unescape(text):
    return saxutils.unescape(text)


# 迭代所有的文件

for i in range(sgml_number_of_files):
    file_name = sgml_file_name_template.format(str(i).zfill(3))
    print("reading file: %s" % file_name)
    with open(data_folder + file_name, "rb") as file:
        content = BeautifulSoup(file.read().lower(), "lxml")
        for newsline in content("reuters"):
            document_categories = []
            document_id = newsline["newid"]
            document_body = strip_tags(
                str(newsline("text")[0].text)).replace("reuter\n&#3;", "")
            document_body = unescape(document_body)

            topics = newsline.topics.contents
            places = newsline.places.contents
            people = newsline.people.contents
            orgs = newsline.orgs.contents
            exchanges = newsline.exchanges.contents
            for topic in topics:
                document_categories.append("to_" + strip_tags(str(topic)))
            for place in places:
                document_categories.append("pl_" + strip_tags(str(place)))
            for person in people:
                document_categories.append("pe_" + strip_tags(str(person)))

            for org in orgs:
                document_categories.append("or_" + strip_tags(str(org)))

            for exchange in exchanges:
                document_categories.append("ex_" + strip_tags(str(exchange)))
            update_frequencies(document_categories)
            document_X.append(document_body)
            document_Y.append(to_category_vector(
                document_categories, selected_categories))

print(document_X[0:2])
print(document_Y[0:5])
print(len(document_X), len(document_Y))

news_categories.sort_values(by="NewsLines", ascending=False, inplace=True)
selected_categories = np.array(news_categories["Name"].head(20))
num_categories = 20
print(news_categories.head(num_categories))


lemmatizer = WordNetLemmatizer()
strip_special_chars = re.compile("[^a-zA-Z0-9]+")
stop_words = set(stopwords.words("english"))


def cleanUpSentence(r, stop_words=None):
    r = r.lower().replace("<br />", " ")
    r = re.sub(strip_special_chars, "", r.lower())
    if stop_words is not None:
        words = word_tokenize(r)
        filtered_sentence = []
        for w in words:
            w = lemmatizer.lemmatize(w)
            if w not in stop_words:
                filtered_sentence.append(w)
        return " ".join(filtered_sentence)
    else:
        return r


totalX = []
# label 的形式：[1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,0., 1., 0.]
# ，每个样本可以对应多个类别
totalY = np.array(document_Y)
for i, doc in enumerate(document_X):
    totalX.append(cleanUpSentence(doc, stop_words))

print(totalX[220])
print(totalY[220])

xLengths = [len(word_tokenize(x)) for x in totalX]
h = sorted(xLengths)
maxLength = h[len(h) - 1]
print("max input length is:", maxLength)

maxLength = h[int(len(h) * 0.70)]
print("覆盖70%文本的长度为：", maxLength)

max_vocab_size = 200000
input_tokenizer = Tokenizer(max_vocab_size)
input_tokenizer.fit_on_texts(totalX)
input_vocab_size = len(input_tokenizer.word_index) + 1
print("input_vocab_size:", input_vocab_size)

totalX = np.array(pad_sequences(
    input_tokenizer.texts_to_sequences(totalX), maxlen=maxLength))


embedding_dim = 256
model = Sequential()
model.add(Embedding(input_vocab_size, embedding_dim, input_length=maxLength))
model.add(GRU(256, dropout=0.9, return_sequences=True))
model.add(GRU(256, dropout=0.9))
model.add(Dense(num_categories, activation="sigmoid"))
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

history = model.fit(totalX, totalY, validation_split=0.1,
                    batch_size=16, epochs=1)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot
