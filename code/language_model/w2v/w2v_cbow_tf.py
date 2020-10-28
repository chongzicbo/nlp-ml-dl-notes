import tensorflow as tf
from data_helper import get_all_apths, get_corpus, data_generator
from data_helper import window, word_size, nb_negative

nb_epoch = 10  # 迭代次数

from tensorflow import keras


def build_model():
    """
        模型网络构建
    :return:
    """
    input_words = keras.layers.Input(shape=(window * 2,), dtype="int32")  # shape=(,window*2)
    input_vecs = keras.layers.Embedding(len(id2word), word_size, name="word2vec")(input_words)  # shape=(,window*2,word_size)
    input_vecs_sum = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(input_vecs)  # CBOW模型直接将上下文词向量求和 shape=(,word_size)

    # 第二个输入，中心词和负样本词
    samples = keras.layers.Input(shape=(nb_negative + 1,), dtype="int32")  # shape=(,nb_negative + 1)
    softmax_weights = keras.layers.Embedding(len(id2word), word_size, name="W")(samples)  # shape=(,nb_negative + 1,word_size)
    softmax_biases = keras.layers.Embedding(len(id2word), 1, name="b")(samples)  # shape=(,nb_negative + 1,1)

    # 将加和得到的词向量与中心词和负样本的词向量分别进行点乘
    input_vecs_sum_dot = keras.layers.Lambda(lambda x: tf.matmul(x[0], tf.expand_dims(x[1], 2)))([softmax_weights, input_vecs_sum])  # shape=(,nb_negative + 1,1)

    add_biases = keras.layers.Lambda(lambda x: tf.reshape(x[0] + x[1], shape=(-1, nb_negative + 1)))([input_vecs_sum_dot, softmax_biases])
    softmax = keras.layers.Lambda(lambda x: tf.nn.softmax(x))(add_biases)

    # 模型编译
    model = keras.layers.Model(inputs=[input_words, samples], outputs=softmax)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    return model


if __name__ == '__main__':
    file_dir = "F:\\data\\machine_learning\\THUCNews\\THUCNews"
    paths = get_all_apths(file_dir)
    print(len(paths), paths[0:10])

    words, corpus, id2word, word2id = get_corpus(paths)

    # print(words)
    # print(id2word)
    x, y, z = data_generator(corpus, word2id, id2word)
    print(x.shape, y.shape, z.shape)

    model = build_model()
    model.fit([x, y], z, epochs=nb_epoch, batch_size=512)
