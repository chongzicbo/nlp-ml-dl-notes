<center><b><font color=#A52A2A size=5 >公众号：数据挖掘与机器学习笔记</font></b></center>

# 1.基于BiLSTM的命名实体识别

Embedding+BiLSTM+BiLSTM+Dense

```python
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Masking
from tensorflow.keras.models import Sequential


def build_model():
    """
    使用Sequential构建模型网络:双向LSTM
    :return:
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(Masking(mask_value=0))
    model.add(Bidirectional(LSTM(lstm_hidden_size, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(lstm_hidden_size, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(tag_size, activation="softmax"))
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model()
    train_x = np.array(padding_sequences_ids)
    train_y = np.array(padding_tags_ids)
    # model.fit(x=train_x, y=train_y, epochs=1, batch_size=batch_size, validation_split=0.2)
```

# 2. 基于BiLSTM-CRF的命名实体识别

Embedding+BiLSTM+BiLSTM+CRF

将Dense换成CRF层

```python
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input, Masking
from tensorflow.keras.models import Model
from crf import CRF


def build_model():
    """
    使用Sequential构建模型网络:双向LSTM+CRF
    :return:
    """
    inputs = Input(shape=(None,), dtype='int32')
    output = Embedding(vocab_size, embedding_dim, trainable=True)(inputs)
    output = Masking(mask_value=0)(output)
    output = Bidirectional(LSTM(lstm_hidden_size, return_sequences=True))(output)
    output = Dropout(dropout_rate)(output)
    output = Bidirectional(LSTM(lstm_hidden_size, return_sequences=True))(output)
    output = Dropout(dropout_rate)(output)
    output = Dense(tag_size, activation=None)(output)
    crf = CRF(dtype="float32")
    output = crf(output)
    model = Model(inputs, output)
    model.compile(loss=crf.loss, optimizer=optimizer, metrics=[crf.accuracy])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model()
    train_x = np.array(padding_sequences_ids)
    train_y = np.array(padding_tags_ids)
    print(train_x.shape, train_y.shape)
    model.fit(x=train_x, y=train_y, epochs=1, batch_size=batch_size, validation_split=0.2)
```

# 3. 基于BiLSTM-Attention的命名实体识别

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Input, Attention


def build_model():
    """
    使用Sequential构建模型网络:双向LSTM+self-attention
    :return:
    """
    query_input = Input(shape=(None,), dtype="int32")
    token_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    query_embeddings = token_embedding(query_input)
    value_embedding = token_embedding(query_input)
    bilstm = Bidirectional(LSTM(lstm_hidden_size, return_sequences=True))

    query_seq_encoding = bilstm(query_embeddings)
    value_seq_encoding = bilstm(value_embedding)

    attention = Attention()([query_seq_encoding, value_seq_encoding])
    output = Dense(tag_size, activation="softmax")(attention)
    model = Model(query_input, output)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model()
    train_x = np.array(padding_sequences_ids)
    train_y = np.array(padding_tags_ids)
    print(train_x.shape, train_y.shape)
    model.fit(x=train_x, y=train_y, epochs=1, batch_size=batch_size, validation_split=0.2)
```

# 4. 基于Bert-BiLSTM-CRF的命名实体识别

这列使用的是Albert，也可以换成Bert或者其它的

```python
import os

os.environ['TF_KERAS'] = '1'  # 必须放在前面,才能使用tf.keras
from bert4keras.models import build_transformer_model
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.models import Model
from crf import CRF


def build_model(use_bilstm=True, use_crf=True):
    albert = build_transformer_model(config_path, checkpoint_path, model='albert', return_keras_model=False)  # 建立模型，加载权重
    output = albert.model.output
    if use_bilstm:
        output = Bidirectional(LSTM(lstm_hidden_size, return_sequences=True))(output)
        output = Dropout(dropout_rate)(output)
    if use_crf:
        activation = None
    else:
        activation = "softmax"
    output = Dense(tag_size, activation=activation, kernel_initializer=albert.initializer)(output)
    if use_crf:
        crf = CRF(dtype="float32")
        output = crf(output)
    model = Model(albert.model.inputs, output)
    model.compile(optimizer=optimizer, loss=crf.loss, metrics=[crf.accuracy])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model()
    train_x1 = np.array(bert_sequence_ids)
    train_x2 = np.array(bert_datatype_ids)
    train_y = np.array(bert_label_ids)
    print(train_x1.shape,train_x2.shape, train_y.shape)
    # model.fit(x=[train_x1, train_x2], y=train_y, epochs=10, batch_size=8, validation_split=0.2)
```



代码：https://github.com/chongzicbo/KG_Tutorial/tree/main/ner

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200828221113544.jpg#pic_center)