from download import download
import re

url = "https://modelscope.cn/api/v1/datasets/SelinaRR/Multi30K/repo?Revision=master&FilePath=Multi30K.zip"

download(url, './', kind='zip', replace=True)

datasets_path = './datasets/'
train_path = datasets_path + 'train/'
valid_path = datasets_path + 'valid/'
test_path = datasets_path + 'test/'

def print_data(data_file_path, print_n=5):
    print("=" * 40 + "datasets in {}".format(data_file_path) + "=" * 40)
    with open(data_file_path, 'r', encoding='utf-8') as en_file:
        en = en_file.readlines()[:print_n]
        for index, seq in enumerate(en):
            print(index, seq.replace('\n', ''))


print_data(train_path + 'train.de')
print_data(train_path + 'train.en')

import os


class Multi30K():
    """Multi30K数据集加载器，加载Multi30K数据集并处理为一个Python迭代对象"""

    def __init__(self, path):
        self.data = self._load(path)

    def _load(self, path):
        def tokenize(text):
            text = text.rstrip()
            return [tok.lower() for tok in re.findall(r'\w+|[^\w\s]', text)]

        def read_data(data_file_path):
            with open(data_file_path, 'r', encoding='utf-8') as data_file:
                data = data_file.readlines()[:-1]
                return [tokenize(i) for i in data]

        members = {i.split('.')[-1]: path + i for i in os.listdir(path)}
        ret = [read_data(members['de']), read_data(members['en'])]
        return list(zip(*ret))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


train_dataset, valid_dataset, test_dataset = Multi30K(train_path), Multi30K(valid_path), Multi30K(test_path)

for de, en in train_dataset:
    print(f'德文：{de}')
    print(f'英文：{en}')
    break

class Vocab:
    """通过词频字典，构建词典"""

    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, word_count_dict, min_freq=1):
        self.word2idx = {}
        for idx, tok in enumerate(self.special_tokens):
            self.word2idx[tok] = idx

        filted_dict = {w: c for w, c in word_count_dict.items() if c >= min_freq}
        for w, _ in filted_dict.items():
            self.word2idx[w] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.bos_idx = self.word2idx['<bos>']
        self.eos_idx = self.word2idx['<eos>']
        self.pad_idx = self.word2idx['<pad>']
        self.unk_idx = self.word2idx['<unk>']

    def _word2idx(self, word):
        """单词映射至数字索引"""
        if word not in self.word2idx:
            return self.unk_idx
        return self.word2idx[word]

    def _idx2word(self, idx):
        """数字索引映射至单词"""
        if idx not in self.idx2word:
            raise ValueError('input index is not in vocabulary.')
        return self.idx2word[idx]

    def encode(self, word_or_list):
        """将单个单词或单词数组映射至单个数字索引或数字索引数组"""
        if isinstance(word_or_list, list):
            return [self._word2idx(i) for i in word_or_list]
        return self._word2idx(word_or_list)

    def decode(self, idx_or_list):
        """将单个数字索引或数字索引数组映射至单个单词或单词数组"""
        if isinstance(idx_or_list, list):
            return [self._idx2word(i) for i in idx_or_list]
        return self._idx2word(idx_or_list)

    def __len__(self):
        return len(self.word2idx)

from collections import Counter, OrderedDict

def build_vocab(dataset):
    de_words, en_words = [], []
    for de, en in dataset:
        de_words.extend(de)
        en_words.extend(en)

    de_count_dict = OrderedDict(sorted(Counter(de_words).items(), key=lambda t: t[1], reverse=True))
    en_count_dict = OrderedDict(sorted(Counter(en_words).items(), key=lambda t: t[1], reverse=True))

    return Vocab(de_count_dict, min_freq=2), Vocab(en_count_dict, min_freq=2)


de_vocab, en_vocab = build_vocab(train_dataset)
print('Unique tokens in de vocabulary:{} and en vocabulary:{}\n'.format(len(de_vocab), len(en_vocab)))

str_seq_en = ['two', 'young', ',', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']
print("word:{}\nindex:{}\n".format(str_seq_en, en_vocab.encode(str_seq_en)))

index = [5, 6, 7, 8, 9, 10]
print("index:{}\nword:{}".format(index, en_vocab.decode(index)))

import mindspore

class Iterator():
    """创建数据迭代器"""
    def __init__(self, dataset, de_vocab, en_vocab, batch_size, max_len=32, drop_reminder=False):
        self.dataset = dataset
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab

        self.batch_size = batch_size
        self.max_len = max_len
        self.drop_reminder = drop_reminder

        length = len(self.dataset) // batch_size
        self.len = length if drop_reminder else length + 1  # 批量数量

    def __call__(self):
        def pad(idx_list, vocab, max_len):
            """统一序列长度，并记录有效长度"""
            idx_pad_list, idx_len = [], []
            for i in idx_list:
                if len(i) > max_len - 2:
                    idx_pad_list.append([vocab.bos_idx] + i[:max_len-2] + [vocab.eos_idx])
                    idx_len.append(max_len)
                else:
                    idx_pad_list.append([vocab.bos_idx] + i + [vocab.eos_idx] + [vocab.pad_idx] * (max_len - len(i) - 2))
                    idx_len.append(len(i) + 2)
            return idx_pad_list, idx_len

        def sort_by_length(src, trg):
            """根据src的字段长度进行排序"""
            data = zip(src, trg)
            data = sorted(data, key=lambda t: len(t[0]), reverse=True)
            return zip(*list(data))

        def encode_and_pad(batch_data, max_len):
            """将批量中的文本数据转换为数字索引，并统一每个序列的长度"""
            src_data, trg_data = zip(*batch_data)
            src_idx = [self.de_vocab.encode(i) for i in src_data]
            trg_idx = [self.en_vocab.encode(i) for i in trg_data]

            src_idx, trg_idx = sort_by_length(src_idx, trg_idx)
            src_idx_pad, src_len = pad(src_idx, de_vocab, max_len)
            trg_idx_pad, _ = pad(trg_idx, en_vocab, max_len)

            return src_idx_pad, src_len, trg_idx_pad

        for i in range(self.len):
            if i == self.len - 1 and not self.drop_reminder:
                batch_data = self.dataset[i * self.batch_size:]
            else:
                batch_data = self.dataset[i * self.batch_size: (i+1) * self.batch_size]

            src_idx, src_len, trg_idx = encode_and_pad(batch_data, self.max_len)
            yield mindspore.Tensor(src_idx, mindspore.int32), \
                mindspore.Tensor(src_len, mindspore.int32), \
                mindspore.Tensor(trg_idx, mindspore.int32)

    def __len__(self):
        return self.len

train_iterator = Iterator(train_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=True)
valid_iterator = Iterator(valid_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=False)
test_iterator = Iterator(test_dataset, de_vocab, en_vocab, batch_size=1, max_len=32, drop_reminder=False)

for src_idx, src_len, trg_idx in train_iterator():
    print(f'src_idx.shape:{src_idx.shape}\n{src_idx}\nsrc_len.shape:{src_len.shape}\n{src_len}\ntrg_idx.shape:{trg_idx.shape}\n{trg_idx}')
    break

from mindspore import nn, Tensor

word_index = Tensor([21, 28, 49, 12, 275, 119, 49, 23, 54, 32])
src_emb = nn.Embedding(len(de_vocab), 4)
enc_outputs = src_emb(word_index)
print(enc_outputs)

import mindspore
from mindspore import nn, ops, Tensor
from mindspore import dtype as mstype
from mindspore import numpy as mnp

class PositionalEncoding(nn.Cell):
    """位置编码"""

    def __init__(self, d_model, dropout_p=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(1 - dropout_p)

        self.pe = ops.Zeros()((max_len, d_model), mstype.float32)

        pos = mnp.arange(0, max_len, dtype=mstype.float32).view((-1, 1))
        angle = ops.pow(10000.0, mnp.arange(0, d_model, 2, dtype=mstype.float32)/d_model)

        self.pe[:, 0::2] = ops.sin(pos/angle)
        self.pe[:, 1::2] = ops.cos(pos/angle)

    def construct(self, x):
        batch_size = x.shape[0]

        pe = self.pe.expand_dims(0)
        pe = ops.broadcast_to(pe, (batch_size, -1, -1))

        x = x + pe[:, :x.shape[1], :]
        return self.dropout(x)

x = ops.Zeros()((1, 10, 4), mstype.float32)
pe = PositionalEncoding(4)
print(pe(x))

class ScaledDotProductAttention(nn.Cell):
    def __init__(self, dropout_p=0.):
        super().__init__()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1-dropout_p)
        self.sqrt = ops.Sqrt()


    def construct(self, query, key, value, attn_mask=None):
        """scaled dot product attention"""
        embed_size = query.shape[-1]
        scaling_factor = self.sqrt(Tensor(embed_size, mstype.float16))

        attn = ops.matmul(query, key.swapaxes(-2, -1) / scaling_factor)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = ops.matmul(attn, value)

        return (output, attn)


attention = ScaledDotProductAttention()
q_s = k_s = v_s = ops.ones((128, 8, 32, 64), mindspore.float32)
attn_mask = ops.ones((128, 8, 32, 32), mindspore.float32)
attn_mask = mindspore.ops.gt(attn_mask, attn_mask)
output, attn = attention(q_s, k_s, v_s, attn_mask)
print(output.shape, attn.shape)

class MultiHeadAttention(nn.Cell):
    def __init__(self, d_model, d_k, n_heads, dropout_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.W_Q = nn.Dense(d_model, d_k * n_heads)
        self.W_K = nn.Dense(d_model, d_k * n_heads)
        self.W_V = nn.Dense(d_model, d_k * n_heads)
        self.W_O = nn.Dense(n_heads * d_k, d_model)
        self.attention = ScaledDotProductAttention(dropout_p=dropout_p)

    def construct(self, query, key, value, attn_mask):
        """
        query: [batch_size, len_q, d_model]
        key: [batch_size, len_k, d_model]
        value: [batch_size, len_k, d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """

        batch_size = query.shape[0]

        q_s = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k)
        k_s = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k)
        v_s = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_k)

        q_s = q_s.transpose((0, 2, 1, 3))
        k_s = k_s.transpose((0, 2, 1, 3))
        v_s = v_s.transpose((0, 2, 1, 3))

        attn_mask = attn_mask.expand_dims(1)
        attn_mask = ops.tile(attn_mask, (1, self.n_heads, 1, 1))

        context, attn = self.attention(q_s, k_s, v_s, attn_mask)

        context = context.transpose((0, 2, 1, 3)).view((batch_size, -1, self.n_heads * self.d_k))

        output = self.W_O(context)

        return output, attn

enc_input = ops.ones((128, 32, 512), mindspore.float32)
attn_mask = ops.ones((128, 32, 32), mindspore.float32)
attn_mask = mindspore.ops.gt(attn_mask, attn_mask)

mha = MultiHeadAttention(512, 64, 8)
output, attn = mha(enc_input, enc_input, enc_input, attn_mask)
print("output.shape:{}\nattn_mask.shape:{}\n".format(output.shape, attn.shape))

def get_attn_pad_mask(seq_q, seq_k, pad_idx):
    """注意力掩码：识别序列中的<pad>占位符

    Args:
        seq_q (Tensor): query序列，shape = [batch size, query len]
        seq_k (Tensor): key序列，shape = [batch size, key len]
        pad_idx (Tensor): key序列<pad>占位符对应的数字索引
    """
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape

    pad_attn_mask = ops.equal(seq_k, pad_idx)

    pad_attn_mask = pad_attn_mask.expand_dims(1)
    pad_attn_mask = ops.broadcast_to(pad_attn_mask, (batch_size, len_q, len_k))

    return pad_attn_mask

q = k = Tensor([[1, 1, 0, 0]], mstype.float32)
pad_idx = 0
mask = get_attn_pad_mask(q, k, pad_idx)
print(mask)

class PoswiseFeedForward(nn.Cell):
    def __init__(self, d_ff, d_model, dropout_p=0.):
        super().__init__()
        self.linear1 = nn.Dense(d_model, d_ff)
        self.linear2 = nn.Dense(d_ff, d_model)
        self.dropout = nn.Dropout(1-dropout_p)
        self.relu = nn.ReLU()

    def construct(self, x):
        """前馈神经网络
        x: [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.linear2(x)
        return output

x = ops.ones((128, 32, 512), mstype.float32)
ffn = PoswiseFeedForward(2048, 512)
print(ffn(x).shape)


class AddNorm(nn.Cell):
    def __init__(self, d_model, dropout_p=0.):
        super().__init__()
        self.layer_norm = nn.LayerNorm((d_model,), epsilon=1e-5)
        self.dropout = nn.Dropout(1 - dropout_p)

    def construct(self, x, residual):
        return self.layer_norm(self.dropout(x) + residual)


class EncoderLayer(nn.Cell):
    def __init__(self, d_model, n_heads, d_ff, dropout_p=0.):
        super().__init__()
        d_k = d_model // n_heads
        if d_k * n_heads != d_model:
            raise ValueError(f"The `d_model` {d_model} can not be divisible by `num_heads` {n_heads}.")
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, n_heads, dropout_p)
        self.pos_ffn = PoswiseFeedForward(d_ff, d_model, dropout_p)
        self.add_norm1 = AddNorm(d_model, dropout_p)
        self.add_norm2 = AddNorm(d_model, dropout_p)

    def construct(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        residual = enc_inputs

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        enc_outputs = self.add_norm1(enc_outputs, residual)
        residual = enc_outputs

        enc_outputs = self.pos_ffn(enc_outputs)

        enc_outputs = self.add_norm2(enc_outputs, residual)

        return enc_outputs, attn


x = ops.ones((128, 32, 512), mstype.float32)
mask = Tensor([False]).broadcast_to((128, 32, 32))
encoder_layer = EncoderLayer(512, 8, 2018)
output, attn = encoder_layer(x, mask)
print(output.shape, attn.shape)

class Encoder(nn.Cell):
    def __init__(self, src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout_p)
        self.layers = nn.CellList([EncoderLayer(d_model, n_heads, d_ff, dropout_p)] * n_layers)
        self.scaling_factor = ops.Sqrt()(Tensor(d_model, mstype.float32))

    def construct(self, enc_inputs, src_pad_idx):
        """enc_inputs : [batch_size, src_len]
        """
        enc_outputs = self.src_emb(enc_inputs.astype(mstype.int32))
        enc_outputs = self.pos_emb(enc_outputs * self.scaling_factor)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs, src_pad_idx)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

encoder = Encoder(len(de_vocab), 512, 8, 2048, 6)

for src_idx, src_len, trg_idx in train_iterator():
    enc_outputs, enc_self_attns = encoder(src_idx, de_vocab.pad_idx)
    print("enc_outputs.shape:{}\n".format(enc_outputs.shape))
    break


def get_attn_subsequent_mask(seq_q, seq_k):
    """生成时间掩码，使decoder在第t时刻只能看到序列的前t-1个元素

    Args:
        seq_q (Tensor): query序列，shape = [batch size, len_q]
        seq_k (Tensor): key序列，shape = [batch size, len_k]
    """
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape
    ones = ops.ones((batch_size, len_q, len_k), mindspore.float32)
    subsequent_mask = mnp.triu(ones, k=1)
    return subsequent_mask


q = k = ops.ones((1, 4), mstype.float32)
mask = get_attn_subsequent_mask(q, k)
print(mask)


class DecoderLayer(nn.Cell):
    def __init__(self, d_model, n_heads, d_ff, dropout_p=0.):
        super().__init__()
        d_k = d_model // n_heads
        if d_k * n_heads != d_model:
            raise ValueError(f"The `d_model` {d_model} can not be divisible by `num_heads` {n_heads}.")
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, n_heads, dropout_p)
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, n_heads, dropout_p)
        self.pos_ffn = PoswiseFeedForward(d_ff, d_model, dropout_p)
        self.add_norm1 = AddNorm(d_model, dropout_p)
        self.add_norm2 = AddNorm(d_model, dropout_p)
        self.add_norm3 = AddNorm(d_model, dropout_p)

    def construct(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, trg_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, trg_len, trg_len]
        dec_enc_attn_mask: [batch_size, trg_len, src_len]
        """
        residual = dec_inputs
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.add_norm1(dec_outputs, residual)
        residual = dec_outputs
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.add_norm2(dec_outputs, residual)
        residual = dec_outputs
        dec_outputs = self.pos_ffn(dec_outputs)
        dec_outputs = self.add_norm3(dec_outputs, residual)

        return dec_outputs, dec_self_attn, dec_enc_attn


x = y = ops.ones((128, 32, 512), mstype.float32)
mask1 = mask2 = Tensor([False]).broadcast_to((128, 32, 32))
decoder_layer = DecoderLayer(512, 8, 2048)
output, attn1, attn2 = decoder_layer(x, y, mask1, mask2)
print(output.shape, attn1.shape, attn2.shape)


class Decoder(nn.Cell):
    def __init__(self, trg_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.):
        super().__init__()
        self.trg_emb = nn.Embedding(trg_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout_p)
        self.layers = nn.CellList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.projection = nn.Dense(d_model, trg_vocab_size)
        self.scaling_factor = ops.Sqrt()(Tensor(d_model, mstype.float32))

    def construct(self, dec_inputs, enc_inputs, enc_outputs, src_pad_idx, trg_pad_idx):
        """
        dec_inputs: [batch_size, trg_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        """
        dec_outputs = self.trg_emb(dec_inputs.astype(mstype.int32))
        dec_outputs = self.pos_emb(dec_outputs * self.scaling_factor)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, trg_pad_idx)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs, dec_inputs)
        dec_self_attn_mask = ops.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, src_pad_idx)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        dec_outputs = self.projection(dec_outputs)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Cell):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.to_float(mindspore.float16)
        self.decoder.to_float(mindspore.float16)

    def construct(self, enc_inputs, dec_inputs, src_pad_idx, trg_pad_idx):
        """
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, trg_len]
        """
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, src_pad_idx)

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs, src_pad_idx,
                                                                  trg_pad_idx)

        dec_logits = dec_outputs.view((-1, dec_outputs.shape[-1]))

        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns

src_vocab_size = len(de_vocab)
trg_vocab_size = len(en_vocab)
src_pad_idx = de_vocab.pad_idx
trg_pad_idx = en_vocab.pad_idx

d_model = 512
d_ff = 2048
n_layers = 6
n_heads = 8

encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.1)
decoder = Decoder(trg_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.1)
model = Transformer(encoder, decoder)

loss_fn = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.0001)

def forward(enc_inputs, dec_inputs):
    """前向网络
    enc_inputs: [batch_size, src_len]
    dec_inputs: [batch_size, trg_len]
    """
    logits, _, _, _ = model(enc_inputs, dec_inputs[:, :-1], src_pad_idx, trg_pad_idx)

    targets = dec_inputs[:, 1:].view(-1)
    loss = loss_fn(logits, targets)

    return loss

grad_fn = ops.value_and_grad(forward, None, optimizer.parameters)

from tqdm import tqdm

def train(iterator, epoch=0):
    model.set_train(True)
    num_batches = len(iterator)
    total_loss = 0
    total_steps = 0

    with tqdm(total=num_batches, unit='step', desc='Train   ') as t:
        for src, src_len, trg in iterator():
            loss, grads = grad_fn(src, trg)
            optimizer(grads)

            total_loss += loss.asnumpy()
            total_steps += 1
            curr_loss = total_loss / total_steps
            t.set_postfix({'loss': f'{curr_loss:.2f}', 'epoch': f'{epoch:03}'})
            t.update(1)

    return total_loss / total_steps

def evaluate(iterator):
    model.set_train(False)
    num_batches = len(iterator)
    total_loss = 0
    total_steps = 0

    with tqdm(total=num_batches, unit='step', desc='Evaluate') as t:
        for src, _, trg in iterator():
            loss = forward(src, trg)
            total_loss += loss.asnumpy()
            total_steps += 1
            curr_loss = total_loss / total_steps
            t.set_postfix({'loss': f'{curr_loss:.2f}'})
            t.update(1)

    return total_loss / total_steps

from mindspore import save_checkpoint

num_epochs = 10
best_valid_loss = float('inf')
ckpt_file_name = './transformer.ckpt'


for i in range(num_epochs):
    train_loss = train(train_iterator, i)
    valid_loss = evaluate(valid_iterator)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_checkpoint(model, ckpt_file_name)

from mindspore import load_checkpoint, load_param_into_net

encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.1)
decoder = Decoder(trg_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_p=0.1)
new_model = Transformer(encoder, decoder)

param_dict = load_checkpoint(ckpt_file_name)
load_param_into_net(new_model, param_dict)

def inference(sentence, max_len=32):
    """模型推理：输入一个德语句子，输出翻译后的英文句子
    enc_inputs: [batch_size(1), src_len]
    """
    new_model.set_train(False)

    if isinstance(sentence, str):
        tokens = [tok.lower() for tok in re.findall(r'\w+|[^\w\s]', sentence.rstrip())]
    else:
        tokens = [token.lower() for token in sentence]

    if len(tokens) > max_len - 2:
        src_len = max_len
        tokens = ['<bos>'] + tokens[:max_len - 2] + ['<eos>']
    else:
        src_len = len(tokens) + 2
        tokens = ['<bos>'] + tokens + ['<eos>'] + ['<pad>'] * (max_len - src_len)

    indexes = de_vocab.encode(tokens)
    enc_inputs = Tensor(indexes, mstype.float32).expand_dims(0)

    enc_outputs, _ = new_model.encoder(enc_inputs, src_pad_idx)

    dec_inputs = Tensor([[en_vocab.bos_idx]], mstype.float32)

    max_len = enc_inputs.shape[1]
    for _ in range(max_len):
        dec_outputs, _, _ = new_model.decoder(dec_inputs, enc_inputs, enc_outputs, src_pad_idx, trg_pad_idx)
        dec_logits = dec_outputs.view((-1, dec_outputs.shape[-1]))

        dec_logits = dec_logits[-1, :]
        pred = dec_logits.argmax(axis=0).expand_dims(0).expand_dims(0)
        pred = pred.astype(mstype.float32)

        dec_inputs = ops.concat((dec_inputs, pred), axis=1)

        if int(pred.asnumpy()[0]) == en_vocab.eos_idx:
            break

    trg_indexes = [int(i) for i in dec_inputs.view(-1).asnumpy()]
    eos_idx = trg_indexes.index(en_vocab.eos_idx) if en_vocab.eos_idx in trg_indexes else -1
    trg_tokens = en_vocab.decode(trg_indexes[1:eos_idx])

    return trg_tokens

example_idx = 0

src = test_dataset[example_idx][0]
trg = test_dataset[example_idx][1]
pred_trg = inference(src)

print(f'src = {src}')
print(f'trg = {trg}')
print(f"predicted trg = {pred_trg}")

from nltk.translate.bleu_score import corpus_bleu

def calculate_bleu(dataset, max_len=50):
    trgs = []
    pred_trgs = []

    for data in dataset[:10]:
        src = data[0]
        trg = data[1]

        pred_trg = inference(src, max_len)
        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return corpus_bleu(trgs, pred_trgs)


bleu_score = calculate_bleu(test_dataset)

print(f'BLEU score = {bleu_score * 100:.2f}')