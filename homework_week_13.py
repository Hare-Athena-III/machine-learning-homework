import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import spacy
import random
import math
import time
import os

# 设置随机种子
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 1. 数据预处理 - 使用 spaCy 进行分词
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]


# 特殊符号定义
SPECIAL_SYMBOLS = ['<pad>', '<sos>', '<eos>', '<unk>']
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# 自定义 Dataset 类
class TranslationDataset(Dataset):
    """从本地文件加载翻译数据集"""

    def __init__(self, src_file, trg_file, src_tokenizer, trg_tokenizer):
        self.src_sentences = []
        self.trg_sentences = []
        with open(src_file, 'r', encoding='utf-8') as f_src, \
                open(trg_file, 'r', encoding='utf-8') as f_trg:
            for src_line, trg_line in zip(f_src, f_trg):
                src_line = src_line.strip()
                trg_line = trg_line.strip()
                if src_line and trg_line:
                    self.src_sentences.append(src_tokenizer(src_line))
                    self.trg_sentences.append(trg_tokenizer(trg_line))

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        return self.src_sentences[idx], self.trg_sentences[idx]


# 构建词汇表
def build_vocab(data_iter, specials, max_size=10000, min_freq=2):
    """从数据集构建词汇表"""
    counter = Counter()
    for tokens in data_iter:
        counter.update(tokens)
    # 按频率排序，保留高频词
    most_common = counter.most_common(max_size - len(specials))
    # 过滤掉低频词
    vocab_list = [word for word, freq in most_common if freq >= min_freq]
    # 构建 stoi 和 itos 映射
    stoi = {sym: i for i, sym in enumerate(specials)}
    itos = list(specials)
    for word in vocab_list:
        stoi[word] = len(itos)
        itos.append(word)
    return stoi, itos


def yield_tokens(dataset, lang_idx):
    """生成 tokens 迭代器"""
    for i in range(len(dataset)):
        src, trg = dataset[i]
        yield src if lang_idx == 0 else trg


def numericalize(vocab_stoi, tokens):
    """将 token 序列转换为索引序列（带 <sos> 和 <eos>）"""
    return [SOS_IDX] + [vocab_stoi.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]


# 2. 设置数据目录并加载数据
DATA_DIR = "./data/multi30k"

print("正在加载数据...")
train_dataset = TranslationDataset(
    os.path.join(DATA_DIR, 'train.de'),
    os.path.join(DATA_DIR, 'train.en'),
    tokenize_de, tokenize_en
)
valid_dataset = TranslationDataset(
    os.path.join(DATA_DIR, 'val.de'),
    os.path.join(DATA_DIR, 'val.en'),
    tokenize_de, tokenize_en
)
test_dataset = TranslationDataset(
    os.path.join(DATA_DIR, 'test2016.de'),
    os.path.join(DATA_DIR, 'test2016.en'),
    tokenize_de, tokenize_en
)
print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(valid_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

# 构建词汇表
print("正在构建词汇表...")
src_stoi, src_itos = build_vocab(
    yield_tokens(train_dataset, 0),
    specials=SPECIAL_SYMBOLS,
    max_size=10000,
    min_freq=2
)
trg_stoi, trg_itos = build_vocab(
    yield_tokens(train_dataset, 1),
    specials=SPECIAL_SYMBOLS,
    max_size=10000,
    min_freq=2
)
print(f"源语言词汇表大小: {len(src_stoi)}")
print(f"目标语言词汇表大小: {len(trg_stoi)}")


# 自定义 collate 函数（用于 DataLoader 的批处理填充）
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_tokens, trg_tokens in batch:
        src_batch.append(torch.tensor(numericalize(src_stoi, src_tokens), dtype=torch.long))
        trg_batch.append(torch.tensor(numericalize(trg_stoi, trg_tokens), dtype=torch.long))
    # pad_sequence 默认 batch_first=False → 形状 [seq_len, batch_size]
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)
    return src_batch, trg_batch


# 创建 DataLoader（替代旧的 BucketIterator）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, drop_last=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, drop_last=True
)

# 2. 模型定义（完全遵循论文架构）
class Transformer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hid_dim=512,
                 n_layers=6,
                 n_heads=8,
                 pf_dim=2048,
                 dropout=0.1,
                 max_length=100):
        super().__init__()
        self.encoder = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout,
                               max_length)
        self.decoder = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout,
                               max_length)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg, src_mask, trg_mask):
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return self.out(output)


# 位置编码（正弦/余弦）
class PositionalEncoding(nn.Module):
    def __init__(self, hid_dim, dropout, max_length=100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_length, hid_dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2) * (-math.log(10000.0) / hid_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(hid_dim, n_heads, dropout)
        self.pwff = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(_src))
        _src = self.pwff(src)
        src = self.norm2(src + self.dropout(_src))
        return src


# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length):
        super().__init__()
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, dropout, max_length)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src = self.dropout(self.tok_embedding(src) * math.sqrt(self.tok_embedding.embedding_dim))
        src = self.pos_embedding(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(hid_dim, n_heads, dropout)
        self.enc_attn = MultiHeadAttention(hid_dim, n_heads, dropout)
        self.pwff = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.norm3 = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout(_trg))
        _trg, attention = self.enc_attn(trg, enc_src, enc_src, src_mask)
        trg = self.norm2(trg + self.dropout(_trg))
        _trg = self.pwff(trg)
        trg = self.norm3(trg + self.dropout(_trg))
        return trg, attention


# 解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length):
        super().__init__()
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, dropout, max_length)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.dropout(self.tok_embedding(trg) * math.sqrt(self.tok_embedding.embedding_dim))
        trg = self.pos_embedding(trg)
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        return trg


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.fc_k(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.fc_v(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


# 前馈网络
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# 3. 掩码生成
def create_mask(src, trg, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(3)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return src_mask, trg_mask


# 4. 初始化模型
INPUT_DIM = len(src_stoi)
OUTPUT_DIM = len(trg_stoi)
HID_DIM = 256   # 论文中为512，此处减小以加快训练
N_LAYERS = 3    # 论文中为6，此处减小
N_HEADS = 8
PF_DIM = 512
DROPOUT = 0.1
MAX_LENGTH = 100

model = Transformer(INPUT_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, N_HEADS,
                    PF_DIM, DROPOUT, MAX_LENGTH).to(device)

# 5. 优化器与损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# 6. 训练函数
def train(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(loader):
        src, trg = src.to(device), trg.to(device)
        src = src.transpose(0, 1)  # [batch_size, src_len]
        trg = trg.transpose(0, 1)  # [batch_size, trg_len]
        optimizer.zero_grad()
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]
        src_mask, trg_mask = create_mask(src, trg_input, PAD_IDX)
        output = model(src, trg_input, src_mask, trg_mask)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_output = trg_output.contiguous().view(-1)
        loss = criterion(output, trg_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


# 7. 评估函数
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            src = src.transpose(0, 1)
            trg = trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]
            src_mask, trg_mask = create_mask(src, trg_input, PAD_IDX)
            output = model(src, trg_input, src_mask, trg_mask)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_output = trg_output.contiguous().view(-1)
            loss = criterion(output, trg_output)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


# 8. 训练循环
N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_loader, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer-model.pt')

# 9. 测试
model.load_state_dict(torch.load('transformer-model.pt'))
test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f}')
