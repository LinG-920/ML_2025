import torch
import torch.nn as nn
import math


class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_len, num_layers=2):
            """
            初始化LSTM模型。

            参数:
            input_dim (int): 输入特征的数量。
            hidden_dim (int): LSTM隐藏层的维度。
            output_len (int): 预测序列的长度（输出维度）。
            num_layers (int): LSTM的层数。
            """
            super(LSTMModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True, dropout = 0.1)
            self.fc = nn.Linear(hidden_dim, output_len)
        
        def forward(self, x):
            # 初始化隐藏状态和细胞状态
            # h0 shape: (num_layers, batch_size, hidden_dim)
            # c0 shape: (num_layers, batch_size, hidden_dim)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.lstm(x, (h0, c0))

            # 获取最后一个时间步的输出
            pred = self.fc(out[:, -1, :])
            return pred


# --- Positional Encoding for Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建位置编码矩阵pe，形状为[max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # position是一个形状为[max_len, 1]的张量，表示位置索引
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 在第0维增加一个维度，使其形状为[1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x是输入张量，形状为[batch_size, seq_len, embed_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- Transformer Model Definition ---
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, d_hid, num_layers, output_len, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_fc = nn.Linear(d_model, output_len)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_dim]
        """
        src = self.input_fc(src)
        src = self.pos_emb(src)
        output = self.encoder(src)
        
        # We take the output of the last time step to make a prediction
        output = self.output_fc(output[:, -1, :])
        return output
    

class CNNGRUTransformerModel(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, kernel_size, gru_hidden_dim, num_gru_layers,
                 d_model, nhead, d_hid, num_transformer_layers, output_len, dropout=0.2):
        super(CNNGRUTransformerModel, self).__init__()
        
        # 1. CNN Part
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, 
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
        )
        
        # 2. GRU Part
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_dim,
            num_layers=num_gru_layers, # 通常GRU层数不用太多
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # 3. Transformer Part
        self.projection = nn.Linear(gru_hidden_dim, d_model) if gru_hidden_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # 4. Output Layer
        self.output_fc = nn.Linear(d_model, output_len)

    def forward(self, src):
        # src: [batch, seq_len, input_dim]
        
        # 1. CNN
        src_permuted = src.permute(0, 2, 1)    # -> [batch, input_dim, seq_len]
        cnn_output = self.cnn(src_permuted)
        cnn_output = cnn_output.permute(0, 2, 1) # -> [batch, seq_len, cnn_out_channels]
        
        # 2. GRU
        gru_output, _ = self.gru(cnn_output) # -> [batch, seq_len, gru_hidden_dim]
        
        # 3. Transformer
        projected_output = self.projection(gru_output)
        transformer_input = self.pos_encoder(projected_output)
        transformer_output = self.transformer_encoder(transformer_input)
        
        # 4. Final Prediction
        output = self.output_fc(transformer_output[:, -1, :])
        
        return output