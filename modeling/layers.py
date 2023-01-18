import torch,sys
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  

        v, attn_weights = self.attention(q, k, v, attn_mask)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)

        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, attn_dropout=0.1, **kwargs):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = kwargs['diagonal_attention_mask']
        self.device = kwargs['device']
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(self.device)
        else:
            mask_time = None

        residual = enc_input
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=mask_time)
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights



class FeedForward_Teddy(nn.Module):
    def __init__(self,n_input,n_output):
        super(FeedForward_Teddy, self).__init__()

        self.fc1 = nn.Linear(n_input, 256)
        self.norm1 = nn.BatchNorm1d(256)
        self.act1 = nn.LeakyReLU(0.2)   

        self.fc2 = nn.Linear(256, 512)
        self.norm2 = nn.BatchNorm1d(512)
        self.act2 = nn.LeakyReLU(0.2)

        self.fc3 = nn.Linear(512, 1024)
        self.norm3 = nn.BatchNorm1d(1024)
        self.act3 = nn.LeakyReLU(0.2)

        self.fc4 = nn.Linear(1024, 5096)
        self.norm4 = nn.BatchNorm1d(5096)
        self.act4 = nn.LeakyReLU(0.2)

        self.fc_final = nn.Linear(5096, n_output)

    def forward(self, z):
        output = self.fc1(z)
        #output = torch.permute(output, (0, 2, 1))
        output = self.norm1(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.act1(output)
    
        output = self.fc2(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.norm2(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.act2(output)
        
        output = self.fc3(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.norm3(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.act3(output)
        
        output = self.fc4(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.norm4(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.act4(output)
        
        output = self.fc_final(output)

        return output


class FeedForward_Camel(nn.Module):
    def __init__(self,n_input,n_output):
        super(FeedForward_Camel, self).__init__()

        self.fc1 = nn.Linear(n_input, 64)
        self.norm1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2)   

        self.fc2 = nn.Linear(64, 32)
        self.norm2 = nn.BatchNorm1d(32)
        self.act2 = nn.LeakyReLU(0.2)

        self.fc3 = nn.Linear(32, 16)
        self.norm3 = nn.BatchNorm1d(16)
        self.act3 = nn.LeakyReLU(0.2)

        self.fc_final = nn.Linear(16, n_output)

    def forward(self, z):

        output = self.fc1(z)
        #output = torch.permute(output, (0, 2, 1))
        output = self.norm1(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.act1(output)
    
        output = self.fc2(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.norm2(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.act2(output)
        
        output = self.fc3(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.norm3(output)
        #output = torch.permute(output, (0, 2, 1))
        output = self.act3(output)
        
        output = self.fc_final(output)

        return output

class TEDDY_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device, num_layers=1):
        super(TEDDY_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) 

        return out
