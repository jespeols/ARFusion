# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointEmbedding(nn.Module):
    
    def __init__(self, config, vocab_size, max_seq_len, pad_idx):
        super(JointEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.emb_dim = config['emb_dim']
        self.dropout_prob = config['dropout_prob']
        
        self.token_emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=self.pad_idx) 
        # self.token_type_emb = nn.Embedding(self.vocab_size, self.emb_dim) 
        self.position_emb = nn.Embedding(self.max_seq_len, self.emb_dim) 
        
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        
    def forward(self, input_tensor):
        # input_tensor: (batch_size, seq_len)
        # token_type_ids: (batch_size, seq_len)
        # position_ids: (batch_size, seq_len)
        seq_len = input_tensor.size(-1)
        
        # token_type not relevant for unimodal data
        # token_type_tensor = torch.zeros_like(input_tensor).to(device) # (batch_size, seq_len)
        # token_type_tensor[:, (seq_len//2 + 1):] = 1 # here, we assume that the sentence is split in half
        
        token_emb = self.token_emb(input_tensor) # (batch_size, seq_len, emb_dim)
        # token_type_emb = self.token_type_emb(token_type_tensor) # (batch_size, seq_len, emb_dim)
        pos_tensor = torch.arange(seq_len, dtype=torch.long, device=device).expand_as(input_tensor) # (batch_size, seq_len)
        position_emb = self.position_emb(pos_tensor) # (batch_size, seq_len, emb_dim)
        
        # emb = token_emb + token_type_emb + position_emb
        emb = token_emb + position_emb
        emb = self.layer_norm(emb) 
        emb = self.dropout(emb)
        return emb
    
################################################################################################################

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        
        self.emb_dim = config['emb_dim']
        self.num_heads = config['num_heads']
        self.dropout_prob = config['dropout_prob']
        
        self.head_dim = self.emb_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.emb_dim, f"Embedding dimension must be divisible by number of heads, got {self.emb_dim} and {self.num_heads}"
        
        self.q = nn.Linear(self.emb_dim, self.emb_dim)
        self.k = nn.Linear(self.emb_dim, self.emb_dim)
        self.v = nn.Linear(self.emb_dim, self.emb_dim)
        
        self.dropout = nn.Dropout(self.dropout_prob)
    
    def forward(self, input_emb: torch.Tensor, attn_mask:torch.Tensor = None):
        B, L, D = input_emb.size() # (B=batch_size, L=seq_len, D=emb_dim)
        
        # project input embeddings to query, key, value, then split into num_heads, reducing the embedding dimension
        query = self.q(input_emb).view(B, L, self.num_heads, self.head_dim).transpose(1,2) # (B, num_heads, L, head_dim)
        key = self.k(input_emb).view(B, L, self.num_heads, self.head_dim).transpose(1,2) # (B, num_heads, L, head_dim)
        value = self.v(input_emb).view(B, L, self.num_heads, self.head_dim).transpose(1,2) # (B, num_heads, L, head_dim)
        
        scale_factor = query.size(-1) ** 0.5
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / scale_factor # (B, num_heads, L, L)
        
        attn_scores = attn_scores.masked_fill_(~attn_mask, -1e9) if attn_mask is not None else attn_scores 
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, num_heads, L, L)
        attn_weights = self.dropout(attn_weights)
        
        attn = torch.matmul(attn_weights, value) # (B, num_heads, L, head_dim)
        attn = attn.transpose(1, 2).contiguous().view(B, L, D) # (B, L, num_heads, head_dim) -> (B, L, D), concatenate the heads
        
        return attn
        

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        
        self.emb_dim = config['emb_dim']
        self.ff_dim = config['ff_dim']
        self.num_heads = config['num_heads']
        self.dropout_prob = config['dropout_prob']
        
        self.attention = MultiHeadAttention(config)
        
        # BertSelfOutput
        self.dense = nn.Linear(self.emb_dim, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        
        # BertIntermediate + BertOutput
        self.feed_forward = nn.Sequential(
            nn.Linear(self.emb_dim, self.ff_dim), # ff_dim is the intermediate dimension, need not be = emb_dim
            nn.GELU(),
            nn.Linear(self.ff_dim, self.emb_dim),
            nn.Dropout(self.dropout_prob)
        )
        
    def forward(self, input_emb: torch.Tensor, attn_mask: torch.Tensor = None):
        x = input_emb
        attn = self.attention(x, attn_mask)
        
        # BertSelfOutput
        hidden_states = self.dense(attn)
        hidden_states = self.dropout(hidden_states)
        x = x + hidden_states
        x = self.layer_norm(x)
        
        res = x
        # BertIntermediate + BertOutput
        x = self.feed_forward(x)
        x = x + res
        x = self.layer_norm(x)
        return x
    
    
class BERT(nn.Module):
    
    def __init__(self, config, vocab_size: int, max_seq_len: int, num_ab: int, pad_idx: int = 1):
        super(BERT, self).__init__()
                
        self.vocab_size = vocab_size
        self.num_ab = num_ab
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
                
        # parameters
        self.emb_dim = config['emb_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.ff_dim = config['ff_dim']
        self.dropout_prob = config['dropout_prob']
        
        # embedding and encoder blocks
        self.embedding = JointEmbedding(config, self.vocab_size, self.max_seq_len, self.pad_idx) 
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(self.num_layers)])
        
        # classifier
        self.hidden_dim = config['hidden_dim'] # for the classification layer
        self.classification_layer = [AbPredictor(self.emb_dim, self.hidden_dim).to(device) for _ in range(num_ab)] 
        
    def forward(self, input_tensor: torch.Tensor, attn_mask:torch.Tensor):
        embedded = self.embedding(input_tensor)
        for layer in self.encoder:
            embedded = layer(embedded, attn_mask)
        encoded = embedded # ouput of the BERT Encoder
        
        cls_token = encoded[:, 0, :] # (batch_size, emb_dim)
        predictions = torch.cat([net(cls_token) for net in self.classification_layer], dim=1) # (batch_size, num_ab)
        return predictions


class AbPredictor(nn.Module): # predicts resistance or susceptibility for an antibiotic
    def __init__(self, emb_dim: int, hidden_dim: int):
        super(AbPredictor, self).__init__()
        
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1), # binary classification (S:0 | R:1)
        )
    
    def forward(self, X):
        # X is the CLS token of the BERT model
        return self.classifier(X)

            
################################################################################################################         
################################################################################################################

class PhenoEmbedding(nn.Module): 
# uses a normal position embedding, an embedding for the antibiotic, and an embedding for the resistance (binary)
    def __init__(self, config, vocab_size):
        super(PhenoEmbedding, self).__init__()
        
        self.emb_dim = config['emb_dim']
        self.vocab_size = vocab_size
        self.dropout_prob = config['dropout_prob']
        
        self.token_emb = nn.Embedding(self.vocab_size, self.emb_dim) 
        self.res_emb = nn.Embedding(2, self.emb_dim) # 2 possible values for resistance: 0 or 1
        # self.token_type_emb = nn.Embedding(self.vocab_size, self.emb_dim) 
        self.position_emb = nn.Embedding(self.vocab_size, self.emb_dim) 
        
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        
    def forward(self, input_tensor, res_mask):
        # input_tensor: (batch_size, seq_len)
        # token_type_ids: (batch_size, seq_len)
        # position_ids: (batch_size, seq_len)
        # res_mask: (batch_size, seq_len) - determines resistance value for each token
        
        seq_len = input_tensor.size(-1)
        
        pos_tensor = self.numeric_position(seq_len, input_tensor)
        # token_type not relevant for unimodal data
        # token_type_tensor = torch.zeros_like(input_tensor).to(device) # (batch_size, seq_len)
        # token_type_tensor[:, (seq_len//2 + 1):] = 1 # here, we assume that the sentence is split in half
        
        token_emb = self.token_emb(input_tensor) # (batch_size, seq_len, emb_dim)
        # token_type_emb = self.token_type_emb(token_type_tensor) # (batch_size, seq_len, emb_dim)
        position_emb = self.position_emb(pos_tensor) # (batch_size, seq_len, emb_dim)
        
        # emb = token_emb + token_type_emb + position_emb
        emb = token_emb + position_emb
        emb = self.dropout(emb)
        emb = self.layer_norm(emb) 
        return emb
                
    def numeric_position(self, dim, input_tensor): # input_tensor: (batch_size, seq_len)
        # dim is the length of the sequence
        position_ids = torch.arange(dim, dtype=torch.long, device=device) # create tensor of [0, 1, 2, ..., dim-1]
        return position_ids.expand_as(input_tensor) # expand to (batch_size, seq_len)
