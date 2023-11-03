# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JointEmbedding(nn.Module):
    
    def __init__(self, config, vocab_size):
        super(JointEmbedding, self).__init__()
        
        self.emb_dim = config['emb_dim']
        self.vocab_size = vocab_size
        self.dropout_prob = config['dropout_prob']
        
        self.token_emb = nn.Embedding(self.vocab_size, self.emb_dim) 
        # self.token_type_emb = nn.Embedding(self.vocab_size, self.emb_dim) 
        self.position_emb = nn.Embedding(self.vocab_size, self.emb_dim) 
        
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        
    def forward(self, input_tensor):
        # input_tensor: (batch_size, seq_len)
        # token_type_ids: (batch_size, seq_len)
        # position_ids: (batch_size, seq_len)
        
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
    
################################################################################################################
################################################################################################################
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
        B, L, D = input_emb.size() # (L=batch_size, L=seq_len, D=emb_dim)
        
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
        self.num_heads = config['num_heads']
        self.hidden_dim = config['hidden_dim']
        self.dropout_prob = config['dropout_prob']
        
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_dim),
            nn.Dropout(self.dropout_prob),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.emb_dim),
            nn.Dropout(self.dropout_prob)
        )
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        
    def forward(self, input_emb: torch.Tensor, attn_mask: torch.Tensor = None):
        x = input_emb
        attn = self.attention(x, attn_mask)
        x = x + attn
        x = self.layer_norm(x)
        res = x
        x = self.feed_forward(x)
        x = x + res
        x = self.layer_norm(x)
        
        return x

class BERT(nn.Module):
    
    def __init__(self, config, vocab_size):
        super(BERT, self).__init__()
                
        self.vocab_size = vocab_size
        
        self.emb_dim = config['emb_dim']
        self.vocab_size = vocab_size
        self.max_seq_len = None # Can be set later
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.dropout_prob = config['dropout_prob']
        
        self.embedding = JointEmbedding(config, vocab_size)
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(self.num_layers)])
        
        self.token_prediction_layer = nn.Linear(self.emb_dim, self.vocab_size) # MLM task
        self.softmax = nn.LogSoftmax(dim=-1) # log softmax improves numerical stability, we use NLLLoss later
        
        self.classification_layer = None # set later for classification task
        
    def forward(self, input_tensor: torch.Tensor, attn_mask:torch.Tensor = None): # None if we are not doing MLM
        embedded = self.embedding(input_tensor)
        for layer in self.encoder:
            embedded = layer(embedded, attn_mask)
        encoded = embedded # ouput of the BERT Encoder
        
        if self.classification_layer: # ASSUMES MLM AND CLASSIFICATION ARE NOT DONE AT THE SAME TIME
            cls_token = encoded[:, 0, :] # (batch_size, emb_dim)
            classification_logits = self.classification_layer(cls_token)
            return classification_logits
        else:
            token_prediction = self.token_prediction_layer(encoded) # (batch_size, seq_len, vocab_size)
            return self.softmax(token_prediction)

################################################################################################################

class AbPredictor(nn.Module):
    def __init__(self, hidden_dim, num_ab, num_outputs=2):
        super(AbPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_ab = num_ab
        self.num_outputs = num_outputs
        
        self.classifiers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.num_outputs),
            ) for _ in range(self.num_ab)]
        )
    
    def forward(self, X, ab_idx):
        return self.classifiers[ab_idx](X)
            