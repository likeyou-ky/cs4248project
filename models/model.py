import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention, NoQueryAttention
from common import GraphConvolution, position_weight, mask, actf

class AFFGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AFFGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        isftext = True if opt.isftext == 'True' else False
        self.gclayers = nn.ModuleList()
        for i in range(opt.nlayers):
            self.gclayers.append(GraphConvolution(opt.hidden_dim*2, opt.hidden_dim*2, isfText=isftext))
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        return position_weight(self.opt, x, aspect_double_idx, text_len, aspect_len)

    def mask(self, x, aspect_double_idx):
        return mask(self.opt, x, aspect_double_idx)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        x, (_, _) = self.text_lstm(text, text_len)
        text_out = x.clone().detach()
        for layer in self.gclayers:
            x = actf(self.opt.actf)(layer(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output

class ATTSenticGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ATTSenticGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.aspect_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        isftext = True if opt.isftext == 'True' else False
        self.gclayers = nn.ModuleList()
        for i in range(opt.nlayers):
            self.gclayers.append(GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim, isfText=isftext))
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.self_att = Attention(opt.embed_dim*2, score_function='bi_linear')

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        return position_weight(self.opt, x, aspect_double_idx, text_len, aspect_len)

    def mask(self, x, aspect_double_idx):
        return mask(self.opt, x, aspect_double_idx)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        aspect = self.embed(aspect_indices)
        text = self.text_embed_dropout(text)
        aspect = self.text_embed_dropout(aspect)
        aspect, (_, _) = self.aspect_lstm(aspect, aspect_len)
        aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float).to(self.opt.device)
        aspect = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        x, (_, _) = self.text_lstm(text, text_len)
        _, score = self.self_att(x, aspect)
        text_out = x.clone().detach()
        adj = torch.mul(adj, score)
        for layer in self.gclayers:
            x = actf(self.opt.actf)(layer(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output

class LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices = inputs[0]
        x = self.embed(text_indices)
        x_len = torch.sum(text_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.fc(h_n[0])
        return out

class SDGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(SDGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        isftext = True if opt.isftext == 'True' else False
        self.gclayers = nn.ModuleList()
        for i in range(opt.nlayers):
            self.gclayers.append(GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim, isfText=isftext))
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.dfc = nn.Linear(4*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        return position_weight(self.opt, x, aspect_double_idx, text_len, aspect_len)

    def mask(self, x, aspect_double_idx):
        return mask(self.opt, x, aspect_double_idx)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj, d_adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        x, (_, _) = self.text_lstm(text, text_len)
        text_out = x.clone().detach()
        for layer in self.gclayers:
            x = actf(self.opt.actf)(layer(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output

class SenticGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(SenticGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        isftext = True if opt.isftext == 'True' else False
        self.gclayers = nn.ModuleList()
        for i in range(opt.nlayers):
            self.gclayers.append(GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim, isfText=isftext))
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        return position_weight(self.opt, x, aspect_double_idx, text_len, aspect_len)

    def mask(self, x, aspect_double_idx):
        return mask(self.opt, x, aspect_double_idx)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        x, (_, _) = self.text_lstm(text, text_len)
        text_out = x.clone().detach()
        for layer in self.gclayers:
            x = actf(self.opt.actf)(layer(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output

class SenticGCN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(SenticGCN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        #self.dropout = nn.Dropout(opt.dropout)
        #self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        #self.text_lstm = DynamicLSTM(768, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        isftext = True if opt.isftext == 'True' else False
        self.gclayers = nn.ModuleList()
        for i in range(opt.nlayers):
            self.gclayers.append(GraphConvolution(opt.bert_dim, opt.bert_dim, isfText=isftext))
        self.fc = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)
    
    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        return position_weight(self.opt, x, aspect_double_idx, text_len, aspect_len)

    def mask(self, x, aspect_double_idx):
        return mask(self.opt, x, aspect_double_idx)

    def forward(self, inputs):
        text_bert_indices, text_indices, aspect_indices, bert_segments_ids, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        #text = self.embed(text_indices)
        #text = self.text_embed_dropout(text)
        #text_out, (_, _) = self.text_lstm(text, text_len)

        #encoder_layer, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        od = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_hidden_states=None)
        encoder_layer = od['last_hidden_state']
        pooled_output = od['pooler_output']
        x = encoder_layer
        text_out = x.clone().detach()
        for layer in self.gclayers:
            x = actf(self.opt.actf)(layer(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output

class SenticGCNGLOVE(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(SenticGCNGLOVE, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        isftext = True if opt.isftext == 'True' else False
        self.gclayers = nn.ModuleList()
        for i in range(opt.nlayers):
            self.gclayers.append(GraphConvolution(opt.hidden_dim, opt.hidden_dim, isfText=isftext))
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        return position_weight(self.opt, x, aspect_double_idx, text_len, aspect_len)

    def mask(self, x, aspect_double_idx):
        return mask(self.opt, x, aspect_double_idx)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1).cpu()
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        x = self.embed(text_indices)
        text = x.clone().detach()
        for layer in self.gclayers:
            x = actf(self.opt.actf)(layer(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output

class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        # text_bert_len = torch.sum(text_bert_indices != 0, dim=-1)
        # text_bert_indices = self.squeeze_embedding(text_bert_indices, text_bert_len)
        # bert_segments_ids = self.squeeze_embedding(bert_segments_ids, text_bert_len)
        _, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits