import torch.nn as nn
import torch.nn.functional as F

'''
	@model: TextCNN
    @params:
        V: (int)Vocab_size
        D: (int)embedding_dim
        filter_sizes: (int list)filter sizes
        Ci: (int) input channel size
        Co: (int) output channel size
        class_nums: (int)class nums
        pre_trained: (bool)是否使用预训练的词嵌入
        pre_weigths: (tensor list)预训练的词嵌入，[tensor(V,D),...]
'''
class TextCNN(nn.Module):
    def __init__(self,V,D,filter_sizes,Ci,Co,class_nums,pre_trained=False,pre_weights=[]):
        super(TextCNN,self).__init__()
        self.class_nums = class_nums
        self.embeddings = []
        for i in range(Ci):
            self.embeddings.append(nn.Embedding(V,D))
        self.embeddings = nn.ModuleList(self.embeddings)
        if pre_trained==True:
            for i in range(len(pre_weights)):
                self.embeddings[i].weight.data.copy_(pre_weights[i])
        self.convs = []
        for i in filter_sizes:
            self.convs.append((nn.Conv2d(Ci,Co,kernel_size=(i,D))))
        self.convs = nn.ModuleList(self.convs)
        if class_nums==2:
            self.fc = nn.Linear(Co*len(filter_sizes),1)
        else:
            self.fc = nn.Linear(Co*len(filter_sizes),class_nums)
        self.dropout = nn.Dropout()
    
    #seq.shape: (Batch,Len)
    def forward(self,seq):
        seq_embs = [emb(seq).unsqueeze(1) for emb in self.embeddings] #shape: [(Batch,1,Len,Dim),...]
        seq_emb = torch.cat(seq_embs,1) # shape: (Batch,Ci,Len,Dim)
        conv_outputs = [F.relu(conv(seq_emb)).squeeze(3) for conv in self.convs] #shape: [(Batch,Co,Len'),...]
        pool_outputs = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in conv_outputs] #shape: [(Batch,Co),...]
        x = torch.cat(pool_outputs,1) #shape: [Batch,Co*len(filter_sizes)]
        x = self.dropout(x)
        if self.class_nums==2:
            return F.sigmoid(self.fc(x))
        else:
            return F.log_softmax(self.fc(x))