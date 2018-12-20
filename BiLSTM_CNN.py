import torch.nn as nn
import torch.nn.functional as F
'''
    @model: BiLSTM
    @params:
        V: (int)vocab_size
        D: (int)embedding_dim
        hidden_dim: (int)
        filter_sizes: (int list)filter sizes
        Co: (int) output channel size
        num_layers: (int)LSTM stack的层数
        bidirecitional: (bool)是否使用双向的lstm
        class_nums: 类别个数
'''
class BiLSTM_CNN(nn.Module):
    def __init__(self,V,D,hidden_dim,filter_sizes,Co,num_layers=1,bi=True,class_nums=2):
        super(BiLSTM_CNN,self).__init__() 
        self.class_nums = class_nums
        self.embedding = nn.Embedding(V,D)
        self.lstm_encoder = nn.LSTM(D,hidden_dim,num_layers=num_layers,bidirectional=bi) #output shape: (len,batch,num_directions*hidden_dim)
        
        # cnn
        self.convs = []
        for i in filter_sizes:
            self.convs.append((nn.Conv2d(1,Co,kernel_size=(i,hidden_dim*((lambda x:2 if x==True else 1)(bi))))))
        self.convs = nn.ModuleList(self.convs)
        self.dropout = nn.Dropout()
        
        # 输出层
        if class_nums==2:
            self.classifier = nn.Linear(Co*len(filter_sizes),1)
        else:
            self.classifier = nn.Linear(Co*len(filter_sizes),class_nums)

    def forward(self, seq):
        hiddens, _ = self.lstm_encoder(self.embedding(seq)) #hiddens shape: (len,batch,num_directions*hidden_dim)
        hiddens = hiddens.transpose(0,1).unsqueeze(1) # shape: (batch,1,len,num_directions*hidden_dim)
        
        conv_outputs = [F.relu(conv(hiddens)).squeeze(3) for conv in self.convs] #shape: [(Batch,Co,Len'),...]
        pool_outputs = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in conv_outputs] #shape: [(Batch,Co),...]
        x = torch.cat(pool_outputs,1) #shape: [Batch,Co*len(filter_sizes)]
        x = self.dropout(x)
        preds = self.classifier(x)
        
        if self.class_nums==2:
            return F.sigmoid(preds)
        else:
            return F.log_softmax(preds)