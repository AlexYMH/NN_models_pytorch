'''
    @model: BiLSTM
    @params:
        V: (int)vocab_size
        D: (int)embedding_dim
        hidden_dim: (int)
        num_layers: (int)LSTM stack的层数
        bidirecitional: (bool)是否使用双向的lstm
        class_nums: 类别个数
        method: (str)怎样得到context vector,
                四种方法：
                    ‘last’: 只取lstm最后的一个hidden作为context vector
                    'max_pool': 对lstm的所有hiddens进行max_pooling，得到context vector
                    ‘avg_pool’: 对lstm的所有hiddens进行avg_pooling，得到context vector
                    ‘sum': 'max_pool'+’avg_pool'
                一般效果排序：‘max_pool’>'avg_pool'>'sum'>last'
'''
class BiLSTM(nn.Module):
    def __init__(self,V,D,hidden_dim,num_layers=1,bi=True,class_nums=2,method='last'):
        super(BiLSTM,self).__init__() 
        self.class_nums = class_nums
        self.method = method
        self.embedding = nn.Embedding(V,D)
        self.encoder = nn.LSTM(D,hidden_dim,num_layers=num_layers,bidirectional=bi) #output shape: (len,batch,num_directions*hidden_dim)
        # 输出层
        if class_nums==2:
            self.classifier = nn.Linear(hidden_dim*((lambda x:2 if x==True else 1)(bi)),1)
        else:
            self.classifier = nn.Linear(hidden_dim*((lambda x:2 if x==True else 1)(bi)),class_nums)

    def forward(self, seq):
        hiddens, _ = self.encoder(self.embedding(seq)) #hdn shape: (len,batch,num_directions*hidden_dim)
        # 基本模型: 取最后一个hidden作为context vector
        if self.method=='last':
            feature = hiddens[-1, :, :]  # 选择最后一个hidden,shape: (batch,num_directions*hidden_dim)
        elif self.method=='max_pool':
            feature = F.max_pool1d(hiddens.transpose(0,1).transpose(1,2),hiddens.size(0)).squeeze(2) # shape: (batch,num_directions*hidden_dim)
        elif self.method=='avg_pool':
            feature = F.avg_pool1d(hiddens.transpose(0,1).transpose(1,2),hiddens.size(0)).squeeze(2) # shape: (batch,num_directions*hidden_dim)
        elif self.method=='sum':
            feature1 = F.max_pool1d(hiddens.transpose(0,1).transpose(1,2),hiddens.size(0)).squeeze(2)
            feature2 = F.avg_pool1d(hiddens.transpose(0,1).transpose(1,2),hiddens.size(0)).squeeze(2)
            feature = feature1+feature2
        
        preds = self.classifier(feature)
        if self.class_nums==2:
            return F.sigmoid(preds)
        else:
            return F.log_softmax(preds)