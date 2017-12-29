from config import *


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, fusion_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.fusion_size = fusion_size
        self.input_size = input_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size + fusion_size, hidden_size)

    def forward(self, input, hidden, fusion_hidden):
        embedded = self. embedding(input).view(1, 1, -1)
        output = embedded
        #print(fusion_fusion_hidden.size())
        #print(fusion_hidden.size())
        output = torch.cat((output, fusion_hidden), dim=2)
        #print("encoder:", hidden.size())
        hidden = (hidden, hidden)
        #print(self.input_size)
        output, hidden = self.lstm(output, hidden)
        return output, hidden[0]

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class FusionLSTM(nn.Module):
    def __init__(self, fusion_size, steps=3):
        super(FusionLSTM, self).__init__()
        self.steps = steps
        self.fusion_size = fusion_size

        self.lstm = nn.LSTM(fusion_size, fusion_size)
        
    def forward(self, input):
        hidden = input.view(1, 1, -1)
        hidden = (hidden, hidden)
        #print(hidden)
        for i in range(self.steps):
            #hidden = (F.relu(hidden[0]), hidden[1])
            _, hidden = self.lstm(hidden[0], hidden)
        return hidden[0]
    
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.fusion_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class MultiEncoder(nn.Module):
    def __init__(self, size_1, size_2, size_3, input_size, output_size, step=3):
        super().__init__()
        self.fusion_size = size_1 + size_2 + size_3
        self.encoder1 = EncoderLSTM(input_size, size_1, self.fusion_size)
        self.encoder2 = EncoderLSTM(input_size, size_2, self.fusion_size)
        self.encoder3 = EncoderLSTM(input_size, size_3, self.fusion_size)
        
        self.hidden_size = self.fusion_size
        self.fuser = FusionLSTM(self.fusion_size, steps=step)

        #self.combine = nn.Linear(fusion_size, 

        
        
    def forward(self, input, hidden_1, hidden_2, hidden_3):
        
        catted_hidden = torch.cat((hidden_1, hidden_2, hidden_3), dim=2)
        #print(catted_hidden.size())
        
        fusion_state = self.fuser(catted_hidden)
        out1, hidden1 = self.encoder1(input, hidden_1, fusion_state)
        out2, hidden2 = self.encoder2(input, hidden_2, fusion_state)
        out3, hidden3 = self.encoder3(input, hidden_3, fusion_state)

        return (out1, out2, out3), hidden1, hidden2, hidden3


    def initHidden(self):
        return self.encoder1.initHidden(), self.encoder2.initHidden(), self.encoder3.initHidden()










    
