from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from hparams import hparams
from dataset import Tacotron2_Dataset,TextMelCollate

def get_mask_from_lengths(lengths): # tensor([20, 18]) lengths.shape: [2]
    max_len = torch.max(lengths).item() # 获取最大长度 20
    # ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)) # 填充上数字
    ids = torch.arange(0, max_len,out=torch.cuda.LongTensor(max_len))
    # ids:tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    #         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], device='cuda:0')
    mask = (ids < lengths.unsqueeze(1)).bool() #[[20],[18]] lengths.unsqueeze(1).shape: [2,1]
    return mask # 将真正的数据位置上设为True， padding的位置上设为False
    # tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
    #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
    #           True,  True,  True,  True,  True,  True,  True,  True, False, False]])
    
class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        
        # 可以根据激活函数的不同调整参数初始化的方法
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal



class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        # attention_weights_cat [B,2,T]
        processed_attention = self.location_conv(attention_weights_cat) #[B,32,T]
        processed_attention = processed_attention.transpose(1, 2)   # [B,T,32]
        processed_attention = self.location_dense(processed_attention) #[B,T,128]
        return processed_attention
        
        

class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        # 将 query 即 decoder 的输出变换维度
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
                                      
        # 将 memory 即 encoder的输出变换维度
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, # [2 1024] [2 24 128]
                               attention_weights_cat): # [2 2 24]
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))  # [B 2,1,128]
        processed_attention_weights = self.location_layer(attention_weights_cat) # [B,T,128]
        # processed_memory   [B,T,128]
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))
        # energies [B,T,1]
        energies = energies.squeeze(-1) #[B T] 2 24
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat) # [2 24]

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = torch.softmax(alignment, dim=1) # [2 24]
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) # [2 1 24] * [ 2  24 512] = [2 1 512]
        attention_context = attention_context.squeeze(1) # [ 2 512]

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, prenet_dim):
        super(Prenet, self).__init__()
        
        
        self.prenet_layer1 = nn.Sequential(
                            LinearNorm(in_dim, prenet_dim, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            )
                            
        self.prenet_layer2 = nn.Sequential(
                        LinearNorm(prenet_dim, prenet_dim, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        )

    def forward(self, x):
        out = self.prenet_layer1(x) # [119 2 256]
        out = self.prenet_layer2(out) # [119 2 256]
        return out
        
        
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, para):
        super(Postnet, self).__init__()
        
        
        self.postnet_layer_1 = nn.Sequential(
                         ConvNorm(para.n_mel_channels, para.postnet_embedding_dim,
                         kernel_size=5, stride=1,padding=2,dilation=1, w_init_gain='tanh'),
                         nn.BatchNorm1d(para.postnet_embedding_dim),
                         nn.Tanh(),
                         nn.Dropout(0.5)
                         )
        
        self.postnet_layer_2 = nn.Sequential(
                         ConvNorm(para.postnet_embedding_dim,para.postnet_embedding_dim,
                         kernel_size=5, stride=1,padding=2,dilation=1, w_init_gain='tanh'),
                         nn.BatchNorm1d(para.postnet_embedding_dim),
                         nn.Tanh(),
                         nn.Dropout(0.5)
                         )
                         
        self.postnet_layer_3 = nn.Sequential(
                         ConvNorm(para.postnet_embedding_dim,para.postnet_embedding_dim,
                         kernel_size=5, stride=1,padding=2,dilation=1, w_init_gain='tanh'),
                         nn.BatchNorm1d(para.postnet_embedding_dim),
                         nn.Tanh(),
                         nn.Dropout(0.5)
                         )
                         
                         
        self.postnet_layer_4 = nn.Sequential(
                         ConvNorm(para.postnet_embedding_dim,para.postnet_embedding_dim,
                         kernel_size=5, stride=1,padding=2,dilation=1, w_init_gain='tanh'),
                         nn.BatchNorm1d(para.postnet_embedding_dim),
                         nn.Tanh(),
                         nn.Dropout(0.5)
                         )
                         
        self.postnet_layer_5 = nn.Sequential(
                         ConvNorm(para.postnet_embedding_dim,para.n_mel_channels,
                         kernel_size=5, stride=1,padding=2,dilation=1, w_init_gain='linear'),
                         nn.BatchNorm1d(para.n_mel_channels),                         
                         nn.Dropout(0.5)
                         )
        

    def forward(self, x):
        out = self.postnet_layer_1(x)
        out = self.postnet_layer_2(out)
        out = self.postnet_layer_3(out)
        out = self.postnet_layer_4(out)
        out = self.postnet_layer_5(out)
        
        return out
        
        
# 编码部分        
class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, para):
        super(Encoder, self).__init__()

        self.Conv_layer1 = nn.Sequential(
                  ConvNorm(para.symbols_embedding_dim,
                           para.encoder_embedding_dim,
                           kernel_size=5, stride=1,
                           padding=2,dilation=1, 
                           w_init_gain='relu'),
                 nn.BatchNorm1d(para.encoder_embedding_dim),
                 nn.ReLU(inplace=True),
                 nn.Dropout(0.5))
                 
        self.Conv_layer2 = nn.Sequential(
                  ConvNorm(para.encoder_embedding_dim,
                           para.encoder_embedding_dim,
                           kernel_size=5, stride=1,
                           padding=2,dilation=1, 
                           w_init_gain='relu'),
                 nn.BatchNorm1d(para.encoder_embedding_dim),
                 nn.ReLU(inplace=True),
                 nn.Dropout(0.5)) 
                
        self.Conv_layer3 = nn.Sequential(
                  ConvNorm(para.encoder_embedding_dim,
                           para.encoder_embedding_dim,
                           kernel_size=5, stride=1,
                           padding=2,dilation=1, 
                           w_init_gain='relu'),
                 nn.BatchNorm1d(para.encoder_embedding_dim),
                 nn.ReLU(inplace=True),
                 nn.Dropout(0.5))

        self.lstm = nn.LSTM(para.encoder_embedding_dim,
                            int(para.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)
        
        
    def forward(self, x, input_lengths):
        # x 的维度为 [B,C,T]
        x = self.Conv_layer1(x) # [B C T] 2 512 30
        x = self.Conv_layer2(x)
        x = self.Conv_layer3(x)
        
        x = x.transpose(1, 2) # [B,T,C]

        # 对batch内的数据按照 input_lengths 进行压缩
        # 在进行lstm计算时，每条数据只计算 input_length 步就可以
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters() # 参数连续化，增加运算速度
        outputs, _ = self.lstm(x) # _:[2 2 256] [2 2 256]
        # pack_padded的反操作，将计算结构重新补0 对齐
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs # [B T C] [2 30 512]
    
    # 测试时只输入1条数据，不用pack padding 的步骤
    def inference(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.Conv_layer3(x)
        
        x = x.transpose(1, 2) # [B,T,C]

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


# 解码部分        
class Decoder(nn.Module):
    def __init__(self, para):
        super(Decoder, self).__init__()
        # 目标特征维度
        self.n_mel_channels = para.n_mel_channels
        
        # 每步解码 n_frames_per_step 帧特征
        self.n_frames_per_step = para.n_frames_per_step
        
        # 编码输出特征的维度, 也就是 attention-context的维度
        self.encoder_embedding_dim = para.encoder_embedding_dim
        
        # 注意力计算用 RNN 的维度
        self.attention_rnn_dim = para.attention_rnn_dim
        
        # 解码 RNN 的维度
        self.decoder_rnn_dim = para.decoder_rnn_dim
        
        # pre-net 的维度
        self.prenet_dim = para.prenet_dim
        
        # 测试过程中最多解码多少步
        self.max_decoder_steps = para.max_decoder_steps
        
        # 测试过程中 gate端 输入多少认为解码结束
        self.gate_threshold = para.gate_threshold
        
        
        # 定义Prenet
        self.prenet = Prenet(
            para.n_mel_channels * para.n_frames_per_step,
            para.prenet_dim)
        
        #  attention rnn 底层RNN
        self.attention_rnn = nn.LSTMCell(
            para.prenet_dim + para.encoder_embedding_dim,
            para.attention_rnn_dim)
        self.dropout_attention_rnn = nn.Dropout(0.1)
        
        # attention 层
        self.attention_layer = Attention(
            para.attention_rnn_dim, para.encoder_embedding_dim,
            para.attention_dim, para.attention_location_n_filters,
            para.attention_location_kernel_size) # 1024 512 128 32 31
        
        # decoder RNN 上层 RNN
        self.decoder_rnn = nn.LSTMCell(
            para.attention_rnn_dim + para.encoder_embedding_dim,
            para.decoder_rnn_dim, 1)
        self.drop_decoder_rnn = nn.Dropout(0.1)
        # 线性映射层 
        self.linear_projection = LinearNorm(
            para.decoder_rnn_dim + para.encoder_embedding_dim,
            para.n_mel_channels * para.n_frames_per_step)

        self.gate_layer = LinearNorm(
            para.decoder_rnn_dim + para.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ 
        构造一个全0的矢量作为 decoder 第一帧的输出
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input # [2 80*3]

    def initialize_decoder_states(self, memory, mask):
        
        B = memory.size(0) #2
        MAX_TIME = memory.size(1) # 20

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_()) # [2 1024]
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_()) # [2 1024]

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_()) # [2 1024]
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_()) # [2 1024]

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_()) # [B T] [2 20]
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_()) # [2 20]
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_()) # [2 512]

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory) # [2 24 128]
        self.mask = mask # [2 24]

    def parse_decoder_inputs(self, decoder_inputs): #[2 80 354]
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2) # [2 354 80]
        # (B, T_out, n_mel_channels) -> (B, T_out/3, n_mel_channels*3) 对音频特征进行变形
        decoder_inputs = decoder_inputs.reshape(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1) # [2 118 240] b t d
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs # [118 2 240] t b d

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1) #[97 2 24]->[2 97 24]
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1) #[2 97]
        gate_outputs = gate_outputs.contiguous() #[2 97]
        
        
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous() # [ 2 97 240]
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels) # [2 291 80]
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2) # [2 80 291]

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1) # decoderinput[2 256] + attentioncontext[2 512] = [2 768]
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)) # attention_hidden[2 1024] attention_cell[2 1024]
        self.attention_hidden = self.dropout_attention_rnn(self.attention_hidden)
        

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1), # [2 1 24]
             self.attention_weights_cum.unsqueeze(1)), dim=1) # [2 2 24]
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask) # attention_hidden[2 1024] memory[2 24 512] processed_memory[2 24 128] attention_weights_cat[2 2 24] mask[2 24]

        self.attention_weights_cum += self.attention_weights # [ 2 24]
        
        
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)  # [2 1024+512=1536]
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)) # decoder_hidden[2 1024], decoder_cell[2 1024]
        self.decoder_hidden = self.drop_decoder_rnn(self.decoder_hidden)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1) # [2 1536]
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context) # [ 2 240] 后处理前的mel帧 80* 3=240

        gate_prediction = self.gate_layer(decoder_hidden_attention_context) #[2 1]
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        
         mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)
        
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0) # [1 2 240(80*3)]
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs) #t b d [118(354/3) 2 240]
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0) # [119 2 240]
        decoder_inputs = self.prenet(decoder_inputs) #T B D [119 2 256]

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths)) #memory:[B2 T24 512] mask:[2 24],memory_lengths 一个batch的encoder的文本数据的真实长度 [24 18]

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1: # 当前batch的mel帧长度98-1
            decoder_input = decoder_inputs[len(mel_outputs)] # 一帧一帧输入
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input) # [2 240] [2 1] [2 T24]
            mel_outputs += [mel_output.squeeze(1)]  # [2 240]
            gate_outputs += [gate_output.squeeze(1)] # [2]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments) # [2 80 291] [2 97] [2 97 24]

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments
        
        
class Tacotron2(nn.Module):
    def __init__(self, para):
        super(Tacotron2, self).__init__()
        
        self.n_frames_per_step = para.n_frames_per_step
        self.n_mel_channels = para.n_mel_channels

        self.embedding = nn.Embedding(
            para.n_symbols, para.symbols_embedding_dim)
        std = sqrt(2.0 / (para.n_symbols + para.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(para)
        self.decoder = Decoder(para)
        self.postnet = Postnet(para)
        
    # 对最终的输出进行 mask 
    def parse_output(self, outputs, output_lengths=None):
       
        # mask = ~get_mask_from_lengths(output_lengths)
        max_len = outputs[0].size(-1)
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
        # ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
        mask = (ids < output_lengths.unsqueeze(1)).bool()
        mask = ~mask
        mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1)) # 【80 2 249】
        mask = mask.permute(1, 0, 2) #[2 80 249]

        outputs[0].data.masked_fill_(mask, 0.0) # mel_outputs
        outputs[1].data.masked_fill_(mask, 0.0) # mel_outputs_postnet
        outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate_outputs gate energies

        return outputs

    def forward(self, text_inputs,text_lengths,mels,output_lengths):
        
        # 进行 text 编码
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2) # [B D T] 2 512 20
        # 得到encoder输出
        encoder_outputs = self.encoder(embedded_inputs, text_lengths) # [2 20 512]
        
     
        # 得到 decoder 输出
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths) # encoder_outputs[2 24 512] mels[2 80 291] memory_lengths[24 18]
        # # [2 80 291] [2 97] [2 97 24]

        gate_outputs = gate_outputs.unsqueeze(2).repeat(1,1,self.n_frames_per_step) # gate部分展开和标签长度匹配 【2 97 3】
        gate_outputs = gate_outputs.view(gate_outputs.size(0),-1) # [2 291]
        
        
        # 进过postnet 得到预测的 mel 输出
        mel_outputs_postnet = self.postnet(mel_outputs) #[2 80 291]
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet # 残差

        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments],output_lengths)
            

    def inference(self, inputs): #[1 20]
        embedded_inputs = self.embedding(inputs).transpose(1, 2) # [1 512 20]
        encoder_outputs = self.encoder.inference(embedded_inputs) # [1 20 512]
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs) # [1 80 3000] [1 1000 1] [ 1 1000 20]

        mel_outputs_postnet = self.postnet(mel_outputs) #[1 80 3000]
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]

        return outputs

if __name__ == "__main__":
    
    para = hparams()
    m_Dataset = Tacotron2_Dataset(para)
    collate_fn = TextMelCollate(para.n_frames_per_step)

    m_DataLoader = DataLoader(m_Dataset,batch_size = 2,shuffle = True, num_workers = 2, collate_fn = collate_fn)
    
    m_model = Tacotron2(para)
    
    for e_epoch in range(5):

        for i,batch_samples in enumerate(m_DataLoader):
            
            text_in = batch_samples[0]
            text_lengths = batch_samples[1]
            mel_in = batch_samples[2]
            mel_lengths = batch_samples[4]
            
            outputs = m_model(text_in,text_lengths ,mel_in,mel_lengths)
            print(outputs[0]) # outputs = [mel_outputs[2 80 249], mel_outputs_postnet[2 80 249], gate_outputs[2 249], alignments[2 83 20]
            
            m_model.eval()
            eval_outputs = m_model.inference(text_in[0].unsqueeze(0))
            print(eval_outputs)
            m_model.train()
            if i>5:
                break
                    
                
            
    
    
    
    
        
        


        
        

        
        
        
        
        
