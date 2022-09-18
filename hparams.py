import torch
import os
from collections import defaultdict

def index_unknown():
    return 0
    
class hparams():
    def __init__(self):
        
        # 数据存储相关
        # self.path_wav = './dataset/tts_BZNSYP/Wave'
        # self.file_trans = './dataset/tts_BZNSYP/ProsodyLabeling/000001-010000.txt'

        self.path_wav = r'.\dataset\tts_BZNSYP\wave'
        self.file_trans = r'.\dataset\tts_BZNSYP\ProsodyLabeling\000001-010000.txt'
        
        self.path_fea = r'.\data_fea'
        self.path_scp = 'scp'
        self.train_scp = r'scp\train.scp'
        self.path_save = 'save2'
        self.test_scp = r'scp\test.scp'
        
        
        # 加载字典
        vocab_file = os.path.join(self.path_scp,'vocab')
    
        # 设置默认值为unknown = 0
        self.dic_phoneme = defaultdict(index_unknown)
        with open(vocab_file, "r", encoding="utf-8") as vocab:
            for line in vocab:
                word, index = line.split()
                index = int(index)
                self.dic_phoneme[word] = index
        
        
        
        
        # 特征提取相关
        self.fs =  48000
        self.n_fft = 4096
        self.win_length = int(self.fs*0.05)
        self.hop_length = int(self.fs*0.0125)
        self.n_mels = 80
        self.fmin = 0.0
        self.fmax = self.fs/2
        
        '''
        # 模型相关
        '''
        self.n_frames_per_step = 3 # 解码时，每步重建3帧音频特征
        
        # 文本 Embedding
        self.n_symbols = 441 # 字典内符号的数目
        self.symbols_embedding_dim = 512 # 将文本符号映射为512维的特征
        # encoder 编码部分
        self.encoder_embedding_dim = 512 # 3个 conv 层的 out—channel，以及lstm层的 out-channel
        
        #  Decoder 解码部分
        self.n_mel_channels = 80   # 目标特征的维度
        
        self.attention_rnn_dim = 1024
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000 
        
        self.gate_threshold = 0.5
        
        # attention 部分 
        self.attention_dim = 128  # 在attention计算时，将encoder输出，和decoder的输出
                                  # 都先变换到 attention_dim         
        self.attention_location_n_filters =32 # conv1d
        self.attention_location_kernel_size = 31
        
        # PosNet 部分
        self.postnet_embedding_dim = 512
        
        '''
            训练相关
        '''
        self.batch_size = 3 # 32
        self.n_epoch = 400
        self.gamma = 0.99
        self.start_lr_decay = 3000
        self.weight_decay = 1e-6
        
        self.lr = 1e-3 # 1e-5
        self.lr_final = 1e-5 # 1e-7
        self.grad_clip_thresh = 1.0
        
        self.step_save = 1500
        

        
        
        
        
        
        
        
