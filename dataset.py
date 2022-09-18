import os
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from hparams import hparams
import random

class Tacotron2_Dataset(Dataset):
    def __init__(self,para):
        self.file_scp = para.train_scp # 'scp\\train.scp'
        files = np.loadtxt(self.file_scp,dtype = 'str',delimiter = '|') # ndarray [10 3] ,10：数据总数
        # ['000001' 'sp1 k a2 er2 p u3 sp2 p ei2 uai4 s un1 uan2 h ua2 t i1 sil', '438 253 50 139 ... 439 440']
        self.file_ids = files[:,0].tolist() # list [10]
        self.index_phone = files[:,2].tolist()
        self.para = para
        
    # 读取特征
    def get_mel(self, file_id):
        file_fea = os.path.join(self.para.path_fea,file_id+'.npy')
        melspec = torch.from_numpy(np.load(file_fea))
        return melspec
    
    # 读取文本编码序列
    def get_text(self, str_phones):
        phone_ids = [int(id) for id in str_phones.split()]
        return torch.IntTensor(phone_ids)
    
    # 获取文本/特征 对
    def get_mel_text_pair(self, file_id,str_phones_ids):
        text = self.get_text(str_phones_ids)
        mel = self.get_mel(file_id)
        return (text, mel)
        
    def __getitem__(self, index):
        return self.get_mel_text_pair(self.file_ids[index],self.index_phone[index])
    
    def __len__(self):
        return len(self.file_ids)
        
        
  
        
class TextMelCollate():
    """ 
        通过补0的方法使一个 batch 内 的  text（输入） 和  mel（目标） 一样长
        对 mel 进行补0的时候 要让最长的mel 是 frames per setep 的整数倍
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step
    def __call__(self, batch):
        # 将一个batch内的数据进行补0对齐

        # Right zero-pad all one-hot text sequences to max input length
        # 文本部分一个batch内，按照降序排列 右端补0对对齐
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True) # 文本长度排序  第一个就是最大长度
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec （b, D, T）
        # 对音频特征的最大帧长进行扩展使其能够被n_frames_per_step整除（解码时每步重构的帧数 #原因就是在训练时 每次预测n_frames_per_step帧
        num_mels = batch[0][1].size(0) #mel的维度 80
        max_target_len = max([x[1].size(1) for x in batch]) # 最大的音频长度
        if max_target_len % self.n_frames_per_step != 0:  #保证特征的帧长是n_frames_per_step的整数倍
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len) # （b,d,t)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len) # (T) 构建gete 目标 判断生成什么时候结束
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1] # 进行数据填充 [D T ]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1 # 最后一帧往后都是1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths # 文本特征 文本长度 音频特征 结束gate特征 音频特征长度
            
            
if __name__ == "__main__":
    para = hparams()
    m_Dataset = Tacotron2_Dataset(para)

    collate_fn = TextMelCollate(para.n_frames_per_step)

    m_DataLoader = DataLoader(m_Dataset,batch_size = 2,shuffle = True, num_workers = 2, collate_fn = collate_fn)
    
    
    for e_epoch in range(5):
        
        for i,batch_samples in enumerate(m_DataLoader):

            print(batch_samples[0].shape) # torch.Size([2, 21]) 文本特征text_padded
            print(batch_samples[1].shape) # torch.Size([2]) 文本长度input_lengths
            print(batch_samples[2].shape) # torch.Size([2, 80, 222]) 音频特征mel_padded
            print(batch_samples[3].shape) # torch.Size([2, 222]) 结束gate特征gate_padded
            print(batch_samples[4].shape) # torch.Size([2]) 音频特征长度output_lengths
            print(batch_samples)
            
            if i>5:
                break
            
            
        
        
    
    
    
    
    