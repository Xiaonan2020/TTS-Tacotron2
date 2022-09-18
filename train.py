import logging
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from hparams import hparams
from dataset import Tacotron2_Dataset,TextMelCollate
from model import Tacotron2
import os 
# lr指数衰减
def adjust_lr_rate(optimizer,lr,gamma,lr_final):
    lr_new = max(lr*gamma,lr_final)
    
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr_new
    return lr_new,optimizer



if __name__ == "__main__":
    
     # 定义log文件
    file_log = "Tacotron2-2.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(file_log),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()   
    
    
    
    # 定义device
    device = torch.device("cuda:0")
    
    # 获取模型参数
    para = hparams()
    
    # 模型实例化
    m_model = Tacotron2(para)
    m_model.to(device)
    
    # 定义优化器
    lr =  para.lr
    m_optimizer = torch.optim.Adam(m_model.parameters(), lr, [0.9, 0.999],weight_decay=para.weight_decay)
    
    # 定义损失函数
    fun_loss_mel_out = nn.MSELoss(reduction='sum')
    fun_loss_mel_posnet_out = nn.MSELoss(reduction='sum')
    fun_loss_gate = nn.BCEWithLogitsLoss()
    
    # 定义数据集
    m_Dataset = Tacotron2_Dataset(para)
    collate_fn = TextMelCollate(para.n_frames_per_step)
    m_DataLoader = DataLoader(m_Dataset,batch_size = para.batch_size ,shuffle = True, num_workers = 1, collate_fn = collate_fn)
    
    n_step = 0
    
    m_model.train()
    for epoch in range(para.n_epoch):
        for i, batch_samples in enumerate(m_DataLoader):
            n_step = n_step+1
            
            text_in = batch_samples[0].to(device)
            text_lengths = batch_samples[1].to(device)
        
            target_mel = batch_samples[2].to(device)
            mel_lengths = batch_samples[4].to(device)
            
            target_gate = batch_samples[3].to(device)
            
            total_size = torch.sum(mel_lengths)*para.n_mel_channels
            # 进行一步计算
            m_model.zero_grad()
            mel_outputs, mel_outputs_postnet, gate_outputs, _ = m_model(text_in,text_lengths ,target_mel,mel_lengths)
            
            loss_mel_out = fun_loss_mel_out(mel_outputs,target_mel)
            loss_mel_out_postnet = fun_loss_mel_out(mel_outputs_postnet,target_mel)
            
            target_gate = target_gate.view(-1,1)
            gate_outputs = gate_outputs.view(-1,1)
            loss_gate = fun_loss_gate(gate_outputs,target_gate)
            
          
            
            # loss_all = loss_mel_out + loss_mel_out_postnet + loss_gate
            loss_all = loss_mel_out/total_size + loss_mel_out_postnet/total_size + loss_gate

           #  m_optimizer.zero_grad()
            
            loss_all.backward()
            # 梯度正则
            grad_norm = torch.nn.utils.clip_grad_norm_(
                   m_model.parameters(), para.grad_clip_thresh)
                    
            m_optimizer.step()
            
            if n_step>para.start_lr_decay:
                lr,m_optimizer = adjust_lr_rate(m_optimizer,lr,para.gamma,para.lr_final)
                
            # log 输出
            logger.info("epoch = %04d step %8d loss_all= %f loss_mse =%f  loss_bce= %f"%(epoch, n_step,loss_all,loss_mel_out/total_size + loss_mel_out_postnet/total_size,loss_gate))
            
            # 模型保存
            if n_step%(para.step_save) ==0:
                path_save = os.path.join(para.path_save,str(n_step))
                os.makedirs(path_save,exist_ok=True)
                
                torch.save({'model':m_model.state_dict(),
                            'opt':m_optimizer.state_dict()},
                            os.path.join(path_save,'model.pick'))
            

            
            
            
            
            
    torch.save({'model':m_model.state_dict(),
                            'opt':m_optimizer.state_dict()},
                            os.path.join(path_save,'model_final.pick'))
            
            
    
    
    
    
    
    
    
    
