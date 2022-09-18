import numpy as np
from hparams import hparams
import librosa
import glob
import os
import re
import xpinyin

# 提取音频特征：80维mel-fank特征
# from collections import defaultdict
def wav2feature(wav_file,para):
    wav,_ = librosa.load(wav_file,sr = None,mono = True) # wav 127680 ; 48000

    fbank = librosa.feature.melspectrogram(y=wav,
                                           sr =para.fs, # 48000
                                           n_fft = para.n_fft, # 4096
                                           win_length = para.win_length, #48000 *0.05=2400
                                           hop_length = para.hop_length, # 48000 * 0.0125 = 600
                                           n_mels=para.n_mels, # 80
                                           fmin=para.fmin, #0.0
                                           fmax=para.fmax) #24000
    #fbank:[D,T] 80,213
    log_fbank = librosa.power_to_db(fbank, ref=np.max) # [D,T] 80,213
    # power_to_db librosa中计算分贝，直接使用两个相同的物理量（例如A1和A0）之比取以10为底的对数并乘以10（也可以是20）
    return log_fbank
    
# 对wav音频处理    
def processing_wavs(wav_files,para):
    feas = [] # list 10000 音频数据
    ids = []
    for file in wav_files:
        print("processing file %s"%(file)) # '.\\dataset\\tts_BZNSYP\\wave\\000001.wav'
        id_wav = os.path.split(file)[-1][:-4] # '000001'
        fea = wav2feature(file,para) # [D,T] 80,213
        feas.append(fea)
        ids.append(id_wav)
    
    # 计算特征的 均值和方差
    fea_array = np.concatenate(feas,axis=1)  # fea的维度 D * T  [80 3419830]
    fea_mean =  np.mean(fea_array,axis=1,keepdims = True) # [80 1]
    fea_std =  np.std(fea_array,axis=1,keepdims = True) # [80 1]
    
    save_path = para.path_fea # '.\\data_fea1'
    os.makedirs(save_path,exist_ok = True)
    
    # 对所有的特征进行正则, 并保存
    for fea,id_wav in zip(feas,ids):
        norm_fea = (fea- fea_mean)/ fea_std # [80 213]
        fea_name = os.path.join(save_path,id_wav+'.npy') # '.\\data_fea1\\000001.npy'
        np.save(fea_name,norm_fea)
    
    static_name = os.path.join(save_path,'static.npy')
    np.save(static_name,np.array([fea_mean,fea_std],dtype=object))
    
'''
文本处理部分
'''

'''
文本处理部分
https://github.com/didi/athena
文本处理部分：该部分将拼音分成生母韵母两部分参考了“滴滴公司”的athena部分代码
'''
# ascii code, used to delete Chinese punctuation
CHN_PUNC_LIST = [183, 215, 8212, 8216, 8217, 8220, 8221, 8230,
    12289, 12290, 12298, 12299, 12302, 12303, 12304, 12305,
    65281, 65288, 65289, 65292, 65306, 65307, 65311]
CHN_PUNC_SET = set(CHN_PUNC_LIST)

MANDARIN_INITIAL_LIST = ["b", "ch", "c", "d", "f", "g", "h", "j",\
    "k", "l", "m", "n", "p", "q", "r", "sh", "s", "t", "x", "zh", "z"]

# prosody phone list
CHN_PHONE_PUNC_LIST = ['sp2', 'sp1', 'sil']
# erhua phoneme
CODE_ERX = 0x513F

def _update_insert_pos(old_pos, pylist):
    new_pos = old_pos + 1
    i = new_pos
    while i < len(pylist)-1:
        # if the first letter is upper, then this is the phoneme of English letter
        if pylist[i][0].isupper():
            i += 1
            new_pos += 1
        else:
            break
    return new_pos

def _pinyin_preprocess(line, words):
    if line.find('.') >= 0:
        # remove '.' in English letter phonemes, for example: 'EH1 F . EY1 CH . P IY1'
        py_list = line.replace('/', '').strip().split('.')
        py_str = ''.join(py_list)
        pinyin = py_str.split()
    else:
        pinyin = line.replace('/', '').strip().split()

    # now the content in pinyin like: ['OW1', 'K', 'Y', 'UW1', 'JH', 'EY1', 'shi4', 'yi2', 'ge4']
    insert_pos = _update_insert_pos(-1, pinyin)
    i = 0
    while i < len(words):
        if ord(words[i]) in CHN_PUNC_SET:
            i += 1
            continue
        if words[i] == '#' and (words[i+1] >= '1' and words[i+1] <= '4'):
            if words[i+1] == '1':
                pass
            else:
                if words[i+1] == '2':
                    pinyin.insert(insert_pos, 'sp2')
                if words[i+1] == '3':
                    pinyin.insert(insert_pos, 'sp2')
                elif words[i+1] == '4':
                    pinyin.append('sil')
                    break
                insert_pos = _update_insert_pos(insert_pos, pinyin)
            i += 2
        elif ord(words[i]) == CODE_ERX:
            if pinyin[insert_pos-1].find('er') != 0: # erhua
                i += 1
            else:
                insert_pos = _update_insert_pos(insert_pos, pinyin)
                i += 1
        # skip non-mandarin characters, including A-Z, a-z, Greece letters, etc.
        elif ord(words[i]) < 0x4E00 or ord(words[i]) > 0x9FA5:
            i += 1
        else:
            insert_pos = _update_insert_pos(insert_pos, pinyin)
            i += 1
    return pinyin

def _pinyin_2_initialfinal(py):
    """
    used to split pinyin into intial and final phonemes
    """
    if py[0] == 'a' or py[0] == 'e' or py[0] == 'E' or py[0] == 'o' or py[:2] == 'ng' or \
            py[:2] == 'hm':
        py_initial = ''
        py_final = py
    elif py[0] == 'y':
        py_initial = ''
        if py[1] == 'u' or py[1] == 'v':
            py_final = list(py[1:])
            py_final[0] = 'v'
            py_final = ''.join(py_final)
        elif py[1] == 'i':
            py_final = py[1:]
        else:
            py_final = list(py)
            py_final[0] = 'i'
            py_final = ''.join(py_final)
    elif py[0] == 'w':
        py_initial = ''
        if py[1] == 'u':
            py_final = py[1:]
        else:
            py_final = list(py)
            py_final[0] = 'u'
            py_final = ''.join(py_final)
    else:
        init_cand = ''
        for init in MANDARIN_INITIAL_LIST:
            init_len = len(init)
            init_cand = py[:init_len]
            if init_cand == init:
                break
        if init_cand == '':
            raise Exception('unexpected')
        py_initial = init_cand
        py_final = py[init_len:]
        if (py_initial in set(['j', 'q', 'x']) and py_final[0] == 'u'):
            py_final = list(py_final)
            py_final[0] = 'v'
            py_final = ''.join(py_final)
    if py_final[-1] == '6':
        py_final = py_final.replace('6', '2')
    return (py_initial, py_final)

def is_all_eng(words):
    #if include mandarin
    for word in words:
        if ord(word) >= 0x4E00 and ord(word) <= 0x9FA5:
            return False
    return True

def pinyin_2_phoneme(pinyin_line, words):
    #chn or chn+eng
    sent_phoneme = ['sp1']
    if not is_all_eng(words):
        sent_py = _pinyin_preprocess(pinyin_line, words)
        for py in sent_py:
            if py[0].isupper() or py in CHN_PHONE_PUNC_LIST:
                sent_phoneme.append(py)
            else:
                initial, final = _pinyin_2_initialfinal(py)
                if initial == '':
                    sent_phoneme.append(final)
                else:
                    sent_phoneme.append(initial)
                    sent_phoneme.append(final)
    else:
        wordlist = words.split(' ')
        word_phonelist = pinyin_line.strip().split('/')
        assert(len(word_phonelist) == len(wordlist))
        i = 0
        while i < len(word_phonelist):
            phone = re.split(r'[ .]', word_phonelist[i])
            for p in phone:
                if p:
                    sent_phoneme.append(p)
            if '/' in wordlist[i]:
                sent_phoneme.append('sp2')
            elif '%' in wordlist[i]:
                if i != len(word_phonelist)-1:
                    sent_phoneme.append('sp2')
                else:
                    sent_phoneme.append('sil')
            i += 1
    return ' '.join(sent_phoneme)



# 处理文本
def trans_prosody(file_trans,dic_phoneme): # dic_phoneme 在hparams.py 生成
    ## 数据
    # 000001	卡尔普#2陪外孙#1玩滑梯#4。
    # 	ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
    # 000002	假语村言#2别再#1拥抱我#4。
    # 	jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
    # 000003	宝马#1配挂#1跛骡鞍#3，貂蝉#1怨枕#2董翁榻#4。
    # 	bao2 ma3 pei4 gua4 bo3 luo2 an1 diao1 chan2 yuan4 zhen3 dong3 weng1 ta4
    # 000004	邓小平#2与#1撒切尔#2会晤#4。
    # 	deng4 xiao3 ping2 yu3 sa4 qie4 er3 hui4 wu4

    is_sentid_line = True  # 用于隔行读取 读取文本
    with open(file_trans, encoding='utf-8') as f,\
            open('biaobei_prosody.csv', 'w') as fw: # biaobei_prosody.csv 要保存到此文件
        for line in f:
            if is_sentid_line: # 隔行读取 # '000001	卡尔普#2陪外孙#1玩滑梯#4。\n'
                sent_id = line.split()[0] # '000001'
                words = line.split('\t')[1].strip() # '卡尔普#2陪外孙#1玩滑梯#4。'
            else:
                # line '	ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1\n'
                sent_phonemes = pinyin_2_phoneme(line, words) # 拼音转换成标签 'sp1 k a2 er2 p u3 sp2 p ei2 uai4 s un1 uan2 h ua2 t i1 sil'

                
                # sent_phonemes转成数字
                sent_sent_phonemes_index = ''
                for phonemes in sent_phonemes.split():
                    sent_sent_phonemes_index = sent_sent_phonemes_index+ str(dic_phoneme[phonemes]) + ' '
                    
                sent_sent_phonemes_index = sent_sent_phonemes_index + str(dic_phoneme['~'])# 添加eos
                print(sent_sent_phonemes_index) # 'sp1 k a2 er2 p u3 sp2 p ei2 uai4 s un1 uan2 h ua2 t i1 sil'
                fw.writelines('|'.join([sent_id, sent_phonemes, sent_sent_phonemes_index]) + '\n')          # # 000002|sp1 j ia2 v3 c un1 ian2 sp2 b ie2 z ai4 iong1 b ao4 uo3 si1|438 252 ... 440
            is_sentid_line = not is_sentid_line # 隔行读取

def index_unknown():
    return 0
if __name__ == "__main__":
    
    para = hparams()
    
    wavs = glob.glob(para.path_wav+'/*wav')
    # list10000 ['.\\dataset\\tts_BZNSYP\\wave\\000001.wav','.\\dataset\\tts_BZNSYP\\wave\\000002.wav',...]
    processing_wavs(wavs,para)
    file_trans = para.file_trans # '.\\dataset\\tts_BZNSYP\\ProsodyLabeling\\000001-010000.txt'
    
    # 字典文件
    vocab_file = os.path.join(para.path_scp,'vocab') # 'scp\\vocab'
   
    trans_prosody(file_trans,para.dic_phoneme)
    
    
    
    
    
    
