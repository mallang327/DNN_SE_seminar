"""
    On-Fly Dataset
"""

import os, glob
import torch
import librosa
import numpy as np

class DatasetMix(torch.utils.data.Dataset):
    def __init__(self, dir_clean,path_noise,sec=1.0,device="cuda:0"):
        self.list_clean = glob.glob(os.path.join(dir_clean,"*.wav"))

        # Noise 파일 16KHz로 불러오기
        self.noise,_ = 

        # 1초
        self.len_item = int(sec*16000)
        self.device = device

        print("Dataset:: {} clean data from {} | noise : {}  ".format(len(self.list_clean),dir_clean,self.noise.shape))

    def __getitem__(self, idx):

        # clean 파일 16KHz로 불러오기
        path_clean = self.list_clean[idx]
        tmp_clean,_ = 

        # data length 맞추기
        if len(tmp_clean) > self.len_item :
            idx_clean = np.random.randint(len(tmp_clean)-self.len_item)
            tmp_clean = tmp_clean[idx_clean:idx_clean + self.len_item]
        elif len(tmp_clean) < self.len_item :
            short = self.len_item - len(tmp_clean)
            tmp_clean =  np.pad(tmp_clean, (0,short))            
        else :
            pass
        
        # noise sampling, noise length clean과 똑같이 맞추기
        idx_noise = np.random.randint(len(self.noise)-self.len_item)
        tmp_noise = self.noise[idx_noise:idx_noise+self.len_item]

        # SNR
        ## 0 ~ 10
        SNR = np.random.rand()*10

        ## DB scale로 나타내기 위해 energy 확인
        energy_clean = # tmp_clean의 energy
        energy_noise = # tmp_noise의 energy

        normal = # clean과 Noise 비율에 맞춰 Normalizing
        weight = # Hint: normal / (...)

        tmp_noise *=weight # 앞서 구한 SNR만큼의 세기로 Noise 세기 변경

        # Mixing
        noisy = tmp_clean + tmp_noise

        # Normalization
        scaling_factor = np.max(np.abs(noisy))

        noisy = noisy/scaling_factor
        clean = tmp_clean/scaling_factor

        # Type 변경 (numpy array --> torch tensor)
        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)

        # STFT Domain change (Spectrogram)
        noisy_spec =  # n_fft = 512

        noisy_mag =   # noisy_spec에서 추출
        noisy_phase = # noisy_spec에서 추출

        noisy_mag = torch.unsqueeze(noisy_mag,dim=0)
        noisy_phase = torch.unsqueeze(noisy_phase,dim=0)

        data = {}

        data["clean_wav"] = clean
        data["noisy_wav"] = noisy
        data["noisy_mag"] = noisy_mag
        data["noisy_phase"] = noisy_phase

        return data

    def __len__(self):
        return len(self.list_clean)

class DatasetFix(torch.utils.data.Dataset):
    def __init__(self, dir_dataset,device="cuda:0"):
        self.list_clean = glob.glob(os.path.join(dir_dataset,"clean","*.wav"))

        self.dir_dataset = dir_dataset
        self.device = device

        print("Dataset:: {} clean data from {}".format(len(self.list_clean),dir_dataset))

    def __getitem__(self, idx):

        # clean matching
        path_clean = self.list_clean[idx]
        clean,_ = librosa.load(path_clean,sr=16000)

        name_clean = path_clean.split("/")[-1]

        noisy,_ = # Hint: Noisy폴더 내의 name_clean 변수에 해당하는 파일을 16KHz로 load
        
        # Type 변경 (numpy array --> torch tensor)
        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)

        noisy_spec =  # DatasetMix와 동일 ## n_fft = 512
        noisy_mag = # noisy_spec에서 추출
        noisy_phase = # noisy_spec에서 추출

        noisy_mag = torch.unsqueeze(noisy_mag,dim=0)
        noisy_phase = torch.unsqueeze(noisy_phase,dim=0)

        data = {}

        data["clean_wav"] = clean
        data["noisy_wav"] = noisy
        data["noisy_mag"] = noisy_mag
        data["noisy_phase"] = noisy_phase

        return data

    def __len__(self):
        return len(self.list_clean)



