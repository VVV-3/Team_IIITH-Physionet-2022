from src.utils.preprocessing import *
import torch
from torch.utils.data import Dataset


class PCGDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.transform  = transform
        self.signals    = list()
        self.labels     = list()
        self.idxs       = list()
        self.mel_spects = list()
        
        signals, labels, pids = get_pcg_data(data_folder)
        signals = resample_pcg_signals(signals)
        signals = apply_bandpass_filter(signals, low_cutoff=1, high_cutoff=400)
        signals = split_signals(signals, final_len=10000, stride=2500)
        
        mel_spects = make_mel_spectrograms(signals, 1000)
        
        for patient_no in range(len(signals)):
            for signal_no in range(len(signals[patient_no][0])):
                self.signals.append(signals[patient_no][0][signal_no])
                self.mel_spects.append(mel_spects[patient_no][0][signal_no])
                self.labels.append(labels[patient_no])
                self.idxs.append(pids[patient_no])
        
        
    def __getitem__(self, index):
        signal, mel_spect, label, pid = self.signals[index], self.mel_spects[index], self.labels[index], self.idxs[index]
        signal = signal.reshape(1,signal.shape[0])
        if self.transform:
            signal, label = self.transform(signal.copy()), self.transform(label)
            signal, label = signal.type(torch.FloatTensor), label.type(torch.FloatTensor)
        return signal, mel_spect, label, pid
        
    def __len__(self):
        return len(self.signals)