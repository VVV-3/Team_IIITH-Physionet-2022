from src.utils.preprocessing import *
import torch
from torch.utils.data import Dataset


class PCGDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.transform  = transform
        self.signals    = list()
        self.labels     = list()
        self.outcomes   = list()
        self.idxs       = list()
        self.mel_spects = list()
        
        signals, labels, outcomes, pids = get_pcg_data(data_folder)
        signals = resample_pcg_signals(signals, 1000)
        signals = apply_bandpass_filter(signals, low_cutoff=1, high_cutoff=300)
        signals = split_signals(signals, final_len=10000, stride=2500)
        
        mel_spects = make_mel_spectrograms(signals, 1000)
        
        for patient_no in range(len(signals)):
            for signal_no in range(len(signals[patient_no][0])):
                tp = np.array([0,0])
                if labels[patient_no][1] == 1:
                    continue
                else:
                    tp[0] = labels[patient_no][0]
                    tp[1] = labels[patient_no][2]
                    
                    # if tp[0] == 1:
                    #     tp = np.array([0])
                    # else:
                    #     tp = np.array([1])
                    
                self.signals.append(signals[patient_no][0][signal_no])
                self.mel_spects.append(mel_spects[patient_no][0][signal_no])
                self.labels.append(tp)
                self.outcomes.append(outcomes[patient_no])
                self.idxs.append(pids[patient_no])
        
    def mel_shape(self):
        return self.mel_spects.shape
        
    def __getitem__(self, index):
        signal, mel_spect, label, outcome, pid = self.signals[index], self.mel_spects[index], self.labels[index], self.outcomes[index], self.idxs[index]
        signal = signal.reshape(1,signal.shape[0])
        if self.transform:
            signal, label, outcome = self.transform(signal.copy()), self.transform(label), self.transform(outcome)
            signal, label, outcome = signal.type(torch.FloatTensor), label.type(torch.FloatTensor), outcome.type(torch.FloatTensor)
        return signal, mel_spect, label, outcome, pid
        
    def __len__(self):
        return len(self.signals)