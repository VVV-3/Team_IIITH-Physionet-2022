from src.utils.helper_code import find_patient_files, load_patient_data, load_recordings, get_murmur, get_outcome, get_patient_id
from scipy.signal import decimate
from heartpy import remove_baseline_wander, filtering as hp_filtering
import numpy as np
import torch
from torch.nn.functional import normalize
import torchaudio.transforms as tat
import librosa


def get_pcg_data(data_folder):
    filenames = find_patient_files(data_folder)
    pcg_signals = list()
    
    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)
    murmurs = list()
    outcomes = list()
    patient_ids = list()
    
    for file in (filenames):
        patient_data = load_patient_data(file)
        patient_recordings = load_recordings(data_folder, patient_data, get_frequencies=True)
        pcg_signals.append(patient_recordings)
        
        # Extract labels and use one-hot encoding.
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)

        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)
        
        patient_ids.append(get_patient_id(patient_data))
        
    return pcg_signals, np.vstack(murmurs), np.vstack(outcomes), patient_ids


def resample_pcg_signals(raw_signals, resampled_freq = 1000):
    pcg_signals = list()
    for patient in raw_signals:
        tp1 = list()
        tp2 = list()
        for i in range(len(patient[0])):
            raw_signal   = patient[0][i]
            current_freq = patient[1][i]
            
            if current_freq%resampled_freq != 0:
                print("downsample factor not an integer!")
                return
        
            resampled_signal = decimate(raw_signal, int(current_freq/resampled_freq))
            tp1.append(resampled_signal)
            tp2.append(resampled_freq)
            
        pcg_signals.append((tp1,tp2))
        
    return pcg_signals


def apply_bandpass_filter(raw_signals, low_cutoff=0.25, high_cutoff=250, normalize=False):
    pcg_signals = list()
    for patient in raw_signals:
        tp1 = list()
        for i in range(len(patient[0])):
            raw_signal   = patient[0][i]
            current_freq = patient[1][i]
            
            # filtered_signal = remove_baseline_wander(raw_signal, cutoff = low_cutoff, sample_rate = current_freq)
            filtered_signal = hp_filtering.filter_signal(raw_signal, cutoff = low_cutoff,
                                                         sample_rate = current_freq, filtertype='highpass')
            filtered_signal = hp_filtering.filter_signal(filtered_signal, cutoff = high_cutoff,
                                                         sample_rate = current_freq, filtertype='lowpass')
            if normalize:
                filtered_signal = filtered_signal/max(abs(filtered_signal))
            
            tp1.append(filtered_signal)
            
        pcg_signals.append((tp1,patient[1]))
        
    return pcg_signals

def split_signals(raw_signals, final_len=10000, stride=2500, pad=False):
    pcg_signals = list()
    for patient in raw_signals:
        tp1 = list()
        for i in range(len(patient[0])):
            raw_signal   = patient[0][i]
            
            cur_pos=0
            ttl_len = len(raw_signal)
            while cur_pos<ttl_len:
                if (cur_pos+final_len) > ttl_len:
                    break
                splt = raw_signal[cur_pos:cur_pos+final_len]
                splt = splt/splt.max()
                tp1.append(splt)
                cur_pos += stride
            
#             if pad:
#                 raw_signal += [0]*((final_len - len(raw_signal)%final_len)%final_len)
                
#             raw_signal = raw_signal[: (len(raw_signal) - len(raw_signal)%final_len)]
#             for j in range(int(len(raw_signal)/final_len)):
#                 tp1.append(raw_signal[j*final_len:(j+1)*final_len])
            
        pcg_signals.append((tp1,[patient[1][0]]*len(tp1)))
    return pcg_signals
        
def make_mel_spectrograms(raw_signals, sampling_rate=1000):
    pcg_signals = list()
    mel_spectrogram_transform = tat.MelSpectrogram(sampling_rate, f_min=1, f_max=400, n_mels=64)
    # mel_spectrogram_transform = tat.Spectrogram()
    n_fft = 1024
    win_length = None
    hop_length = 500
    n_mels = 512
    n_mfcc = 256

    mfcc_transform = tat.MFCC(
        sample_rate=1000,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },
    )
    for patient in raw_signals:

        tp1 = list()
        for i in range(len(patient[0])):
            raw_signal      = patient[0][i]
            mel_spect  = mel_spectrogram_transform(torch.from_numpy(raw_signal.copy()).type(torch.FloatTensor))
            mel_spect  = librosa.power_to_db(mel_spect)
            
            fin = torch.from_numpy(mel_spect.copy()).type(torch.FloatTensor)
            fin = fin/fin.max()
            # fin = mel_spect
            # fin  = normalize(fin, p=0)
            tp1.append(fin)
        pcg_signals.append((tp1,patient[1]))
        
    return pcg_signals
