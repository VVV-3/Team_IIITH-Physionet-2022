#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from src.utils.dataset import PCGDataset
import torch
from torch.utils.data import DataLoader
from src.models.ast_models import ASTModel
from src.models.resnet import ResNet1d_length, ResNet1d_filter
from src.utils.preprocessing import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)
    
    trainset = PCGDataset(data_folder, transform=torch.from_numpy)
    trainloader = DataLoader(dataset=trainset, batch_size=64, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_config = {
                    'num_epochs':30,
                    'learning_rate':1e-4,
                    }
    
    # model2 = ASTModel(label_dim=3,input_fdim=33, input_tdim=128, imagenet_pretrain=False)
    arch_config = {
                    'n_input_channels':1,
                    'signal_length':10000,
                    'net_filter_size':[8 , 32, 64],
                    'net_signal_length':[ 5000, 500, 100],
                    'kernel_size':[65, 33, 17, 9],
                    'n_classes':2,
                    'dropout_rate':0.4
                    }

    model_murmur = ResNet1d_filter(input_dim=(arch_config['n_input_channels'], arch_config['signal_length']), 
                     blocks_dim=list(zip(arch_config['net_filter_size'], arch_config['net_signal_length'])),
                     kernel_size=arch_config['kernel_size'],
                     n_classes=arch_config['n_classes'],
                     dropout_rate=arch_config['dropout_rate'])
    model_murmur = model_murmur.to(device)
    
    
    class_weights=torch.tensor([4, 1],dtype=torch.float)
    class_weights = class_weights.to(device)
    criterion_murmur = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer_murmur = torch.optim.Adam(model_murmur.parameters(), lr=train_config['learning_rate'], weight_decay=0.005)
    
    for epoch in (range(train_config['num_epochs'])):
        model_murmur.train()
        for i, (inputs, specs, murmurs, outcomes, pid) in enumerate(trainloader): 
            
            inputs = inputs.to(device)
            labels = murmurs.to(device)

            outputs_murmur = model_murmur(inputs)
            loss_murmur    = criterion_murmur(outputs_murmur, labels)
            optimizer_murmur.zero_grad()
            loss_murmur.backward()
            optimizer_murmur.step()
        print("Murmur",epoch)
    
    
    model_outcome = ResNet1d_length(input_dim=(arch_config['n_input_channels'], arch_config['signal_length']), 
                     blocks_dim=list(zip(arch_config['net_filter_size'], arch_config['net_signal_length'])),
                     kernel_size=arch_config['kernel_size'],
                     n_classes=arch_config['n_classes'],
                     dropout_rate=arch_config['dropout_rate'])
    model_outcome = model_outcome.to(device)
    
    class_weights=torch.tensor([1.5, 1],dtype=torch.float)
    class_weights = class_weights.to(device)
    criterion_outcome = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer_outcome = torch.optim.Adam(model_outcome.parameters(), lr=train_config['learning_rate'], weight_decay=0.005)
    
    for epoch in (range(train_config['num_epochs'])):
        model_outcome.train()
        for i, (inputs, specs, murmurs, outcomes, pid) in enumerate(trainloader): 
            
            inputs = inputs.to(device)
            labels = outcomes.to(device)

            outputs_outcome = model_outcome(inputs)
            loss_outcome    = criterion_outcome(outputs_outcome, labels)
            optimizer_outcome.zero_grad()
            loss_outcome.backward()
            optimizer_outcome.step()
        print("Outcome",epoch)

    

    # Save the model.
    save_challenge_model(model_folder, murmur_classes, model_murmur, outcome_classes, model_outcome)

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):

    classes = model['murmur_classes']
    mast    = model['murmur_classifier']
    outcome_classes = model['outcome_classes']
    outcome_classifier = model['outcome_classifier']
    
    signals = [([],[])]
    for i in recordings:
        signals[0][0].append(i)
        signals[0][1].append(4000)
        
    signals = resample_pcg_signals(signals, 1000)
    signals = apply_bandpass_filter(signals, low_cutoff=1, high_cutoff=400)
    signals = split_signals(signals, final_len=10000, stride=2500)
#     mel_spects = make_mel_spectrograms(signals, 1000)
    mel_spects=signals
    # print(len(mel_spects[0][0]))
    if len(mel_spects[0][0]) > 0:
        inps = torch.from_numpy(np.array(mel_spects[0][0])).type(torch.FloatTensor)
        inps = inps.reshape((inps.shape[0],1,inps.shape[1]))
#         print(inps.shape)
    else:
        return classes+outcome_classes, np.array([0,1,0,1,0]), np.array([0,1,0,1,0])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inps = inps.to(device)
    mast = mast.to(device)
    mast.eval()
    outputs = mast(inps)
    
    
    a =np.array(outputs.detach().cpu())
    tp = np.zeros((a.shape[0],a.shape[1]+1))
    tp[:,0] = a[:,0]
    tp[:,2] = a[:,1]
    a = tp.copy()
    pp = np.argmax(a, axis = 1)
    ls=list(pp)
    probs = np.array([ls.count(0), ls.count(1), ls.count(2)])/len(ls)
    val = probs.max()
    if(probs[0]>0.4):
        val = probs[0]
    pos = np.where( probs == val)

    # Choose label with higher probability.
    labels = np.zeros(len(classes), dtype=np.int_)
    labels[pos] = 1
    murmur_probabilities = probs
    murmur_labels = labels
    
    
    outcome_classifier = outcome_classifier.to(device)
    outcome_classifier.eval()
    outputs = outcome_classifier(inps)
    
    a =np.array(outputs.detach().cpu())
    tp = np.zeros((a.shape[0],a.shape[1]))
    tp[:,0] = a[:,0]
    tp[:,1] = a[:,1]
    a = tp.copy()
    pp = np.argmax(a, axis = 1)
    ls=list(pp)
    probs = np.array([ls.count(0), ls.count(1)])/len(ls)
    val = probs.max()
    if(probs[0]>0.5):
        val = probs[0]
    pos = np.where( probs == val)

    # Choose label with higher probability.
    labels = np.zeros(len(outcome_classes), dtype=np.int_)
    labels[pos] = 1
    outcome_probabilities = probs
    outcome_labels = labels

    # Concatenate classes, labels, and probabilities.
    classes = classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))
    

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier):
    d = {'murmur_classes': murmur_classes, 'murmur_classifier': murmur_classifier, 'outcome_classes': outcome_classes, 'outcome_classifier': outcome_classifier}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)
