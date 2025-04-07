import os
import sys
from random import shuffle
from typing import Tuple, Union
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import utils
from sklearn.model_selection import KFold
import librosa
from scipy.signal import resample
import random
import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
n = 3
np.random.seed(n)
random.seed(n)
tf.random.set_seed(n)
check_random_state(n)

def features(X, sample_rate: float) -> np.ndarray:
    stft = np.abs(librosa.stft(X))

    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))


    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset,pitchmin, meanrms, maxrms, stdrms
    ])
    # Spectral roll-off
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=X, sr=sample_rate).T, axis=0)

    # Spectral bandwidth
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate).T, axis=0)

    # CENS
    cens = np.mean(librosa.feature.chroma_cens(y=X, sr=sample_rate).T, axis=0)


    total_mfccs = 90  # Total number of MFCCs you want to extract
    n_fft = min(2048, len(X))  # Adjust n_fft based on the length of the signal

    # Calculate the MFCCs for the entire audio file
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50, n_fft=n_fft)
    mfcc = mfcc.flatten()  # Flatten the MFCC feature

    # Resample the MFCCs to have a fixed size
    if len(mfcc) < total_mfccs:
        mfcc = np.pad(mfcc, (0, total_mfccs - len(mfcc)))  # If not enough MFCCs, pad with zeros
    else:
        mfcc = resample(mfcc, total_mfccs)  # If too many MFCCs, resample to the desired size

    ext_features = np.concatenate((ext_features, rolloff, bandwidth, cens))
    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast, mfcc))

    return ext_features


def extract_features(file: str, pad: bool = False) -> np.ndarray:
    X, sample_rate = librosa.load(file, sr=None)
    max_ = X.shape[0] / sample_rate
    if pad:
        length = (max_ * sample_rate) - X.shape[0]
        X = np.pad(X, (0, int(length)), 'constant')
    return features(X, sample_rate)

def get_max_min(files: list) -> Tuple[float]:
    min_, max_ = 100, 0

    for file in files:
        sound_file, samplerate = librosa.load(file, sr=None)
        t = sound_file.shape[0] / samplerate
        if t < min_:
            min_ = t
        if t > max_:
            max_ = t

    return max_, min_

def get_data_path(data_path: str, class_labels: list) -> list:


    wav_file_path = []

    cur_dir = os.getcwd()
    sys.stderr.write('Curdir: %s\n' % cur_dir)
    os.chdir(data_path)

    for _, directory in enumerate(class_labels):
        os.chdir(directory)

        for filename in os.listdir('.'):
            if not filename.endswith('wav'):
                continue
            filepath = os.path.join(os.getcwd(), filename)
            wav_file_path.append(filepath)

        os.chdir('..')
    os.chdir(cur_dir)

    shuffle(wav_file_path)
    return wav_file_path

def load_feature(config, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:


    feature_path = os.path.join(config.feature_folder, "train.p" if train == True else "predict.p")

    features = pd.DataFrame(
        data = joblib.load(feature_path),
        columns = ['file_name', 'features', 'emotion']
    )

    X = list(features['features'])
    Y = list(features['emotion'])


    scaler_path = os.path.join(config.checkpoint_path, 'SCALER_LIBROSA.m')


    if train == True:

        scaler = StandardScaler().fit(X)

        utils.mkdirs(config.checkpoint_path)
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)

        X = np.array(X)
        Y = np.array(Y)
        # Initialize KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Use KFold splitting for cross-validation
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            yield x_train, x_test, y_train, y_test
    else:

        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        return X

from sklearn.preprocessing import LabelEncoder

def get_data(config, data_path: str, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:

    if train == True:
        files = get_data_path(data_path, config.class_labels)
        max_, min_ = get_max_min(files)
        count = 0
        mfcc_data = []
        for file in files:
            # Use the directory name as the label
            label = os.path.basename(os.path.dirname(file))
            count+= 1
            print(count)
            features = extract_features(file, max_)
            mfcc_data.append([file, features, label])

        # Create a DataFrame from mfcc_data
        df = pd.DataFrame(mfcc_data, columns=['file', 'features', 'label'])

        # Create a label encoder
        le = LabelEncoder()

        # Fit the encoder to the labels and transform the labels into integers
        df['label'] = le.fit_transform(df['label'])

        # Convert the DataFrame into a list of tuples
        mfcc_data = list(df.itertuples(index=False, name=None))

    else:
        features = extract_features(data_path)
        mfcc_data = [[data_path, features, -1]]

    # If config.feature_folder directory doesn't exist, create it
    utils.mkdirs(config.feature_folder)
    # Path to save the features
    feature_path = os.path.join(config.feature_folder, "train.p" if train == True else "predict.p")
    # Save the features
    pickle.dump(mfcc_data, open(feature_path, 'wb'))

    return load_feature(config, train=train)
