!pip install spleeter
!pip install pydub
import os
import numpy as np
import pydub
import librosa
import tensorflow as tf

sample_rate = 48000
def resample_au(path):

    audio, sr = librosa.load(path)
    if len(audio) < sample_rate:
        audio = np.pad(audio, (1,48000-len(audio)), "constant")
    else:
        audio = audio[:sample_rate]
    return audio

label = ['Abdollah Ramezani','Azra Khedadmand','Davood Fazeli','Maryam Saeedi','Rezaie ','kiana jhnshid','Javad Nematollahi','Matin Ghorbani ','Mohammad_prf','Mohammad','Mona','Omid nomiri ','Nima','Parisa Baqerzade','Parisa','Khadijeh Valipour','Shima Bazzazan','Sajedeh Gharabadiyan']
model = tf.keras.models.load_model("/content/drive/MyDrive/audio.h5")
ASD = resample_au('/content/drive/MyDrive/Abdollah Ramezani.ogg')
ASD = ASD.reshape(1,48000,1)
# process
preds = model(ASD)
preds = preds.cpu().numpy()
output = np.argmax(preds)