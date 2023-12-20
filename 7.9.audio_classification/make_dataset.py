import os 
import pydub
import matplotlib.pyplot as plt
import numpy as np

voice_Mona = pydub.AudioSegment.from_file("raw_data/Mona.m4a")
voice_Mona.export("data/Mona.ogg")

voice_Khadijeh_Valipour_1 = pydub.AudioSegment.from_file("raw_data/Khadijeh Valipour.ogg")
voice_Khadijeh_Valipour_2 = pydub.AudioSegment.from_file("raw_data/Khadijeh Valipour2.ogg")
result = voice_Khadijeh_Valipour_1+voice_Khadijeh_Valipour_2
result.export("data/Khadijeh Valipour.ogg")

voice_Azra_Khedadmand_1 = pydub.AudioSegment.from_file("raw_data/Azra Khedadmand.ogg")
voice_Azra_Khedadmand_2 = pydub.AudioSegment.from_file("raw_data/Azra Khedadmand2.ogg")
result = voice_Azra_Khedadmand_1+voice_Azra_Khedadmand_2
result.export("data/Azra Khedadmand.ogg")

voice_Nima_Abdollahzadeh_1 = pydub.AudioSegment.from_file("raw_data/Nima Abdollahzadeh.ogg")
voice_Nima_Abdollahzadeh_2 = pydub.AudioSegment.from_file("raw_data/Nima Abdollahzadeh2.ogg")
result = voice_Nima_Abdollahzadeh_1+voice_Nima_Abdollahzadeh_1
result.export("data/Nima.ogg")

voice_Parsa_Dehmolai_1 = pydub.AudioSegment.from_file("raw_data/Parsa Dehmolai.ogg")
voice_Parsa_Dehmolai_2 = pydub.AudioSegment.from_file("raw_data/Parsa Dehmolai2.ogg")
result = voice_Parsa_Dehmolai_1+voice_Parsa_Dehmolai_1
result.export("data/Parsa.ogg")

names = []
files = os.listdir("data")
for file in files:
    audio = pydub.AudioSegment.from_file(os.path.join("data" , file))
    audio = audio.set_sample_width(2)
    audio = audio.set_frame_rate(48000)
    audio = audio.set_channels(1)
    chunks = pydub.silence.split_on_silence(audio, min_silence_len=2000, silence_thresh=-45)
    result = sum(chunks)
    file_name = file.split(".")[0]

    result.export("wav_data/" + file_name+".wav" , format="wav")

for file in os.listdir("wav_data"):
        audio = pydub.AudioSegment.from_file(os.path.join("wav_data", file))
        person_name = file.split(".")[0]
        os.makedirs(os.path.join("dataset",person_name), exist_ok=True)
        chunks = pydub.utils.make_chunks(audio , 1000)
        for i , chunk in enumerate(chunks):
                if len(chunk) >= 1000:
                        chunk.export(os.path.join("dataset",person_name,f"voice_{i}.wav") , format="wav")
