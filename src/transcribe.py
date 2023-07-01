import glob
import subprocess
import whisper
import sys
import wave, os

model = whisper.load_model("base")

# wav_files = []
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(parent_dir, 'ADReSSo/ADReSSo21-diagnosis-train/diagnosis/train/audio/ad/fix')
transcription_dir = os.path.join(data_dir, 'transcription')

# for filename in glob.glob(os.path.join(path, '*.wav')):
#     wav_files.append(filename)
#pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
# print(wav_files)


# Get a list of all the audio files in the "data" folder
audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

# Initialize an empty list to store the transcriptions
transcriptions = []

# Loop over all the audio files in the folder
for audio_file in audio_files:
    audio_file_path = os.path.join(data_dir, audio_file)
    result = model.transcribe(audio_file_path)
    transcription = str(result)
    transcriptions.append(transcription)
    transcription_file = os.path.join(transcription_dir, audio_file)
    with open(transcription_file, 'w') as f:
        for line in transcription:
            f.write(line)
            f.write('\n')

#print(transcriptions)