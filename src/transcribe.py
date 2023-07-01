import glob
import subprocess
import whisper
import wave, os

model = whisper.load_model("base")

# wav_files = []
path = '/Users/lukasprobst/Documents/Studium/Bachelor Informatik/HiWi TECO/TECO/ADReSSo/ADReSSo21-diagnosis-train/diagnosis/train/audio/ad'
# for filename in glob.glob(os.path.join(path, '*.wav')):
#     wav_files.append(filename)
#pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
# print(wav_files)


# Get a list of all the audio files in the "data" folder
audio_files = [f for f in os.listdir(path) if f.endswith('.wav')]

# Initialize an empty list to store the transcriptions
transcriptions = []

# Loop over all the audio files in the folder
for audio_file in audio_files:
    audio_file_path = os.path.join(path, audio_file)
    result = model.transcribe(audio_file_path)
    transcription = str(result)
    transcriptions.append(transcription)