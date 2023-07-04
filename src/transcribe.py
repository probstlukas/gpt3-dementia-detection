import os
import whisper


model = whisper.load_model("base")

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(parent_dir, 'ADReSSo/ADReSSo21-diagnosis-train/diagnosis/train/audio/ad/fix')
transcription_dir = os.path.join(data_dir, 'transcription')

# Get a list of all the audio files in the data folder
audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

# Loop over all the audio files in the folder
for audio_file in audio_files:
    audio_file_path = os.path.join(data_dir, audio_file)
    result = model.transcribe(audio_file_path, fp16=False)
    transcription = str(result["text"])
    # Use file name, but change extension from .wav to .txt for transcription
    transcription_file = os.path.join(transcription_dir, os.path.splitext(audio_file)[0] + ".txt")

    with open(transcription_file, 'w') as f:
        f.write(transcription)

print("-" * 32)
print("Transcription done.")
