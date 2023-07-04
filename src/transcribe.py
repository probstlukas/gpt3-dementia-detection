import os
import whisper


model = whisper.load_model("base")

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(parent_dir, 'ADReSSo')

# Get a list of all the audio files in the data folder
audio_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.wav'):
            audio_files.append(os.path.join(root, file))
print(f"Successfully fetched {len(audio_files)} (.wav) audio files!")


# Loop over all the audio files in the folder
for audio_file_path in audio_files:
    result = model.transcribe(audio_file_path, fp16=False)
    transcription = str(result["text"])
    transcription_dir = os.path.join(os.path.dirname(audio_file_path), 'transcription')

    # Check if the respective transcription folder exists, otherwise create it
    if not os.path.exists(transcription_dir):
        os.makedirs(transcription_dir)

    filename = os.path.splitext(os.path.basename(audio_file_path))[0]
    transcription_file = os.path.join(transcription_dir, filename)

    with open(transcription_file, 'w') as f:
        f.write(transcription)
        print(f"Transcribed {transcription_file}...")

print("-" * 32)
print("Transcription done.")
