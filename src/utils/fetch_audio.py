import os


def fetch_audio_files(path):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.wav'):
                audio_files.append(os.path.join(root, f))
    print(f"Successfully fetched {len(audio_files)} (.wav) audio files!")
    return audio_files
