# Detection of dementia using Data Science

## Getting Started

### Installation
1. Get an OpenAI API Key.
2. Set environment variable 'OPENAI\_API\_KEY' depending on your OS:
    - macOS: echo "export OPENAI\_API\_KEY='your key'" | cat >> ~/.zshrc
    - Linux: echo "export OPENAI\_API\_KEY='your key'" | cat >> ~/.bashrc


### Remarks
In my case, the original ADReSSo audio files had an incompatible format. Therefore I used: 
```
for i in *.wav; do ffmpeg -i "$i" "$i"; done
find . -name '*.wav' | xargs ffmpeg -i "$i" "$i"; done
```
in the ADReSSo folder to fix the format for all .wav-files. Since we cannot reformat the files and replace them at the same time, we have to save them temporarily and replace the old files afterwards:
```
find . -name '*.wav' -exec sh -c 'mkdir -p fix && ffmpeg -i "$0" "fix/$(basename "$0")"' {} \;
```
