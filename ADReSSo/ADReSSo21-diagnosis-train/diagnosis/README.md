# This directory contains the training data for the [IS2021 ADReSSo Challenge](https://edin.ac/3p1cyaI).

The data are in the following directories structure and files:

```
diagnosis
└── train
    ├── audio
    │   ├── ad
    │   └── cn
    └── segmentation
        ├── ad
        └── cn
```

They contain the enhanced, volume normalised audio data for the
diagnosis and MMSE score prediction tasks, and a table of MMSE scores
for model training (adresso-train-mmse-scores.csv). The abbreviation
'cn' denotes 'control' patients, and 'ad' patients with a (probable)
Alzheimer's dementia diagnosis.

Also included are the utterance segmentation files (diarisation), in
CSV format. These files are for those who choose to do the segmented
prediction sub-task. The segmented prediction and speech-only
sub-tasks will be assessed separately.

