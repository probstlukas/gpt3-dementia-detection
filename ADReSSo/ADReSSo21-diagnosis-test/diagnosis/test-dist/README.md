
For submitting results for the diagnosis task (AD and MMSE
prediction), you will need to create a copies of
test_results_task1.csv and/or test_results_task2.csv, completing the
second column ('Prediction') with your model's predictions for each
instance:

- for the classification task, assign 0 to a non-AD (cn) subject, or 1
  to an AD subject.

- for the MMSE score prediction task, enter the predicted score.

You can submit up to 5 sets of results, but you must submit them all at once,
in separate files. For instance, if you are submitting
results for all tasks (you don't necessarily have to enter in all
tasks) you could send:

test_results-task1-1.csv, ..., test_results-task1-5.csv
and
test_results-task2-1.csv, ..., test_results-task2-5.csv


Note:

Please note that you also need to submit the ASR output
(transcription) of the audio files in case you are using a linguistic
approach for both train and test data. These folders named should be named
asr_task1, and asr_task2. If you used different ASR outputs for different models, you can submit each trancript by naming the folders as
follows:

asr_task1-1, ..., asr_results-task1-5
asr_task2-1, ..., asr_results-task2-5


We *strongly encourage* you to also share your model and code through
a publicly accessible repository (such as gitlab or github), and if
possible use a literate programming "notebook" environment such as R
Markdown or Jupyter Notebook.
