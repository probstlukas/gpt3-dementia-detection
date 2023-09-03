import logging
import os
import sys
import typing
from pathlib import Path
from utils.formatter import Formatter
import tiktoken

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(Formatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)

dirname = Path(__file__).parent.resolve()

# Transcription -----------------------------------------------------------------
# Set by user in CLI transcription prompt
whisper_model_name = None
whisper_model = None

# Embedding ---------------------------------------------------------------------

max_tokens = 500
embedding_engine = 'text-embedding-ada-002'

# Classification ----------------------------------------------------------------

# Number of splits for the K-Fold CV
n_splits = 10

# Data --------------------------------------------------------------------------

# Root data dir
data_dir = (dirname / "ADReSSo").resolve()

# Diagnosis task
diagnosis_train_data = (
        data_dir / "diagnosis-train" / "diagnosis" / "train" / "audio").resolve()  # Dementia and control group
diagnosis_test_data = (
        data_dir / "diagnosis-test" / "diagnosis" / "test-dist" / "audio").resolve()  # Test data
diagnosis_train_scores = (
        data_dir / "diagnosis-train" / "diagnosis" / "train" / "adresso-train-mmse-scores.csv").resolve()
empty_test_results_file = (data_dir / "diagnosis-test" / "diagnosis" / "test-dist" / "test_results_task1.csv").resolve()
test_results_task1 = (data_dir / "task1.csv").resolve()

# MMSE score prediction task ----------------------------------------------------
# Progression
# (Decline is defined as a difference in MMSE score between baseline and year-2 greater than or equal to 5 points.)
decline_data = (
        data_dir / "progression-train" / "progression" / "train" / "audio" / "decline").resolve()  # Baseline data from patients who exhibited
# cognitive decline between their baseline assessment and their year-2 visit to the clinic.
no_decline_data = (
        data_dir / "progression-train" / "progression" / "train" / "audio" / "no_decline").resolve()  # Speech from patients with no decline
# during that period.

"""
Ignore diagnosis-test & progression-test folders for now as this is for MMSE (Mini-Mental-Status-Examination) score 
prediction, which should result in a predicted score instead of a classification of each transcript.
"""

# processed
transcription_dir = (dirname / "processed" / "transcription").resolve()
diagnosis_train_transcription_dir = (transcription_dir / "train").resolve()
diagnosis_test_transcription_dir = (transcription_dir / "test").resolve()
train_scraped_path = (dirname / "processed" / "train_scraped.csv").resolve()
test_scraped_path = (dirname / "processed" / "test_scraped.csv").resolve()
train_embeddings_path = (dirname / "processed" / "train_embeddings.csv").resolve()
test_embeddings_path = (dirname / "processed" / "test_embeddings.csv").resolve()

# results
embedding_results_dir = (dirname / "results" / "embedding").resolve()
models_size_file = (embedding_results_dir / 'embedding_models_size.csv').resolve()


def set_up():
    ### Embedding
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    logger.info("Loading cl100k_base tokenizer...")
    logger.info(f"Max tokens per embedding: {max_tokens}.")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    logger.info(f"Loading GPT embedding engine {embedding_engine}...")

    # Create folder structure if some folders are missing
    Path(dirname / "processed").resolve().mkdir(exist_ok=True)
    Path(dirname / "results").resolve().mkdir(exist_ok=True)
    embedding_results_dir.mkdir(exist_ok=True)
    Path(embedding_results_dir / 'plots').resolve().mkdir(exist_ok=True)

    return tokenizer


def secret_key() -> typing.Optional[str]:
    # Get secret API key to use OpenAI functions from environment variables
    value = os.environ.get('OPENAI_API_KEY', None)

    if not value:
        logger.warning("Optional environment variable 'OPENAI_API_KEY' is missing.")

    return value
