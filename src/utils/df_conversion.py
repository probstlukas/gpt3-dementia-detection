from config import logger


def df_to_csv(df, file_path):
    """
    Save a DataFrame to a CSV file.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be saved.
        file_path (str): The path to the CSV file.

    Returns:
        None
    """
    # Save the DataFrame to a CSV file without including the index column
    df.to_csv(file_path, index=False)
    logger.info(f"Writing {file_path}...")