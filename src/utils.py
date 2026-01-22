import pandas as pd
import numpy as np
from typing import Tuple

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    loads data from a csv file and preprocesses it.
    
    args:
        file_path (str): the path to the csv file.
        
    returns:
        tuple[pd.dataframe, pd.dataframe]: a tuple containing the price data and the returns data.
    """
    try:
        data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        # calculate daily returns
        returns = data.pct_change().dropna()
        if isinstance(data, pd.Series):
            data = data.to_frame(name=data.name or 'price')
        if isinstance(returns, pd.Series):
            returns = returns.to_frame(name=returns.name or 'return')
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how='all')
        return data, returns
    except FileNotFoundError:
        print(f"error: the file at {file_path} was not found.")
        raise