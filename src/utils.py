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
        # calculate daily log returns r_t = ln(P_t / P_{t-1})
        returns = np.log(data / data.shift(1)).dropna()
        if isinstance(data, pd.Series):
            data = data.to_frame(name=data.name or 'price')
        if isinstance(returns, pd.Series):
            returns = returns.to_frame(name=returns.name or 'return')
        data = data.replace([np.inf, -np.inf], np.nan).dropna(how='all')
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how='all')
        return data, returns
    except FileNotFoundError:
        print(f"error: the file at {file_path} was not found.")
        raise