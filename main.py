import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils import load_and_preprocess_data
from src.models import GarchModel, CopulaModel
from src.simulation import SimulationEngine

def main():
    """
    Main entry point for the Monte Carlo simulation.
    """
    np.random.seed(123)

    # configuration
    config = {
        'input_file': 'data.csv',
        'output_file': 'output/scenarios.csv',
        'n_scenarios': 10000,
        'n_steps': 12,  # one year
    }
        
    # load data and log-returns
    _, returns = load_and_preprocess_data(config['input_file'])
        
    # fit GARCH models
    garch_models = {}
    std_residuals = pd.DataFrame()
    for col in tqdm(returns.columns, desc="Fitting GARCH models"):
        model = GarchModel()
        model.fit(returns[col])
        garch_models[col] = model
        std_residuals[col] = model.std_resid

    # fit copula model
    copula = CopulaModel()
    copula.fit(std_residuals)

    # setup simulation engine
    engine = SimulationEngine(garch_models, copula)
    
    # generate scenarios
    scenarios = engine.generate_scenarios(
        n_scenarios=config['n_scenarios'],
        n_steps=config['n_steps']
    )
    
    # save results
    scenarios.to_csv(config['output_file'], index=False)
        
if __name__ == "__main__":
    main()
