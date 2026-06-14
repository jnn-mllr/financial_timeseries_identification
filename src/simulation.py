import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats

from src.models import GarchModel, CopulaModel

class SimulationEngine:
    """Monte Carlo simulation engine."""
    
    def __init__(self, models: Dict[str, GarchModel], copula: CopulaModel):
        self.models = models
        self.copula = copula
        self.asset_names = list(models.keys())
        self.n_assets = len(self.asset_names)
        
    def generate_scenarios(self, n_scenarios: int, n_steps: int) -> pd.DataFrame:
        """Generate Monte Carlo scenarios using the GARCH-Copula model."""
        
        # generate synthetic correlated Uniforms for all steps and scenarios form our copula
        # total samples needed = n_scenarios * n_steps
        n_samples = n_scenarios * n_steps
        uniforms_flat = self.copula.sample(n_samples) # here we solve via MLE for the copula nu - different from the asset GARCH nus
        
        # reshape for vectorization of loops (steps, scenarios, assets)
        u_t = uniforms_flat.reshape(n_scenarios, n_steps, self.n_assets).transpose(1, 0, 2)
        
        # setup parameter arrays for broadcasting
        omegas = np.zeros(self.n_assets)
        alphas = np.zeros(self.n_assets)
        betas = np.zeros(self.n_assets)
        mus = np.zeros(self.n_assets)
        scales = np.zeros(self.n_assets)
        nus = np.zeros(self.n_assets)
        
        # initial state of the vectors
        current_sigma2 = np.zeros((n_scenarios, self.n_assets))
        current_resid2 = np.zeros((n_scenarios, self.n_assets))
        
        for i, asset in enumerate(self.asset_names):
            model = self.models[asset]
            p = model.params
            # store model parameters
            omegas[i] = p['omega']
            alphas[i] = p['alpha[1]']
            betas[i] = p['beta[1]']
            mus[i] = p['mu']
            scales[i] = model.scale_factor
            nus[i] = p['nu']
            # store final values for GARCH forecasts
            current_sigma2[:, i] = model.last_vol**2
            current_resid2[:, i] = model.last_resid**2

        # pre-calc t-scores (retransform uniforms u_t to t-scores)
        z_t = np.zeros_like(u_t)
        for i in range(self.n_assets):
            z_t[:, :, i] = stats.t.ppf(u_t[:, :, i], df=nus[i])
        
        # time-stepping loop of simulation of returns
        simulated_returns = np.zeros((n_steps, n_scenarios, self.n_assets))
        for t in range(n_steps):
            
            # garch update: next_varaince = baseline_variance + alpha * last_shock^2 + beta * last_variance
            next_sigma2 = omegas + alphas * current_resid2 + betas * current_sigma2
            next_vol = np.sqrt(next_sigma2)
            
            # calc returns using the pre-calculated shocks
            z_step = z_t[t]
            scaled_ret = mus + next_vol * z_step
            simulated_returns[t] = scaled_ret / scales
            
            # update states
            current_resid2 = (next_vol * z_step)**2
            current_sigma2 = next_sigma2
        
        # save as pandas dataframes
        flat_returns = simulated_returns.transpose(1, 0, 2).reshape(-1, self.n_assets)
        scenario_idx = np.repeat(np.arange(1, n_scenarios + 1), n_steps)
        step_idx = np.tile(np.arange(1, n_steps + 1), n_scenarios)
        df = pd.DataFrame(flat_returns, columns=self.asset_names)
        df.insert(0, 'Step', step_idx)
        df.insert(0, 'Scenario', scenario_idx)
        return df
