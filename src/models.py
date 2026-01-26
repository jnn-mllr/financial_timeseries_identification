from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from arch import arch_model
from scipy import stats, optimize
from typing import Dict

class BaseModel(ABC):
    """Abstract base class for models."""
    
    @abstractmethod
    def fit(self, data: pd.Series):
        pass

class GarchModel(BaseModel):
    """GARCH(1,1) model wrapper with explicit scaling for numerical stability."""
    
    def __init__(self, p: int = 1, q: int = 1, dist: str = 't'):
        self.p = p
        self.q = q
        self.dist = dist
        self.model_result = None
        self.params = {}
        self.std_resid = None
        self.scale_factor = 100.0 # for percentage returns for better numerical stability
        self.last_vol = None
        self.last_resid = None

    def fit(self, data: pd.Series):
        """Fits GARCH on scaled data (Percentage Returns)."""
        self.fit_data = (data * self.scale_factor).dropna() # to avoid DataScaleWarning
        model = arch_model(self.fit_data, vol='Garch', p=self.p, q=self.q, dist=self.dist, rescale=False)
        self.model_result = model.fit(disp='off')
        self.params = self.model_result.params
        self.std_resid = self.model_result.std_resid
        # store the last volatility and residual for forecasts in Monte Carlo simulation
        self.last_vol = self.model_result.conditional_volatility.iloc[-1]
        self.last_resid = self.model_result.resid.iloc[-1]

    def get_params(self) -> Dict[str, float]:
        return self.params.to_dict()

class CopulaModel:
    """Student's t-Copula model with MLE for degrees of freedom."""
    
    def __init__(self):
        self.corr = None
        self.df = None
        self.fitted_copula = None

    def fit(self, std_residuals: pd.DataFrame):
        """
        Fit the copula:
        1. Transform residuals to Uniforms (PIT).
        2. Calculate Correlation matrix (Pearson on Uniforms approx).
        3. Optimize Degrees of Freedom (nu) via MLE.
        """
        # transform to uniforms with function
        uniforms = std_residuals.apply(self._to_uniform)
        
        # copula correlation matrix
        self.corr = uniforms.corr(method='pearson').to_numpy().copy()
        np.fill_diagonal(self.corr, 1.0)
        
        # find the optimal degrees of freedom for the copula
        self.df = self._optimize_nu(uniforms, self.corr)
        
        # student‑t copula in latent t‑space
        print(f"Copula df (nu): {self.df}")
        self.fitted_copula = stats.multivariate_t(shape=self.corr, df=self.df, allow_singular=True)
        return self.df
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate correlated Uniform samples U ~ [0,1]."""
        # sample copula in latent space -> uniforms
        latent_samples = self.fitted_copula.rvs(n_samples)
        uniform_samples = stats.t.cdf(latent_samples, df=self.df)
        return uniform_samples

    def _to_uniform(self, x: pd.Series) -> pd.Series:
        """Empirical CDF transform."""
        x = x.dropna()
        ranks = stats.rankdata(x, method="average") # converts the rank into percentage
        return pd.Series(ranks / (len(x) + 1), index=x.index)

    def _optimize_nu(self, uniforms: pd.DataFrame, corr_matrix: np.ndarray) -> float:
        """Find optimal degrees of freedom using MLE."""
        
        def neg_loglikelihood(uniforms: pd.DataFrame, nu: float, corr_matrix: np.ndarray) -> float:
            if nu <= 2.01:
                return np.inf # require df > 2
            # convert uniforms to t-scores
            u_data = uniforms.to_numpy()
            q = stats.t.ppf(u_data, df=nu) 
            # multivariate t distribution for joint probabilities
            mv = stats.multivariate_t(shape=corr_matrix, df=nu, allow_singular=True) 
            log_joint = mv.logpdf(q)
            # marginal t distributions for each variable
            log_marginal = stats.t.logpdf(q, df=nu).sum(axis=1)
            return -np.sum(log_joint - log_marginal)

        opt_res = optimize.minimize_scalar(
            lambda nu: neg_loglikelihood(uniforms, nu, corr_matrix),
            bounds=(2.1, 50.0), 
            method='bounded'
        )
        return float(opt_res.x)
