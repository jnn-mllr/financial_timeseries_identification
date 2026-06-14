# Monte Carlo Simulation Engine

This project implements a Monte Carlo simulation engine for various financial time series, based on a GARCH-Copula model.

## Time Series Identification & Methodology

The financial time series in this repository are identified and analyzed using established quantitative finance methods:
- **Univariate Analysis (GARCH):** Individual asset behaviors, such as volatility clustering and fat-tailed distributions, are identified and modeled separately using GARCH models.
- **Multivariate Dependency (Copula):** The complex, non-linear relationships and joint tail risks between different time series are captured using a Student's t-Copula.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## How to Run the Analysis

The exploratory data analysis and model selection process are documented in Jupyter notebooks inside the `notebooks` directory.

1.  **Exploratory Data Analysis:** `notebooks/01_eda.ipynb`
2.  **Model Selection and Validation:** `notebooks/02_model_selection.ipynb`
3.  **Scenario Analysis:** `03_scenario_analysis.ipynb`

To run the notebooks, you need a Jupyter environment. You can start it with:
```bash
jupyter notebook
```

## How to Run the Simulation

To run the Monte Carlo simulation and generate scenarios, run the `main.py` script:

```bash
python main.py
```

The generated scenarios will be saved in `output/scenarios.csv`.

## Model Logic

The simulation engine uses a GARCH-Copula approach to model the financial time series.

### GARCH(1,1)

Each individual time series is modeled using a GARCH(1,1) model with a Student's t-distribution for the shocks. This choice is motivated by these facts:
-   **Volatility Clustering:** GARCH models are well-suited to capture the tendency of volatility to appear in clusters.
-   **Fat Tails:** The Student's t-distribution allows for fatter tails than the normal distribution, which is a common feature of financial returns.

### Dependency Structure: Student's t-Copula

The dependency between the time series is modeled using a Student's t-copula. The copula is fitted to the standardized residuals of the GARCH models. This offers several advantages:
-   **Flexibility:** It separates the modeling of the marginal distributions from the modeling of the dependency structure.
-   **Tail Dependence:** The t-copula is capable of capturing tail dependence, meaning that extreme events are more likely to occur together. This is a crucial feature for risk management applications.

The simulation process is as follows:
1.  Generate random samples from the fitted t-copula.
2.  Use the inverse CDF of the marginal distributions (from the GARCH models) to transform the uniform copula samples into standardized residuals.
3.  Use the GARCH models to generate the simulated returns from the standardized residuals.
