# OptionPricingToolbox

OptionPricingToolbox is a comprehensive Python library designed for financial analysts, quantitative researchers, and students interested in the valuation of financial options. This toolbox offers robust implementations of several option pricing models, including Black-Scholes, Monte Carlo simulations, and Binomial Trees, each capable of handling different types of options like European, Asian, American, Barrier, and Lookback options.

## Features

- **Black-Scholes Model**: Price European call and put options using the analytical solution provided by the Black-Scholes formula. Ideal for options that can only be exercised at expiration.
- **Monte Carlo Simulations**: Useful for pricing exotic options where analytical solutions might not exist, such as Asian, Barrier, and Lookback options. This method uses randomness to simulate the path of the underlying asset prices.
- **Binomial Tree Model**: A versatile method for pricing American options that may be exercised early. This model uses a discrete-time framework to model the movements in the price of the underlying asset.


## Usage

Import the library and use it to calculate option prices:

```python
from OptionPricingToolbox import OptionPricingToolbox

# Example: Pricing a European call option using Black-Scholes
toolbox = OptionPricingToolbox(spot_price=100, strike_price=100, maturity=1, volatility=0.2, interest_rate=0.05)
call_price = toolbox.black_scholes_european_call()
print(f"European Call Option Price: {call_price}")
```

### Detailed Examples

Below are more detailed examples showing how to use the library to price various types of options:

- **European Put Option**:
  ```python
  put_price = toolbox.black_scholes_european_put()
  print(f"European Put Option Price: {put_price}")
  ```

- **Asian Option via Monte Carlo**:
  ```python
  asian_option_price = toolbox.monte_carlo_asian_option()
  print(f"Asian Option Price: {asian_option_price}")
  ```

- **American Option using Binomial Tree**:
  ```python
  american_option_price = toolbox.binomial_tree_american_option(option_type='call')
  print(f"American Call Option Price: {american_option_price}")
  ```

## Contributing

Contributions are welcome! To contribute to this project, please fork the repository and submit a pull request.


## References

- "Options, Futures, and Other Derivatives" by John C. Hull
- "The Concepts and Practice of Mathematical Finance" by Mark S. Joshi
