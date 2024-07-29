# Import the OptionPricingToolbox class
from OptionPricingToolbox import OptionPricingToolbox

# Create an instance of the OptionPricingToolbox with initial parameters
toolbox = OptionPricingToolbox(spot_price=100, strike_price=100, maturity=1, volatility=0.2, interest_rate=0.05)

# Pricing a European Call Option using the Black-Scholes model
european_call_price = toolbox.black_scholes_european_call()
print(f"Price of the European Call Option: {european_call_price:.2f}")

# Pricing a European Put Option using the Black-Scholes model
european_put_price = toolbox.black_scholes_european_put()
print(f"Price of the European Put Option: {european_put_price:.2f}")

# Pricing an Asian Option using Monte Carlo simulation
asian_option_price = toolbox.monte_carlo_asian_option(num_simulations=50000, num_steps=100)
print(f"Price of the Asian Option: {asian_option_price:.2f}")

# Pricing a Barrier Option (up-and-out) using Monte Carlo simulation
barrier_option_price = toolbox.monte_carlo_barrier_option(barrier_level=120, option_type='up_and_out', num_simulations=50000, num_steps=100)
print(f"Price of the Up-and-Out Barrier Option: {barrier_option_price:.2f}")

# Pricing a Lookback Option (floating strike) using Monte Carlo simulation
lookback_option_price = toolbox.monte_carlo_lookback_option(option_type='floating', num_simulations=50000, num_steps=100)
print(f"Price of the Floating Strike Lookback Option: {lookback_option_price:.2f}")

# Pricing an American Call Option using the Binomial Tree model
american_call_price = toolbox.binomial_tree_american_option(option_type='call', num_steps=500)
print(f"Price of the American Call Option: {american_call_price:.2f}")
