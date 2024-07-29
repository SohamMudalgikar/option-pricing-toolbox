import numpy as np
from scipy.stats import norm

class OptionPricingToolbox:
    def __init__(self, spot_price, strike_price, maturity, volatility, interest_rate):
        """
        Initializes the OptionPricingToolbox with the given parameters.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        maturity (float): The time to maturity of the option in years.
        volatility (float): The volatility of the underlying asset.
        interest_rate (float): The risk-free interest rate.
        """
        self.S = spot_price
        self.K = strike_price
        self.T = maturity
        self.sigma = volatility
        self.r = interest_rate

    def black_scholes_european_call(self, spot_price=None, strike_price=None, maturity=None, volatility=None, interest_rate=None):
        """
        Calculates the price of a European call option using the Black-Scholes formula.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        maturity (float): The time to maturity of the option in years.
        volatility (float): The volatility of the underlying asset.
        interest_rate (float): The risk-free interest rate.

        Returns:
        float: The price of the European call option.
        """
        S = spot_price if spot_price is not None else self.S
        K = strike_price if strike_price is not None else self.K
        T = maturity if maturity is not None else self.T
        sigma = volatility if volatility is not None else self.sigma
        r = interest_rate if interest_rate is not None else self.r

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    def black_scholes_european_put(self, spot_price=None, strike_price=None, maturity=None, volatility=None, interest_rate=None):
        """
        Calculates the price of a European put option using the Black-Scholes formula.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        maturity (float): The time to maturity of the option in years.
        volatility (float): The volatility of the underlying asset.
        interest_rate (float): The risk-free interest rate.

        Returns:
        float: The price of the European put option.
        """
        S = spot_price if spot_price is not None else self.S
        K = strike_price if strike_price is not None else self.K
        T = maturity if maturity is not None else self.T
        sigma = volatility if volatility is not None else self.sigma
        r = interest_rate if interest_rate is not None else self.r

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price

    def monte_carlo_asian_option(self, spot_price=None, strike_price=None, maturity=None, volatility=None, interest_rate=None, num_simulations=100000, num_steps=252):
        """
        Prices an Asian option using Monte Carlo simulation.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        maturity (float): The time to maturity of the option in years.
        volatility (float): The volatility of the underlying asset.
        interest_rate (float): The risk-free interest rate.
        num_simulations (int): The number of simulation paths. Default is 100,000.
        num_steps (int): The number of time steps in each simulation. Default is 252.

        Returns:
        float: The price of the Asian option.
        """
        S = spot_price if spot_price is not None else self.S
        K = strike_price if strike_price is not None else self.K
        T = maturity if maturity is not None else self.T
        sigma = volatility if volatility is not None else self.sigma
        r = interest_rate if interest_rate is not None else self.r

        dt = T / num_steps
        drift = np.exp((r - 0.5 * sigma ** 2) * dt)
        daily_returns = np.exp(sigma * np.sqrt(dt) * np.random.normal(0, 1, (num_simulations, num_steps)))
        path = S * np.cumprod(drift * daily_returns, axis=1)

        average_price = np.mean(path, axis=1)
        payoff = np.maximum(average_price - K, 0)
        discounted_payoff = np.exp(-r * T) * payoff

        asian_option_price = np.mean(discounted_payoff)
        return asian_option_price

    def monte_carlo_barrier_option(self, barrier_level, option_type='up_and_out', spot_price=None, strike_price=None, maturity=None, volatility=None, interest_rate=None, num_simulations=100000, num_steps=252):
        """
        Prices a barrier option using Monte Carlo simulation.

        Parameters:
        barrier_level (float): The barrier level.
        option_type (str): The type of barrier option ('up_and_out' or 'down_and_out'). Default is 'up_and_out'.
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        maturity (float): The time to maturity of the option in years.
        volatility (float): The volatility of the underlying asset.
        interest_rate (float): The risk-free interest rate.
        num_simulations (int): The number of simulation paths. Default is 100,000.
        num_steps (int): The number of time steps in each simulation. Default is 252.

        Returns:
        float: The price of the barrier option.
        """
        S = spot_price if spot_price is not None else self.S
        K = strike_price if strike_price is not None else self.K
        T = maturity if maturity is not None else self.T
        sigma = volatility if volatility is not None else self.sigma
        r = interest_rate if interest_rate is not None else self.r

        dt = T / num_steps
        drift = np.exp((r - 0.5 * sigma ** 2) * dt)
        path = S * np.cumprod(drift * np.exp(sigma * np.sqrt(dt) * np.random.normal(0, 1, (num_simulations, num_steps))), axis=1)

        if option_type == 'up_and_out':
            barrier_crossed = np.any(path > barrier_level, axis=1)
            path = np.where(barrier_crossed[:, None], 0, path)

        payoff = np.maximum(path[:, -1] - K, 0)
        discounted_payoff = np.exp(-r * T) * payoff

        barrier_option_price = np.mean(discounted_payoff)
        return barrier_option_price

    def monte_carlo_lookback_option(self, option_type='floating', spot_price=None, strike_price=None, maturity=None, volatility=None, interest_rate=None, num_simulations=100000, num_steps=252):
        """
        Prices a lookback option using Monte Carlo simulation.

        Parameters:
        option_type (str): The type of lookback option ('fixed' or 'floating'). Default is 'floating'.
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        maturity (float): The time to maturity of the option in years.
        volatility (float): The volatility of the underlying asset.
        interest_rate (float): The risk-free interest rate.
        num_simulations (int): The number of simulation paths. Default is 100,000.
        num_steps (int): The number of time steps in each simulation. Default is 252.

        Returns:
        float: The price of the lookback option.
        """
        S = spot_price if spot_price is not None else self.S
        K = strike_price if strike_price is not None else self.K
        T = maturity if maturity is not None else self.T
        sigma = volatility if volatility is not None else self.sigma
        r = interest_rate if interest_rate is not None else self.r

        dt = T / num_steps
        drift = np.exp((r - 0.5 * sigma ** 2) * dt)
        path = S * np.cumprod(drift * np.exp(sigma * np.sqrt(dt) * np.random.normal(0, 1, (num_simulations, num_steps))), axis=1)

        if option_type == 'fixed':
            min_prices = np.min(path, axis=1)
            payoff = np.maximum(min_prices - K, 0)
        elif option_type == 'floating':
            max_prices = np.maximum.accumulate(path, axis=1)
            payoff = np.maximum(max_prices[:, -1] - K, 0)

        discounted_payoff = np.exp(-r * T) * payoff
        lookback_option_price = np.mean(discounted_payoff)
        return lookback_option_price

    def binomial_tree_american_option(self, option_type='call', spot_price=None, strike_price=None, maturity=None, volatility=None, interest_rate=None, num_steps=1000):
        """
        Prices an American option using the binomial tree model.

        Parameters:
        option_type (str): The type of American option ('call' or 'put'). Default is 'call'.
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        maturity (float): The time to maturity of the option in years.
        volatility (float): The volatility of the underlying asset.
        interest_rate (float): The risk-free interest rate.
        num_steps (int): The number of time steps in the binomial tree. Default is 1000.

        Returns:
        float: The price of the American option.
        """
        S = spot_price if spot_price is not None else self.S
        K = strike_price if strike_price is not None else self.K
        T = maturity if maturity is not None else self.T
        sigma = volatility if volatility is not None else self.sigma
        r = interest_rate if interest_rate is not None else self.r

        dt = T / num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        ST = np.zeros((num_steps + 1, num_steps + 1))
        for i in range(num_steps + 1):
            for j in range(i + 1):
                ST[j, i] = S * (u ** (i - j)) * (d ** j)
        
        # Initialize option values at maturity
        if option_type == 'call':
            payoff = np.maximum(ST[:, num_steps] - K, 0)
        else:
            payoff = np.maximum(K - ST[:, num_steps], 0)
        
        # Backward induction to calculate option price
        for i in range(num_steps - 1, -1, -1):
            for j in range(i + 1):
                payoff[j] = np.exp(-r * dt) * (p * payoff[j] + (1 - p) * payoff[j + 1])
                if option_type == 'call':
                    payoff[j] = np.maximum(payoff[j], ST[j, i] - K)
                else:
                    payoff[j] = np.maximum(payoff[j], K - ST[j, i])
        
        return payoff[0]
