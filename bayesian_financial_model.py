import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors

# --- 1. Define the Bayesian Model ---
class BayesianFinancialModel:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        
        # We'll focus on learning these parameters as a proof of concept
        # 1. asset_growth (positive)
        # 2. depreciation_rate (0 to 1)
        # 3. account_receivables_pct (0 to 1)
        # 4. tax_rate (0 to 1)
        
        # Observation noise scale (also learned)
        self.noise_scale = tf.Variable(0.1, dtype=tf.float64, name="noise_scale")

    def model_prior(self):
        """Defines the prior distributions for the parameters."""
        return tfd.JointDistributionNamed({
            # Prior for asset growth (LogNormal to ensure positive)
            'asset_growth': tfd.LogNormal(loc=tf.cast(-5.0, tf.float64), scale=1.0),
            
            # Prior for depreciation rate (Beta or Logit-Normal for [0, 1])
            'depreciation_rate': tfd.Sample(tfd.Uniform(low=tf.cast(0.0, tf.float64), high=0.2)),
            
            # Prior for AR % (Uniform [0, 1])
            'ar_pct': tfd.Uniform(low=tf.cast(0.0, tf.float64), high=0.5),
            
            # Prior for Tax % (Uniform [0, 1])
            'tax_pct': tfd.Uniform(low=tf.cast(0.0, tf.float64), high=0.4)
        })

    def get_likelihood(self, params, historical_sales, historical_nca, historical_depr, historical_ar, historical_tax, historical_ni):
        """
        Calculates the likelihood of the observed data given the parameters.
        This follows the logic in trainable_financial_model.py but probabilistically.
        """
        # 1. Asset Growth: delta_nca ~ Normal(sales * asset_growth, sigma)
        delta_nca_true = historical_nca[1:] - historical_nca[:-1]
        sales_growth = historical_sales[1:]
        pred_delta_nca = sales_growth * params['asset_growth']
        
        # 2. Depreciation: depr ~ Normal(nca_prev * depr_rate, sigma)
        nca_prev = historical_nca[:-1]
        pred_depr = nca_prev * params['depreciation_rate']
        
        # 3. AR: ar ~ Normal(sales * ar_pct, sigma)
        pred_ar = historical_sales * params['ar_pct']
        
        # 4. Tax: tax ~ Normal(ni * tax_pct, sigma)
        pred_tax = historical_ni * params['tax_pct']

        # Define likelihoods for each observation type
        # We use a broad scale for simplicity, or we could learn individual scales
        likelihoods = [
            tfd.Normal(loc=pred_delta_nca, scale=self.noise_scale * tf.math.reduce_std(delta_nca_true)),
            tfd.Normal(loc=pred_depr, scale=self.noise_scale * tf.math.reduce_std(historical_depr)),
            tfd.Normal(loc=pred_ar, scale=self.noise_scale * tf.math.reduce_std(historical_ar)),
            tfd.Normal(loc=pred_tax, scale=self.noise_scale * tf.math.reduce_std(historical_tax))
        ]
        
        # Observations
        observations = [delta_nca_true, historical_depr[1:], historical_ar, historical_tax]
        
        # Sum log probabilities
        log_prob = 0.0
        for dist, obs in zip(likelihoods, observations):
            log_prob += tf.reduce_sum(dist.log_prob(obs))
            
        return log_prob

    def fit(self, sales, nca, depr, ar, tax, ni, epochs=1000):
        """Uses Variational Inference to find the posterior."""
        
        # Define the surrogate posterior (Independent Normal distributions in unconstrained space)
        # TFP handles the transformation from unconstrained to constrained space automatically
        
        # We need bijectors to map from Real -> Constrained Space
        # asset_growth > 0 (Exp)
        # others [0, 1] (Sigmoid)
        
        surrogate_posterior = tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=self.model_prior().event_shape,
            bijector={
                'asset_growth': tfb.Exp(),
                'depreciation_rate': tfb.Sigmoid(),
                'ar_pct': tfb.Sigmoid(),
                'tax_pct': tfb.Sigmoid()
            },
            dtype=tf.float64
        )

        # Define the target log-prob function (P(params | data) propto P(data | params) * P(params))
        def target_log_prob(**params):
            prior_log_prob = self.model_prior().log_prob(params)
            likelihood_log_prob = self.get_likelihood(params, sales, nca, depr, ar, tax, ni)
            return prior_log_prob + likelihood_log_prob

        # Optimize the surrogate posterior to minimize KL divergence (Maximize ELBO)
        optimizer = tf.optimizers.Adam(learning_rate=0.05)
        
        @tf.function
        def train_step():
            return tfp.vi.fit_surrogate_posterior(
                target_log_prob,
                surrogate_posterior=surrogate_posterior,
                optimizer=optimizer,
                num_steps=1)

        losses = []
        for i in range(epochs):
            loss = train_step()
            losses.append(loss)
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.numpy():.4f}")
        
        self.surrogate_posterior = surrogate_posterior
        return losses

    def sample_posterior(self, num_samples=1000):
        """Draw samples from the learned posterior."""
        return self.surrogate_posterior.sample(num_samples)

# --- 2. Example Usage ---
def run_bayesian_demo():
    # Load dummy data similar to the user's project
    sales = tf.constant([2.6e11, 2.7e11, 3.6e11, 3.9e11, 3.8e11], dtype=tf.float64)
    nca = tf.constant([1.7e11, 1.8e11, 2.1e11, 2.1e11, 2.0e11], dtype=tf.float64)
    depr = tf.constant([1.0e10, 1.1e10, 1.1e10, 1.1e10, 1.1e10], dtype=tf.float64)
    ar = tf.constant([4.5e10, 3.7e10, 5.1e10, 6.0e10, 6.0e10], dtype=tf.float64)
    tax = tf.constant([1.0e10, 0.9e10, 1.4e10, 1.9e10, 1.6e10], dtype=tf.float64)
    ni = tf.constant([5.5e10, 5.7e10, 9.4e10, 9.9e10, 9.6e10], dtype=tf.float64)

    model = BayesianFinancialModel(None)
    print("Starting Variational Inference...")
    model.fit(sales, nca, depr, ar, tax, ni, epochs=500)
    
    # Draw samples and show results
    samples = model.sample_posterior(1000)
    
    print("\nLearned Posterior Distributions (Mean +/- Std):")
    for key in samples.keys():
        mean = tf.reduce_mean(samples[key])
        std = tf.math.reduce_std(samples[key])
        print(f"{key:>20}: {mean.numpy():.4f} +/- {std.numpy():.4f}")

    # Probability Forecast Example
    print("\nBayesian Forecasting (showing uncertainty)...")
    last_sales = sales[-1]
    # Predict next year's AR with uncertainty
    future_ar_samples = last_sales * samples['ar_pct']
    mean_ar = tf.reduce_mean(future_ar_samples)
    p05 = tfp.stats.percentile(future_ar_samples, 5)
    p95 = tfp.stats.percentile(future_ar_samples, 95)
    
    print(f"Predicted AR for Next Year: {mean_ar.numpy():.2e}")
    print(f"90% Confidence Interval: [{p05.numpy():.2e}, {p95.numpy():.2e}]")

if __name__ == "__main__":
    # Check if TFP is installed before running
    try:
        import tensorflow_probability
        run_bayesian_demo()
    except ImportError:
        print("Error: tensorflow-probability is not installed.")
        print("Please install it using: pip install tensorflow-probability")

