import numpy as np
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

"""
NeuroChaoticReservoirComputer (NCRC)
=====================================
Core idea:
- A fixed random reservoir matrix W_res (like ESN) provides echo-state / fading memory.
- Instead of tanh activation, each neuron runs through the GLS chaotic map for `n` iterations.
- The reservoir state h[t] is updated as:
      z[t]  = W_res @ h[t-1]  +  W_in * u[t]       (pre-activation mix)
      h[t]  = GLS_iterate(sigmoid(z[t]), n_iter)     (chaotic activation)
- W_out is solved with ridge regression on the collected reservoir states.
- Prediction is recursive: previous output feeds back as the next input.

Why this fixes the original problem:
- Original: each timestep's GLS trace was independent → no memory between steps.
- Here: W_res @ h[t-1] carries memory of past states into the current chaotic activation,
  giving the reservoir echo-state dynamics while keeping the chaotic richness.
"""


# ---------------------------------------------------------------------------
# GLS neuron helpers
# ---------------------------------------------------------------------------

def gls_step(x: float, b: float = 0.5) -> float:
    """One step of the Generalised Lüroth Series (GLS) tent map."""
    eps = 1e-8
    x = float(np.clip(x, eps, 1 - eps))
    return (1 - x) / (1 - b) if x >= b else x / b


def gls_iterate(x: float, n: int, b: float = 0.5) -> np.ndarray:
    """Run GLS for `n` steps starting from x. Returns array of shape (n,)."""
    trace = np.empty(n)
    trace[0] = x
    for i in range(1, n):
        trace[i] = gls_step(trace[i - 1], b)
    return trace


def gls_activate(z_vec: np.ndarray, n: int, b: float = 0.5) -> np.ndarray:
    """
    Apply GLS activation to a vector of pre-activations.
    Each element z_i → sigmoid → GLS n-step trace → use LAST value as activation.
    The last value keeps the chaotic nonlinearity while staying scalar per neuron.
    """
    sig = 1.0 / (1.0 + np.exp(-z_vec))          # map to (0,1) for GLS
    return np.array([gls_iterate(s, n, b)[-1] for s in sig])


# ---------------------------------------------------------------------------
# NCRC model
# ---------------------------------------------------------------------------

class NCRC:
    """
    NeuroChaoticReservoirComputer

    Parameters
    ----------
    n_reservoir : int    – number of reservoir neurons
    n_gls       : int    – GLS iterations per neuron per timestep (≥ 10 recommended)
    b           : float  – GLS bifurcation parameter (0 < b < 1)
    spectral_r  : float  – target spectral radius of W_res (< 1 for stability)
    sparsity    : float  – fraction of W_res entries set to zero
    reg         : float  – ridge regression regularisation
    washout     : int    – initial timesteps to discard (let reservoir warm up)
    """

    def __init__(
        self,
        n_reservoir: int = 100,
        n_gls: int = 10,
        b: float = 0.5,
        spectral_r: float = 0.9,
        sparsity: float = 0.8,
        reg: float = 1e-6,
        washout: int = 10,
        seed: int = 42,
    ):
        self.n_reservoir = n_reservoir
        self.n_gls = n_gls
        self.b = b
        self.spectral_r = spectral_r
        self.sparsity = sparsity
        self.reg = reg
        self.washout = washout
        self.rng = np.random.default_rng(seed)
        self._build_reservoir()

    # ------------------------------------------------------------------
    def _build_reservoir(self):
        N = self.n_reservoir

        # Sparse random reservoir matrix, rescaled to desired spectral radius
        W = self.rng.standard_normal((N, N))
        mask = self.rng.random((N, N)) < self.sparsity
        W[mask] = 0.0
        eigvals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigvals))
        if rho > 1e-10:
            W = W * (self.spectral_r / rho)
        self.W_res = W

        # Input weights: each reservoir neuron gets a random signed input weight
        self.W_in = self.rng.uniform(-1, 1, (N,))

    # ------------------------------------------------------------------
    def _run_reservoir(self, inputs: np.ndarray) -> np.ndarray:
        """
        Drive the reservoir with a 1-D input sequence.
        Returns state matrix of shape (T, n_reservoir).
        """
        T = len(inputs)
        N = self.n_reservoir
        states = np.zeros((T, N))
        h = np.zeros(N)

        for t, u in enumerate(inputs):
            z = self.W_res @ h + self.W_in * float(u)
            h = gls_activate(z, self.n_gls, self.b)   # chaotic activation
            states[t] = h

        return states

    # ------------------------------------------------------------------
    def fit(self, train_X: np.ndarray, train_y: np.ndarray):
        """
        Train W_out via ridge regression.

        train_X : (T,)  input sequence (scaled to [0,1])
        train_y : (T,)  target sequence (next-step values)
        """
        states = self._run_reservoir(train_X)           # (T, N)

        # Discard washout steps
        w = self.washout
        S = states[w:]                                  # (T-w, N)
        # Prepend bias column
        S_bias = np.hstack([np.ones((S.shape[0], 1)), S])   # (T-w, N+1)
        Y = train_y[w:].reshape(-1, 1)                  # (T-w, 1)

        # Ridge regression: W_out = (Y^T S)(S^T S + λI)^{-1}
        reg_matrix = self.reg * np.eye(S_bias.shape[1])
        self.W_out = (Y.T @ S_bias) @ linalg.inv(S_bias.T @ S_bias + reg_matrix)
        # shape: (1, N+1)

        # Keep last reservoir state for seeding prediction
        self._last_state = states[-1].copy()
        return self

    # ------------------------------------------------------------------
    def predict(self, seed_input: float, test_len: int) -> np.ndarray:
        """
        Recursive prediction starting from a single seed value.

        seed_input : float  last known (scaled) value to prime the reservoir
        test_len   : int    number of steps to predict
        """
        N = self.n_reservoir
        h = self._last_state.copy()
        Y = np.zeros(test_len)
        u = float(seed_input)

        for t in range(test_len):
            z = self.W_res @ h + self.W_in * u
            h = gls_activate(z, self.n_gls, self.b)
            h_bias = np.concatenate([[1.0], h])
            y = float(self.W_out @ h_bias)
            Y[t] = y
            u = y          # feed prediction back as next input

        return Y


# ---------------------------------------------------------------------------
# Benchmark data – logistic map (chaotic)
# ---------------------------------------------------------------------------

def logistic_map(a: float = 3.95, x0: float = 0.5422, length: int = 1000) -> np.ndarray:
    x = np.zeros(length)
    x[0] = x0
    for t in range(length - 1):
        x[t + 1] = a * x[t] * (1 - x[t])
    return x


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(43)

    data = logistic_map()
    train_len = 200
    test_len  = 100

    train_raw = data[:train_len]
    test_raw  = data[train_len: train_len + test_len]

    # Targets: one-step-ahead
    train_X_raw = train_raw[:-1]
    train_y_raw = train_raw[1:]
    test_y_raw  = test_raw         # ground truth for comparison

    # Scale to [0, 1]  (required for GLS)
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    train_X = scaler.fit_transform(train_X_raw.reshape(-1, 1)).flatten()
    train_y = scaler.transform(train_y_raw.reshape(-1, 1)).flatten()
    seed_val = scaler.transform([[train_raw[-1]]])[0, 0]   # last training point

    # ------- build & train -------
    model = NCRC(
        n_reservoir = 200,
        n_gls       = 10,       # ← your required 10 GLS iterations
        b           = 0.5,
        spectral_r  = 0.95,
        sparsity    = 0.7,
        reg         = 1e-6,
        washout     = 20,
        seed        = 7,
    )
    model.fit(train_X, train_y)

    # ------- predict -------
    y_pred_scaled = model.predict(seed_val, test_len)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # ------- metrics -------
    min_len = min(len(test_y_raw), len(y_pred))
    mse = mean_squared_error(test_y_raw[:min_len], y_pred[:min_len])
    print(f"MSE : {mse:.6f}")

    # ------- plot -------
    plt.figure(figsize=(12, 4))
    plt.plot(test_y_raw[:min_len], label="Ground truth", linewidth=1.5)
    plt.plot(y_pred[:min_len],     label="NCRC prediction", linewidth=1.5, linestyle="--")
    plt.title(f"NeuroChaoticReservoirComputer — Logistic Map  (MSE = {mse:.4e})")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/ncrc_prediction.png", dpi=150)
    plt.show()
    print("Plot saved.")