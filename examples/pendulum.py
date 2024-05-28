from pontryagin import run

# Learn to swing up the pendulum with off-trajectoy open-loop RL
run(
    seed=42,
    env='cartpole',
    method='rls',   # Off-trajectory open-loop RL
    u0=(100, 0.1),  # Initial action sequence (horizon: 100, std: 0.1)
    steps=20_000,   # Number of optimization steps
    monitor=100,    # Monitoring interval (plot progress every 100 steps)
    checkpoint=False,
    plot=True,
    method_kwargs=dict(
        lr=0.001,   # Step size
        std=0.001,  # Noise scale
        s0=0.001,   # Initial precision
        alpha=0.8   # Forgetting factor
    )
)
