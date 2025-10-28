# Domain-Specific Hyperparameters
This document contains the domain-specific hyperparameters used in BREEZE on ExORL.

### Hyperparameter Descriptions

- **$K_{\text{train}}$**: Num. rejection sampling (train)
- **$\rho_a$**: Dataset action ratio
- **$K_{\text{eval}}$**: Num. rejection sampling (eval)
- **Expectile $\tau$**: Expectile value for value function
- **$F$-reg Coef. $\omega_q$**: F-regularization coefficient
- **Diffusion Steps $T$**: Num. diffusion steps
- **Temperature $\alpha$**: Temperature balance regularization and optimization

## Full Dataset Hyperparameter Table

| Domain-Dataset | $K_{\text{train}}$ | $\rho_a$ | $K_{\text{eval}}$ | Expectile $\tau$ | $F$-reg Coef. $\omega_q$ | Diffusion Steps $T$ | Temperature $\alpha$ |
|---------------|-------------------|----------|------------------|------------------|-------------------------|---------------------|---------------------|
| Walker-RND | 9 | 0.1 | 64 | 0.99 | 0.001 | 5 | 0.05 |
| Walker-APS | 9 | 0.1 | 64 | 0.99 | 0.001 | 5 | 0.05 |
| Walker-PROTO | 9 | 0.1 | 64 | 0.99 | 0.001 | 5 | 0.05 |
| Walker-DIAYN | 9 | 0.1 | 64 | 0.99 | 0.001 | 5 | 0.05 |
| Jaco-RND | 1 | 0 | 64 | 0.99 | 0.001 | 10 | 0.05 |
| Jaco-APS | 9 | 0.1 | 64 | 0.99 | 0.0001 | 5 | 0.05 |
| Jaco-PROTO | 1 | 0 | 64 | 0.99 | 0.001 | 10 | 0.05 |
| Jaco-DIAYN | 9 | 0.1 | 64 | 0.99 | 0.0001 | 5 | 0.05 |
| Quadruped-RND | 1 | 0 | 64 | 0.99 | 0.001 | 10 | 0.05 |
| Quadruped-APS | 1 | 0 | 64 | 0.99 | 0.001 | 10 | 0.05 |
| Quadruped-PROTO | 1 | 0 | 64 | 0.99 | 0.001 | 10 | 0.05 |
| Quadruped-DIAYN | 1 | 0 | 64 | 0.99 | 0.001 | 10 | 0.05 |
| Kitchen-mixed | 1 | 0 | 16 | 0.7 | 0.001 | 5 | 0.1 |
| Kitchen-partial | 2 | 0.2 | 4 | 0.7 | 0.001 | 5 | 0.08 |


## 100k Small Sample Dataset Hyperparameter Table

| Domain-Dataset | $K_{\text{train}}$ | $\rho_a$ | $K_{\text{eval}}$ | Expectile $\tau$ | $F$-reg Coef. $\omega_q$ | Diffusion Steps $T$ |Temperature $\alpha$ |
|---------------|-------------------|----------|------------------|------------------|-------------------------|---------------------|---------------------|
| Walker-RND | 2 | 0.2 | 32 | 0.99 | 0.0001 | 10 | 0.05 |
| Walker-APS | 2 | 0.2 | 32 | 0.99 | 0.0001 | 10 | 0.05 |
| Walker-PROTO | 2 | 0.2 | 32 | 0.99 | 0.001 | 10 | 0.05 |
| Walker-DIAYN | 8 | 0.2 | 32 | 0.99 | 0.001 | 10 | 0.05 |
| Jaco-RND | 1 | 0 | 32 | 0.99 | 0.0001 | 10 | 0.05 |
| Jaco-APS | 1 | 0 | 32 | 0.99 | 0.0001 | 10 | 0.05 |
| Jaco-PROTO | 1 | 0 | 32 | 0.99 | 0.0001 | 10 | 0.05 |
| Jaco-DIAYN | 1 | 0 | 32 | 0.99 | 0.0001 | 10 | 0.05 |
| Quadruped-RND | 2 | 0.2 | 32 | 0.99 | 0.001 | 10 | 0.05 |
| Quadruped-APS | 1 | 0 | 32 | 0.99 | 0.001 | 10 | 0.05 |
| Quadruped-PROTO | 2 | 0.2 | 32 | 0.99 | 0.001 | 10 | 0.05 |
| Quadruped-DIAYN | 2 | 0.2 | 32 | 0.99 | 0.0001 | 10 | 0.05 |
---

