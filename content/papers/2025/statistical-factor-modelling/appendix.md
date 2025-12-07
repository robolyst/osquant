# Appendix

## Synthetic historical returns

## Orthogonal Procrustes problem

Given two matrices of the same shape $\boldsymbol{L}\_{t+1}$ and $\boldsymbol{L}\_{t}$ the orthogonal Procrustes problem is:
$$
\begin{aligned}
\boldsymbol{Q}^* &= \arg\min\_{\boldsymbol{Q}} || \boldsymbol{L}\_{t+1} \boldsymbol{Q} - \boldsymbol{L}_t ||_F^2 \\\
\text{s.t.} & \  \boldsymbol{Q}^\top \boldsymbol{Q} = \boldsymbol{I} \\\
\end{aligned}
$$

This is known as the [orthogonal Procrustes problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem) and has a closed form solution. The Frobenius norm can be written as a trace:
$$
|| \boldsymbol{L}\_{t+1} \boldsymbol{Q} - \boldsymbol{L}_t ||_F^2 =
\text{Tr} \left((\boldsymbol{L}\_{t+1} \boldsymbol{Q} - \boldsymbol{L}_t)^\top (\boldsymbol{L}\_{t+1} \boldsymbol{Q} - \boldsymbol{L}_t)\right)
$$
expanding out the trace gives:
$$
= \text{Tr}(\boldsymbol{Q}^\top \boldsymbol{L}\_{t+1}^\top \boldsymbol{L}\_{t+1} \boldsymbol{Q}) - 2 \text{Tr}(\boldsymbol{L}_t^\top \boldsymbol{L}\_{t+1} \boldsymbol{Q}) + \text{Tr}(\boldsymbol{L}_t^\top \boldsymbol{L}_t)
$$
We can use the cyclic property of traces to rewrite the first term:
$$
\text{Tr}(\boldsymbol{Q}^\top \boldsymbol{L}\_{t+1}^\top \boldsymbol{L}\_{t+1} \boldsymbol{Q}) = \text{Tr}(\boldsymbol{Q}\boldsymbol{Q}^\top \boldsymbol{L}\_{t+1}^\top \boldsymbol{L}\_{t+1})
$$
and since $\boldsymbol{Q}$ is orthonormal, $\boldsymbol{Q}\boldsymbol{Q}^\top = \boldsymbol{I}$. Therefore, the first term is constant with respect to $\boldsymbol{Q}$. The third term is also constant with respect to $\boldsymbol{Q}$. Therefore, the only $\boldsymbol{Q}$ dependent term is the middle term:
$$
|| \boldsymbol{L}\_{t+1} \boldsymbol{Q} - \boldsymbol{L}_t ||_F^2 = \text{const} - 2 \text{Tr}(\boldsymbol{L}_t^\top \boldsymbol{L}\_{t+1} \boldsymbol{Q})
$$

Now, minimising the Frobenius norm is equivalent to:
$$
\begin{aligned}
\boldsymbol{Q}^* &= \arg\max\_{\boldsymbol{Q}}  \ \text{Tr}(\boldsymbol{L}_t^\top \boldsymbol{L}\_{t+1} \boldsymbol{Q}) \\\
\text{s.t.} & \  \boldsymbol{Q}^\top \boldsymbol{Q} = \boldsymbol{I} \\\
\end{aligned}
$$
Now set $\boldsymbol{C} = \boldsymbol{L}_t^\top \boldsymbol{L}\_{t+1}$ and take the SVD of $\boldsymbol{C} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^\top$:
$$
\boldsymbol{C} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^\top
$$
The problem then becomes:
$$
\begin{aligned}
\boldsymbol{Q}^* &= \arg\max\_{\boldsymbol{Q}}  \ \text{Tr}(\boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^\top \boldsymbol{Q}) \\\
\text{s.t.} & \  \boldsymbol{Q}^\top \boldsymbol{Q} = \boldsymbol{I} \\\
\end{aligned}
$$
Using the cyclic property of traces again gives:
$$
= \text{Tr}(\boldsymbol{\Sigma} \boldsymbol{V}^\top \boldsymbol{Q} \boldsymbol{U})
$$
Now, let $\boldsymbol{W} = \boldsymbol{V}^\top \boldsymbol{Q} \boldsymbol{U}$ and write:

$$
\text{Tr}(\boldsymbol{\Sigma} \boldsymbol{W}) = \\sum_{i=1}^K \sigma_i W\_{ii}
$$
where $\sigma_i$ are the diagonal entries of $\boldsymbol{\Sigma}$. 

Since $\boldsymbol{Q}$, $\boldsymbol{U}$ and $\boldsymbol{V}$ are all orthonormal, $\boldsymbol{W}$ is also orthonormal. This means that the entries of $\boldsymbol{W}$ are bounded by $|W\_{ij}| \leq 1$. Since the singular values $\sigma_i$ are non-negative, we can maximise the trace by setting $W\_{ii} = 1$ for all $i$. If the diagonal entries of $\boldsymbol{W}$ are all 1, and the matrix is orthonormal, then $\boldsymbol{W} = \boldsymbol{I}$. Therefore:
$$
\boldsymbol{V}^\top \boldsymbol{Q} \boldsymbol{U} = \boldsymbol{I} \implies \boldsymbol{Q} = \boldsymbol{U} \boldsymbol{V}^\top
$$
Again, $\boldsymbol{U}$ and $\boldsymbol{V}$ are orthonormal, so $\boldsymbol{Q}$ is orthonormal as required and is the solution to the orthogonal Procrustes problem.

To state the whole problem:
$$
\begin{aligned}
\arg\min\_{\boldsymbol{Q}} & \ || \boldsymbol{L}\_{t+1} \boldsymbol{Q} - \boldsymbol{L}_t ||_F^2 \\\
\text{s.t.} & \  \boldsymbol{Q}^\top \boldsymbol{Q} = \boldsymbol{I} \\\
\end{aligned}
$$
The solution is to take the following SVD: 
$$
\boldsymbol{L}_t^\top \boldsymbol{L}\_{t+1} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^\top
$$
and solve:
$$
\boldsymbol{Q} = \boldsymbol{U} \boldsymbol{V}^\top
$$
