# COM_POISSON
 adaptive filtering for com-poisson


One strange thing for the simulation... You can see simulation in demo.m.
Basically, after filtering/ smoothing, the fitting results for parameters are:

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda.png" width="350"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu.png" width="350"/>

But when investigating the fitting for mean($Y_k$), Var($Y_k$) and mean($log(Y_K!)$). Things looks much better:
<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/meanY.png" width="250"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/varY.png" width="250"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/meanLogYfac.png" width="250"/>

Maybe we should consider reparameterize to $\mu$ and $\nu$, where $\mu = E(Y)$?

