# COM_POISSON
 adaptive filtering for com-poisson

The derivation mainly has 4 parts: 1) general adaptive filtering for COM-Poisson; 2) Fisher scoring version; 3) linear regression version and 4) adaptive filtering for generalized count distribution

Here, I showed 3 examples:

## Example 1: jump lambda + jump nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_1.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_1.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_1.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_1.png" width="400"/>

## Example 2: jump lambda + linear nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_2.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_2.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_2.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_2.png" width="400"/>


## Example 3: jump lambda + cosine nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_3.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_3.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_3.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_3.png" width="400"/>

Well, all looks good. However, the Hessian is not stable and tend to be influenced by outliers (the term log(y!) blows up quickly). Here, I first tried to use 'Fisher scoring', i.e. replace Hessian (observed information) with expected (Fisher) information, which is motivated by 'Fisher scoring' in IRLS. It seems that Fisher scoring doesn't change the results a lot.

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_fisher.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_fisher.png" width="400"/>

But the Fisher scoring matrix is still not stable... The same problem happens for initial value estimation. Basically, I initialized the value by Newton-Raphson. It's quite unstable (not robust to outlier), even when I replace Hessian with Fisher information (equivalent to IRLS)... So, for the initial value estimation, I simply delete outliers (outside [Q1 - 1.5IQR, Q3 + 1.5IQR]). But...sometimes it still give a ill-conditioned matrix....

## Thoughts & Ideas

**Approximation for moments**: A recent paper (2020), https://www.sciencedirect.com/science/article/abs/pii/S0167947317302608, when \lambda >=2 and \nu <= 1

I found this review paper is very useful: https://onlinelibrary.wiley.com/doi/10.1002/wics.1533

For interpretation, maybe we can consider to reparametrize to mu and nu, where E(Y) = mu*(Delta t)? (https://journals.sagepub.com/doi/abs/10.1177/1471082X17697749) Well, I think that doesn't add too much...

Another thing I'm thinking: Will it benefit a lot to extend COM-Poisson into zero-inflated COM-Poisson? Since as far as I know, the spikes are usually pretty sparse. Conceptually, I'm thinking 2 ways: 1) track P(Y_k = 0) directly; 2) use another linear model logit(E(Y_k = 0)) = Z*\alpha (wow, now we have a tri-linear model). Maybe way-2) is more informative, for example, we can include pre-synaptic spiking information into Z, and see the influence of presynaptic spikes onto the sparsity of post-spikes?

Finally, is it valuable to specify some form of g(y) in the generalized count distribution?





