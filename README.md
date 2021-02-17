# COM_POISSON
 adaptive filtering for com-poisson

The derivation includes 2 files: 1) a initial version & 2) a more meaniful, compact version (matrix form)

Here, I showed 3 examples:

## Example 1: jump lambda + jump nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_1.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_1.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_1.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_1.png" width="400"/>

## Example 2: jump lambda + linear nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_2.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_2.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_2.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_2.png" width="400"/>


## Example 3: jump lambda + cosine nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_3.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_3.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_3.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_3.png" width="400"/>


Well, all looks good (although lambda for example 3 is not super ideal).
Here's one problem. Basically, I initialized the value by Newton-Raphson. However, it's quite unstable... The Hessian will be singular in some cases. Maybe we can think of using IRLS? The same problem occurs for filtering/ smoothing, if initialized with a bad value... (there are tons of research about this, we can just use them)


Another issue is that the updating of lambda is influenced by nu a lot (especially in example 3, because of bad simulation?). Maybe we can consider to reparametrize to mu and nu, where E(Y) = mu*(Delta t)? Then since mu and nu are orthogonal, that would be better. (https://journals.sagepub.com/doi/abs/10.1177/1471082X17697749)


