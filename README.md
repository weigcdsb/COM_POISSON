# COM_POISSON
 adaptive filtering for com-poisson

The derivation includes 3 files: 1) a initial version， 2) a more meaniful, compact version (matrix form) and 3) replace hessian (observed information) with expected (Fisher) information.

Here, I showed 3 examples:

## Example 1: jump lambda + jump nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_1.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_1.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_1.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_1.png" width="400"/>

## Example 2: jump lambda + linear nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_2.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_2.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_2.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_2.png" width="400"/>


## Example 3: jump lambda + cosine nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_3.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_3.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_3.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_3.png" width="400"/>

Well, all looks good, although lambda for example 3 is not super ideal. This may cause by the Hessian is not robust to outliers. To resolve that, I replace replace hessian (observed information) with expected (Fisher) information, which is motivated by 'Fisher scoring' in IRLS. It seems using fisher information resolves  the problem a lot:

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta1_fisher.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/theta2_fisher.png" width="400"/>

Here's one problem. Basically, I initialized the value by Newton-Raphson. However, it's quite unstable... Even I replace Hessian with Fisher information...(But there are tons of research about this, and I will do some literature review later)

For interpretation, maybe we can consider to reparametrize to mu and nu, where E(Y) = mu*(Delta t)? (https://journals.sagepub.com/doi/abs/10.1177/1471082X17697749) Well, that doesn't add too much...


