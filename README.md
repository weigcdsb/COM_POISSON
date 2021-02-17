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


Well, looks good expect for example 3.
Here's another problem. The algorithm is very easy to get singular matrix. That means, it correct "too hard"... Maybe we can add some constraint for theta updating? Maybe that will make the W matrix meaningless...
Also, I need to mention that the initial value is set as theta_true(1,:). The same reason: arbitrary initial value may make the algorithm get singular matrix.

Maybe this is because when nu -> 0, the estimation of summation such as Z,... will shoot to inifinity.

Another issue is that the updating of lambda is influenced by nu a lot (especially in example 3, because of bad simulation?). Maybe we can consider to reparametrize to mu and nu, where E(Y) = mu*(Delta t)? Then since mu and nu are orthogonal, that would be better. (https://journals.sagepub.com/doi/abs/10.1177/1471082X17697749)


