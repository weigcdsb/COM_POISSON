# COM_POISSON
 adaptive filtering for com-poisson

Here, I showed 2 examples:

## Example 1: jump lambda + jump nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_1.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_1.png" width="400"/>

## Example 2: jump lambda + cos nu

<img src="https://github.com/weigcdsb/COM_POISSON/blob/main/lambda_2.png" width="400"/><img src="https://github.com/weigcdsb/COM_POISSON/blob/main/nu_2.png" width="400"/>

Well, looks good.
But here's one problem. The algorithm is very easy to get singular matrix. That means, it correct "too hard"... Maybe we can add some constraint for theta updating? Maybe that will make the W matrix meaningless...
Also, I need to mention that the initial value is set as theta_true(1,:). The same reason: arbitrary initial value may make the algorithm get singular matrix.


