function [beta,sigma,omegah,h0,htilde,e] = svreg(y,x,sigma,htilde,beta0,iVbeta0,h0,omegah,priors,stability)
%% this functions performs bayesian regression with random walk SV

[T,n] = size(y);
k     = size(x,2);

% sample beta
iSig     = sparse(1:T*n,1:T*n,1./sigma');          % inverse of covariance Matrix
xiSig    = x'*iSig;
Vpost    = (iVbeta0 + xiSig*x)\eye(k); 
beta_hat = Vpost*(iVbeta0*beta0 + xiSig*y);                       % posterior mean
C        = chol(Vpost,'lower');


if stability == 1
    check = -1;
    while check < 0
        beta = beta_hat + C*randn(k,1);
        if k == 1
            if beta < 1, check = 1; end
        elseif k == 2
            if (beta(1) + beta(2) < 1 && beta(2) - beta(1) < 1 && abs(beta(2))<1) == 1, check = 1; end
        else
            error('p has to be smaller than 3');
        end
    end
else
    beta = beta_hat + C*randn(k,1);
end


e = y - x*beta;

% sample sigma using random walk sv with noncentered parametrization (see Chan, 2018)
[htilde,h0,omegah] = SVRW_gam(log(e.^2 + 0.001),htilde,h0,omegah,priors.b0,priors.Vh0,priors.Vh,priors.Vomegah);     % log variance
sigma              = exp((h0 + omegah*htilde)/2);                                                                                                         
end

