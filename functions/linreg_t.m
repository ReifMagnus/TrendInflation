function [beta,sigma,L,nu] = linreg_t(y,x,sigma,beta0,iVbeta0,L,nu,nu0,S0,stability)
%% this functions performs bayesian regression with t-distributed erros

T = size(y,1);
k = size(x,2);

% sample beta
iL       = sparse(1:T,1:T,1./L);
xiL      = x'*iL;

Dbeta    = (iVbeta0 + xiL*x/sigma)\eye(k);
C        = chol(Dbeta,'lower')';
beta_hat = Dbeta*(iVbeta0*beta0 + xiL*y/sigma);                       % posterior mean


if stability == 1
    check = -1;
    while check < 0
        beta     = beta_hat + C*randn(k,1);
        if k == 1
            if beta < 1, check = 1; end
        elseif k == 2
            if (beta(1)+beta(2)<0.95 && beta(2)-beta(1)<1 && abs(beta(2))<1) == 1, check = 1; end
        else
            error('p has to be smaller than 3');
        end
    end
else
    beta     = beta_hat + C*randn(k,1);
end


e = y - x*beta;

% sample lambda
L    = 1./gamrnd((nu + 1)/2,2./(nu + e.^2/sigma));
iL   = sparse(1:T,1:T,1./L);

% sample Sigma
sigma = 1/gamrnd(nu0 + T/2,1/(S0 + e'*iL*e/2));

% sample nu
nu_ub  = 50;            % upper bound for degrees of freedom
nu_lb  = 2;             % lower bound for degrees of freedom
nu     = sample_nu(L,nu,nu_ub,nu_lb);


end

