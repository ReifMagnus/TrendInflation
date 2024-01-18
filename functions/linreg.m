function [beta,sigma] = linreg(y,x,sigma,beta0,iVbeta0,nu0,S0,stability)
%% this functions performs ordinary bayesian regression

T = size(y,1);
k = size(x,2);

sigma2 = sigma;

% sample beta
Dbeta    = (iVbeta0 + x'*x/sigma2)\eye(k); 
beta_hat = Dbeta*(iVbeta0*beta0 + x'*y/sigma2);
C        = chol(Dbeta,'lower');

if stability == 1
    check = -1;
    while check < 0
        beta = beta_hat + C*randn(k,1);
        if k == 1
            if beta < 1, check = 1; end
        elseif k == 2
            if (beta(1) + beta(2)<1 && beta(2) - beta(1) < 1 && abs(beta(2))<1) == 1, check = 1; end
        else
            error('p has to be smaller than 3');
        end
    end
else
    beta = beta_hat + C*randn(k,1);
end


% sample sigma
e      = y - x*beta;
sigma2 = 1/gamrnd(nu0/2 + T/2,1/(S0 + sum(e.^2)'/2));
sigma  = sqrt(sigma2);

end

