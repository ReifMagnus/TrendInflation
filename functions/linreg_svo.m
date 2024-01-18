function [beta,sigma,omega,h0,htilde,scale,p] = linreg_svo(y,x,sigma,htilde,beta0,iVbeta0,h0,omega,scale,p,priors,stability,ident)
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

e = (y - x*beta);

% rescale error
e_scaled = e./scale;

if ident == 1
    priors.Vh0 = 10^-9;
end

% sample sigma using random walk sv with noncentered parametrization (see Chan, 2018) conditional on rescaled errors
[htilde,h0,omega,~,~,S] = SVRW_gam(log(e_scaled.^2 + 0.001),htilde,h0,omega,priors.b0,priors.Vh0,priors.Vh,priors.Vomegah);     % log variance
sigma                   = exp((h0 + omega*htilde)/2);    % standard deviations


% normal mixture
mj       = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
sqrtsigj = sqrt([5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261]);

mean_cs = mj(S)';
sd_cs   = sqrtsigj(S)';

% sample outlier conditional on non-rescaled errors
scale = draw_scale(e,sigma,mean_cs,sd_cs,priors.scl_eps_vec,p);

% draw probability of outlier
p = draw_ps(scale,priors.ps_prior,length(sqrtsigj));

end

