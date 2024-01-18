function [beta,sigma,omega,h0,htilde,scale,p] = linreg_svo_uniform(y,x,sigma,htilde,beta0,iVbeta0,h0,omega,scale,p,priors,stability,ident)
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


% -- Parameters for model
% 10-component mixture approximation to log chi-squared(1) from Omori, Chib, Shephard, and Nakajima JOE (2007)
r_p = [0.00609 0.04775 0.13057 0.20674 0.22715 0.18842 0.12047 0.05591 0.01575 0.00115]';
r_m = [1.92677 1.34744 0.73504 0.02266 -0.85173 -1.97278 -3.46788 -5.55246 -8.68384 -14.65000]';
r_v = [0.11265 0.17788 0.26768 0.40611 0.62699 0.98583 1.57469 2.54498 4.16591 7.33342]';
r_s = sqrt(r_v);

[ind,S] = draw_lcs_indicators(e_scaled,sigma'./scale,r_p,r_m,r_s);


ng = 5;      % Number of grid points for approximate uniform prior
g = linspace(1e-3,0.2,ng)';
g = g/sqrt(4);
g=2*g;
g = [g ones(ng,1)/ng;];


omega = draw_g(e_scaled,g,ind,r_m,r_s,ident);
sigma = draw_sigma(e_scaled,omega,ind,r_m,r_s,ident);	

scale = draw_scale(e,sigma,r_m(S),r_s(S),priors.scl_eps_vec,p);
p     = draw_ps(scale,priors.ps_prior,length(r_s));

%  % sample sigma using random walk sv with noncentered parametrization (see Chan, 2018) conditional on rescaled errors
% [htilde,h0,omega,~,~,S] = SVRW_gam(log(e_scaled.^2 + 0.001),htilde,h0,omega,priors.b0,priors.Vh0,priors.Vh,priors.Vomegah);     % log variance
% sigma                   = exp((h0 + omega*htilde)/2);    % standard deviations
%
%
% % normal mixture
% mj       = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
% sqrtsigj = sqrt([5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261]);

% mean_cs = mj(S)';
% sd_cs   = sqrtsigj(S)';
%
% % sample outlier conditional on non-rescaled errors
% scale = draw_scale(e,sigma,mean_cs,sd_cs,priors.scl_eps_vec,p);
%
% % draw probability of outlier
% p = draw_ps(scale,priors.ps_prior,length(sqrtsigj));

end

