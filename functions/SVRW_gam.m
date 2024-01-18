function [htilde,h0,omega,omegahhat,Domegah,S] = SVRW_gam(Ystar,htilde,h0,omega,b0,Vh0,Vh,Vomega)
% This function draws the log volatilities with a gamma prior on the error variance.
% See: Chan, J.C.C. (2018). Specification Tests for Time-Varying Parameter Models with Stochastic Volatility, Econometric Reviews, 37(8), 807-823
% model:
% ystar_t  = h0 + omega*htilde_t + e_t,       e_t ~ N(0,1),
% htilde_t = htilde_t-1 + eps_t,            eps_t ~ N(0,1).

% priors:
% h_1   ~ N(0,Vh),
% omega ~ N(0,Vomega),      note: implied prior mean for omega^2 (E[omega^2]) is given by G(1/2,1/(2*Vomega)), with G being the Gamma density
% h_0   ~ N(b0,Vh0).        note: h0 is not initival value!

% implied mean of variance (omega^2): E[omega^2] = 0.5/[1/(2*Vomegah)]    
T = length(htilde);
%% normal mixture
pj       = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
mj       = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
sigj     = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
sqrtsigj = sqrt(sigj);

%% sample S from a 7-point distrete distribution
temprand = rand(T,1);
q        = repmat(pj,T,1).*normpdf(repmat(Ystar,1,7),repmat(h0+omega*htilde,1,7) + repmat(mj,T,1), repmat(sqrtsigj,T,1));
q        = q./repmat(sum(q,2),1,7);
S        = 7 - sum(repmat(temprand,1,7) < cumsum(q,2),2) + 1;

%% sample htilde
% y^*c      = h0 + omegah*htilde + d + eps,    eps ~ N(0,Omega),
% H*htilde  = nu,                             nu ~  N(0,S),
%       d_t = E[z_t], Omega = diag(omega_1,...,omega_n),
%  omega_t  = var[z_t],   S = diag(Vh,1,...,1)
Hh        = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
invSh     = sparse(1:T,1:T,[1/Vh; ones(T-1,1)]);
dconst    = mj(S)'; 
invOmega  = sparse(1:T,1:T,1./sigj(S));
Kh        = Hh'*invSh*Hh + invOmega*omega^2;
htildehat = Kh\(invOmega*omega*(Ystar - dconst - h0));
htilde    = htildehat + chol(Kh,'lower')'\randn(T,1);

%% sample h0 and omegah
Xbeta         = [ones(T,1) htilde];
invVbeta      = diag([1/Vh0 1/Vomega]);
XbetainvOmega = Xbeta'*invOmega;
invDbeta      = invVbeta + XbetainvOmega*Xbeta;
betahat       = invDbeta\(invVbeta*[b0;0] + XbetainvOmega*(Ystar - dconst));
beta          = betahat + chol(invDbeta,'lower')'\randn(2,1);
h0            = beta(1);
omega         = beta(2);

U      = -1 + 2*(rand>0.5);
htilde = U*htilde;
omega = U*omega;

%% compute the mean and variance of the conditional posterior of omegah
Xbeta     = [ones(T,1) htilde];
Dbeta     = (invVbeta + Xbeta'*invOmega*Xbeta)\speye(2);
omegahhat = betahat(2);
Domegah   = Dbeta(2,2);
end