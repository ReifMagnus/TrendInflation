function [b,mu,rho,sig_obs,sig_s,accept] = tvpreg_ar1(y,x,sig_obs,sig_s,mu,rho,mud0,Vmud,rhod0,Vrhod,nu0,S0_obs,S0_s,accept)
%% This function sample the coefficients of a univariate, homosecadastic time-varying parameter regression using precsion sampling
% coefficients follow AR(1)-process subject to stationarity constraint
% y_t      = b_t * x_t + eps_t,       eps_t ~ N(0,sig_obs)
% b_t - mu = rho(*b_t-1 - mu) + e_t,      e_t ~ N(0,sig_s)


% (c) Magnus Reif, 2022

%% preliminiaries
[T,~]  = size(x);


X       = SURform(x);
Hrho    = speye(2*T) - sparse(2+1:T*2,1:(T-1)*2,repmat(rho,T-1,1),2*T,2*T);
HiSigcH = Hrho'*sparse(1:2*T,1:2*T,[(1-rho)./sig_s; repmat(1./sig_s,T-1,1)])*Hrho;

%% draw states (using precision sampling)
Kb     = HiSigcH + X'*X/sig_obs;
delb   = Hrho\[mu; repmat((1-rho).*mu,T-1,1)];
CKb    = chol(Kb,'lower');
bhat   = CKb'\(CKb\(HiSigcH*delb + X'*y/sig_obs));
B      = bhat + CKb'\randn(2*T,1);
b      = reshape(B,2,T)';

%% sample mu
for i = 1:2
    Dmu   = 1/(1/Vmud(i) + ((T-1)*(1-rho(i))^2 + (1-rho(i)^2))/sig_s(i));
    muhat = Dmu*(mud0(i)/Vmud(i) + (1-rho(i)^2)/sig_s(i)*b(1,i) + (1-rho(i))/sig_s(i)*sum(b(2:end,i)-rho(i)*b(1:end-1,i)));
    mu(i) = muhat + sqrt(Dmu)*randn;
end

%% sample rho
for i = 1:2
    Xrho    = b(1:end-1,i) - mu(i);
    yrho    = b(2:end,i) - mu(i);
    Drho 	= 1/(1/Vrhod(i) + Xrho'*Xrho/sig_s(i));
    rhohat  = Drho*(rhod0(i)/Vrhod(i) + Xrho'*yrho/sig_s(i));
    rhod    = rhohat + sqrt(Drho)*randn;
    g       = @(x) -.5*log(sig_obs./(1-x.^2))  -.5*(1-x.^2)/sig_s(i)*(b(1,i)-mu(i))^2;
    if abs(rhod) < .98
        alpMH = exp(g(rhod)-g(rho(i)));
        if alpMH > rand
            rho(i)    = rhod;
            accept(i) = accept(i) + 1;
        end
    end
end

%% sample observation error variance
e2      = sum((y - X*B).^2);
sig_obs = 1./gamrnd(nu0 + T/2, 1./(S0_obs + e2/2));

%% sample transition error variance
for i = 1:2
    e2       = sum([(b(1,i)-mu(i))*sqrt(1-rho(i)^2);    b(2:end,i)-rho(i)*b(1:end-1,i)-mu(i)*(1-rho(i))].^2);
    sig_s(i) = 1./gamrnd(nu0 + T/2, 1./(S0_s(i) + e2/2));
end


%% helper functions
    function Xout = SURform(X)
        [r,cc] = size(X);
        idi   = kron((1:r)',ones(cc,1));
        idj   = (1:r*cc)';
        Xout  = sparse(idi,idj,reshape(X',r*cc,1));
    end

end
