function [beta,Sigma,U] = AR_factor(y,Sigma,beta0,iVbeta,df_post,scale_prior,p,stability,ident,sv)
%% generates a draw of a AR using precision sampling
% Inputs:
% y       = factors
% Sigma   = Reduced-form residuals (stochastic volatilities)
% Outputs:
% beta    = AR-coefficients
% Sigma   = variances


T = size(y,1);

%%  arrange data

Y = y(:,1);
X = y(:,2:p+1);


invSig  = sparse(1:T,1:T,1./Sigma);
XinvSig = X'*invSig;
%% -----------------------------------------------------------------------------------------------------------------------------
% sample AR-cofficients
Kbeta    = iVbeta + XinvSig*X;
beta_hat = Kbeta\(iVbeta*beta0 + XinvSig*Y);                       % posterior mean
cKbeta   = chol(Kbeta,'lower')';

if stability == 1
    check = -1;
    while check < 0
        beta = beta_hat + cKbeta\randn(p,1);
        if p == 1
            if beta < 1, check = 1; end
        elseif p == 2
            if (beta(1)+beta(2)<0.95 && beta(2)-beta(1)<1 && abs(beta(2))<1) == 1, check = 1; end
        else
            error('p has to be smaller than 3');
        end
    end
else
    beta = beta_hat + cKbeta\randn(p,1);
end

U   = (Y - X*beta);                                          % innovations
                                       
if ~sv
    if ident == 1
        Sigma  = 1/gamrnd(df_post, 1./(scale_prior + sum(U.^2)'/2));
    else
        Sigma = 1;
    end
end
    

