function [alpha_eps,alpha_tau] = draw_alpha(y,prior_var_alpha,tau_unique,eps_common,tau_common,sigma_eps_unique)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
     n_y = size(y,2);
     nobs = size(y,1);
     
     ytmp = y - tau_unique;
     xtmp = [eps_common tau_common];
     alpha_mean = zeros(2*n_y,1);
     alpha_var = prior_var_alpha;
     for i = 1:n_y;
      yi = ytmp(:,i);
      sd_i = sigma_eps_unique(:,i);
      w_yi = yi./sd_i;
      w_x = xtmp ./ repmat(sd_i,1,2);
      tmp = ([w_yi w_x]);
      yp = tmp(:,1);
      xp = tmp(:,2:3);
      xpxi = inv(xp'*xp);
      xpy = xp'*yp;
      bhat = xpxi*xpy;
      var_bhat = xpxi;
      % Update estimate of alpha
      S = zeros(2,2*n_y);
      S(1,i) = 1;                       % select entry of trend
      if i ~= 1
        S(2,n_y+i) = 1;                   % select entry of gap
      end
      cov_ab = S*alpha_var;
      var_a = cov_ab*S' + var_bhat;
      var_a_inv = inv(var_a);
      k_ba = cov_ab'*var_a_inv;
      alpha_mean = alpha_mean + k_ba*(bhat - S*alpha_mean);
      alpha_var = alpha_var - k_ba*cov_ab;
     end
     % Draw new Alpha Coefficients
     chol_alpha_var = chol(alpha_var);
     alpha_draw = alpha_mean + chol_alpha_var'*randn(2*n_y,1);
%      ab = [alpha_draw(1+n_y:2*n_y,1) alpha_draw(1:n_y,1)]';
     alpha_eps = repmat(alpha_draw(1:n_y,1)',nobs,1);
     alpha_tau = repmat(alpha_draw(1+n_y:2*n_y,1)',nobs,1);
end

