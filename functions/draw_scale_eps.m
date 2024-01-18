function [scale_e] = draw_scale_eps(e,sigma,ind_e,r_m,r_s,scale_e_vec,prob_scale_e_vec)

% Draw Variances from Stochastic Volatility Model using chi-squared mixtureapprox;
%   e = data;
%   sigma = Standard Deviation of innovation for ln(volatilty);
%   ind_e = mixture indicators;
%   r_m = mixture means
%   r_s = mixture sd
%   scale_e_vec = list of scale values
%   prob_scale_e_vec = prior probabilities on scale values;
%;

n = size(scale_e_vec,1);
T=size(e,1);

% Compute mean of ln(e^2) associated with each scale value
scl_mean = log(scale_e_vec.^2);

% Compute Mean and SD from chi-squared 1 mixture
mean_cs = ind_e*r_m;
sd_cs = ind_e*r_s;

% ln(e.^2) residual
c=0.001;
lnres2=log(e.^2 + c);      % c = 0.001 factor from ksv, restud(1998), page 370) 
res_e = lnres2 - mean_cs - log(sigma.^2);

% Compute scale-specific demeaned version
res_e_mat = repmat(res_e,1,n)-repmat(scl_mean',T,1);
tmp1 = 1./sqrt(sd_cs);
tmp2 = res_e_mat./repmat(sd_cs,1,n);
tmp_scl = repmat(tmp1,1,n);
tmp_exp = exp(-0.5*(tmp2.^2));
den_mat = tmp_scl.*tmp_exp;
den_prob = den_mat.*repmat(prob_scale_e_vec',T,1);
den_marg = sum(den_prob,2);
p_post = den_prob./repmat(den_marg,1,n);  % Posterior probability of different scale factors 

% Draw Scale Factors from Posterior
cum_prob = cumsum(p_post,2);
u = rand(T,1);
tmp2 = repmat(u,1,n);
aa = repmat(u,1,n) < cum_prob;
bb = n+1-sum(aa,2);
scale_e = zeros(T,1);
for t = 1:T;
    scale_e(t) = scale_e_vec(bb(t));
end;
 
end