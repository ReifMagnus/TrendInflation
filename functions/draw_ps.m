function prob_scl_eps_vec = draw_ps(scale_eps,ps_prior,n_scl_eps)
%% this functions draws the outlier probability using a beta prior

n1  = sum(scale_eps == 1);
n2  = size(scale_eps,1) - n1;
ps  = betarnd(ps_prior(1) + n1,ps_prior(2) + n2,1,1);
ps2 = (1-ps)/(n_scl_eps-1);
ps2 = ps2*ones(n_scl_eps-1,1);

prob_scl_eps_vec = [ps; ps2];

end