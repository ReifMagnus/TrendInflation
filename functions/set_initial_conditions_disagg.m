%% set initial conditions (draw from prior)
scale_var = 0.1;                                                           % reduce prior variance for initial draw

phi               = phi0;                                                  % AR coefficients of output gap
loadings          = zeros(n_real,s);                                            % factor loadings
loadings(1,2)     = 1;                                                     % gdp loading on factor is fixed to unity

for i = 2:n_real
    loadings(i,:) = loadings(i,:) + (scale_var*chol(diag(1./iVlambda0(:,i)))'*randn(s,1))';
end

ab                       = zeros(2,n_infl+n_exp);
% ab(1:2,1:2)              = eye(n_exp);
ab(1,2) = 1;
ab(2,1) = 1;
ab(1,n_exp+1:end)        = a0 + scale_var*chol(1/iVab0(1,1))*randn(n_infl,1);
ab(2,n_exp+1:end)        = b0 + scale_var*chol(1/iVab0(2,2))*randn(n_infl,1);

% Phillip-Curve
a_g = PC0_output_gap + scale_var*chol(diag(VPC0_output_gap))'*randn(pc_p,1);
a_p = PC0_infl_gap + scale_var*chol(diag(VPC0_infl_gap))'*randn(pc_q,1);


if n_costp > 0
    a_c = PC0_costp_gap + scale_var*chol(diag(VPC0_costp_gap))'*randn(pc_r,1);
else
    a_c = [];
end

% cost-push process
if n_costp > 0
    h_g   = CP0_output_gap + scale_var*chol(diag(VCP0_output_gap))'*randn(cp_q,1);
    h_rho = CP0_costp_gap + scale_var*chol(diag(VCP0_costp_gap))'*randn(cp_r,1);
else
    [h_rho,h_g] = deal([]);
end

% constants
c   = repmat(c0 + scale_var*eye(length(c0))/iVC0*randn(length(c0),1),1,max([1 spec.tvp*T]))';                % infl. exp. equation
d   = d0 + scale_var*diag(1./iVd0)*randn(length(d0),1);                                                     % drift of real trends
if n_dRWd ~= 0,     d(1) = 0; end
f_d = fd0 + scale_var*eye(length(fd0))/iVfd0*randn(length(fd0),1);
f_z = fz0 + scale_var*eye(length(fz0))/iVfz0*randn(length(fz0),1);            % trend inflation equation

% error processes
psi  = psi0 + scale_var*eye(length(psi0))/iVpsi0*randn(length(psi0),1);
zeta = zeta0 + scale_var*eye(length(zeta0))/iVzeta0*randn(length(zeta0),1);

% measurement error variance
sig_obs                 = zeros(n,1);
sig_obs(infl_ix)        = scale_prior_pi/(df_prior-1);
sig_obs(real_ix(2:end)) = scale_prior_real(2:end)/(df_prior-1);                   % real variables (no shock for GDP)
sig_obs(exp_ix)         = scale_prior_pie/(df_prior-1);                           % (noisy) inflation expecations
sig_obs(costp_ix)       = scale_prior_cp/(df_prior-1);                            % cost-push variable

if n_expcycle > 0
    sig_obs(exp_ix) = 0;
end

if n_costpcycle > 0
    sig_obs(costp_ix) = 0;
end

% sig_obs = sqrt(sig_obs);                                                   % state-space is expressed in terms of standard deviations

% transition error variance
sig_trans                  = zeros(K,1);
sig_trans(fac_ix(1))       = scale_prior_fac/(df_prior-1);          % output gap error variance
sig_trans(real_trend_ix)   = scale_prior_real_trend/(df_prior-1);   % real trend variances

if n_costp > 0
    sig_trans(costp_trend_ix)   = scale_prior_cptrend/(df_prior-1);                  % cost-push trend
    sig_trans(costp_gap_ix(1))  = scale_prior_cpgap/(df_prior-1);                    % cost-push gap
end
sig_trans(infl_trend_ix) = scale_prior_pi_trend/(df_prior-1);                 % inflation trend
sig_trans(infl_gap_ix)   = scale_prior_pi_gap/(df_prior-1);                       % inflation gap(s)

if n_expcycle > 0
    sig_trans(infl_expcyc_ix(1)) = scale_prior_expcyc/(df_prior-1);
end

if n_costpcycle > 0
    sig_trans(costp_cycle_ix(1)) =  scale_prior_cpcyc/(df_prior-1);
end
sig_trans = sqrt(sig_trans);                                               % state-space is expressed in terms of standard deviations

sig_trans(exp_trend_ix) = 0.9*10/(df_prior-1);
%% inital state vector (last training sample observation)
x0                = zeros(K,1);                                     % factor
x0(fac_ix)        = 0;
x0(real_trend_ix) = y0(end,real_ix);                      % trends

if n_dRWd ~= 0
    x0(dRWd_ix) = y0(end,real_ix(1));                      % trends
end

if n_costp > 0
    x0(costp_trend_ix) = y0(end,costp_ix);                                % cost-push trend
    x0(costp_gap_ix)   = 0;                                              % cost-push gap
end
x0(infl_trend_ix) = 0;
x0(infl_gap_ix)   = 0;

% inital state variance
scale               = 5;
Sig0_output_gap     = chol(scale*(sigma_nu(real_ix(1)))*toeplitz(g_autocorr_phi(1:s)))';

if n_dRWd ~= 0
    Sig0_realtrends     = diag(scale.*[sqrt(sigma_nu(real_ix(1))) sqrt(sigma_nu(real_ix))]);
else
    Sig0_realtrends     = diag(scale.*sqrt(sigma_nu(real_ix)));
end

Sig0_infl_gap   = 1;                                    % fix initial condition of common gap to 0
Sig0_infl_trend = diag([1 sqrt(sigma_nu(infl_ix))*1]);    % fix initial condition of common trend to 0
Sig0_exp_trend  = diag(ones(length(exp_trend_ix),1)*10);   


if n_costp > 0
    Sig0_costp_trend    = sqrt(sigma_nu(costp_ix));
    Sig0_costp_cycle    = chol((sigma_nu(costp_ix))*toeplitz(g_autocorr_rho(1:max([cp_r pc_r]))))';
else
    Sig0_costp_trend = [];
    Sig0_costp_cycle = [];
end

if n_expcycle > 0
    Sig0_infl_exp_cyc   = chol((scale*sigma_nu(exp_ix))*toeplitz(g_autocorr_psi(1:2)))';
else
    Sig0_infl_exp_cyc = [];
    psi               = [];
end

if n_costpcycle > 0
    Sig0_costp_cyc   = chol((1/4*sigma_nu(costp_ix))*toeplitz(g_autocorr_zeta(1:2)))';
else
    Sig0_costp_cyc = [];
    zeta           = [];
end

cSig0 = blkdiag(Sig0_infl_gap,Sig0_infl_trend,Sig0_exp_trend,Sig0_output_gap,Sig0_realtrends,Sig0_costp_trend,Sig0_costp_cycle,Sig0_infl_exp_cyc,Sig0_costp_cyc);
Sig0  = cSig0*cSig0';




%% SV-part
Lix = [];
L   = [];

if SV == 1

    % measurement erros
%     h0_obs                    = zeros(1,n_infl + n_exp);
    h0_obs        = log(var(y(:,1:n_infl + n_exp),'omitnan'))/2;
    omegah_obs    = sqrt(ones(1,length(h0_obs))*.2);                                  % std deviation of rw process
    htilde_obs    = zeros(T,n_infl + n_exp);%mvnrnd(h0_obs',diag(sig_obs(1:n_infl + n_exp)),T);                % initial vola
    htilde_obs    = h0_obs + sqrt(omegah_obs).*htilde_obs;


    sig_obs                   = repmat(sig_obs,1,T);
    sig_obs(1:n_infl+n_exp,:) = exp(htilde_obs)';
    
    % transition errors
    h0_trans     = zeros(1,infl_trends + infl_gap_c + exp_trends + 1);    
    omegah_trans = sqrt(ones(1,length(h0_trans))*.2);                                                                              % std deviation of rw process
    htilde_trans = mvnrnd(h0_trans',diag(sig_trans([infl_trend_ix(1) infl_gap_ix infl_trend_ix(2:end) exp_trend_ix fac_ix(1)])),T);
    sig_trans    = repmat(sig_trans,1,T);
%     sig_trans([infl_trend_ix(1) infl_gap_ix infl_trend_ix(2:end) exp_trend_ix fac_ix(1)],:) = exp(htilde_trans)';    
    
    % Initial Value for outlier probability
    ps                       = prior.sv.ps_prior(1)/(prior.sv.ps_prior(1) + prior.sv.ps_prior(2));
    ps2                      = (1-ps)/(length(prior.sv.scl_eps_vec) - 1);
    ps2                      = ps2*ones(length(prior.sv.scl_eps_vec) - 1,1);
    p_scl_obs                = repmat([ps; ps2],1,n_infl + n_exp);
    p_scl_trans              = repmat([ps; ps2],1,K);
    scl_obs                  = ones(T,n_infl + n_exp);
    sig_obs_scl              = sig_obs;
    sig_obs_scl(1:n_infl + n_exp,:)  = sig_obs(1:n_infl + n_exp,:).*scl_obs(:,1:n_infl + n_exp)';
    
    scl_trans                    = ones(T,K);
    sig_trans_scl                = sig_trans;
    sig_trans_scl(infl_gap_ix,:) = sig_trans(infl_gap_ix,:).*scl_trans(:,end)';
    
    SV_obs   = sum(diff(sig_obs,1,2),2) ~= 0;
    SV_trans = sum(diff(sig_trans,1,2),2) ~= 0;
elseif SV == 2
    SVs  = numel([spec.real_ix(1),spec.exp_ix,spec.infl_ix']);
    nu   = 5*ones(SVs,1);
    for i = 1:SVs
        L(:,i) = 1./gamrnd(nu(i)/2,2/nu(i),T,1);
    end
    Lix = [fac_ix(1) infl_trend_ix infl_gap_ix];
end

if spec.tvp
    sig_trans(K-n_infl,:) = 0.2;
    sig_rw                = [0.01 0.001]';
    rhoc                  = [.98 .98]';
    muc                   = [0 1]';
    acc                   = [0 0];
end

