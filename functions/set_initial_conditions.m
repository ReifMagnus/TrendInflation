%% set initial conditions (draw from prior)
scale_var = 0.1;                                                           % reduce prior variance for initial draw

phi               = phi0;                                                  % AR coefficients of output gap
loadings          = zeros(n,s);                                            % factor loadings
loadings(1,2)     = 1;                                                     % gdp loading on factor is fixed to unity

for i = 2:n_real
    loadings(i,:) = loadings(i,:) + (scale_var*chol(diag(1./iVlambda0(:,i)))'*randn(s,1))';
end

% Phillip-Curve(s)
a_g = PC0_output_gap + scale_var*chol(diag(VPC0_output_gap))'*randn(pc_p,1);
a_p = PC0_infl_gap + scale_var*chol(diag(VPC0_infl_gap))'*randn(pc_q,1);

if n_costp > 0
    a_c = PC0_costp_gap + scale_var*chol(diag(VPC0_costp_gap))'*randn(pc_r,1);
else
    a_c = [];
end

if n_infl > 1
    b_g = PC0_output_gap + scale_var*chol(diag(VPC0_output_gap_head))'*randn(pc_p,1);
    b_p = PC0_infl_gap + scale_var*chol(diag(VPC0_infl_gap))'*randn(pc_q,1);
    b_c = PC0_costp_gap + scale_var*chol(diag(VPC0_costp_gap))'*randn(pc_r,1);
else
    [b_g,b_p,b_c] = deal([]);
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
sig_obs                 = zeros(n,1);                          % no shock inflation
sig_obs(real_ix(2:end)) = eps_real_meas(2:end);                     % real variables (no shock for GDP)
sig_obs(exp_ix)         = eps_pie_mean;                        % (noisy) inflation expecations
sig_obs(costp_ix)       = eps_cp_mean;                         % import price inflation

if n_expcycle > 0
    sig_obs(exp_ix) = 0;
end

if n_costpcycle > 0
    sig_obs(costp_ix) = 0;
end


% transition error variance
sig_trans                  = zeros(K,1);
sig_trans(fac_ix(1))       = eta_g_mean;                            % output gap error variance
sig_trans(real_trend_ix)   = eta_real_trend;                        % real trend variances
if n_costp > 0                                                              
    sig_trans(costp_trend_ix)   = eps_cptrend_mean;                  % cost-push trend
    sig_trans(costp_gap_ix(1))  = eps_cpgap_mean;                    % cost-push gap
end
sig_trans(infl_trend_ix)                   = eps_pi_trend_mean;                 % inflation trend
sig_trans(infl_gap_ix(1:pc_q:pc_q*n_infl)) = eps_pi_mean;                       % inflation gap(s)     

if n_expcycle > 0
    sig_trans(infl_expcyc_ix(1)) = eps_expcyc_mean;
end

if n_costpcycle > 0
    sig_trans(costp_cycle_ix(1)) = eps_costpcyc_mean;
end

%% inital state vector (last training sample observation)
x0                = zeros(K,1);                                     % factor
x0(real_trend_ix) = y0(end,real_ix);                      % trends

if n_dRWd ~= 0
    x0(real_trend_ix(1)+1) = y0(end,real_ix(1));                      % trends
end

if n_costp > 0   
    x0(costp_trend_ix) = y0(end,costp_ix);                                % cost-push trend
    x0(costp_gap_ix)   = 0;                                              % cost-push gap
end
x0(infl_trend_ix)                     = y0(end,exp_ix);                            
x0(infl_gap_ix(1:pc_q:pc_q*n_infl))   = y0(end,infl_ix);

% inital state variance
scale               = 5;
Sig0_output_gap     = chol(scale*sigma_nu(real_ix(1))*toeplitz(g_autocorr_phi(1:s)))';

if n_dRWd ~= 0
    Sig0_realtrends     = diag(scale.*[sqrt(sigma_nu(real_ix(1))) sqrt(sigma_nu(real_ix))]);
else
    Sig0_realtrends     = diag(scale.*sqrt(sigma_nu(real_ix))); 
end

Sig0_infl_gap       = kron(eye(pc_q),diag(sqrt(sigma_nu(infl_ix))));
Sig0_infl_trend     = sqrt(sigma_nu(exp_ix));

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


cSig0               = blkdiag(Sig0_output_gap,Sig0_realtrends,Sig0_costp_trend,Sig0_costp_cycle,Sig0_infl_trend,Sig0_infl_gap,Sig0_infl_exp_cyc,Sig0_costp_cyc);
Sig0                = cSig0*cSig0';


%% SV-part
Lix = [];
L   = [];

if SV == 1
    h0      = [log((sigma_nu(spec.real_ix(1))))/2 log((sigma_nu(spec.exp_ix)))/2 log(sigma_nu(spec.infl_ix))./2];     % initial variance
%     h0      = [log((sigma_nu(spec.real_ix(1))))/2 log(.1)/2 log((sigma_nu(spec.infl_ix)))/2];     % initial variance

    omegah  = sqrt(ones(1,length(h0))*.2);                                                                              % std deviation of rw process  
    htilde  = zeros(T,length(h0));                                                                                          
    h       = h0 + omegah(1)*htilde;                                                                                    % initial log variance

    sig_trans                                          = repmat(sig_trans,1,T);    
    sig_trans([fac_ix(1) infl_trend_ix infl_gap_ix(1:pc_q:pc_q*n_infl)],:) = exp(h)';        
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
    
