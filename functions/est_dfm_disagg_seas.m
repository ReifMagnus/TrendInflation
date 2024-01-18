function [results, posterior, spec] = est_dfm_disagg_seas(spec)

rng(1234)

v2struct(spec)

set_model_disagg_seas
set_priors_disagg
set_initial_conditions_disagg
set_state_space_disagg


%% preallocation
post_X                        = nan(T,K,(draws-burnin)/thin);
post_Xhat                     = nan(hmax+1,K,(draws-burnin)/thin);
post_Q                        = (nan(K,max([1,T*SV(SV==1)]),(draws-burnin)/thin));                          % transition error variance
post_R                        = nan(n,T,(draws-burnin)/thin);                            % observation error variance
post_d                        = nan(n_real,(draws-burnin)/thin);                       % drift coefficients
post_lambda                   = nan(n_real,s,(draws-burnin)/thin);                          % factor loadings
post_L                        = nan(T,size(L,2),(draws-burnin)/thin);
post_h_g                      = nan(cp_q,(draws-burnin)/thin);                           % import price in PC
post_Yhat                     = nan(T+hmax,n,(draws-burnin)/thin);
[post_a_g,post_b_g]           = deal(nan(pc_p,(draws-burnin)/thin));                           % output gap in PC
[post_a_p,post_b_p]           = deal(nan(pc_q,(draws-burnin)/thin));                           % PC-persistence
[post_a_c,post_b_c]           = deal(nan(pc_r,(draws-burnin)/thin));                           % import price in PC
[post_Phi,post_h_Rho]         = deal(nan(p,(draws-burnin)/thin));
[post_f_z,post_f_d,post_c]    = deal(nan(2,(draws-burnin)/thin));
post_c                        = squeeze(nan(max([1,spec.tvp*T]),2,(draws-burnin)/thin));
post_psi                      = nan(2,(draws-burnin)/thin);
post_ab                       = nan(n_infl+n_exp,2,(draws-burnin)/thin);
post_pi_trend                 = nan(T,n_infl,(draws-burnin)/thin);
post_pi_gap                   = nan(T,n_infl,(draws-burnin)/thin);
post_scl_obs                = nan(T,n,(draws-burnin)/thin);
post_scl_trans              = nan(T,K,(draws-burnin)/thin);
post_omega_obs                = nan(n_infl + n_exp,(draws-burnin)/thin);
post_omega_trans              = nan(n_infl + 3 + n_exp,(draws-burnin)/thin);

% PC0(2) = 0;
% iVPC0  = inv(eye(4).*10^-9);

R(real_ix(1),real_ix(1),:) = 10^-9;
cSig0(exp_trend_ix(2),exp_trend_ix(2)) = 10^-9;

% sd_ddp_median = median(std(diff(y(:,1:n_infl)),'omitnan'));
scale_y       = 1;%sd_ddp_median/5;

% y(:,1:n_infl) = y(:,1:n_infl)./scale_y;

omega_tau = 10/scale_y;
omega_eps = 10/scale_y;
sigma_tau = 0.4/scale_y;
sigma_eps = 0.4/scale_y;

var_alpha_tau = ((omega_tau^2)*ones(n_infl,n_infl)) + ((sigma_tau^2)*eye(n_infl));
var_alpha_eps = ((omega_eps^2)*ones(n_infl,n_infl)) + ((sigma_eps^2)*eye(n_infl));

prior_var_alpha                                      = zeros(2*n_infl,2*n_infl);
prior_var_alpha(1:n_infl,1:n_infl)                   = var_alpha_eps;
prior_var_alpha(n_infl+1:2*n_infl,n_infl+1:2*n_infl) = var_alpha_tau;

% -- Parameters for model
% 10-component mixture approximation to log chi-squared(1) from Omori, Chib, Shephard, and Nakajima JOE (2007)
r_p = [0.00609 0.04775 0.13057 0.20674 0.22715 0.18842 0.12047 0.05591 0.01575 0.00115]';
r_m = [1.92677 1.34744 0.73504 0.02266 -0.85173 -1.97278 -3.46788 -5.55246 -8.68384 -14.65000]';
r_v = [0.11265 0.17788 0.26768 0.40611 0.62699 0.98583 1.57469 2.54498 4.16591 7.33342]';
r_s = sqrt(r_v);

r_p7  = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
r_m7  = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
r_v7  = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
r_s7  = sqrt(r_v7);

   
ng = 5;      % Number of grid points for approximate uniform prior
g  = linspace(1e-3,.2,ng)';
g  = g/sqrt(4);
g  = 2*g;
g  = [g ones(ng,1)/ng;];

g1 = linspace(1e-3,.4,ng)';
g1 = g1/sqrt(4);
g1 = 2*g1;
g1 = [g1 ones(ng,1)/ng;];


%% intialize the state vector
%
ix    = 0;
for m = 1:draws
    
    
    %% draw states
    [X,~,~,~] = DurbinKoopman(y',x0,cSig0,F,H,Q,R,A,B,K,T,n);
%     X1        = [X; (B + F*X(end,:)' + Q(:,:,end)*randn(K,1))'];
    
%     [X, ~] = simulation_smoother(y',A,H,R,B,F,Q,x0,cSig0,2);    
%     % append a draw of states in S+1 - needed for the regressions with lags
%     X1 = [X; (B + F*X(end,:)' + Q(:,:,end)*randn(nQ_shocks,1))'];

    X1        = [X; (B + F*X(end,:)' + diag(sig_trans(:,end))*randn(K,1))'];            % ignore outliers

    
    %% extract objects from state vector
    output_gap   = X1(2:end,fac_ix(1));
    real_trends  = X1(:,real_trend_ix);
%     costp_trend  = X1(2:end,costp_trend_ix);
%     costp_gap    = X1(2:end,costp_gap_ix);
    infl_trend   = X1(:,infl_trend_ix);
    infl_gap     = X1(2:end,infl_gap_ix);
    
    output_gap_lag   = X(:,fac_ix);
    real_trends_lag  = X(:,real_trend_ix);
%     costp_trend_lag  = X(:,costp_trend_ix);
%     costp_gap_lag    = X(:,costp_gap_ix);
    infl_trend_lag   = X(:,infl_trend_ix);
    infl_gap_lag     = X(:,infl_gap_ix);
    
    %% draw observables conditionally on the states and parameters
    obsdraw = y;
    for i = 1:length(nan_id)
        per               = nan_id(i);
        if ~tvp, j = 1; else, j = per; end
        inan              = find(isnan(y(per,:)));
%         obsdraw(per,inan) = (A(inan,j) + H(inan,:,j)*X(per,:)' + R(inan,:,per)*randn(nR_shocks,1))';
        obsdraw(per,inan) = (A(inan,j) + H(inan,:,j)*X(per,:)' + R(inan,:,per)*randn(n,1))';
    end
    
    %% draw coefficients of measurement equations
    
    
    % inflation exp. (no outliers!)
%     i                                               = exp_ix(1);
%     e                                               = (obsdraw(:,i) - X(:,infl_trend_ix(1)));       % draw loadings conditional on common inflation noise
%     [htilde_obs(:,i),h0_obs(i),omegah_obs(i),~,~,~] = SVRW_gam(log(e.^2 + 0.001),htilde_obs(:,i),h0_obs(i),omegah_obs(i),prior.sv.b0,prior.sv.Vh0,prior.sv.Vh,prior.sv.Vomegah);     % log variance
%     sig_obs(i,:)                                    = exp((h0_obs(i) + omegah_obs(i)*htilde_obs(:,i))/2);            % standard deviations
%     
%     
%     i                                               = exp_ix(2);
%     e                                               = (obsdraw(:,i) - X(:,infl_gap_ix(1)));       % draw loadings conditional on common inflation noise
%     [htilde_obs(:,i),h0_obs(i),omegah_obs(i),~,~,~] = SVRW_gam(log(e.^2 + 0.001),htilde_obs(:,i),h0_obs(i),omegah_obs(i),prior.sv.b0,prior.sv.Vh0,prior.sv.Vh,prior.sv.Vomegah);     % log variance
%     sig_obs(i,:)                                    = exp((h0_obs(i) + omegah_obs(i)*htilde_obs(:,i))/2);            % standard deviations

    
    % nominal variables: loadings on common trend and gap
    [alpha_eps, alpha_tau] = draw_alpha(obsdraw(:,infl_ix),prior_var_alpha,infl_trend_lag(:,2:end),infl_gap_lag,infl_trend_lag(:,1),sig_obs_scl(infl_ix,:)');
    ab(1,n_exp+1:end)      = alpha_eps(1,:);
    ab(2,n_exp+1:end)      = alpha_tau(1,:);


    % nominal variables: ar(1) coefficients of gap
    xx = [output_gap_lag(:,1:pc_p) infl_gap_lag(:,1)];
    yy = infl_gap(:,1);

    % Phillips-curve
    iSig     = sparse(1:T,1:T,1./sig_trans_scl(infl_gap_ix,:)');          % inverse of covariance Matrix
    xiSig    = xx'*iSig;
    Vpost    = (iVPC0 + xiSig*xx)\eye(length(PC0)); 
    beta_hat = Vpost*(xiSig*yy);                       % posterior mean
    C        = chol(Vpost,'lower');  
    a_p      = 1.1;
    while a_p > 1
        PC =  beta_hat + C*randn(length(PC0),1);
        a_p = PC(pc_p+1);
    end
    a_g = PC(1:pc_p);

    eps_common        = yy - xx*PC;
    eps_common_scaled = eps_common./scl_trans(:,infl_gap_ix);
        
    % Save some values    
    y_tau_common = alpha_tau.*repmat(X(:,infl_trend_ix(1)),1,n_infl);    
    y_eps_common = alpha_eps.*repmat(X(:,infl_gap_ix),1,n_infl);   
    y_tau_unique = X(:,infl_trend_ix(2:end));

    eps_unique(:,1)         = obsdraw(:,1) - X(:,infl_trend_ix(1)) - X(:,exp_trend_ix(1));
    eps_unique(:,2)         = obsdraw(:,2) - X(:,infl_gap_ix)  - X(:,exp_trend_ix(2));    
    eps_unique(:,infl_ix)   = obsdraw(:,infl_ix) - y_eps_common - y_tau_common - y_tau_unique;
    eps_unique_scaled       = eps_unique./scl_obs(:,[exp_ix; infl_ix]);
    
%     eps_common        = X(:,infl_gap_ix);

    dtau_common = diff(X1(:,infl_trend_ix(1)));
    dtau_unique = diff(X1(:,infl_trend_ix(2:end)));

    % draw indicator variables
    ind_dtau_common = draw_lcs_indicators(dtau_common,sig_trans(infl_trend_ix(1),:)',r_p,r_m,r_s);    
    ind_eps_common  = draw_lcs_indicators(eps_common_scaled,sig_trans(infl_gap_ix,:)',r_p,r_m,r_s);

    ind_eps_unique  = nan(T,size(r_p,1),n_infl + n_exp);
    ind_dtau_unique = nan(T,size(r_p,1),n_infl);

    for i = 1:n_exp
        ind_eps_unique(:,:,i)  = draw_lcs_indicators(eps_unique_scaled(:,i),sig_obs(i,:)',r_p,r_m,r_s);
    end

    for i = 1:n_infl
        ind_eps_unique(:,:,i+n_exp)  = draw_lcs_indicators(eps_unique_scaled(:,i+n_exp),sig_obs(i+n_exp,:)',r_p,r_m,r_s);
        ind_dtau_unique(:,:,i)       = draw_lcs_indicators(dtau_unique(:,i),sig_trans(infl_trend_ix(i+1),:)',r_p,r_m,r_s);
    end

    
    i_init = 1;   % Variance = 1 as initial condition to identify factor loadings (no identification assumption required)
    g_dtau_common = draw_g(dtau_common,g,ind_dtau_common,r_m,r_s,i_init);    
    g_eps_common  = draw_g(eps_common_scaled,g1,ind_eps_common,r_m,r_s,i_init);
    
    i_init = 1;   % Vague prior for initial variance;
    g_dtau_unique = nan(n_infl,1);    
    g_eps_unique  = nan(n_infl+n_exp,1);

    for i = 1:n_exp
        g_eps_unique(i)  = draw_g(eps_unique_scaled(:,i),g1,ind_eps_unique(:,:,i),r_m,r_s,i_init);
    end

    for i = 1:n_infl
        g_eps_unique(i+n_exp) = draw_g(eps_unique_scaled(:,i+n_exp),g1,ind_eps_unique(:,:,i+n_exp),r_m,r_s,i_init);
        g_dtau_unique(i)      = draw_g(dtau_unique(:,i),g,ind_dtau_unique(:,:,i),r_m,r_s,i_init);
    end

    i_init = 1;   % Variance = 1 as initial condition to identify factor loadings
    sig_trans(infl_trend_ix(1),:) = draw_sigma(dtau_common,g_dtau_common,ind_dtau_common,r_m,r_s,i_init);	    
    sig_trans(infl_gap_ix,:)      = draw_sigma(eps_common_scaled,g_eps_common,ind_eps_common,r_m,r_s,i_init);	
    
    i_init = 1;  % Vague prior for initial variance;
    for i = 1:n_exp
        sig_obs(i,:) = draw_sigma(eps_unique_scaled(:,i),g_eps_unique(i),ind_eps_unique(:,:,i),r_m,r_s,i_init);  
    end

    for i = 1:n_infl
      sig_obs(i+n_exp,:)              = draw_sigma(eps_unique_scaled(:,i+n_exp),g_eps_unique(i+n_exp),ind_eps_unique(:,:,i+n_exp),r_m,r_s,i_init);  
      sig_trans(infl_trend_ix(i+1),:) = draw_sigma(dtau_unique(:,i),g_dtau_unique(i),ind_dtau_unique(:,:,i),r_m,r_s,i_init);
    end
    
    scl_trans(:,infl_gap_ix) = draw_scale_eps(eps_common,sig_trans(infl_gap_ix,:)',ind_eps_common,r_m,r_s,[1:10]',p_scl_trans(:,infl_gap_ix));
    scl_obs     	         = ones(T,n);
    for i = 1:n_infl
      scl_obs(:,i+n_exp) = draw_scale_eps(eps_unique(:,i+n_exp),sig_obs(i+n_exp,:)',ind_eps_unique(:,:,i+n_exp),r_m,r_s,[1:10]',p_scl_obs(:,i+n_exp));
    end

    p_scl_trans(:,infl_gap_ix) = draw_ps(scl_trans(:,infl_gap_ix),prior.sv.ps_prior,10);
    for i = 1:n_infl
        p_scl_obs(:,i+n_exp) = draw_ps(scl_obs(:,i+n_exp),prior.sv.ps_prior,10);
    end

    % exp trends
    dtau_exp = diff(X1(:,exp_trend_ix));
    for i = 1:n_exp
        sigma2(i)                    = 1/gamrnd(5/2 + T/2,1/(scale_prior_exptrend*1 + sum(dtau_exp(:,i).^2)'/2));
        sig_trans(exp_trend_ix(i),:) = sqrt(sigma2(i));      
    end


    omegah_trans(infl_trend_ix(1))     = g_dtau_common;
    omegah_trans(infl_gap_ix)          = g_eps_common;
    omegah_trans(infl_trend_ix(2:end)) = g_dtau_unique;
    omegah_obs(1:n_exp+n_infl)         = g_eps_unique;

%     e = obsdraw(:,1:n_infl) - ab(1,:).*X(:,1) - ab(2,:).*X(:,2) - X(:,infl_trend_ix(2:end));
%     for i = 1:n_infl      
%         [~,sig_obs(i,:),omegah_obs(i),~,~,scl_obs(:,i),p_scl_obs(:,i)] = svo_uniform(eps_unique(:,i),[],sig_obs(i,:),[],[],[],[],omegah_obs(i),scl_obs(:,i),p_scl_obs(:,i),prior.sv,0,1);  
%     end

%     x      = X(:,[infl_trend_ix(1), infl_gap_ix]);                           % select common inflation states    
%     pi_gap = obsdraw(:,infl_ix) - X(:,infl_trend_ix(2:end));                 % draw loadings conditional on idioyncratic inflation trends    

%     for i = 1 + n_exp:n_exp + n_infl          
%         if i == 1 + n_exp
%             yy = pi_gap(:,i) - x(:,2);                                       % draw loadings conditional on common inflation gap                            
%             xx = x(:,1);
%             [ab(1,i),sig_obs(i,:),omegah_obs(i),h0_obs(i),htilde_obs(:,i),scl_obs(:,i),p_scl_obs(:,i)] = linreg_svo(yy,xx,sig_obs_scl(i,:),htilde_obs(:,i),ab0(1),iVab0(1),h0_obs(i),omegah_obs(i),scl_obs(:,i),p_scl_obs(:,i),prior.sv,0,0);
%         else
%             [ab(:,i),sig_obs(i,:),omegah_obs(i),h0_obs(i),htilde_obs(:,i),scl_obs(:,i),p_scl_obs(:,i)] = linreg_svo_uniform(pi_gap(:,i - n_exp),x,sig_obs_scl(i,:),htilde_obs(:,i),ab0,iVab0,h0_obs(i),omegah_obs(i),scl_obs(:,i),p_scl_obs(:,i),prior.sv,0,0);
%         end
%     end
       
    % real variables: factor loadings and measurement error
    y_cycle = obsdraw(:,real_ix) - real_trends_lag;               % draw loadings conditional on trend, ignore estimated states
    for i = 2:n_real                                              % GDP loading is fixed to unity, only real variables load on factor
        [loadings(i,1:p+1),sig_obs(real_ix(i),:)] = linreg(y_cycle(:,i),output_gap_lag(:,1:s),sig_obs(real_ix(i),1),lambda0,diag(iVlambda0(:,i)),df_prior,scale_prior_real(i),0);
    end
    
    %% draw coefficients of transition equations
    % Phillips-Curve (inflation gap process)
%     if n_costp > 0
%         x = [output_gap_lag(:,1:pc_p) infl_gap_lag(:,1:pc_q) costp_gap_lag(:,1:pc_r)];
%     else
%         x = [output_gap_lag(:,1:pc_p) infl_gap_lag(:,1)];
%     end
    
%     z = infl_gap_ix;
%     e = infl_gap_lag;
%     [~,sig_trans(z,:),omegah_trans(z),~,~,scl_trans(:,z),p_scl_trans(:,z)] = svo_uniform(e,[],sig_trans(z,:),[],[],[],[],omegah_trans(z),scl_trans(:,z),p_scl_trans(:,z),prior.sv,0,0);  

%     check = 1;
%     while check > 0
%         switch spec.sv            
%             case 'RW'       
%                 
%                 [PC,sig_trans(z,:),omegah_trans(z),h0_trans(z),htilde_trans(:,z),scl_trans(:,z),p_scl_trans(:,z)] = linreg_svo_uniform(infl_gap(:,1),x,sig_trans_scl(z,:),htilde_trans(:,z),PC0,iVPC0,h0_trans(z),omegah_trans(z),scl_trans(:,z),p_scl_trans(:,z),prior.sv,0,0);
%             case 't'
%                 [PC,sig_trans(infl_gap_ix),L(:,3),nu(3)] = linreg_t(infl_gap(:,1),x,sig_trans(infl_gap_ix),PC0,iVPC0,L(:,3),nu(3),df_prior,scale_prior_pi(1),0);  % draw coefficients
%             otherwise
%                 [PC,sig_trans(infl_gap_ix,:)] = linreg(infl_gap(:,1),x,sig_trans(infl_gap_ix,1),PC0,iVPC0,df_prior,scale_prior_pi(1),0);  % draw coefficients
%         end
%         a_g = PC(1:pc_p);
%         a_p = PC(pc_p + (1:pc_q));
%         a_c = PC(pc_p + pc_q + (1:pc_r));
%         if a_p <.99, check = 0; end
%     end
%    
    % inflation trends (driftless random walks with SV)
%     log_resid = log(diff(infl_trend).^2 + 0.001);
%     e = diff(infl_trend);
%     for i = 1:infl_trends    
%         z = infl_trend_ix(i);
%         if i == 1, ident = 0; else, ident = 1;  end
%         [~,h(:,z),omegah_trans(z)] = linreg_sv_uniform(e(:,i),sig_trans(z,:),[],[],omegah_trans(z),[],[],ident);
%         sig_trans(z,:)             = h(:,z);        
%         [htilde_trans(:,z),h0_trans(z),omegah_trans(z)] = SVRW_gam(log_resid(:,i),htilde_trans(:,z),h0_trans(z),omegah_trans(z),prior.sv_trend.b0,prior.sv_trend.Vh0,prior.sv_trend.Vh,prior.sv_trend.Vomegah);     % log variance
%         sig_trans(infl_trend_ix(i),:)                   = exp((h0_trans(z) + omegah_trans(z)*htilde_trans(:,z))/2);        
%     end
    
    % factor
    z        = fac_ix(1);
    yy       = output_gap;
    xx       = output_gap_lag(:,1:p);
    iSig     = sparse(1:T,1:T,1./sig_trans_scl(z,:)');          % inverse of covariance Matrix
    xiSig    = xx'*iSig;
    Vpost    = (iVphi0 + xiSig*xx)\eye(p); 
    phi_hat  = Vpost*(iVphi0*phi0 + xiSig*yy);                       % posterior mean
    C        = chol(Vpost,'lower');  
    check    = 0;
    while check == 0
        phi =  phi_hat + C*randn(p,1);
        if (phi(1) + phi(2) < 1 && phi(2) - phi(1) < 1 && abs(phi(2))<1) == 1, check = 1; end
    end
    eps_fac        = yy - xx*phi;
    eps_fac_scaled = eps_fac./scl_trans(:,z);
    [htilde_trans(:,z),h0_trans(z),omegah_trans(z),~,~,S]  = SVRW_gam(log(eps_fac_scaled.^2 + 0.001),htilde_trans(:,z),h0_trans(z),omegah_trans(z),prior.sv_trend.b0,prior.sv_trend.Vh0,prior.sv_trend.Vh,prior.sv_trend.Vomegah);     % log variance
    ind_eps_fac = zeros(T,7);
    for i = 1:T
        ind_eps_fac(i,S(i)) = 1;
    end
    sig_trans(z,:)                                          = exp((h0_trans(z) + omegah_trans(z)*htilde_trans(:,z))/2);        
    scl_trans(:,z)                                          = draw_scale_eps(eps_fac,sig_trans(z,:)',ind_eps_fac,r_m7',r_s7',[1:7]',p_scl_trans(1:7,z));
    ps_prior =        [39.2 40-39.2];  % outlier every ten years        
    p_scl_trans(1:7,z)                                      = draw_ps(scl_trans(:,z),ps_prior,7);
    
    % factor
%     switch spec.sv
%         case 'RW'
%             z = fac_ix(1);
%             [phi,sig_trans(z,:),omegah_trans(z),h0_trans(z),htilde_trans(:,z)] = linreg_sv(output_gap,output_gap_lag(:,1:p),sig_trans(z,:),htilde_trans(:,z),phi0,iVphi0,h0_trans(z),omegah_trans(z),prior.sv,stability);
%         case 't'
%             [phi,sig_trans(z),L(:,1),nu(1)] = linreg_t(output_gap,output_gap_lag(:,1:p),sig_trans(fac_ix(1)),phi0,iVphi0,L(:,1),nu(1),df_prior,scale_prior_fac,1);
%         otherwise
%             [phi,sig_trans(z,:)] = linreg(output_gap,output_gap_lag(:,1:p),sig_trans(fac_ix(1)),phi0,iVphi0,df_prior,scale_prior_fac,0);
%     end
   
    % trends of real variables
    if n_dRWd ~= 0              % time-varying GDP trend (only shock variance)        
        z               = real_trend_ix(1);   
        e               = [x0(z);X(:,z)] - [x0(z+1);x0(z);X(1:T-1,z)]; 
        f_tau           = @(x) -T/2*log(x) - sum(diff(e).^2)./(2*x);       
        sigtau2_grid    = linspace(rand/10000,sigtau2_ub-rand/10000,n_grid);
        lp_sigtau2      = f_tau(sigtau2_grid);
        p_sigtau2       = exp(lp_sigtau2-max(lp_sigtau2));
        p_sigtau2       = p_sigtau2/sum(p_sigtau2);
        cdf_sigtau2     = cumsum(p_sigtau2);
        sig_trans(z,:)  = sqrt(sigtau2_grid(find(rand<cdf_sigtau2,1)));    
    end
    H2    = speye(T) - 2*sparse(2:T,1:(T-1),ones(1,T-1),T,T) + sparse(3:T,1:(T-2),ones(1,T-2),T,T);
    H2H2  = H2'*H2;
    Xtau0 = [(2:T+1)' -(1:T)'];    
    B0    = 10*eye(2);
    a0    = [y0(end,real_ix(1));y0(end,real_ix(1))];
    Ktau0 = B0\speye(2) + Xtau0'*H2H2*Xtau0/sig_trans(z,1)^2;
    tau0_hat = Ktau0\(B0\a0 + Xtau0'*H2H2*X(:,real_trend_ix(1))/sig_trans(z,1)^2);
    x0(z:z+1) = tau0_hat + chol(Ktau0,'lower')'\randn(2,1);
    
    % trends of real variables
    resid = diff(real_trends);
    for i = n_dRWd + 1:n_real
        [d(i),sig_trans(real_trend_ix(i),:)] = linreg(resid(:,i),ones(T,1),sig_trans(real_trend_ix(i),1),d0(i),iVd0(i),df_prior,scale_prior_real_trend(i),0);
    end
    
    sig_obs_scl    = sig_obs.*scl_obs';        
    sig_trans_scl  = sig_trans.*scl_trans';        
        
    %% update state-space
    [H,F,R,Q,A,B] = get_state_space_disagg(Hraw,Fraw,loadings,ab,a_g,a_p,a_c,f_z,c,d,f_d,phi,[],zeta,h_g,h_rho,...
        sig_obs_scl,shocks_to_R,sig_trans_scl,shocks_to_Q,n,n_real,K,s,pc_p,L,Lix,sspace);
    
    %%
    if m > burnin && mod(m,thin) == 0
        ix = ix + 1;
        if spec.forecast                                                   % compute forecasts
            % extrapolate the time-varying components
            Qhat = repmat(diag(sig_trans(:,end)),1,1,hmax  + 1);           % get the final error variance matrices for extrapolation
            Rhat = repmat(diag(sig_obs(:,end)),1,1,hmax  + 1);             % ignore the scaling factor for extrapolations
                        
            if strcmp(spec.sv,'RW')    
                % extrapolate transition error variance for nominal indicators       
                tmp_n = exp(log(sig_trans(1:n_infl+n_exp,end)) +  (omegah_trans(1:n_infl+n_exp).*[zeros(1,n_infl+n_exp); cumsum(randn(hmax,n_infl+n_exp),1)])'./2);
                % extrapolate transition error variance for activity indicators               
                tmp1   = htilde_trans(end,:) + [zeros(1,size(htilde_trans,2)); cumsum(randn(hmax,size(htilde_trans,2)),1)];   % extrapolate log standard deviations
                hnew1  = exp((h0_trans(end) + omegah_trans(fac_ix(1)).*tmp1(:,end))./2);                                            % re-transform into variance
                htrans = [tmp_n' hnew1];
                % extrapolate measurement errors variance for nominal indicators 
                tmp_n = exp(log(sig_obs(1:n_infl+n_exp,end)) +  (omegah_obs(1:n_infl+n_exp).*[zeros(1,n_infl+n_exp); cumsum(randn(hmax,n_infl+n_exp),1)])'./2);
                % extrapolate measurement errors variance for activty indicators            
%                 tmp2  = htilde_obs(end,:) + [zeros(1,size(htilde_obs,2)); cumsum(randn(hmax,size(htilde_obs,2)),1)];         % extrapolate log standard deviations
%                 hnew2 = exp((h0_obs + omegah_obs.*tmp2)./2);                                                                 % re-transform into variance
                hobs  = tmp_n';
            end
            for i = 1:size(htrans,2)
                if SV_trans(i)
                    Qhat(i,i,:) = htrans(:,i);
                end
            end
            for i = 1:size(hobs,2)
                if SV_obs(i)
                    Rhat(i,i,:) = hobs(:,i);
                end
            end
                        
            Anew = repmat(A(:,end),1,hmax + 1);
            Hnew = repmat(H(:,:,end),1,1,hmax + 1);
            if tvp                                                             % extrapolate coefficients
                cnew = [c(end,:); zeros(hmax,2)];
                for i = 1:hmax
                    cnew(i+1,:) = cnew(i,:) + mvnrnd(zeros(2,1),diag(sig_rw));
                end
                Anew                              = repmat(A(:,end),1,hmax + 1);
                Anew(end,:)                       = cnew(:,1);
                Hnew                              = repmat(H(:,:,end),1,1,hmax + 1);
                Hnew(n,K - n_infl - n_expcycle,:) = cnew(:,2);
            end
            
            yhat  = [obsdraw(end,:)', nan(n,hmax)];
            Xhat  = [X(end,:)', nan(size(X,2),hmax)];
            Qdraw = randn(size(Q,2),hmax);
            Rdraw = randn(size(R,2),hmax);
            for i = 1:hmax
                Xhat(:,i+1) = B + F*Xhat(:,i) + Qhat(:,:,i)*Qdraw(:,i);                         % extrapolate states
                yhat(:,i+1) = Anew(:,i) + Hnew(:,:,i)*Xhat(:,i+1) + Rhat(:,:,i)*Rdraw(:,i);     % extrapolate observables
            end
            post_Yhat(:,:,ix) = [obsdraw(1:end-1,:);  yhat'];
            post_Xhat(:,:,ix) = Xhat';            
        end
        % store posterior draws
        post_X(:,:,ix)         = X;                         % state draws
        post_Phi(:,ix)         = phi;                       % ar-coeff of factor
        post_Q(:,:,ix)         = sig_trans;                 % transition error variance
        post_R(:,:,ix)         = sig_obs;                   % observation error variance
        post_omega_obs(:,ix)   = omegah_obs;                % standard deviations of vola process        
        post_omega_trans(:,ix) = omegah_trans;             % standard deviations of vola process
        post_d(:,ix)           = d;                         % drift coefficients
        post_lambda(:,:,ix)    = loadings;                  % factor loadings
        post_a_g(:,ix)         = a_g;                       % output gap in PC
        post_a_p(:,ix)         = a_p;                       % PC-persistence
        post_f_z(:,ix)         = f_z;
        post_f_d(:,ix)         = f_d;
        post_ab(:,:,ix)        = ab';
        post_scl_obs(:,:,ix)   = scl_obs;
        post_scl_trans(:,:,ix) = scl_trans;        
        post_pi_trend(:,:,ix)  = X(:,infl_trend_ix(1))*ab(2,n_exp+1:end);       
        post_pi_gap(:,:,ix)    = y(:,infl_ix) -  X(:,infl_gap_ix)*ab(1,n_exp+1:end) - post_pi_trend(:,:,ix) - X(:,infl_trend_ix(2:end));    
        if tvp
            post_c(:,:,ix) = c;                         % inflation exp. coefficients
        else
            post_c(:,ix) = c;                         % inflation exp. coefficients
        end
        if n_costp > 0
            post_h_Rho(:,ix)     = h_rho;
            post_h_g(:,ix)       = h_g;
            post_a_c(:,ix)       = a_c;                       % PC-persistence
        end
        if SV == 2
            post_L(:,:,ix) = L;
        end
        if n_expcycle > 0
            post_psi(:,ix) = psi;
        end
    end
    
    if mod(m,10) == 0,  fprintf('Iteration %d of %d completed\n',m,draws);     end
    
end

%% re-attribute standard deviation to inflation series and re-transform series
if spec.forecast
    post_Yhatnew          = post_Yhat;
    post_Yhatnew(:,1:2,:) = post_Yhat(:,1:2,:);

    % re-transform log series
    y_level = nan(T+hmax,n,size(post_Yhat,3));
    for i = 1:n
        if spec.trans_vec(1,i) == 1 && spec.trans_vec(2,i) == 1
            y_level(1:end,i,:) = exp(log(spec.data_Q(spec.tau,i)) + cumsum(squeeze(post_Yhatnew(1:end,i,:))./400));
%             for j = 1:size(post_Yhat,3)
%                 y_level(1:end,i,j) = exp(log(spec.data_Q(spec.tau,i)) +  cumsum(post_Yhatnew(1:end,i,j))./400);
%             end
        elseif spec.trans_vec(1,i) == 1 && spec.trans_vec(2,i) == 0
            y_level(1:end,i,:,:) = exp(post_Yhatnew(:,i,:)./100);
        elseif spec.trans_vec(1,i) == 0 && spec.trans_vec(2,i) == 0
            y_level(1:end,i,:,:) = post_Yhatnew(:,i,:);
        end
    end
    results.y_level = y_level;
    results.states  = [prctile(post_X,[5,16,50,68,95],3); prctile(post_Xhat(2:end,:,:),[5,16,50,68,95],3)];   
    results.weights = [spec.weights; repmat(spec.weights(end,:),T + spec.hmax - size(spec.weights,1),1)];
end

%% compute aggregate trend and cycle (see eq (13) of Stock/Watson (2016)), core inflation, and headline inflation
for t = 1:T + spec.hmax
    if t <= T
        tmp               = squeeze(post_pi_trend(t,:,:));  
        trend_common(t,:) = results.weights(t,1:end)*tmp;              
    
        tmp               = squeeze(post_X(t,infl_trend_ix(2:end),:));
        trend_idio(t,:)   = results.weights(t,1:end)*tmp;  
    end
    inf_head(t,:)     = sum(results.y_level(t,infl_ix,:).*results.weights(t,:),2);      
end

results.agg_trend = trend_common + trend_idio;
results.inf_head  = inf_head;
    
%% extract important objects
results.output_gap      = squeeze(post_X(:,fac_ix(2),:));
results.trend_inflation = squeeze(post_X(:,infl_trend_ix,:));
results.real_trends     = squeeze(post_X(:,real_trend_ix,:));
results.states          = [prctile(post_X,[5,16,50,68,95],3); prctile(post_Xhat(2:end,:,:),[5,16,50,68,95],3)];   

%% posterior
% Posterior.states = post_X;            % activate for output of full state vector
posterior.Phi         = post_Phi;
posterior.h_Rho       = post_h_Rho;
posterior.h_g         = post_h_g;
posterior.lambda      = post_lambda;
posterior.a_c         = post_a_c;
posterior.a_p         = post_a_p;
posterior.a_g         = post_a_g;
posterior.b_g         = post_b_g;
posterior.b_p         = post_b_p;
posterior.b_c         = post_b_c;
posterior.c           = post_c;
posterior.d           = post_d;
posterior.f_z         = post_f_z;
posterior.f_d         = post_f_d;
posterior.ab          = post_ab;
posterior.R           = squeeze(post_R);
posterior.Q           = squeeze(post_Q);
posterior.Omega_obs   = post_omega_obs;
posterior.Omega_trans = post_omega_trans;
posterior.L           = post_L;
posterior.psi         = post_psi;
posterior.scale_obs   = post_scl_obs;
posterior.scale_trans = post_scl_trans;

posterior.means.ab          = mean(posterior.ab,3);
posterior.means.a_c         = mean(posterior.a_c,2);
posterior.means.a_p         = mean(posterior.a_p,2);
posterior.means.a_g         = mean(posterior.a_g,2);
posterior.means.b_c         = mean(posterior.b_c,2);
posterior.means.b_p         = mean(posterior.b_p,2);
posterior.means.b_g         = mean(posterior.b_g,2);
posterior.means.c           = squeeze(mean(posterior.c,3));
posterior.means.d           = mean(posterior.d,2);
posterior.means.f_z         = mean(posterior.f_z,2);
posterior.means.f_d         = mean(posterior.f_d,2);
posterior.means.h_Rho       = mean(posterior.h_Rho,2);
posterior.means.h_g         = mean(posterior.h_g,2);
posterior.means.lambda      = mean(posterior.lambda,3);
posterior.means.Phi         = mean(posterior.Phi,2);
posterior.means.psi         = mean(posterior.psi,2);
posterior.means.R           = mean(posterior.R,3);
posterior.means.Q           = mean(posterior.Q,3);
posterior.means.Omega_obs   = mean(posterior.Omega_obs,2);
posterior.means.Omega_trans = mean(posterior.Omega_trans,2);
posterior.means.L           = mean(posterior.L,3);
posterior.means.scale_obs   = mean(posterior.scale_obs,3);
posterior.means.scale_trans = mean(posterior.scale_trans,3);

%% save posterior
% if exist('posteriors/posterior_current.mat','file') == 2
%     posterior_new = posterior;
%     load('posteriors/posterior_current','posterior','estimation_date');
%     save(['posteriors/posterior_',estimation_date],'posterior');
%     posterior = posterior_new;
%     estimation_date = datestr(now,'yyyy_mm_dd');
%     save('posteriors/posterior_current','posterior','estimation_date');
% else
%     estimation_date = datestr(now,'yyyy_mm_dd');
%     save('posteriors/posterior_current','posterior','estimation_date');
% end
end