function [results, posterior, spec] = est_dfm_disagg(spec)

v2struct(spec)

set_model_disagg
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
post_ab                       = nan(n_infl+1,2,(draws-burnin)/thin);
post_pi_trend                 = nan(T,n_infl,(draws-burnin)/thin);
post_pi_gap                   = nan(T,n_infl,(draws-burnin)/thin);

% PC0(2) = 0;
% iVPC0  = inv(eye(4).*10^-9);

R(real_ix(1),real_ix(1),:) = 10^-9;

mj       = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
sqrtsigj = sqrt([5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261]);

%% intialize the state vector
%
ix    = 0;
for m = 1:draws
    
    
    %% draw states
    [X,~,~,~] = DurbinKoopman(y',x0,cSig0,F,H,Q,R,A,B,K,T,n);
    X1        = [X; (B + F*X(end,:)' + Q(:,:,end)*randn(K,1))'];
    
    
%     [X, ~] = simulation_smoother(y',A,H,R,B,F,Q,x0,cSig0,2);    
%     % append a draw of states in S+1 - needed for the regressions with lags
%     X1 = [X; (B + F*X(end,:)' + Q(:,:,end)*randn(nQ_shocks,1))'];
    
    
    %% extract objects from state vector
    output_gap   = X1(2:end,fac_ix(1));
    real_trends  = X1(:,real_trend_ix);
    costp_trend  = X1(2:end,costp_trend_ix);
    costp_gap    = X1(2:end,costp_gap_ix);
    infl_trend   = X1(:,infl_trend_ix);
    infl_gap     = X1(2:end,infl_gap_ix);
    
    output_gap_lag   = X(:,fac_ix);
    real_trends_lag  = X(:,real_trend_ix);
    costp_trend_lag  = X(:,costp_trend_ix);
    costp_gap_lag    = X(:,costp_gap_ix);
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
    
    
    % inflation exp.
    i                                               = exp_ix;
    e                                               = (obsdraw(:,i) - X(:,infl_trend_ix(1)))./scl_obs(:,i);       % draw loadings conditional on common inflation noise
    [htilde_obs(:,i),h0_obs(i),omegah_obs(i),~,~,S] = SVRW_gam(log(e.^2 + 0.001),htilde_obs(:,i),h0_obs(i),omegah_obs(i),prior.sv.b0,prior.sv.Vh0,prior.sv.Vh,prior.sv.Vomegah);     % log variance
    sig_obs(i,:)                                    = exp((h0_obs(i) + omegah_obs(i)*htilde_obs(:,i))/2);            % standard deviations
    scl_obs(:,i)                                    = draw_scale(e,sig_obs(i,:)',mj(S)',sqrtsigj(S)',prior.sv.scl_eps_vec,p_scl_obs(:,i));                     % sample outlier
    p_scl_obs(:,i)                                  = draw_ps(scl_obs(:,i),prior.sv.ps_prior,length(sqrtsigj));
    
    
    % nominal variables: loadings on common trend and gap
    x      = X(:,[infl_trend_ix(1), infl_gap_ix]);                           % select common inflation states    
    pi_gap = obsdraw(:,infl_ix) - X(:,infl_trend_ix(2:end));                 % draw loadings conditional on idioyncratic inflation trends    
    for i = 1 + n_exp:n_exp + n_infl          
        if i == 1 + n_exp
            yy = pi_gap(:,i) - x(:,2);
            xx = x(:,1);
            [ab(1,i),sig_obs(i,:),omegah_obs(i),h0_obs(i),htilde_obs(:,i),scl_obs(:,i),p_scl_obs(:,i)] = linreg_svo(yy,xx,sig_obs_scl(i,:),htilde_obs(:,i),ab0(1),iVab0(1),h0_obs(i),omegah_obs(i),scl_obs(:,i),p_scl_obs(:,i),prior.sv,0,0);
        else
            [ab(:,i),sig_obs(i,:),omegah_obs(i),h0_obs(i),htilde_obs(:,i),scl_obs(:,i),p_scl_obs(:,i)] = linreg_svo(pi_gap(:,i - n_exp),x,sig_obs_scl(i,:),htilde_obs(:,i),ab0,iVab0,h0_obs(i),omegah_obs(i),scl_obs(:,i),p_scl_obs(:,i),prior.sv,0,0);
        end
    end
       
    % real variables: factor loadings and measurement error
    y_cycle = obsdraw(:,real_ix) - real_trends_lag;  % draw loadings conditional on trend, ignore estimated states
    for i = 2:n_real                                              % GDP loading is fixed to unity, only real variables load on factor
        [loadings(i,1:p+1),sig_obs(real_ix(i),:)] = linreg(y_cycle(:,i),output_gap_lag(:,1:s),sig_obs(real_ix(i),1),lambda0,diag(iVlambda0(:,i)),df_prior,scale_prior_real(i),0);
    end
    
    %% draw coefficients of transition equations
    % Phillips-Curve (inflation gap process)
    if n_costp > 0
        x = [output_gap_lag(:,1:pc_p) infl_gap_lag(:,1:pc_q) costp_gap_lag(:,1:pc_r)];
    else
        x = [output_gap_lag(:,1:pc_p) infl_gap_lag(:,1)];
    end
    
    check = 1;
    while check > 0
        switch spec.sv
            case 'RW'       
                z = infl_gap_ix;
                [PC,sig_trans(z,:),omegah_trans(z),h0_trans(z),htilde_trans(:,z),scl_trans(:,z),p_scl_trans(:,z)] = linreg_svo(infl_gap(:,1),x,sig_trans_scl(z,:),htilde_trans(:,z),PC0,iVPC0,h0_trans(z),omegah_trans(i),scl_trans(:,z),p_scl_trans(:,z),prior.sv,0,1);
            case 't'
                [PC,sig_trans(infl_gap_ix),L(:,3),nu(3)] = linreg_t(infl_gap(:,1),x,sig_trans(infl_gap_ix),PC0,iVPC0,L(:,3),nu(3),df_prior,scale_prior_pi(1),0);  % draw coefficients
            otherwise
                [PC,sig_trans(infl_gap_ix,:)] = linreg(infl_gap(:,1),x,sig_trans(infl_gap_ix,1),PC0,iVPC0,df_prior,scale_prior_pi(1),0);  % draw coefficients
        end
        a_g = PC(1:pc_p);
        a_p = PC(pc_p + (1:pc_q));
        a_c = PC(pc_p + pc_q + (1:pc_r));
        if a_p <.99, check = 0; end
    end
    
    % inflation trends (driftless random walks with SV)
    log_resid = log(diff(infl_trend).^2 + 0.001);
    for i = 1:infl_trends    
        z = infl_trend_ix(i);
        [htilde_trans(:,z),h0_trans(z),omegah_trans(z)] = SVRW_gam(log_resid(:,i),htilde_trans(:,z),h0_trans(z),omegah_trans(z),prior.sv_trend.b0,prior.sv_trend.Vh0,prior.sv_trend.Vh,prior.sv_trend.Vomegah);     % log variance
        sig_trans(infl_trend_ix(i),:)                   = exp((h0_trans(z) + omegah_trans(z)*htilde_trans(:,z))/2);
    end
    
    % factor
    switch spec.sv
        case 'RW'
            z = fac_ix(1);
            [phi,sig_trans(fac_ix(1),:),omegah_trans(z),h0_trans(z),htilde_trans(:,z)] = linreg_sv(output_gap,output_gap_lag(:,1:p),sig_trans(z,:),htilde_trans(:,z),phi0,iVphi0,h0_trans(z),omegah_trans(z),prior.sv,stability);
        case 't'
            [phi,sig_trans(fac_ix(1)),L(:,1),nu(1)] = linreg_t(output_gap,output_gap_lag(:,1:p),sig_trans(fac_ix(1)),phi0,iVphi0,L(:,1),nu(1),df_prior,scale_prior_fac,1);
        otherwise
            [phi,sig_trans(fac_ix(1),:)] = linreg(output_gap,output_gap_lag(:,1:p),sig_trans(fac_ix(1)),phi0,iVphi0,df_prior,scale_prior_fac,0);
    end
   
    % trends of real variables
    resid = diff(real_trends);
    for i = n_dRWd + 1:n_real
        [d(i),sig_trans(real_trend_ix(i),:)] = linreg(resid(:,i),ones(T,1),sig_trans(real_trend_ix(i),1),d0(i),iVd0(i),df_prior,scale_prior_real_trend(i),0);
    end
    
    sig_obs_scl                  = sig_obs;
    sig_obs_scl(1:n_infl,:)      = sig_obs(1:n_infl,:).*scl_obs(:,1:n_infl)';        
    
    sig_trans_scl                = sig_trans;
    sig_trans_scl(infl_gap_ix,:) = sig_trans(infl_gap_ix,:).*scl_trans(:,1)';        
        
    %% update state-space
    [H,F,R,Q,A,B] = get_state_space_disagg(Hraw,Fraw,loadings,ab,a_g,a_p,a_c,f_z,c,d,f_d,phi,[],zeta,h_g,h_rho,...
        sig_obs_scl,shocks_to_R,sig_trans_scl,shocks_to_Q,n,n_real,K,s,pc_p,L,Lix,sspace);
    
    %%
    if m > burnin && mod(m,thin) == 0
        ix = ix + 1;
        if spec.forecast
            % compute forecasts
            Qhat = zeros(K,K,hmax + 1);
            Rhat = zeros(n,n,hmax + 1);            
            Qhat = repmat(Q(:,:,end),1,1,hmax  + 1);
            Rhat = repmat(R(:,:,end),1,1,hmax  + 1);
            
            if strcmp(spec.sv,'RW')    
                %extrapolate transition error variance                
                tmp1  = htilde_trans(end,:) + [zeros(1,size(htilde_trans,2)); cumsum(randn(hmax,size(htilde_trans,2)),1)];         % extrapolate log standard deviations
                hnew1 = exp((h0_trans + omegah_trans.*tmp1)./2);
                %extrapolate measurement errors variance
                tmp2  = htilde_obs(end,:) + [zeros(1,size(htilde_obs,2)); cumsum(randn(hmax,size(htilde_obs,2)),1)];         % extrapolate log standard deviations
                hnew2 = exp((h0_obs + omegah_obs.*tmp2)./2);                
            end
            for i = 1:size(hnew1,2)
                if SV_trans(i)
                    Qhat(i,i,:) = hnew1(:,i);
                end
            end
            for i = 1:size(hnew2,2)
                if SV_obs(i)
                    Rhat(i,i,:) = hnew2(:,i);
                end
            end
%           Qhat = Qhat(:,shocks_to_Q,:);
%             Rhat = Rhat(:,shocks_to_R,:);

            
%             %extrapolate measurement errors
%             htilde_new    = htilde_p(end,:) + [zeros(1,size(htilde_p,2)); cumsum(randn(hmax,size(htilde_p,2)),1)];         % extrapolate volatilities
%             hnew          = exp((h0_p + omegah_p.*htilde_new)./2);
%             for i = 1:hmax+1
%                 Rhat(1:n_infl,1:n_infl,i) = diag(hnew(i,:));
%             end
%             Rhat = Rhat(:,shocks_to_R,:);
            
            Anew = repmat(A(:,end),1,hmax + 1);
            Hnew = repmat(H(:,:,end),1,1,hmax + 1);
            if tvp                                                             % extrapolate coefficients
                cnew = [c(end,:); zeros(hmax,2)];
                for h = 1:hmax
                    cnew(h+1,:) = cnew(h,:) + mvnrnd(zeros(2,1),diag(sig_rw));
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
            for h = 1:hmax
                Xhat(:,h+1) = B + F*Xhat(:,h) + Qhat(:,:,h)*Qdraw(:,h);                         % extrapolate states
                yhat(:,h+1) = Anew(:,h) + Hnew(:,:,h)*Xhat(:,h+1) + Rhat(:,:,h)*Rdraw(:,h);     % extrapolate observables
            end
            post_Yhat(:,:,ix) = [obsdraw(1:end-1,:);  yhat'];
            post_Xhat(:,:,ix) = Xhat';            
        end
        % store posterior draws
        post_X(:,:,ix)        = X;                         % state draws
        post_Phi(:,ix)        = phi;                       % ar-coeff of factor
        post_Q(:,:,ix)        = sig_trans;                 % transition error variance
        post_R(:,:,ix)        = sig_obs;                   % observation error variance
        post_d(:,ix)          = d;                         % drift coefficients
        post_lambda(:,:,ix)   = loadings;                  % factor loadings
        post_a_g(:,ix)        = a_g;                       % output gap in PC
        post_a_p(:,ix)        = a_p;                       % PC-persistence
        post_f_z(:,ix)        = f_z;
        post_f_d(:,ix)        = f_d;
        post_ab(:,:,ix)       = ab';
        post_scale(:,:,ix)    = [scl_obs scl_trans];
        post_pi_trend(:,:,ix) = X(:,infl_trend_ix(1))*ab(2,2:end);       
        post_pi_gap(:,:,ix)   = y(:,infl_ix) -  X(:,infl_gap_ix)*ab(1,2:end) - post_pi_trend(:,:,ix) - X(:,infl_trend_ix(2:end));    
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
            for j = 1:size(post_Yhat,3)
                y_level(1:end,i,j) = exp(log(spec.data_Q(spec.tau,i)) +  cumsum(post_Yhatnew(1:end,i,j))./400);
            end
        elseif spec.trans_vec(1,i) == 1 && spec.trans_vec(2,i) == 0
            y_level(1:end,i,:) = exp(post_Yhatnew(:,i,:)./100);
        end
    end
    results.y_level = y_level;
    results.states  = [prctile(post_X,[5,16,50,68,95],3); prctile(post_Xhat(2:end,:,:),[5,16,50,68,95],3)];   
end

%% compute aggregate trend and cycle (see eq (13) of Stock/Watson (2016))
for t = 1:T    
    tmp               = squeeze(post_pi_trend(t,:,:));  
    trend_common(t,:) = spec.weights(t,1:end)*tmp;              
    
    tmp               = squeeze(post_X(t,infl_trend_ix(2:end),:));
    trend_idio(t,:)   = spec.weights(t,1:end)*tmp;    
end

results.agg_trend = trend_common + trend_idio;

    
%% extract important objects
results.output_gap      = squeeze(post_X(:,fac_ix(2),:));
results.trend_inflation = squeeze(post_X(:,infl_trend_ix,:));
results.real_trends     = squeeze(post_X(:,real_trend_ix,:));
results.states          = [prctile(post_X,[5,16,50,68,95],3); prctile(post_Xhat(2:end,:,:),[5,16,50,68,95],3)];   

%% posterior
% Posterior.states = post_X;            % activate for output of full state vector
posterior.Phi    = post_Phi;
posterior.h_Rho  = post_h_Rho;
posterior.h_g    = post_h_g;
posterior.lambda = post_lambda;
posterior.a_c    = post_a_c;
posterior.a_p    = post_a_p;
posterior.a_g    = post_a_g;
posterior.b_g    = post_b_g;
posterior.b_p    = post_b_p;
posterior.b_c    = post_b_c;
posterior.c      = post_c;
posterior.d      = post_d;
posterior.f_z    = post_f_z;
posterior.f_d    = post_f_d;
posterior.ab     = post_ab;
posterior.R      = squeeze(post_R);
posterior.Q      = squeeze(post_Q);
posterior.L      = post_L;
posterior.psi    = post_psi;
posterior.scale  = post_scale;

posterior.means.a_c    = mean(posterior.a_c,2);
posterior.means.a_p    = mean(posterior.a_p,2);
posterior.means.a_g    = mean(posterior.a_g,2);
posterior.means.b_c    = mean(posterior.b_c,2);
posterior.means.b_p    = mean(posterior.b_p,2);
posterior.means.b_g    = mean(posterior.b_g,2);
posterior.means.c      = squeeze(mean(posterior.c,3));
posterior.means.d      = mean(posterior.d,2);
posterior.means.f_z    = mean(posterior.f_z,2);
posterior.means.f_d    = mean(posterior.f_d,2);
posterior.means.ab     = mean(posterior.ab,3);
posterior.means.h_Rho  = mean(posterior.h_Rho,2);
posterior.means.h_g    = mean(posterior.h_g,2);
posterior.means.lambda = mean(posterior.lambda,3);
posterior.means.Phi    = mean(posterior.Phi,2);
posterior.means.psi    = mean(posterior.psi,2);
posterior.means.R      = mean(posterior.R,3);
posterior.means.Q      = mean(posterior.Q,3);
posterior.means.L      = mean(posterior.L,3);
posterior.means.scale  = mean(posterior.scale,3);

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