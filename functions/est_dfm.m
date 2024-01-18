function [results, posterior] = est_dfm(spec)

v2struct(spec)

set_model
set_priors
set_initial_conditions
set_state_space


%% preallocation
post_X                        = nan(T,K,(draws-burnin)/thin);
post_Xhat                     = nan(hmax+1,K,(draws-burnin)/thin);
post_Q                        = (nan(K,max([1,T*SV(SV==1)]),(draws-burnin)/thin));                          % transition error variance
post_R                        = nan(n,(draws-burnin)/thin);                            % observation error variance
post_d                        = nan(n_real,(draws-burnin)/thin);                       % drift coefficients
post_lambda                   = nan(n,s,(draws-burnin)/thin);                          % factor loadings
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


%% intialize the state vector
% 
f_d                     = [0; 0];
sig_trans(costp_trend_ix,:)   = 10^-9;

ix    = 0;
for m = 1:draws
    
    
    %% draw states    
    [X, ~] = simulation_smoother(y',A,H,R,B,F,Q,x0,cSig0,2);
    
    % append a draw of states in S+1 - needed for the regressions with lags
    X1 = [X; (B + F*X(end,:)' + Q(:,:,end)*randn(nQ_shocks,1))'];
    
    %% extract objects from state vector
    output_gap   = X1(2:end,1);
    real_trends  = X1(:,real_trend_ix);
    costp_trend  = X1(2:end,costp_trend_ix);
    costp_gap    = X1(2:end,costp_gap_ix);
    infl_trend   = X1(2:end,infl_trend_ix);
    infl_gap     = X1(2:end,infl_gap_ix);
    
    output_gap_lag  = X(:,fac_ix);
    real_trends_lag = X(:,real_trend_ix);
    costp_trend_lag = X(:,costp_trend_ix);
    costp_gap_lag   = X(:,costp_gap_ix);
    infl_trend_lag  = X(:,infl_trend_ix);
    infl_gap_lag    = X(:,infl_gap_ix);
    
    %% draw observables conditionally on the states and parameters
    obsdraw = y;
    for i = 1:length(nan_id)
        per               = nan_id(i);
        if ~tvp, j = 1; else, j = per; end
        inan              = find(isnan(y(per,:)));
        obsdraw(per,inan) = (A(inan,j) + H(inan,:,j)*X(per,:)' + R(inan,:)*randn(nR_shocks,1))';
    end
    
    %% draw coefficients of measurement equations
    
    % real variables: factor loadings and measurement error
    y_cycle = obsdraw(:,n_infl + (1:n_real)) - real_trends_lag;  % draw loadings conditional on trend, ignore estimated states
    for i = 2:n_real                                              % GDP loading is fixed to unity, only real variables load on factor
        [loadings(i,1:p+1),sig_obs(i+ n_infl)] = linreg(y_cycle(:,i),output_gap_lag(:,1:s),sig_obs(i+n_infl),lambda0,diag(iVlambda0(:,i)),df_prior,scale_prior_real(i),0);
    end
    
    % inflation expectations
    x = [ones(T,1) infl_trend_lag];
    if n_expcycle > 0
        Ystar           = filter([1 -psi'],1,obsdraw(:, exp_ix));        % remove serial correlation in measurement error by computing quasi-differences
        Xstar           = filter([1 -psi'],1,x);                        % remove serial correlation in measurement error by computing quasi-differences
        sig_obs(exp_ix) = 1;
    else
        Ystar = obsdraw(:,exp_ix);
        Xstar = x;
    end
       
    if tvp
        [c,muc,rhoc,sig_obs(exp_ix,:),sig_rw,acc] = tvpreg_ar1(Ystar,Xstar,sig_obs(exp_ix,:),sig_rw,muc,rhoc,mud0,Vmud,rhod0,Vrhod,df_prior,scale_prior_pie,scale_prior_c,acc);
    else
        [c,sig_obs(exp_ix)] = linreg(Ystar,Xstar,sig_obs(exp_ix),c0,iVC0,df_prior,scale_prior_pie,0); % draw coefficients
    end
     
    %% draw coefficients of transition equations
    % factor
    shock_ix = 1;
    switch spec.sv
        case 'RW'
            [phi,sig_trans(shock_ix,:),omegah(1),h0(1),htilde(:,1)] = linreg_sv(output_gap,output_gap_lag(:,1:p),sig_trans(shock_ix,:),htilde(:,1),phi0,iVphi0,h0(1),omegah(1),prior.sv,stability);
        case 't'
            [phi,sig_trans(shock_ix),L(:,1),nu(1)] = linreg_t(output_gap,output_gap_lag(:,1:p),sig_trans(shock_ix),phi0,iVphi0,L(:,1),nu(1),df_prior,scale_prior_fac,1);
        otherwise
            [phi,sig_trans(shock_ix,:)] = linreg(output_gap,output_gap_lag(:,1:p),sig_trans(shock_ix),phi0,iVphi0,df_prior,scale_prior_fac,0);
    end
    
    if n_dRWd ~= 0              % time-varying GDP trend (only shock variance)        
        shock_ix              = s + 1;    
        e                     = [x0(4);X(:,4)] - [x0(5);x0(4);X(1:T-1,4)]; 
        f_tau                 = @(x) -T/2*log(x) - sum((e(2:T) - e(1:T-1)).^2)./(2*x);       
        sigtau2_grid          = linspace(rand/1000,sigtau2_ub-rand/1000,n_grid);
        lp_sigtau2            = f_tau(sigtau2_grid);
        p_sigtau2             = exp(lp_sigtau2-max(lp_sigtau2));
        p_sigtau2             = p_sigtau2/sum(p_sigtau2);
        cdf_sigtau2           = cumsum(p_sigtau2);
        sig_trans(shock_ix,:) = sigtau2_grid(find(rand<cdf_sigtau2, 1 ));    
    end
    
    % trends of real variables
    resid = diff(real_trends);
    for i = n_dRWd + 1:n_real
        shock_ix                     = n_dRWd + s + i;
        [d(i),sig_trans(shock_ix,:)] = linreg(resid(:,i),ones(T,1),sig_trans(shock_ix,1),d0(i),iVd0(i),df_prior,scale_real_trend(i),0);
    end
    
    if n_costp > 0
        % cost-push trend
        shock_ix = shock_ix + 1;
%         [f_d,sig_trans(shock_ix,:)] = linreg(costp_trend,[ones(T,1) costp_trend_lag],sig_trans(shock_ix),fd0,iVfd0,df_prior,scale_prior_cptrend,0);
        % cost-push gap
        shock_ix = shock_ix + 1;
        [CP,sig_trans(shock_ix,:)] = linreg(costp_gap(:,1),[costp_gap_lag(:,1:cp_r) output_gap_lag(:,1:cp_q)],sig_trans(shock_ix),CP0,iVCP0,df_prior,scale_prior_cpgap,1);
        h_rho = CP(1:cp_r);
        h_g   = CP(cp_r + 1:end);
    end
    
    % trend inflation
    x        = [ones(T,1) infl_trend_lag];
    shock_ix = shock_ix + 1 + (length(costp_gap_ix)-1)*n_costp;
    switch spec.sv
        case 'RW'
            [f_z,sig_trans(shock_ix,:),omegah(2),h0(2),htilde(:,2)] = linreg_sv(infl_trend,x,sig_trans(shock_ix,:),htilde(:,2),fz0,iVfz0,h0(2),omegah(2),prior.sv,0);
        case 't'
            [f_z,sig_trans(shock_ix),L(:,2),nu(2)] = linreg_t(infl_trend,x,sig_trans(shock_ix),fz0,iVfz0,L(:,2),nu(2),df_prior,scale_prior_pi_trend,0);
        otherwise
            [f_z,sig_trans(shock_ix,:)] = linreg(infl_trend,x,sig_trans(shock_ix,1),fz0,iVfz0,df_prior,scale_prior_pi_trend,0);
    end
    
    % Phillips-Curve I
    if n_costp > 0
        x = [output_gap_lag(:,1:pc_p) infl_gap_lag(:,1:pc_q) costp_gap_lag(:,1:pc_r)];
    else
        x = [output_gap_lag(:,1:pc_p) infl_gap_lag(:,1)];
    end
    shock_ix = shock_ix + 1;
    switch spec.sv
        case 'RW'
            [PC,sig_trans(shock_ix,:),omegah(3),h0(3),htilde(:,3)] = linreg_sv(infl_gap(:,1),x,sig_trans(shock_ix,:),htilde(:,3),PC0,iVPC0,h0(3),omegah(3),prior.sv,0);  % draw coefficients
        case 't'
            [PC,sig_trans(shock_ix),L(:,3),nu(3)] = linreg_t(infl_gap(:,1),x,sig_trans(shock_ix),PC0,iVPC0,L(:,3),nu(3),df_prior,scale_prior_pi(1),0);  % draw coefficients
        otherwise
            [PC,sig_trans(shock_ix,:)] = linreg(infl_gap(:,1),x,sig_trans(shock_ix,1),PC0,iVPC0,df_prior,scale_prior_pi(1),0);  % draw coefficients
    end
    a_g = PC(1:pc_p);
    a_p = PC(pc_p + (1:pc_q));
    a_c = PC(pc_p + pc_q + (1:pc_r));
    
    % Phillips-Curve II
    if n_infl > 1
        if n_costp > 0
            x = [output_gap_lag(:,1:pc_p) infl_gap_lag(:,pc_q+(1:pc_q)) costp_gap_lag(:,1:pc_r)];
        else
            x = [output_gap_lag(:,1:pc_p) infl_gap_lag(:,pc_q+(1:pc_q))];
        end
        shock_ix = shock_ix + 1 + max([0 pc_q-1]);
        switch spec.sv
            case 'RW'
                [PC,sig_trans(shock_ix,:),omegah(4),h0(4),htilde(:,4)] = linreg_sv(infl_gap(:,pc_q+1),x,sig_trans(shock_ix,:),htilde(:,4),PC0_head,iVPC0_head,h0(4),omegah(4),prior.sv,0);  % draw coefficients
            case 't'
                [PC,sig_trans(shock_ix),L(:,4),nu(4)] = linreg_t(infl_gap(:,pc_q+1),x,sig_trans(shock_ix),PC0,iVPC0_head,L(:,4),nu(4),df_prior,scale_prior_pi(n_infl),0);  % draw coefficients
            otherwise
                [PC,sig_trans(shock_ix,:)] = linreg(infl_gap(:,pc_q+1),x,sig_trans(shock_ix),PC0,iVPC0_head,df_prior,scale_prior_pi(n_infl),0);  % draw coefficients
        end
        b_g = PC(1:pc_p);
        b_p = PC(pc_p + (1:pc_q));
        b_c = PC(pc_p + pc_q + (1:pc_r));
    end
    
    % inflation expectations error cycle
    if n_expcycle > 0
        shock_ix                  = shock_ix + 1 + max([0 pc_q-1]);
        keep                      = ~isnan(y(:,exp_ix));
        Y                         = X(keep,infl_expcyc_ix(1));
        x                         = [lag0(Y,1) lag0(Y,2)];
        [psi,sig_trans(shock_ix)] = linreg(Y(3:end,1),x(3:end,:),sig_trans(shock_ix),psi0,iVpsi0,df_prior,scale_prior_expcyc,1);
    end
    
    % cost-push error cycle
    if n_costpcycle > 0
        shock_ix                  = shock_ix + 2;
        Y                         = X(:,costp_cycle_ix(1));
        x                         = [lag0(Y,1) lag0(Y,2)];
        [zeta,sig_trans(shock_ix)] = linreg(Y(3:end,1),x(3:end,:),sig_trans(shock_ix),zeta0,iVzeta0,df_prior,scale_prior_costpcyc,1);
    end
    
    %% update state-space
    [H,F,R,Q,A,B] = get_state_space_extended(Hraw,Fraw,loadings,a_g,a_p,a_c,b_g,b_p,b_c,f_z,c,d,f_d,phi,psi,zeta,h_g,h_rho,delta,...
                    sig_obs,shocks_to_R,sig_trans,shocks_to_Q,n,n_real,K,s,pc_p,L,Lix,sspace);

    %%
    if m > burnin && mod(m,thin) == 0
        % compute forecasts
        Qhat = zeros(K,K,hmax + 1);
        Qhat(:,shocks_to_Q,:) = repmat(Q(:,:,end),1,1,hmax  + 1);
        if strcmp(spec.sv,'RW')
            htilde_new    = htilde(end,:) + [zeros(1,size(htilde,2)); cumsum(randn(hmax,size(htilde,2)),1)];         % extrapolate volatilities
            hnew          = exp(0.5.*(h0 + omegah.*htilde_new));
            
            Qhat(fac_ix(1),fac_ix(1),:)           = hnew(:,1);
%             Qhat(infl_trend_ix,infl_trend_ix,:)   = hnew(:,2);
            Qhat(infl_gap_ix(1),infl_gap_ix(1),:) = hnew(:,3);
            if n_infl >1
                Qhat(infl_gap_ix(pc_q+1),infl_gap_ix(pc_q+1),:) = hnew(:,4);
            end
        end
        Qhat = Qhat(:,shocks_to_Q,:);
        
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
            Xhat(:,h+1) = B + F*Xhat(:,h) + Qhat(:,:,h)*Qdraw(:,h);             % extrapolate states
            yhat(:,h+1) = Anew(:,h) + Hnew(:,:,h)*Xhat(:,h+1) + R*Rdraw(:,h);     % extrapolate observables
        end
        
        % store posterior draws
        ix                   = ix + 1;
        post_X(:,:,ix)       = X;                         % state draws
        post_Phi(:,ix)       = phi;                       % ar-coeff of factor
        post_Q(:,:,ix)       = sig_trans;                 % transition error variance
        post_R(:,ix)         = sig_obs;                   % observation error variance
        post_d(:,ix)         = d;                         % drift coefficients
        post_lambda(:,:,ix)  = loadings;                  % factor loadings
        post_a_g(:,ix)       = a_g;                       % output gap in PC
        post_a_p(:,ix)       = a_p;                       % PC-persistence
        post_f_z(:,ix)       = f_z;
        post_f_d(:,ix)       = f_d;
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
        if n_infl > 1
            post_b_g(:,ix) = b_g;
            post_b_p(:,ix) = b_p;
            post_b_c(:,ix) = b_c;
        end
        if SV == 2
            post_L(:,:,ix) = L;
        end
        if n_expcycle > 0
            post_psi(:,ix) = psi;
        end
        post_Yhat(:,:,ix) = [obsdraw(1:end-1,:);  yhat'];
        post_Xhat(:,:,ix) = Xhat';
    end
    
    if mod(m,10) == 0,  fprintf('Iteration %d of %d completed\n',m,draws);     end
    
end

%% re-attribute standard deviation to inflation series and re-transform series
post_Yhatnew          = post_Yhat;
post_Yhatnew(:,1:2,:) = post_Yhat(:,1:2,:);

% %% re-transform log series
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
posterior.R      = post_R;
posterior.Q      = squeeze(post_Q);
posterior.L      = post_L;
posterior.psi    = post_psi;

posterior.means.a_c = mean(posterior.a_c,2);
posterior.means.a_p = mean(posterior.a_p,2);
posterior.means.a_g = mean(posterior.a_g,2);
posterior.means.b_c = mean(posterior.b_c,2);
posterior.means.b_p = mean(posterior.b_p,2);
posterior.means.b_g = mean(posterior.b_g,2);
posterior.means.c   = squeeze(mean(posterior.c,3));
posterior.means.d   = mean(posterior.d,2);
posterior.means.f_z = mean(posterior.f_z,2);
posterior.means.f_d = mean(posterior.f_d,2);
posterior.means.h_Rho = mean(posterior.h_Rho,2);
posterior.means.h_g = mean(posterior.h_g,2);
posterior.means.lambda = mean(posterior.lambda,3);
posterior.means.Phi = mean(posterior.Phi,2);
posterior.means.psi = mean(posterior.psi,2);
posterior.means.R = mean(posterior.R,2);
posterior.means.Q = mean(posterior.Q,3);
posterior.means.L = mean(posterior.L,3);


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