function [Results, Posterior] = run_dfm(spec,Posterior)

v2struct(spec)

set_model
set_priors
set_initial_conditions
set_state_space

v2struct(Posterior)

eps_obs   = R;
eps_trans = Q;

if ismatrix(Q)
    simulation_smoother = @disturbance_smoother;
else
    simulation_smoother = @disturbance_smoother_sv;
end

keep = (draws-burnin)/thin;
%% preallocation
post_Yhat = nan(T+hmax,n,(draws-burnin)/thin);
post_X    = nan(T,K,(draws-burnin)/thin);

for m = 1:keep
    
    %% build state-space
    if n_infl > 1 && n_costp > 0
        [H,F,R,Q,A,B] = get_state_space_extended(Hraw,Fraw,lambda(:,:,m),a_g(:,m),a_p(m),a_c(m),b_g(:,m),b_p(m),b_c(m),f_z(:,m),c(:,m),d(:,m),f_d(:,m),...
            Phi(:,m),h_g(:,m),h_Rho(:,m),delta,eps_obs(:,m),shocks_to_R,eps_trans(:,m),shocks_to_Q,n,n_real,K,s,pp,L,Lix);
    elseif n_infl > 1 && n_costp == 0
        [H,F,R,Q,A,B] = get_state_space_extended(Hraw,Fraw,lambda(:,:,m),a_g(:,m),a_p(m),b_g(:,m),b_p(m),f_z(:,m),c(:,m),d(:,m),f_d(:,m),...
            Phi(:,m),h_g(:,m),h_Rho(:,m),delta,eps_obs(:,m),shocks_to_R,eps_trans(:,m),shocks_to_Q,n,n_real,K,s,pp,L,Lix);
    elseif n_infl == 1 && n_costp == 1
        [H,F,R,Q,A,B] = get_state_space_extended(Hraw,Fraw,lambda(:,:,m),a_g(:,m),a_p(m),a_c(m),[],[],[],f_z(:,m),c(:,m),d(:,m),f_d(:,m),...
            Phi(:,m),h_g(:,m),h_Rho(:,m),delta,eps_obs(:,m),shocks_to_R,eps_trans(:,m),shocks_to_Q,n,n_real,K,s,pp,L,Lix);
    elseif n_infl == 1 && n_costp == 0
        [H,F,R,Q,A,B] = get_state_space_extended(Hraw,Fraw,lambda(:,:,m),a_g(:,m),a_p(m),[],[],[],[],f_z(:,m),c(:,m),d(:,m),f_d(:,m),...
            Phi(:,m),h_g(:,m),h_Rho(:,m),[],eps_obs(:,m),shocks_to_R,eps_trans(:,m),shocks_to_Q,n,n_real,K,s,pp,L,Lix);
    end
    
    X = simulation_smoother(y',A,H,R,B,F,Q,x0,cSig0,2);
    
    %% draw observables conditionally on the states and parameters
    obsdraw = y;
    for i = 1:length(nan_id)
        per               = nan_id(i);
        inan              = find(isnan(y(per,:)));
        obsdraw(per,inan) = (A(inan) + H(inan,:)*X(per,:)' + R(inan,:)*randn(size(R,2),1))';
    end
    
    % compute forecasts
    Qhat  = repmat(Q(:,:,end),1,1,hmax);                                     % extrapolate state-covariance
    yhat = [obsdraw(end,:)', nan(n,hmax)];
    Xhat = [X(end,:)', nan(size(X,2),hmax)];
    for h = 1:hmax
        Xhat(:,h+1) = B + F*Xhat(:,h) + Qhat(:,:,h)*randn(size(Q,2),1);
        yhat(:,h+1) = A + H*Xhat(:,h+1) + R*randn(size(R,2),1);
    end
    
    post_X(:,:,m)    = X;
    post_Yhat(:,:,m) = [obsdraw(1:end-1,:);  yhat'];
    
    if mod(m,100) == 0,  fprintf('Iteration %d of %d completed\n',m,keep);     end
    
end

%% re-attribute standard deviation to inflation series and re-transform series
post_Yhatnew          = post_Yhat;
% post_Yhatnew(:,1:2,:) = post_Yhat(:,1:2,:)./delta;

% %% re-transform log series
y_level = nan(T+hmax,n,size(post_Yhat,3));
for i = 1:n
    if spec.trans_vec(1,i) == 1 && spec.trans_vec(2,i) == 1
        for j = 1:size(post_Yhat,3)
            y_level(1:end,i,j) = exp(log(spec.data_Q(spec.tau,i)) +  cumsum(post_Yhatnew(1:end,i,j))./400);
        end
    elseif spec.trans_vec(1,i) == 1 && spec.trans_vec(2,i) == 0
        y_level(1:end,i,:) = exp(post_Yhatnew(:,i,:)./100);
    else
        y_level(1:end,i,:) = (post_Yhatnew(:,i,:));
    end
end

Results.y_level = y_level;

%% extract important objects
Results.output_gap      = squeeze(post_X(:,fac_ix(2),:));
Results.trend_inflation = squeeze(post_X(:,infl_trend_ix,:));
Results.real_trends     = squeeze(post_X(:,real_trend_ix,:));

