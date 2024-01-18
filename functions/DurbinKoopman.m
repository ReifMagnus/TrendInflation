function [x_tT,x_tt,Sig_tt,Sig_t1] = DurbinKoopman(y,x0,cSig0,F,H,Q,R,A,B,k,T,n)
% Durbin and Koopman (2003) Simulation Smoother
% Ouput: 
% x_tT   = smoothed state vector
% x_tt   = filtered state vector
% Sig_tt = filtered covariance matrix of states
% Sig_tT = smoothed covariance matrix of states
% Sig_t1 = lag filtered covariance matrix of states
% y(t) = A + H * x(t) + u(t)		var(u) = R;
% x(t) = B + F * x(t-1) + e(t)      var(e) = Q

%% Create Fake Observations

% measure dimensions
[nobs,T]              = size(y);
[nobs2, nobsshocks,~] = size(R);
if (nobs2~=nobs), error('Input size mismatch, transpose y?'), end
[~,nstatesshocks,~] = size(Q);

% Generate yplus and xplus - y and a drawn from their unconditional distribution *using the 'demeaned' model*,
% i.e. zero initial state and zero constant terms!
yplus      = nan(n,T);
xplus      = nan(k,T+1);
xplus(:,1) = cSig0*randn(size(cSig0,2),1); % draw the first state with a1=0
for t = 1:T
    yplus(:,t)   = H*xplus(:,t) + R(:,:,t)*randn(nobsshocks,1);
    xplus(:,t+1) = F*xplus(:,t) + Q(:,:,t)*randn(nstatesshocks,1);
end
xplus(:,end) = [];
ystar        = y - yplus;

Sig0 = cSig0*cSig0';


%% Run the filter
[x_tT, ~] = runKF_DK(ystar,F,H,Q,R,x0,Sig0,A,B);

x_tT = x_tT + xplus';
[x_tt,Sig_tt,Sig_t1] = deal(ones(1,1));
% [xtt_s_star,xtt_star, Sigttminus1_star, Sigtt_star] = SimulationSmoother(ystar,x0,Sig0,F,H,Q,R,A,B,k,T);
% 
% Sig_t1 = Sigttminus1_star;
% Sig_tt = Sigtt_star;              % filtered variance of states
% 
% x_tt = xtt_star   + xplus';       % filtered states
% x_tT = xtt_s_star + xplus';       % smoothed states

end

