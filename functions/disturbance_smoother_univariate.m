function [xdraw, loglik] = disturbance_smoother_univariate(y,A,H,R,B,F,Q,x0,cSig0,opt_output)
%% Performs Kalman filtering and simulation smoothing.
% Algorithm 2 of Durbin and Koopman (2002) with the efficient modification that saves one pass of the filter/smoother.
%
% Allows for missing data in y
% Model:
% y(t) = A + H x(t) + R eps(t),  eps(t) ~ N(0,I)
% x(t+1) = B + F x(t) + Q eta(t), eta(t) ~ N(0,I)
% x(1) ~ N(x0,cSig0*cSig0')

% INPUTS:
% y:           data, nobs x T (where nobs is the number of observables and T is the
%              number of time periods)
% A, H, R:     parameters of the observation equations
% B, F, Q:     parameters of the state equations
% x1, Sig0:    parameters of the distribution of the initial states
% opt_output:  0: return the log likelihood [NaN,loglik]
%              1: return the smoothed state and the log likelihood [aaa,loglik]
%              2: return a draw of the states from the simulation smoother [aaa,NaN]
% OUTPUTS:
% xdraw  = the mean of the states conditional on y, nstates x T
% loglik = log likelihood of each observation of y, 1 x T
%
% Reference: Jarocinski (2015), A note on implementing the Durbin and Koopman simulation smoother, 
% Computational Statistics and Data Analysis 91 (2015) 1-3.

% measure dimensions
[nobs,T]            = size(y);
[nobs2, nobsshocks] = size(R);
if (nobs2~=nobs), error('Input size mismatch, transpose y?'), end
[nstates, nstatesshocks] = size(Q);

if opt_output == 2
    % Generate yplus and xplus - y and a drawn from their unconditional distribution *using the 'demeaned' model*,
    % i.e. zero initial state and zero constant terms!
    yplus = nan(T);
    xplus = nan(nstates,T+1);
    xplus(:,1) = cSig0*randn(size(cSig0,2),1); % draw the first state with a1=0
    for t = 1:T
        yplus(t)     = H(t,:)*xplus(:,t) + R*randn(nobsshocks,1);
        xplus(:,t+1) = F*xplus(:,t) + Q*randn(nstatesshocks,1);
    end
    xplus(:,end) = [];
    yy = y - yplus;
else
    yy = y;
end

% allocate space
loglik = nan(1,T);
vvv    = nan(nobs,T);                   % one-step-ahead forecast error of y
FFFinv = nan(nobs,T);              % inverse of the variance of the one-step-ahead forecast error of y
KKK    = nan(nstates,T);           % Kalman gain

%------------------------------------------------------------------------------------------------
% Kalman filter on yy
% compute frequently used matrices
HH  = R*R';
RQR = Q*Q';
% initialize Kalman filter
xt = x0;                % xt|I(t-1)
Pt = cSig0*cSig0';      % Pt|I(t-1)
for t = 1:T
    iobs = find(~isnan(yy(t)));
    if length(iobs) == nobs
        vt    = yy(t) - A - H(t,:)*xt;
        Ftinv = (H(t,:)*Pt*H(t,:)' + HH)\eye(nobs);
        Kt    = F*Pt*H(t,:)'*Ftinv;
        % update xt,Pt; from now on their interpretation is "x(t+1),P(t+1)"
        xt = B + F*xt + Kt*vt;
        Pt = F*Pt*(F - Kt*H(t,:))' + RQR;
        % store the quantities needed for smoothing
        vvv(:,t)      = vt;
        FFFinv(:,t) = Ftinv;
        KKK(:,t) = Kt;
        if opt_output<2
            loglik(t) = -0.5*nobs*log(2*pi) + sum(log(diag(chol(Ftinv)))) -0.5*(vt'*Ftinv*vt);
        end
    else
        % update at,Pt; from now on their interpretation is "a(t+1),P(t+1)"
        xt        = B(:,t) + F(:,:,t)*xt;
        Pt        = F(:,:,t)*Pt*F(:,:,t)' + RQR;
        loglik(t) = 0;
    end
end

if opt_output==0, xdraw = []; return, end

% -------------------------------------------------------------------------------------------------------------------
% Kalman smoother
% backwards recursion on r
rrr = zeros(nstates,T);
for t = T-1:-1:1
    iobs = find(~isnan(yy(t+1)));
    if length(iobs) == nobs
        rrr(:,t) = H(t,:)'*FFFinv(:,t+1)*vvv(:,t+1) + (F - KKK(:,t+1)*H(t,:))'*rrr(:,t+1);
    else        
        rrr(:,t) = F(:,:,t)'*rrr(:,t+1); % backward recursion with no observables
    end
end

% one more iteration to get r0
iobs = find(~isnan(yy(1)));
r0   = H(t,:)'*FFFinv(iobs,1)*vvv(iobs,1) + (F - KKK(:,1)*H(t,:))'*rrr(:,1);

xdraw = nan(nstates,T);                                                       % allocate space for smoothed states

% forwards recursion to compute smoothed states from r - see Durbin/Koopmann (2002),eq.8
xdraw(:,1) = x0 + cSig0*cSig0'*r0;                                            % initialize the forward recursion
for t = 2:T
    xdraw(:,t) = B + F*xdraw(:,t-1) + RQR*rrr(:,t-1);
end

if opt_output == 2
    xdraw  = xdraw + xplus;
    loglik = [];
end

xdraw = xdraw';
end