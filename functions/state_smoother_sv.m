function [xdraw, loglik] = state_smoother_sv(y,A,H,R,B,F,Q,x0,cSig0,opt_output)
%% Performs Kalman filtering and simulation smoothing.
% Algorithm 2 of Durbin and Koopman (2002) with the efficient modification that saves one pass of the filter/smoother.
%
% Allows for missing data in y
% Model:
% y(t)   = A + H x(t) + R eps(t),  eps(t) ~ N(0,I)
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
[nobs,T]              = size(y);
[nobs2, nobsshocks,~] = size(R);
if (nobs2~=nobs), error('Input size mismatch, transpose y?'), end
[nstates, nstatesshocks,~] = size(Q);

% adjust matrices in case of tvp
if size(A,2) < T, A = repmat(A,1,T); end
if size(B,2) < T, B = repmat(B,1,T); end
if ismatrix(H), H = repmat(H,1,1,T); end
if ismatrix(F), F = repmat(F,1,1,T); end 
if ismatrix(R), R = repmat(R,1,1,T); end

if opt_output==2
    % Generate yplus and xplus - y and a drawn from their unconditional distribution *using the 'demeaned' model*,
    % i.e. zero initial state and zero constant terms!
    yplus = nan(nobs,T);
    xplus = nan(nstates,T+1);
    xplus(:,1) = cSig0*randn(size(cSig0,2),1); % draw the first state with a1=0
    for t = 1:T
        yplus(:,t)   = H(:,:,t)*xplus(:,t) + R(:,:,t)*randn(nobsshocks,1);
        xplus(:,t+1) = F(:,:,t)*xplus(:,t) + Q(:,:,t)*randn(nstatesshocks,1);
    end
    xplus(:,end) = [];
    yy = y - yplus;
else
    yy = y;
end

% allocate space
loglik = nan(1,T);
vvv    = nan(nobs,T);                   % one-step-ahead forecast error of y
FFFinv = nan(nobs,nobs,T);              % inverse of the variance of the one-step-ahead forecast error of y
KKK    = nan(nstates,nobs,T);           % Kalman gain

%------------------------------------------------------------------------------------------------
% Kalman filter on yy
% compute frequently used matrices
% HH  = R*R';
% RQR = zeros(size(Q));
% initialize Kalman filter
xt = x0;                % xt|I(t-1)
Pt = cSig0*cSig0';      % Pt|I(t-1)
for t = 1:T
    iobs = find(~isnan(yy(:,t)));
    RQRt = Q(:,:,t)*Q(:,:,t)';
    HH   = R(:,:,t)*R(:,:,t)';
    if length(iobs) == nobs
        vt    = yy(:,t) - A(:,t) - H(:,:,t)*xt;
        Ftinv = (H(:,:,t)*Pt*H(:,:,t)' + HH)\eye(nobs);
        Kt    = F(:,:,t)*Pt*H(:,:,t)'*Ftinv;
        % update xt,Pt; from now on their interpretation is "x(t+1),P(t+1)"
        xt = B(:,t) + F(:,:,t)*xt + Kt*vt;
        Pt = F(:,:,t)*Pt*(F(:,:,t) - Kt*H(:,:,t))' + RQRt;
        % store the quantities needed for smoothing
        vvv(:,t)      = vt;
        FFFinv(:,:,t) = Ftinv;
        KKK(:,:,t) = Kt;
        if opt_output<2
            loglik(t) = -0.5*nobs*log(2*pi) + sum(log(diag(chol(Ftinv)))) -0.5*(vt'*Ftinv*vt);
        end
    elseif ~isempty(iobs)
        nobst = length(iobs);
        ZZt   = H(iobs,:,t);
        HHt   = HH(iobs,iobs);
        vt    = yy(iobs,t) - A(iobs,t) - ZZt*xt;
        Ftinv = (ZZt*Pt*ZZt' + HHt)\eye(nobst);
        Kt    = F(:,:,t)*Pt*ZZt'*Ftinv;
        % update xt,Pt; from now on their interpretation is "x(t+1),P(t+1)"
        xt = B(:,t) + F(:,:,t)*xt + Kt*vt;
        Pt = F(:,:,t)*Pt*(F(:,:,t) - Kt*ZZt)' + RQRt;
        % store the quantities needed for smooting
        vvv(iobs,t)         = vt;
        FFFinv(iobs,iobs,t) = Ftinv;
        KKK(:,iobs,t) = Kt;
        if opt_output<2
            loglik(t) = -0.5*nobs*log(2*pi) + sum(log(diag(chol(Ftinv)))) -0.5*(vt'*Ftinv*vt);                  % store the log likelihood
        end
    else
        % update at,Pt; from now on their interpretation is "a(t+1),P(t+1)"
        xt        = B + F(:,:,t)*xt;
        Pt        = F(:,:,t)*Pt*F(:,:,t)' + RQRt;
        loglik(t) = 0;
    end
    Sigtt(:,:,t) = Pt;
end

if opt_output==0, xdraw = []; return, end

% -------------------------------------------------------------------------------------------------------------------
% Kalman smoother
Sig_s(:,:,T) = Sigtt(:,:,T);
xtt_s(:,T)   = xtt(:,T);

for h = 1:(T-1)		
	
	J = (Sigtt(:,:,T-h)*F')/(Sigttminus1(:,:,T-h+1));											  % Kalman gain
	
	xtt_s(:,T-h)   = xtt(:,T-h) + J * (xtt_s(:,T-h+1) - F*xtt(:,T-h) - B);							  % update for x
	Sig_s(:,:,T-h) = Sigtt(:,:,T-h) + J * (Sig_s(:,:,T-h+1) - Sigttminus1(:,:,T-h+1)) * J';         % update for Sig
	
end
xtt_s = xtt_s';
xtt   = xtt';


end