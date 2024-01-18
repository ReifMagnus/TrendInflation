function [xtt_s,xtt,Sig_s,Sigtt] = SimulationSmoother(y,x0,Sig0,F,H,Q,R,A,B,k,T)
%%  This function does the time-invariant Kalman-Filter recursion
%   The State Space system is written
%   y(t) = A + H * x(t) + u(t)			var(u) = R;
%   x(t) = B + F * x(t-1) + e(t)        var(e) = Q
%
%   x0 and Sig0 are the initial conditions for the state vector mean and variance

% preallocation
Sigtt        = nan(k,k,T);
Sigttminus1  = Sigtt;
xtt          = nan(k,T);
xttminus1    = xtt;
Sig_s        = zeros(k,k,T);       % E[xhat_t,xhat_t']
xtt_s        = zeros(k,T);         % Ehat_t[x_{t+1}]

%% check for missings

NaNid = logical(isnan(y));                              % get indices of missing values
Obsid = logical(~isnan(y));								% get indices of non-missing values

%% forward filtering
for t = 1:T
    Qt = Q(:,:,t)*Q(:,:,t)';	
    Rt = R(:,:,t)*R(:,:,t)';	
    yt = y(t,:)';    
    % adjust the system for missing values in the data (following Durbin/Koopman)
    if sum(NaNid(t,:)) == 0
        H1 = H;												% use full H matrix
        R1 = Rt;
        A1 = A;
    else
        H1          = H;
        R1			= Rt(Obsid(t,:),Obsid(t,:));
        A1          = A;
        H1(NaNid(t,:),:) = [];										% skip rows of missing variables
        yt(NaNid(t,:))   = [];
        A1(NaNid(t,:))   = [];
    end
    
    % Prediction
    if t == 1												% Initialisation
        xttminus1(:,t)     = B + F * x0;
        Sigttminus1(:,:,t) = F*Sig0*F' + Qt;
    else													% Predict the State
        xttminus1(:,t)     = B + F * xtt(:,t-1);
        Sigttminus1(:,:,t) = F * Sigtt(:,:,t-1) * F' + Qt;
    end

    yhat = yt - H1*xttminus1(:,t) - A1;							% Forecast Error
        
    % Updating (Standard form)
    fhat         = H1 * Sigttminus1(:,:,t) * H1' + R1;    
    K            = Sigttminus1(:,:,t) * (H1'/fhat);          % Kalman Gain     
    xtt(:,t)     = xttminus1(:,t) + K * yhat;          % Update State    
    Sigtt(:,:,t) = (eye(k) - K * H1) * Sigttminus1(:,:,t) * (eye(k) - K * H1)' + K*R1*K';
    
    
    % Updating  (Square Root Form)
%     Zf           = chol(Sigttminus1(:,:,t),'lower');
%     [C,Tau,~]    = svd(Zf'*H1'*diag(1./diag(R1))*H1*Zf);
%     Za           = Zf*C*diag((1 + diag(Tau)).^(-0.5));
%     Sigtt_temp   = Za*Za';
%     
%     K  = Sigtt_temp * H1' * diag(1./diag(R1));
%      
%     xtt(:,t)     = xttminus1(:,t) + K * yhat;
%     Sigtt(:,:,t) = Sigtt_temp;
    
end


%% backward smoothing

% Final values
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


