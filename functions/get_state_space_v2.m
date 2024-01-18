function [H,F,R,Q,A,B] = get_state_space_v2(Hraw,Fraw,Qraw,loadings,a_g,a_p,f_z,c,d,phi,eps,eta,n,real_indicators,k,s,pp)
%% builds state-space system according to following notation
% y_t = A + H * x_t   + eps_t, eps_t ~ N(0,R)
% x_t = B + F + x_t-1 + eta_t, eta_t ~ N(0,Q)

% observation matrix
H                          = Hraw;
H(2:real_indicators+1,1:s) = loadings(1:real_indicators,:);
H(end,end-1)               = c(2);

% observations constants
A        = zeros(n,1);
A(end) = c(1);

% transition matrix
F              = Fraw;
F(1,1:2)       = phi;
F(end,1:pp)    = a_g;
F(end-1,end-1) = f_z(2);
F(end,end)     = a_p;


% transition constants
B                        = zeros(k,1);
B(s+(1:real_indicators)) = d(1:end);
B(end-1)                 = f_z(1);

% observation error
R                     = diag(sqrt(eps));
R(:,~logical(sum(R))) = []; % drop shocks that have zero variance

% meausurement error
Q                                = Qraw;
Q(1,1)                           = sqrt(eta(1));
Q(s+(1:real_indicators+2),2:end) = diag(sqrt(eta(2:end)));

