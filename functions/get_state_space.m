function [H,F,R,Q,A,B] = get_state_space(Hraw,Fraw,Qraw,loadings,a_p,c,d,phi,eps,eta,n,real_indicators,k,s)
%% builds state-space system according to following notation
% y_t = A + H * x_t   + eps_t, eps_t ~ N(0,R)
% x_t = B + F + x_t-1 + eta_t, eta_t ~ N(0,Q)

% observation matrix
H                          = Hraw;
H(1:real_indicators+1,1:s) = loadings(1:end-1,:);
H(end-1,end)               = -a_p;
H(end,end-1)   = c(2);

% observations constants
A      = zeros(n,1);
A(end) = c(1);

% transition matrix
F              = Fraw;
F(1,1:2)       = phi;

% transition constants
B                        = zeros(k,1);
B(s+(1:real_indicators)) = d;

% observation error
R = diag(eps);

% meausurement error
Q                                                  = Qraw;
Q(1,1)                                             = eta(1);
Q(s+(1:real_indicators+1),s+(1:real_indicators+1)) = diag(eta(2:end));
