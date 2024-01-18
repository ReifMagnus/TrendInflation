function [b,sigma,sigma_rw] = tvpreg_ma(y,x,sigma,sigma_rw,b0,iV0,nu0,S0)
%% This function sample the coefficients of a univariate, homosecadastic time-varying parameter regression using precsion sampling
% y_t = b_t * x_t + eps_t + psi*eps_t-1,   eps_t ~ N(0,sigma)
% b_t = b_t-1 + e_t,                         e_t ~ N(0,sigma_rw)
% b_0 ~ N(b0,V0)

% (c) Magnus Reif, 2022

%% preliminiaries
[T,K]  = size(x);

hzeros = sparse(K,(T-1)*K);                 % first row of zeros
vzeros = sparse(T*K,K);                     % last column zeros

B0  = [b0; zeros((T-1)*K,1)];               % initial state vector
% iV0 = 1./ones(K,1);                         % inital state vector variance

%----------------------------------------------------------------------------------------------------------------------
%% Build state-space for precision sampling (following Chan and Jeliazkov (2009), i.e. stack everything over T)
% Observation Matrix
H = SURform(x);

% Observation error variance 
iR = sparse(1:T,1:T,1./sigma);

% Transition matrix
Ft = kron(speye(T-1),eye(K));
F  = speye(T*K) - cat(2,cat(1,hzeros,Ft),vzeros);     
 
% Transition error variance
Q = repmat(sigma_rw,T-1,1);

% 
psi  = zeros(q,1);
Hpsi = buildHpsi(psi,T);


%----------------------------------------------------------------------------------------------------------------------
%% draw states (using precision sampling)
iQ    = sparse(1:T*K,1:T*K,[iV0; 1./Q]);                                   % prepend prior variance
FiQ   = F'*iQ;
HiR   = H'*iR;
P     = FiQ*F  + HiR*H;                                                    % precision matrix
cP    = chol(P,'lower');
b_hat = cP'\(cP\(FiQ*B0 + HiR*y));                                         % solve for posterior mean by forward substitution
bdraw = b_hat + cP'\randn(T*K,1);                                          % vectorized draw of states
b     = reshape(bdraw,K,T)';                                               % final state draw

% (using disturbance sampling)
% A      = zeros(1,1);
% B      = zeros(K,1);
% H      = x;
% F      = eye(K);
% R      = sqrt(sigma);
% Q      = diag(sqrt(sigma_rw));
% cSig0  = chol(diag(1./ones(K,1)));
% 
% b     = disturbance_smoother_univariate(y',A,H,R,B,F,Q,b0,cSig0,2);
% bdraw = reshape(b',K*T,1);
% H = SURform(x);

%----------------------------------------------------------------------------------------------------------------------
%% sample transition error variance

nud0 = 5*ones(K,1);                             % prior degrees of freedom
Sd0  = [.001 .001]'.*(nud0-1);                  % prior scale 

e2 = sum(diff(b).^2,1);
for i = 1:K
    sigma_rw(i) = 1./gamrnd(nud0(i) + T/2, 1/(Sd0(i) + e2(i)/2));
end

%% sample observation error variance

e2    = (y - H*bdraw).^2;
sigma = 1./gamrnd(nu0 + T/2, 1/(S0 + sum(e2)'/2));



%% helper functions
function Xout = SURform(X)
    [r,c] = size(X);
    idi   = kron((1:r)',ones(c,1));
    idj   = (1:r*c)';
    Xout  = sparse(idi,idj,reshape(X',r*c,1));
end

function Hpsi = buildHpsi(psi,T)
    q    = length(psi);
    Hpsi = speye(T);
    for j = 1:q
        Hpsi = Hpsi + psi(j)*sparse(j+1:T,1:T-j,ones(1,T-j),T,T);
    end
end

end
