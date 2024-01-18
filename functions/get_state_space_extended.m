function [H,F,R,Q,A,B] = get_state_space_extended(Hraw,Fraw,loadings,a_g,a_p,a_c,b_g,b_p,b_c,f_z,c,d,f_d,...
    phi,psi,zeta,h_g,h_rho,delta,eps,eps_shocks,eta,eta_shocks,n,n_real,k,s,pp,L,Lix,sspace)
%% builds state-space system according to following notation
% y_t = A + H * x_t   + eps_t, eps_t ~ N(0,R*R')
% x_t = B + F + x_t-1 + eta_t, eta_t ~ N(0,Q*Q')

if ~isempty(b_g)
    n_infl = 2;
else
    n_infl = 1;
end

if numel(c) > 3
    tvp = 1;
    T        = size(c,1);
else
    tvp = 0;
end

if ~isempty(psi), n_expcycle = 1;   else, n_expcycle = 0; end
if ~isempty(zeta), n_costpcycle = 1;   else, n_costpcycle = 0; end


%% observation matrix
H = Hraw;
if ~isempty(delta)
    H(1:n_infl,end - n_infl - n_expcycle) = delta(1:2);
end

H(n_infl + 1:n_real + n_infl,1:s) = loadings(1:n_real,:);

if tvp
    H                             = repmat(H,1,1,T);
    H(end,sspace.infl_trend_ix,:) = c(:,2);
else
    H(end,sspace.infl_trend_ix) = c(2);
end

%% observations constants
A      = zeros(n,1);
if tvp
    A        = repmat(A,1,T);
    A(end,:) = c(:,1);
else
    A(end) = c(1);
end

%% transition matrix
F                                                           = Fraw;
F(1,1:2)                                                    = phi;
F(sspace.costp_trend_ix,sspace.costp_trend_ix)              = f_d(2);
F(sspace.infl_trend_ix,sspace.infl_trend_ix)                = f_z(2);

F(sspace.infl_gap_ix(1),1:pp)                              = a_g;
F(sspace.infl_gap_ix(1),sspace.infl_gap_ix(1:length(a_p))) = a_p;
if ~isempty(a_c)
    F(sspace.infl_gap_ix(1),sspace.costp_gap_ix(1:length(a_c)))    = a_c;
    F(sspace.costp_gap_ix(1),sspace.costp_gap_ix(1:length(h_rho))) = h_rho;
    F(sspace.costp_gap_ix(1),1:length(h_g))                        = h_g;
end
if ~isempty(b_g)
    F(sspace.infl_gap_ix(1+length(a_p)),1:pp)                                             = b_g;
    F(sspace.infl_gap_ix(1+length(a_p)),sspace.infl_gap_ix(length(a_p)+ (1:length(b_p)))) = b_p;
    if ~isempty(b_c)
        F(sspace.infl_gap_ix(1+length(b_p)),s + n_real + 1 +(1:length(a_c))) = b_c;
    end
end

if n_expcycle == 1
    F(sspace.infl_expcyc_ix(1),sspace.infl_expcyc_ix) = psi;
end

if n_costpcycle == 1
    F(sspace.infl_costpcyc_ix(1),sspace.infl_costpcyc_ix) = zeta;
end

%% transition constants
B                          = zeros(k,1);
B(sspace.real_trend_ix)    = d;
B(sspace.costp_trend_ix)   = f_d(1);
B(sspace.infl_trend_ix)    = f_z(1);

%% observation error
R                = diag((eps));
R(:,~eps_shocks) = []; % drop shocks that have zero variance

%% transition error
T = size(eta,2);
if T > 1 
    Q = zeros(k,k,T);
    Seta = eta;
    for i = 1:size(eta,2)
        Q(:,:,i) = diag(Seta(:,i));
    end
    Q(:,~eta_shocks,:) = [];                 % drop shocks that have zero variance
elseif ~isempty(L)
    T              = size(L,1);   
    Q              = zeros(k,k,T);    
    eta_mat        = repmat(eta,1,T);
    eta_mat(Lix,:) = (eta_mat(Lix,:)'.*L)';
    for i = 1:T
        Q(:,:,i) = diag(sqrt(eta_mat(:,i)));
    end
    Q(:,~eta_shocks,:) = [];
else
    Q                = diag(sqrt(eta));
    Q(:,~eta_shocks) = [];
end
end
