%% observation matrix
Hraw                                   = zeros(n,K);
Hraw(real_ix(1),fac_ix(2))             = 1;                 % fix loading on GDP to unity
Hraw(infl_ix,infl_trend_ix(2:end))     = eye(n_infl);
Hraw(real_ix,real_trend_ix)            = eye(n_real);
Hraw(exp_ix,exp_trend_ix)              = eye(n_exp);

if n_costp > 0
    Hraw(costp_ix,[costp_trend_ix,costp_gap_ix(2)]) = ones(1,2);
end

if n_expcycle > 0
    Hraw(exp_ix,infl_expcyc_ix(1)) = 1;
end

if n_costpcycle > 0
    Hraw(costp_ix,costp_cycle_ix(1)) = 1;
end

% if n_exp > 0 
%     Hraw(exp_ix:exp_ix+1,exp_ix:exp_ix+1) = eye(2);
% end

%% transition matrix
Fraw                                              = zeros(K,K);
Fraw(sspace.fac_ix(2:end),sspace.fac_ix(1:end-1)) = eye(s-1);          % identities for output gap
Fraw(real_trend_ix,real_trend_ix)                 = eye(n_real);       % real trend identities
Fraw(infl_trend_ix(1:end),infl_trend_ix(1:end))   = eye(length(infl_trend_ix));
Fraw(exp_trend_ix,exp_trend_ix)                   = eye(length(exp_trend_ix));


if n_dRWd ~=0
    Fraw(real_trend_ix(1),real_trend_ix(1))   = 2;                           % real trend identities
    Fraw(real_trend_ix(1),real_trend_ix(1)+1) = -1;                          % real trend identities
    Fraw(real_trend_ix(1)+1,real_trend_ix(1)) = 1;                           % real trend identities
end    

if pc_q > 1
    Fraw(infl_gap_ix(2:pc_q),infl_gap_ix(pc_q-1:-1:1)) = eye(pc_q-1);
    if n_infl > 1
        Fraw(infl_gap_ix(pc_q+2:end),infl_gap_ix(pc_q+1:-1:pc_q+1)) = eye(pc_q-1);
    end
end

if n_costp > 0
    Fraw(costp_gap_ix(2:end),costp_gap_ix(1:end-1)) = eye(length(costp_gap_ix)-1);
end

if n_expcycle > 0
    Fraw(infl_expcyc_ix(2),infl_expcyc_ix(1)) = 1;
end

if n_costpcycle >0 
	Fraw(costp_cycle_ix(2),costp_cycle_ix(1)) = 1;
end



[H,F,R,Q,A,B] = get_state_space_disagg(Hraw,Fraw,loadings,ab,a_g,a_p,a_c,f_z,c,d,f_d,phi,psi,zeta,h_g,h_rho,...
                    sig_obs,shocks_to_R,sig_trans,shocks_to_Q,n,n_real,K,s,pc_p,L,Lix,sspace);
