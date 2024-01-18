function Hpsi = buildHpsi(psi,T)
q = length(psi);
Hpsi = speye(T);
for j = 1:q
    Hpsi = Hpsi + psi(j)*sparse(j+1:T,1:T-j,ones(1,T-j),T,T);
end
end