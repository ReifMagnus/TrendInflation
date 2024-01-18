function [nu,flag] = sample_nu(lam,nu,ub,lb)
%% sample degrees of freedom
flag   = 0;
T      = size(lam,1);
sum1   = sum(log(lam));
sum2   = sum(1./lam);

S_nu   = 1;
nut    = nu;
while abs(S_nu) > 10^(-5)   % stopping criteria
    S_nu = T/2*(log(nut/2) + 1 - psi(nut/2)) - .5*(sum1+sum2);
    H_nu = T/(2*nut) - T/4*psi(1,nut/2);
    nut  = nut - H_nu\S_nu;
    if nut<2
        nut = 5;
        H_nu =  T/(2*nut) - T/4*psi(1,nut/2);
        break;
    end
end

Dnu = -1/H_nu;
nuc = nut + sqrt(Dnu)*randn; 
if nuc > lb && nuc < ub
for i = 1:8000
    l_old = T*(nu/2.*log(nu/2) - gammaln(nu/2)) - (nu/2+1)*sum1 - nu/2*sum2;
    l_new = T*(nuc/2.*log(nuc/2) - gammaln(nuc/2)) - (nuc/2+1)*sum1 - nuc/2*sum2;
    l_MH  = l_new - l_old - .5*(nu-nut)^2/Dnu + .5*(nuc-nut)^2/Dnu;        
    if exp(l_MH) > rand
        nu   = nuc;
        flag = 1;
    end    
end
    
end