function [ind_e,bb] = draw_lcs_indicators(e,sigma_e,r_p,r_m,r_s);
% ln[(e/sigma_2)^2] is a mixture of normals
%  r_p are prior mixture probabilities
%  r_m are means
%  r_s are standard deviations
%  Find posterior means for mixture probabilities and compute a draw 
%   of indicators from posterior
%
n = size(r_p,1);
T = size(e,1);
c=0.001;
x = log(e.^2 + c) - log(sigma_e.^2);   % c = 0.001 factor from ksv, restud(1998), page 370)
% Compute likelihood for each mixture and each time period
xrep = repmat(x,1,n);
mrep = repmat(r_m',T,1);
srep = repmat(r_s',T,1);
prep = repmat(r_p',T,1);
qf = ((xrep-mrep)./srep).^2;
xlike = (exp(-0.5*qf))./srep;
pxlike = xlike.*prep;
xmlike = sum(pxlike,2);
xmlikerep = repmat(xmlike,1,n);
p_post = pxlike./xmlikerep;

% If data are missing, posterior = prior (which is in prep); 
inan = isnan(p_post);
p_post(inan==1) = prep(inan==1);

% Draw Indicators from posterior
cum_prob = cumsum(p_post,2);
u = rand(T,1);
tmp2 = repmat(u,1,n);
aa = repmat(u,1,n) < cum_prob;
bb = n+1-sum(aa,2);
ind_e = zeros(T,n);
for t = 1:T;
    ind_e(t,bb(t)) = 1;
end;

end

