function [g_draw] = draw_g(e,g_prior,ind_e,r_m,r_s,i_init)
%% Draw g from Stochastic Volatility Model using chi-squared mixture-approx
%  and using a discrecte grid
%   e = data;
%   g = standard deviation of innovation for ln(volatilty);
%   ind_e = mixture indicators;
%   r_m = mixture means
%   r_s = mixture sd
%   sigma_e_min = minimum value of sigma_e
%   i_init = 1 if vague prior used; = 0 is ln(sig2) = 0 is used;

big     = 1.0e+6;  % large number
T       = size(e,1);
c       = 0.001;
lnres2  = log(e.^2 + c);      % c = 0.001 factor from ksv, restud(1998), page 370)
mean_t  = ind_e*r_m;
sigma_t = ind_e*r_s;
var_t   = sigma_t.^2;

% Kalman Filtering
ye      = lnres2-mean_t;
p1t     = zeros(T+1,1);
p2t     = zeros(T+1,1);
x1t     = zeros(T+1,1);
x2t     = zeros(T+1,1);
x3_draw = zeros(T+1,1);

% Compute log-likelihood values for each g value
n_g      = size(g_prior,1);
g_values = g_prior(:,1);
p_values = g_prior(:,2);
llf_vec  = NaN*zeros(n_g,1);

% Check once for missing values
if sum(isnan(ye)) == 0
    
    % No missing values
    for ig = 1:n_g
        gam = g_values(ig)^2;
        % -- Compute Covariance Matrix  --
        x1=0;
        if i_init == 1
            p1 = big;
        elseif i_init == 0
            p1 = 0;
        end
        llf = 0;
        for t=1:T
            x2  = x1;
            p2  = p1+gam;
            e   = ye(t)-x2;
            h   = p2+var_t(t);
            k   = p2/h;
            p1  = p2-k*p2;
            x1  = x2+k*e;
            llf = llf - 0.5*(log(h) + (e*e)/h);
        end
        llf_vec(ig) = llf;
    end
    
else
    
    % Check for and skip missing values
    for ig = 1:n_g
        gam = g_values(ig)^2;
        % -- Compute Covariance Matrix  --
        x1 = 0;
        if i_init == 1
            p1 = big;
        elseif i_init == 0
            p1 = 0;
        end
        llf = 0;
        for t = 1:T
            x2 = x1;
            p2 = p1+gam;
            if isnan(ye(t)) == 1
                x1 = x2;
                p1 = p2;
            else
                e   = ye(t)-x2;
                h   = p2+var_t(t);
                k   = p2/h;
                p1  = p2-k*p2;
                x1  = x2+k*e;
                llf = llf - 0.5*(log(h) + (e*e)/h);
            end
        end
        llf_vec(ig) = llf;
    end
    
    
end

llf_vec = llf_vec - max(llf_vec);   % normalize so largest value is 0 ;
lf_vec  = exp(llf_vec);
lf_marg = lf_vec'*p_values;
g_post  = (lf_vec.*p_values)./lf_marg;
g_post   = g_post/sum(g_post);   % Make sure this adds to unity

cum_prob = cumsum(g_post);
u        = rand(1,1);
aa       = u < cum_prob;
bb       = n_g+1-sum(aa);
bb       = min([n_g;bb]);
bb       = max([1;bb]);
g_draw   = g_values(bb);

end