function [sigma_e] = draw_sigma(e,g,ind_e,r_m,r_s,i_init)

% Draw Variances from Stochastic Volatility Model using chi-squared mixtureapprox;
%   e = data;
%   g = Standard Deviation of innovation for ln(volatilty);
%   ind_e = mixture indicators;
%   r_m = mixture means
%   r_s = mixture sd
%   sigma_e_min = minimum value of sigma_e
%   i_init = 1 if vague prior used; = 0 is ln(sig2) = 0 is used;
%;

big = 1.0e+6;  % large number
small = 1.0e-10;
T=size(e,1);
c=0.001;
gam = g*g;
lnres2=log(e.^2 + c);      % c = 0.001 factor from ksv, restud(1998), page 370)
mean_t = ind_e*r_m;
sigma_t = ind_e*r_s;
var_t = sigma_t.^2;

% Kalman Filtering
ye=lnres2-mean_t;
p1t=zeros(T+1,1);
p2t=zeros(T+1,1);
x1t=zeros(T+1,1);
x2t=zeros(T+1,1);
x3_draw=zeros(T+1,1);

% -- Compute Covariance Matrix  --
x1=0;
if i_init == 1
    p1=big;
elseif i_init == 0
    p1 = 0;
end
x1t(1,1)=x1;
p1t(1,1)=p1;

if sum(isnan(ye)) == 0
    
    % No missing values
    for t=1:T
        x2=x1;
        p2=p1+gam;
        h=p2+var_t(t);
        k=p2/h;
        p1=p2-k*p2;
        x1=x2+k*(ye(t)-x2);
        p1t(t+1,1)=p1;
        p2t(t+1,1)=p2;
        x1t(t+1,1)=x1;
        x2t(t+1,1)=x2;
    end
    
else
    
    % Check for and skip missing values
    for t=1:T
        x2=x1;
        p2=p1+gam;
        if isnan(ye(t)) == 1
            x1 = x2;
            p1 = p2;
        else
            h=p2+var_t(t);
            k=p2/h;
            p1=p2-k*p2;
            x1=x2+k*(ye(t)-x2);
        end
        p1t(t+1,1)=p1;
        p2t(t+1,1)=p2;
        x1t(t+1,1)=x1;
        x2t(t+1,1)=x2;
    end
    
end

utmp=randn(T+1,1);
x3mean=x1;
p3=p1;
chol_p=sqrt(p3);
x3=x3mean+chol_p*utmp(T+1,1);
x3_draw(T+1,1)=x3;

for t=T:-1:2
    x2=x2t(t+1,1);
    p2=p2t(t+1,1);
    x1=x1t(t,1);
    p1=p1t(t,1);
    if p2 > small
        p2i=1/p2;
        k=p1*p2i;
        x3mean=x1+k*(x3-x2);
        p3=p1-k*p1;
    else
        x3mean = x1;
        p3 = p1;
    end
    chol_p=sqrt(p3);
    x3=x3mean+chol_p*utmp(t,1);
    x3_draw(t,1)=x3;
end

sigma_e=exp(x3_draw(2:T+1)/2);

end