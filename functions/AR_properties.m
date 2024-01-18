function [vardiff, autocorr] = AR_properties(phi, N)
%% PURPOSE: Return some statistics about the AR(P) process
% y(t) = phi(1)*y(t-1) + phi(2)*y(t-2) + ... phi(P)*y(t-P) + u, var(u)=1.
% RETURNS:
% vardiff - the variance of y(t)-y(t-1)
% autocorr - 1 x N vector with autocorrelations of order 0 to N-1

P = length(phi);

if P==1
    vardiff = 2/(1+phi);
    autocorr = repmat(phi,1,N-1);
    autocorr = [1 cumprod(autocorr)];
elseif P==2
    vardiff = 2*(1 - phi(1) - phi(2))/((1 + phi(2))*((1 - phi(2))^2 - phi(1)^2));
    autocorr = nan(1,N);
    autocorr(1) = 1;
    if N>1
        autocorr(2) = phi(1)/(1-phi(2)); % Hamilton (1994) eq. 3.4.27
        for n = 3:N
            autocorr(n) = phi(1)*autocorr(n-1) + phi(2)*autocorr(n-2); % Hamilton (1994) eq. 3.4.28
        end
    end
else % for P>2 use Monte Carlo
    T = 1000000;
    % simulate for T periods
    phix = flipud(phi(:));
    y0 = zeros(P,1);
    u = randn(T,1);
    y = [y0; nan(T,1)];
    for t = 1:T
        y(P+t) = sum(y(t:P+t-1).*phix) + u(t);
    end
    y(1:P) = [];
    % done, now compute the stats
    vardiff = var(diff(y));
    autocorr = nan(1,N-1);
    for m = 1:N-1
        autocorr(m) = corr(y(1:end-m),y(m+1:end));
    end
    autocorr = [1 autocorr];
end
