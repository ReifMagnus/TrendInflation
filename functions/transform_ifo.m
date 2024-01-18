function Xc = transform_ifo(X,Code)
%% Transforms  data according to input code
%
% INPUT
%       X        : Panel of raw data                      (T x N)
%       Code     : Vector containing transformation codes (T x 3)
%                  - col 1: take logs (Yes = 1) or percentage growth rates (Yes = 3)
%                  - col 2: Degree of differening (1 or 2)

% OUTPUT
%       Xc       : Panel of transformed series            (T x N)
%________________________________________________________________________________

Xc  = nan*zeros(size(X));
n   = size(X,1);

for j = 1:size(X,2)
    k = 0;
    z = X(:,j);
    
    % percentage growth rates
    if Code(j,1) == 3
        dx = (z(2:end)./z(1:end-1)-1) * 100;
        Xc(:,j) = [nan*ones(1,1);dx];
    end
    
    % log
    if Code(j,1) == 1
        z = log(z) * 100;
    end
    
    % Differencing
    if Code(j,1) ~= 3 && Code(j,1) ~= 0 
        for i = 1:Code(j,2)
            ni = size(z,1);
            z  = (z(2:ni)-z(1:ni-1));
            k  = k + 1;
        end
        
        % Add leading NaNs
        z      = [nan*ones(k,1);z];
        Xc(:,j) = z;
    end
    
    if Code(j,1) == 0
        Xc(:,j) = z;
    end
    
    if ~isreal(z)
        string = ['Wrong transformation for one series ', num2str(j)];
        error(string)
    end
    
end
