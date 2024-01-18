%% set model specifications
ident     = 1;                   % identification via factor loadings

switch spec.sv
    case 'RW'
        SV = 1;
    case 't'
        SV = 2;
    otherwise
        SV = 0;
end

dRWd   = strcmp(spec.indicators.Trend_Spec,'dRWd');
n_dRWd = sum(dRWd);                     % number of time-varying drifts

if n_dRWd ~= 0
    sigtau2_ub = .0001;                 % upper bound for variance of gdp trend
    n_grid     = 500;
end

% choose the Kalman filter function
if SV == 1 || SV == 2
    simulation_smoother = @disturbance_smoother_sv;
else
    simulation_smoother = @disturbance_smoother;
end

p           = spec.OutputGap.AR_lags;                % lags of factor
q           = spec.InflTrend.AR_lags;                % lag of trend inflation

pc_q          = spec.PC.AR_lags;                       % lags of inflation gap in PC
pc_p          = spec.PC.OutputGap_lags;                % lags of output gap in PC
pc_r          = spec.PC.CostPush_lags*n_costp;         % lags of cost-push variable gap in PC

cp_q = spec.CostPush.OutputGap_lags;
cp_r = spec.CostPush.AR_lags;

s              = p + 1;                                % output gap in state vector
real_trends    = n_real;

infl_trend_c   = 1;
infl_trends    = n_infl + infl_trend_c;
infl_gap_c     = 1;

exp_trends = sum(strcmp(spec.indicators.Category,'Exp') & strcmp(spec.indicators.Trend_Spec,'RW'));


costp_gap       = max([cp_r,pc_r])*n_costp;
costp_trend     = n_costp;

% shock in measurement equation
shock_infl  = strcmp(spec.indicators.Measurement(infl_ix,:),'Error')';                                     % no shock to inflation
shock_real  = strcmp(spec.indicators.Measurement(real_ix,:),'Error')';
shock_costp = strcmp(spec.indicators.Measurement(costp_ix,:),'Error')';                                    % shock to import prices
shocks_exp  = strcmp(spec.indicators.Measurement(exp_ix,:),'Error')';                                   % shock to inflation expectations proxy
shocks_oil  = strcmp(spec.indicators.Measurement(exp_ix,:),'Error')';                                   % shock to inflation expectations proxy

% shocks in transition equation
shock_gap          = [ones(1,1) zeros(1,max([s,p])-1)];
shock_trend        = [[1 zeros(n_dRWd,1)] ones(1,n_real-1)];
shocks_costp_trend = ones(1,1)*n_costp(n_costp~=0);
shocks_costp_gap   = [ones(1,1) zeros(1,costp_gap-1)]*n_costp(n_costp~=0);
shocks_infl_trend  = [ones(1,infl_trends)];
shocks_infl_gap    = repmat([1,zeros(pc_q-1)],1,infl_gap_c);
shock_exp_trend    = ones(1,exp_trends);

if strcmp(spec.indicators.Measurement(exp_ix,:),'Cycle')
    shocks_exp_cycle = [ones(1,1), 0];
else
    shocks_exp_cycle = [];
end


if strcmp(spec.indicators.Measurement(costp_ix,:),'Cycle')
    shocks_costp_cycle = [ones(1,1), 0];
else
    shocks_costp_cycle = [];
end


n_expcycle   = length(shocks_exp_cycle);
n_costpcycle = length(shocks_costp_cycle);

% map shocks to covariance matrices in state-space
shocks_to_R = logical([shock_infl shock_real shock_costp shocks_exp]');
shocks_to_Q = logical([shocks_infl_gap shocks_infl_trend shock_exp_trend shock_gap shock_trend shocks_costp_trend shocks_costp_gap shocks_exp_cycle shocks_costp_cycle]);

R_ix = find(shocks_to_R);
Q_ix = find(shocks_to_Q);

nR_shocks = length(R_ix);
nQ_shocks = length(Q_ix);

% elements of the state vector
K = s + real_trends + n_dRWd + costp_gap + costp_trend + infl_trends + infl_gap_c + n_expcycle + n_costpcycle + exp_trends;          % length of state vector

infl_gap_ix      = 1;
infl_trend_ix    = 2:infl_trends+1;
exp_trend_ix     = 1 + infl_trends + (1:exp_trends);
fac_ix           = 1 + infl_trends + exp_trends + (1:s);                                           % factor
real_trend_ix    = 1 + infl_trends + exp_trends + s + (1:n_real + n_dRWd);
if n_dRWd ~= 0
    dRWd_ix = real_trend_ix(2);
    real_trend_ix(2) = [];
end
costp_trend_ix   = 1 + infl_trends + exp_trends + s + n_real + n_dRWd + (1:costp_trend)*n_costp(n_costp~=0);          % cost-push trend
costp_gap_ix     = 1 + infl_trends + exp_trends + s + n_real + n_dRWd + +costp_trend + (1:costp_gap)*n_costp(n_costp~=0);          % cost-push gap
infl_expcyc_ix   = 1 + infl_trends + exp_trends + s + n_real + n_dRWd + +costp_trend + costp_gap + (1:n_expcycle);
costp_cycle_ix   = 1 + infl_trends + exp_trends + s + n_real + n_dRWd + +costp_trend + costp_gap + n_expcycle + (1:n_costpcycle);

sspace.fac_ix           = fac_ix;
sspace.real_trend_ix    = real_trend_ix;
sspace.costp_trend_ix   = costp_trend_ix;
sspace.costp_gap_ix     = costp_gap_ix;
sspace.infl_trend_ix    = infl_trend_ix;
sspace.exp_trend_ix     = exp_trend_ix;
sspace.infl_gap_ix      = infl_gap_ix;
sspace.infl_expcyc_ix   = infl_expcyc_ix;
sspace.infl_costpcyc_ix = costp_cycle_ix;
sspace.real_ix          = real_ix;
sspace.infl_ix          = infl_ix;
sspace.exp_ix           = exp_ix;

