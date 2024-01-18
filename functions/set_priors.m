%% set priors (according to Jarocinski/Lenza)

delta_nu = mean(diff(y0),1,'omitnan');                                                 % mean of first differences of training sample
sigma_nu = var(diff(y0(:,1:end-1)),0,1,'omitnan');                                     % variance of first differnces of training sample

if sum(~isnan(y0(:,end))) == 0
    sigma_inf_exp = var(diff(y(1:40,end)),'omitnan');
else
    sigma_inf_exp = var(diff(y0(~isnan(y0(:,end)),end),1));
end

sigma_nu = [sigma_nu sigma_inf_exp];                          % append infl. exp. (calibrate prior over entire sample)

infl_to_gdp = sigma_nu(infl_ix)/sigma_nu(real_ix(1));               % ratio of variance of inflation to gdp
cp_to_gdp   = sigma_nu(costp_ix)/sigma_nu(real_ix(1));                  % ratio of variance of cost-push variable to gdp

% standardize inflation series
delta = [];

%------------------------------------------------------------------------------------------------------------------
%% observations equations

% factor model for real indicators
lambda0      = spec.prior.loadings.mean;
iVlambda0    = repmat(1./((spec.prior.loadings.tightness^spec.prior.loadings.decay)*sigma_nu(real_ix)./sigma_nu(real_ix(1))),p+1,1);         % prior variance

% inflation expectations
c0   = spec.prior.infexp.mean ;
VC0  = spec.prior.infexp.variance;
iVC0 = eye(size(VC0))/VC0;

%------------------------------------------------------------------------------------------------------------------
%% transition equations

% drift terms for real trends
d0   = [delta_nu(2:5)'; zeros(n_real-4,1);];                                  % non-zero for drifting variables
iVd0 = 1./[sigma_nu(2:5)'; ones(n_real-4,1)*10^-9];                           % drift for stationary indicators fixed at zero

% drift of cost-push trend
fd0   = spec.prior.costp_trend.mean;                                % shrink towards driftless random walk
Vfd0  = spec.prior.costp_trend.variance;
iVfd0 = eye(size(Vfd0))/Vfd0;

% output gap process
phi0                   = spec.prior.output_gap.mean;                % Vphi0  = [0.0806 -0.0597; -0.0597 0.0464];   % numbers taken from JL
Vphi0                  = spec.prior.output_gap.variance;            % numbers taken from JL
iVphi0                 = eye(size(Vphi0))/Vphi0;
[g_phi,g_autocorr_phi] = AR_properties(phi0,s);           % get autocovariance

% trend inflation process
fz0   = spec.prior.inftrend.mean;                                           % shrink towards driftless random walk
Vfz0  = spec.prior.inftrend.variance;
iVfz0 = eye(size(Vfz0))/Vfz0;

%------------------------------------------------------------------------------------------------------------------
% Phillips curve(s)
% means
PC0_output_gap    = spec.prior.pc.output_gap.mean;                  % priors on output gap in PC equations                                                         
PC0_infl_gap      = spec.prior.pc.infl_gap.mean;                                                          % prior on inflation gap
PC0_costp_gap     = spec.prior.pc.costp_gap.mean;
PC0               = [PC0_output_gap; PC0_infl_gap; PC0_costp_gap];

% variances
VPC0_output_gap = spec.prior.pc.output_gap.variance;  
VPC0_infl_gap   = spec.prior.pc.infl_gap.variance;  
VPC0_costp_gap  = spec.prior.pc.costp_gap.variance; 

if isfield(spec.prior,'shocks'),    VPC0_output_gap(2) = infl_to_gdp(1);    end       % first lag of output gap

iVPC0              = diag(1./[VPC0_output_gap; VPC0_infl_gap; VPC0_costp_gap]);

if n_infl > 1
    PC0_output_gap_head  = spec.prior.pc_head.output_gap.mean;                                                          % priors on output gap in PC equations
    PC0_infl_gap_head    = spec.prior.pc_head.infl_gap.mean;   
    PC0_costp_gap_head   = spec.prior.pc_head.costp_gap.mean;
    PC0_head             = [PC0_output_gap_head; PC0_infl_gap_head; PC0_costp_gap_head];    
    VPC0_output_gap_head = spec.prior.pc_head.output_gap.variance;  
    VPC0_infl_gap_head   = spec.prior.pc_head.infl_gap.variance;  
    VPC0_costp_gap_head  = spec.prior.pc_head.costp_gap.variance;   
    iVPC0_head           = diag(1./[VPC0_output_gap_head; VPC0_infl_gap_head; VPC0_costp_gap_head]);
else
    VPC0_output_gap_head = [];
end
%------------------------------------------------------------------------------------------------------------------
% cost-push process: means
CP0_output_gap = spec.prior.cost_push.output_gap.mean;
CP0_costp_gap  = spec.prior.cost_push.costpush_gap.mean;
CP0            = [CP0_costp_gap; CP0_output_gap];

% variances
VCP0_output_gap = spec.prior.cost_push.output_gap.variance ;                                 % scale prior varaiance with ratio of variances
VCP0_costp_gap  = spec.prior.cost_push.costpush_gap.variance;
iVCP0           = diag(1./[VCP0_costp_gap; VCP0_output_gap]);

[g_rho,g_autocorr_rho] = AR_properties(CP0_costp_gap,max([cp_r pc_r+1]));           % get autocovariance

% cost-push error cycle
zeta0                    = spec.prior.cost_push_error.mean;
iVzeta0                  = eye(2)/spec.prior.cost_push_error.variance;
[g_zeta,g_autocorr_zeta] = AR_properties(zeta0,2);           % get autocovariance

%------------------------------------------------------------------------------------------------------------------
% inflation expecations error cycle
psi0                   = spec.prior.infexp_error.mean;
iVpsi0                 = eye(2)/spec.prior.infexp_error.variance;
[g_psi,g_autocorr_psi] = AR_properties(psi0,2);           % get autocovariance

%------------------------------------------------------------------------------------------------------------------
%% shock variances

% prior means
if ~isfield(spec.prior,'shocks')
    df_prior          = 5;                                                                      % degrees of freedom
    eps_real_meas     = sigma_nu(2:n_real+1)'/4;                                              % mean of measurement error
    eta_real_trend    = sigma_nu(2:n_real+1)'.*spec.indicators.Trend_share(spec.real_ix);       % mean of shocks to trends    
    eta_g_mean        = (1-0.5)/(g_phi)*sigma_nu(2);                                            % mean of shocks to output gap
    eps_pi_mean       = sigma_nu(infl_ix)./4;                                                         % mean of shock to  inflation gap    
    eps_pie_mean      = sigma_nu(exp_ix)/4;                                                     % mean of shock to inflation exp. proxy
    eps_pi_trend_mean = sigma_nu(exp_ix)/2;                                                     % mean of shock to trend inflation    
    eps_expcyc_mean   = 0.01;

    eps_cp_mean       = sigma_nu(costp_ix-1)*1;                % mean of shock to cost-push measurement error 
    eps_cptrend_mean  = sigma_nu(costp_ix)/4;                            % mean of shock to cost-push trend
    eps_cpgap_mean    = sigma_nu(costp_ix)/1;                            % mean of shock to cost-push cycle
    eps_costpcyc_mean = sigma_nu(costp_ix)/2;

    % implied scale coefficients
    % measurement equations
    scale_prior_real = eps_real_meas*(df_prior-2);          % real variables (except GDP)
    scale_prior_pie  = eps_pie_mean*(df_prior-2);           % inflation exp.
    scale_prior_cp   = eps_cp_mean*(df_prior-2);            % cost push variable

    % transition equations
    scale_prior_fac      = eta_g_mean*(df_prior-2);
    scale_prior_pi       = eps_pi_mean*(df_prior-2);
    scale_real_trend     = eta_real_trend*(df_prior-2);
    scale_prior_pi_trend = eps_pi_trend_mean*(df_prior-2);
    scale_prior_cptrend  = eps_cptrend_mean*((df_prior-2));
    scale_prior_cpgap    = eps_cpgap_mean*(df_prior-2);
    scale_prior_expcyc   = eps_expcyc_mean*(df_prior-2);
    scale_prior_costpcyc = eps_costpcyc_mean*(df_prior-2);
else
    df_prior          = spec.prior.shocks.df_prior;                                                                      % degrees of freedom
    eps_real_meas     = repmat(spec.prior.shocks.mean.real_indicators,1,n_real);                                         % mean of measurement error
    eta_g_mean        = (1-0.5)/(g_phi)*sigma_nu(2);                                            % mean of shocks to output gap
    eps_pi_mean       = spec.prior.shocks.mean.inf_gap;                                                    % mean of shock to  inflation gap
    eps_pi_trend_mean = spec.prior.shocks.mean.inf_trend;                                                    % mean of shock to trend inflation
    eps_pie_mean      = spec.prior.shocks.mean.inf_exp;                                                  % mean of shock to inflation exp. proxy
    eps_expcyc_mean   = spec.prior.shocks.mean.inf_exp_cyc;    
    eta_real_trend    = spec.prior.shocks.mean.real_trends;            % mean of shocks to trends

    eps_cp_mean       = sigma_nu(costp_ix-1)*1;                          % mean of shock to cost-push measurement error 
    eps_cptrend_mean  = spec.prior.shocks.mean.costp_trend;                          % mean of shock to cost-push trend
    eps_cpgap_mean    = spec.prior.shocks.mean.costp_gap;                            % mean of shock to cost-push cycle
    eps_costpcyc_mean = spec.prior.shocks.mean.costp_cyc;
    
    % measurement equations
    scale_prior_real = repmat(spec.prior.shocks.scale.real_indicators,1,n_real);          % real variables (except GDP)
    scale_prior_pie  = spec.prior.shocks.scale.inf_exp;                                   % inflation exp.
    scale_prior_cp   = spec.prior.shocks.scale.costp;                         % cost push variable

    % transition equations
    scale_prior_fac      = spec.prior.shocks.scale.output_gap;
    scale_prior_pi       = repmat(spec.prior.shocks.scale.inflation,n_infl);
    scale_real_trend     = repmat(spec.prior.shocks.scale.real_trends,1,n_real);
    scale_prior_pi_trend = spec.prior.shocks.scale.inf_trend;
    scale_prior_cpgap    = spec.prior.shocks.scale.costp_gap;
    scale_prior_cptrend  = spec.prior.shocks.scale.costp_trend;    
    scale_prior_expcyc   = spec.prior.shocks.scale.inf_exp_cyc;
    scale_prior_costpcyc = spec.prior.shocks.scale.costp_cyc;
end    
    
%% stochastic volatility
if SV == 1
    prior.sv.b0       = log(1);                                                  % h0 ~ N(b0,Vh0) --> initial volatility
    prior.sv.Vh0      = 10;                                                 
    prior.sv.Vh       = 10;    
    prior.sv.Vomegah  = .1;               % omega_h ~ N(0,omegah2) --> std. dev. of sv process, implied mean: 0.5/(1/2*Vomegah)
elseif SV == 2
    prior.sv.nu0 = 3;                                                   % prior degrees of freedom
    prior.sv.S0  = 1*(prior.sv.nu0 - 1);
end

%% time-varying coefficients    
if tvp
    prior_df      = 10;
    mud0          = [0 1]';                     % prior mean on mean of coefficient processes
    Vmud          = [.1^2 .1^2]';               % prior variance on mean of coefficient processes
    rhod0         = [.95 .95]';                 % prior mean on AR-coefficient
    Vrhod         = [.1^2 .1^2]';               % prior variance on AR-coefficient
    scale_prior_c = [.001 .001]'.*(prior_df-1);  % prior scale
end

