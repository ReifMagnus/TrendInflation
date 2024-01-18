%% set priors (according to Jarocinski/Lenza)

delta_nu = mean(diff(y0),1,'omitnan');                                                 % mean of first differences of training sample
sigma_nu = var(diff(y0),0,1,'omitnan');                                     % variance of first differnces of training sample
sigma_nu = ones(size(sigma_nu)).*1;

inf_exp_interpolated = fillmissing(yraw(:,infl_ix(end)),'linear');
sigma_inf_exp        = var(diff(inf_exp_interpolated(1:spec.tau)));


% sigma_nu(infl_ix(end)) = sigma_inf_exp;                          % append infl. exp. (calibrate prior over entire sample)

infl_to_gdp = sigma_nu(infl_ix)/sigma_nu(real_ix(1));               % ratio of variance of inflation to gdp
cp_to_gdp   = sigma_nu(costp_ix)/sigma_nu(real_ix(1));                  % ratio of variance of cost-push variable to gdp

% standardize inflation series
delta        = [];

%------------------------------------------------------------------------------------------------------------------
%% observations equations

% factor model for real indicators
lambda0      = spec.prior.loadings.mean;
iVlambda0    = repmat(1./((spec.prior.loadings.tightness^spec.prior.loadings.decay)*sigma_nu(real_ix)./sigma_nu(real_ix(1))),p+1,1);         % prior variance

% sectoral inflation
a0  = spec.prior.loading_gap.mean ;           % loadings on common trend
b0  = spec.prior.loading_trend.mean;           % loadings on common cycle
ab0 = [a0; b0];

Vb0   = spec.prior.loading.gap.variance;
Va0   = spec.prior.loading.trend.variance;
iVab0 = diag(1./[Va0 Vb0]);

% inflation expectations
c0   = spec.prior.infexp.mean(1);
VC0  = spec.prior.infexp.variance(1);
iVC0 = eye(size(VC0))/VC0;

%------------------------------------------------------------------------------------------------------------------
%% transition equations

% drift terms for real trends
d0   = [delta_nu(real_ix(1:4))'; zeros(n_real-4,1);];                                  % non-zero for drifting variables
iVd0 = 1./[sigma_nu(real_ix(1:4))'; ones(n_real-4,1)*10^-9];                           % drift for stationary indicators fixed at zero

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
PC0_costp_gap     = spec.prior.pc.costp_gap.mean * n_costp(n_costp~=0);
PC0               = [PC0_output_gap; PC0_infl_gap; PC0_costp_gap];

% variances
VPC0_output_gap = spec.prior.pc.output_gap.variance;  
VPC0_infl_gap   = spec.prior.pc.infl_gap.variance;  
VPC0_costp_gap  = spec.prior.pc.costp_gap.variance * n_costp(n_costp~=0); 

% if isfield(spec.prior,'shocks'),    VPC0_output_gap(2) = infl_to_gdp(1);    end       % first lag of output gap
iVPC0              = diag(1./[VPC0_output_gap; VPC0_infl_gap; VPC0_costp_gap]);

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
df_prior          = T/10;                                                                      % degrees of freedom

% implied scale coefficients
% measurement equations
scale_prior_real = 0.09*ones(n_real,1);                 % real variables (except GDP)
scale_prior_pie  = 0.09;                                % inflation exp.
scale_prior_cp   = 0.09;                                % cost push variable
scale_prior_pi   = 0.09*ones(n_infl,1);                 % inflations series


% transition equations
scale_prior_fac        = 0.09;
scale_prior_real_trend = 0.09*ones(n_real,1)/4;
scale_prior_pi_trend   = 0.01*ones(n_infl + 1,1)*(df_prior-1);
% scale_prior_pi_trend   = 0.09*ones(n_infl + 1,1)/10;

scale_prior_pi_gap     = 0.09*ones(1,1);
scale_prior_cptrend    = 0.09;
scale_prior_cpgap      = 0.09;
scale_prior_expcyc     = 0.09;
scale_prior_exptrend   = spec.prior.shocks.scale.inf_exp_trend;
scale_prior_cpcyc      = 0.09;


%% stochastic volatility
if SV == 1
    prior.sv.b0         = log(1);           % h0 ~ N(b0,Vh0) --> initial volatility, h0 is not initial variance, but "autonomous" variance
    prior.sv.Vh0        = 10;               % h0 ~ N(b0,Vh0) --> initial volatility
    prior.sv.Vh         = 10;               % initial variance of h_1
    prior.sv.Vomegah    = .2;               % omega_h ~ N(0,omegah2) --> std. dev. of sv process, implied mean of variance: 0.5/[1/(2*Vomegah)]    
    
    prior.sv_trend         = prior.sv;
    prior.sv_trend.b0      = log(.1);           % h0 ~ N(b0,Vh0) --> initial volatility, h0 is not initial variance, but "autonomous" variance    
    prior.sv_trend.Vh      = 1;               % initial log variance (h_1)
    prior.sv_trend.Vomegah = .01;               % omega_h ~ N(0,omegah2) --> std. dev. of sv process, implied mean: 0.5/[1/(2*Vomegah)]    
      
    % Parameters for scale mixture of epsilon component    
    freq                  = 4;                                               % 4: quarterly data
    prior.sv.scl_eps_vec  = (1:10)';                                          % draw from a 10-point grid
    ps_mean               = 1 - 1/(4*freq);                                  % Outlier every 4 years
    prior.sv.ps_prior_obs = freq*10;                                         % Sample size of 10 years for prior
    ps_prior_a            = ps_mean*prior.sv.ps_prior_obs;                   % a in beta prior
    ps_prior_b            = (1 - ps_mean)*prior.sv.ps_prior_obs;             % b in beta prior
    prior.sv.ps_prior     = [ps_prior_a ps_prior_b];    
    
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

