###
### Common experiment parameter
###

param_model = {}
param_model['dimension'] = 40
param_model['forcing'] = 8
param_model['dt'] = 0.005

param_letkf = {}
param_letkf['obs_error'] = 1.0
param_letkf['obs_number'] =8
param_letkf['localization_length_scale'] = 1
param_letkf['localization_length_cutoff'] = 1
param_letkf['inflation_factor'] = 1.1
param_letkf['additive_inflation_factor'] = 0.0
param_letkf['missing_value'] = -9.99e8

param_exp = {}
param_exp['exp_length'] = 30000
param_exp['ensembles'] = 10
param_exp['expdir'] = './DATA/test'
param_exp['obs_type'] = 'obs_p40_1'
param_exp['da_type'] = 'nocorr'
param_exp['dt_nature'] = 0.05
param_exp['dt_obs'] = 0.05
param_exp['dt_assim'] = 0.05
param_exp['spinup_length'] = 2000

