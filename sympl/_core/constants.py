from .array import DataArray
from .units import is_valid_unit

default_constants = {
    'stefan_boltzmann_constant': DataArray(5.670367e-8, attrs={'units': 'W m^-2 K^-4'}),
    'gravitational_acceleration': DataArray(9.80665, attrs={'units': 'm s^-2'}),
    'heat_capacity_of_dry_air_at_constant_pressure': DataArray(1004.64, attrs={'units': 'J kg^-1 K^-1'}),
    'heat_capacity_of_water_vapor_at_constant_pressure': DataArray(1846.0, attrs={'units': 'J kg^-1 K^-1'}),
    'specific_enthalpy_of_water_vapor': DataArray(2500.0, attrs={'units': 'J kg^-1'}),
    'heat_capacity_of_liquid_water': DataArray(4185.5, attrs={'units': 'J kg^-1 K^-1'}),
    'freezing_temperature_of_liquid_water': DataArray(273.0, attrs={'units': 'K'}),
    'thermal_conductivity_of_liquid_water': DataArray(0.57, attrs={'units': 'W m^-1 K^-1'}),
    'heat_capacity_of_ice': DataArray(2108, attrs={'units': 'J kg^-1 K^-1'}),
    'thermal_conductivity_of_ice': DataArray(2.22, attrs={'units': 'W m^-1 K^-1'}),
    'heat_capacity_of_snow': DataArray(2108, attrs={'units': 'J kg^-1 K^-1'}),
    'thermal_conductivity_of_snow': DataArray(0.2, attrs={'units': 'W m^-1 K^-1'}),
    'reference_air_pressure': DataArray(1.0132e5, attrs={'units': 'Pa'}),
    'thermal_conductivity_of_dry_air': DataArray(0.026, attrs={'units': 'W m^-1 K^-1'}),
    'gas_constant_of_dry_air': DataArray(287., attrs={'units': 'J kg^-1 K^-1'}),
    'gas_constant_of_water_vapor': DataArray(461.5, attrs={'units': 'J kg^-1 K^-1'}),
    'planetary_rotation_rate': DataArray(7.292e-5, attrs={'units': 's^-1'}),
    'planetary_radius': DataArray(6.371e6, attrs={'units': 'm'}),
    'latent_heat_of_vaporization_of_water': DataArray(2.5e6, attrs={'units': 'J kg^-1'}),
    'latent_heat_of_fusion_of_water': DataArray(333550.0, attrs={'units': 'J kg^-1'}),
    'density_of_liquid_water': DataArray(1e3, attrs={'units': 'kg m^-3'}),
    'density_of_ice': DataArray(916.7, attrs={'units': 'kg m^-3'}),
    'density_of_snow': DataArray(100.0, attrs={'units': 'kg m^-3'}),
    'solar_constant': DataArray(1367., attrs={'units': 'W m^-2'}),
    'planck_constant': DataArray(6.62607004e-34, attrs={'units': 'J s'}),
    'speed_of_light': DataArray(299792458., attrs={'units': 'm s^-1'}),
    'seconds_per_day': DataArray(86400., attrs={'units': 'dimensionless'}),
    'avogadro_constant': DataArray(6.022140857e23, attrs={'units': 'mole^-1'}),
    'boltzmann_constant': DataArray(1.38064852e-23, attrs={'units': 'J K^-1'}),
    'loschmidt_constant': DataArray(2.6516467e25, attrs={'units': 'm^-3'}),
    'universal_gas_constant': DataArray(8.3144598, attrs={'units': 'J mole^-1 K^-1'}),
}


def set_constant(name, value, units):
    if is_valid_unit(units):
        default_constants[name] = DataArray(value, attrs={'units': units})
    else:
        raise TypeError('{} is not a valid unit.'.format(units))
