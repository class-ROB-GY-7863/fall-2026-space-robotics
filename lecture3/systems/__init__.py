from systems.double_integrator import DoubleIntegrator
from systems.single_integrator import SingleIntegrator
from systems.spacecraft import Spacecraft
from systems.spacecraft_attitude import SpacecraftAttitude
from systems.spacecraft_translation import SpacecraftTranslation
from systems.unicycle import Unicycle, UnicycleParking

SYSTEMS = {
    "single_integrator": SingleIntegrator,
    "double_integrator": DoubleIntegrator,
    "unicycle": Unicycle,
    "unicycle_parking": UnicycleParking,
    "spacecraft_translation": SpacecraftTranslation,
    "spacecraft_attitude": SpacecraftAttitude,
    "spacecraft": Spacecraft,
}
