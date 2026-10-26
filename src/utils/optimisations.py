# Performance optimisations that can be switched off one by one, to profile each against the baseline (the code before they existed).
#
# Selected with the environment variable ACTIVESPLAT_OPTIMISATIONS, which every node inherits from roslaunch:
#   unset / "all"     every optimisation (default)
#   "baseline"/"none" none
#   "-render_cache,-snapshot_cache"   all except these
#   "render_cache,single_pass_render"   only these
# Each node reads it; the mapper writes the resolved set and a label to <run>/optimisations.json. Standard library only, because the
# VGGT-Omega worker's env loads trace.py by path and this file is imported by the ActiveSplat packages.
import json
import os

ENV_VARIABLE = 'ACTIVESPLAT_OPTIMISATIONS'

OPTIMISATIONS = {
    'single_pass_render': 'rasterize the image pass only: the silhouette/depth pass of render() is never used by the map renders',
    'prepared_params': 'compute the camera-independent derived tensors (colours, rotations, scales) once per map, not per render (needs single_pass_render)',
    'render_cache': 'reuse a top-down map render while its inputs (map version, height cut, camera) are unchanged',
    'snapshot_cache': 'take the visible-map snapshot only when the map or the height sliders changed, not on every pass of the main loop',
    'no_clone_cut': 'cut the map by height without first cloning every parameter (the boolean-mask cut already copies)',
}
REQUIRES = {'prepared_params': 'single_pass_render'}

def parse(spec:str) -> set:
    spec = (spec or 'all').strip()
    if spec in ('', 'all'):
        enabled = set(OPTIMISATIONS)
    elif spec in ('baseline', 'none'):
        enabled = set()
    else:
        tokens = [token.strip() for token in spec.split(',') if token.strip()]
        unknown = [token.lstrip('-') for token in tokens if token.lstrip('-') not in OPTIMISATIONS]
        if unknown:
            raise ValueError(f'{ENV_VARIABLE}: unknown optimisation {unknown}; known: {sorted(OPTIMISATIONS)}')
        # NOTE: any name without "-" means "only these"; "-name" alone means "all except"
        enabled = {token for token in tokens if not token.startswith('-')} or set(OPTIMISATIONS)
        enabled -= {token[1:] for token in tokens if token.startswith('-')}
    for name, needed in REQUIRES.items():
        if name in enabled and needed not in enabled:
            enabled.discard(name)
    return enabled

class Optimisations:

    def __init__(self) -> None:
        self.spec = os.environ.get(ENV_VARIABLE, 'all')
        self.__enabled = parse(self.spec)

    def enabled(self, name:str) -> bool:
        if name not in OPTIMISATIONS:
            raise KeyError(name)
        return name in self.__enabled

    @property
    def label(self) -> str:
        if self.__enabled == set(OPTIMISATIONS):
            return 'all optimisations'
        if not self.__enabled:
            return 'baseline'
        if len(self.__enabled) > len(OPTIMISATIONS) / 2:
            return 'all except ' + ', '.join(name for name in OPTIMISATIONS if name not in self.__enabled)
        return 'only ' + ', '.join(name for name in OPTIMISATIONS if name in self.__enabled)

    def describe(self) -> dict:
        return {'label': self.label, 'spec': self.spec, 'enabled': [name for name in OPTIMISATIONS if name in self.__enabled],
                'disabled': [name for name in OPTIMISATIONS if name not in self.__enabled]}

    def save(self, results_dir:str) -> None:
        with open(os.path.join(results_dir, 'optimisations.json'), 'w') as f:
            json.dump(self.describe(), f, indent=2)

optimisations = Optimisations()
