"""Language-Mass Precision (LMP) for process mining.

``LMP_mu(L, M) = mu(L~ intersect L(M)) / mu(L(M))``: the share of the model
language's prior mass that the log supports. Conventionally imported as::

    import language_mass_precision as lmp
"""

from language_mass_precision import algo, datasets, objects, utils
from language_mass_precision.estimator import LanguageMassPrecision, NotFittedError
from language_mass_precision.meta import CITATION, __version__
from language_mass_precision.objects import (
    CompiledModel,
    GeometricLengthPrior,
    LMPResult,
    PowerLawLengthPrior,
    Status,
    TracePrior,
)
from language_mass_precision.simplified import (
    compile_model,
    imprecision_attribution,
    precision_language_mass,
    rank_models,
)
from language_mass_precision.utils import traces_to_dataframe

__all__ = [
    "CITATION",
    "CompiledModel",
    "GeometricLengthPrior",
    "LMPResult",
    "LanguageMassPrecision",
    "NotFittedError",
    "PowerLawLengthPrior",
    "Status",
    "TracePrior",
    "__version__",
    "algo",
    "compile_model",
    "datasets",
    "imprecision_attribution",
    "objects",
    "precision_language_mass",
    "rank_models",
    "traces_to_dataframe",
    "utils",
]
