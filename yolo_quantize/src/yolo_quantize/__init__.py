"""Public interfaces for the Full35 quantization project."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .activation_adapter import (
    ActivationOutputAdapter,
    ActivationOutputPolicy,
    AppliedActivationQuantization,
)
from .activations import (
    QSiLUFixedPointConfig,
    QSiLUPQ,
    QSiLUPQProfile,
    emulate_qsilu_pq_legacy,
)
from .balance_selection import (
    BalanceCandidate,
    BalanceSelection,
    QuantizationBalanceSelector,
)
from .diagnostic_manifest import (
    DiagnosticManifest,
    DiagnosticManifestBuilder,
    DiagnosticManifestSpec,
)
from .exact_preparation import ExactPreparationLayout, ExactWeightPreparation
from .exact_routing import (
    ExactRoutingManifest,
    ExactRoutingSource,
    ExactWeightRouting,
)
from .experiment_plan import (
    ActivationExperimentCell,
    ActivationScreenPlan,
    CoupledExperimentPlanner,
    CoupledWeightPlan,
    WeightExperimentCell,
)
from .full35_adapter import (
    Full35ActivationAdapter,
    Full35ActivationBuild,
    Full35ActivationPolicy,
)
from .intake import (
    ActivationIntake,
    ActivationIntakeReport,
    ActivationJobSummary,
    ActivationQueueSummary,
    FileDigest,
)
from .integer_boundaries import (
    AccumulatorAudit,
    AccumulatorSiteAudit,
    AddResult,
    ConcatResult,
    Full35IntegerBoundaryManifest,
    IntegerBoundaryContract,
    IntegerTensorSpec,
    RequantizationResult,
)
from .metric_gate import (
    FULL35_MAP50_95_KEYS,
    FULL35_MAP50_KEYS,
    FULL35_METRIC_KEYS,
    Full35MetricCandidate,
    Full35MetricGate,
    Full35MetricGateResult,
    Full35MetricGateSpec,
    Full35MetricSnapshot,
)
from .qat_calibration import (
    QATActivationCalibrationReport,
    calibrate_activation_outputs,
)
from .qat_graph import (
    BNFoldedHardwareContractGuard,
    BNOnlyFoldReport,
    PreparedFoldedQATTrainingGraph,
    fold_batch_norm_only,
    materialize_qat_deployment_graph,
    prepare_folded_qat_training_graph,
)
from .qat_metrics import (
    MAP50_95_GATE_METRICS,
    MAP50_GATE_METRICS,
    QAT_GATE_METRICS,
    Map50AccuracyGate,
    Map50AccuracyGateReport,
    Map50CheckpointSelectors,
    Map50SelectionResult,
)
from .qat_optimizer import (
    QuantizerOptimizerGroupReport,
    is_quantizer_parameter,
    split_quantizer_parameter_groups,
)
from .qat_plan import (
    Full35QATPlan,
    QATArm,
    QATDataContract,
    QATGateSpec,
    QATMonitoringSpec,
    QATOptimizerSpec,
    QATTrainingSpec,
)
from .qat_schedule import (
    QATTrainabilityController,
    QATTrainabilityState,
    QuantizationEpochController,
    QuantizationEpochState,
)
from .qat_validation import (
    QATDeploymentView,
    QATJointValidatorAdapter,
    build_qat_deployment_view,
)
from .qat_weights import (
    AppliedFoldedQATWeightPolicy,
    FoldedQATWeightAdapter,
    ProgressiveQuantizationSchedule,
    QATConv2d,
    QATLinear,
    TrainableWeightFakeQuantizer,
)
from .quantizers import (
    LSQPlusActivationQuantizer,
    LSQPlusSpec,
    grad_scale,
    round_to_nearest_even_ste,
)
from .scaled_codebook import optimal_scaled_codebook_scales
from .sd4_encoding import SD4Encoding
from .search_data import PreparedBBAT5SearchView, prepare_bbat5_search_view
from .validation_source import Full35DeploymentValidationSource
from .weight_formats import (
    FixedSD4ScaleMethod,
    WeightAnalysisPlan,
    WeightFormatAnalysis,
    WeightFormatAnalyzer,
    WeightFormatMeasurement,
)
from .weight_quantization import (
    AppliedWeightQuantization,
    AppliedWeightQuantizationPolicy,
    ExactTernaryWeightSpec,
    FilterwiseTWNWeightSpec,
    FixedSD4WeightSpec,
    Full35WeightRegionCatalog,
    PaperTWNWeightSpec,
    UniformWeightSpec,
    WeightQuantizationAdapter,
    WeightRegionAssignment,
    WeightSite,
)
from .weight_views import (
    Full35WeightViewAdapter,
    Full35WeightViews,
    WeightViewParityManifest,
)

if TYPE_CHECKING:
    from .qat_runtime import Full35QATRuntime, LoadedQATDeploymentParent
    from .weight_sensitivity import (
        WeightParent,
        WeightSensitivityCell,
        WeightSensitivityStudy,
    )

_LAZY_EXPORT_MODULES = {
    "Full35QATRuntime": ".qat_runtime",
    "LoadedQATDeploymentParent": ".qat_runtime",
    "Full35SearchValidationPlan": ".search_validation",
    "WeightParent": ".weight_sensitivity",
    "WeightSensitivityCell": ".weight_sensitivity",
    "WeightSensitivityStudy": ".weight_sensitivity",
}


def __getattr__(name: str) -> Any:
    """Load the executable weight-sensitivity module only when requested."""

    if name in _LAZY_EXPORT_MODULES:
        module = import_module(_LAZY_EXPORT_MODULES[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FULL35_MAP50_95_KEYS",
    "FULL35_MAP50_KEYS",
    "FULL35_METRIC_KEYS",
    "MAP50_95_GATE_METRICS",
    "MAP50_GATE_METRICS",
    "QAT_GATE_METRICS",
    "AccumulatorAudit",
    "AccumulatorSiteAudit",
    "ActivationExperimentCell",
    "ActivationIntake",
    "ActivationIntakeReport",
    "ActivationJobSummary",
    "ActivationOutputAdapter",
    "ActivationOutputPolicy",
    "ActivationQueueSummary",
    "ActivationScreenPlan",
    "AddResult",
    "AppliedActivationQuantization",
    "AppliedFoldedQATWeightPolicy",
    "AppliedWeightQuantization",
    "AppliedWeightQuantizationPolicy",
    "BNFoldedHardwareContractGuard",
    "BNOnlyFoldReport",
    "BalanceCandidate",
    "BalanceSelection",
    "ConcatResult",
    "CoupledExperimentPlanner",
    "CoupledWeightPlan",
    "DiagnosticManifest",
    "DiagnosticManifestBuilder",
    "DiagnosticManifestSpec",
    "ExactPreparationLayout",
    "ExactRoutingManifest",
    "ExactRoutingSource",
    "ExactTernaryWeightSpec",
    "ExactWeightPreparation",
    "ExactWeightRouting",
    "FileDigest",
    "FilterwiseTWNWeightSpec",
    "FixedSD4ScaleMethod",
    "FixedSD4WeightSpec",
    "FoldedQATWeightAdapter",
    "Full35ActivationAdapter",
    "Full35ActivationBuild",
    "Full35ActivationPolicy",
    "Full35DeploymentValidationSource",
    "Full35IntegerBoundaryManifest",
    "Full35MetricCandidate",
    "Full35MetricGate",
    "Full35MetricGateResult",
    "Full35MetricGateSpec",
    "Full35MetricSnapshot",
    "Full35QATPlan",
    "Full35QATRuntime",
    "Full35SearchValidationPlan",
    "Full35WeightRegionCatalog",
    "Full35WeightViewAdapter",
    "Full35WeightViews",
    "IntegerBoundaryContract",
    "IntegerTensorSpec",
    "LSQPlusActivationQuantizer",
    "LSQPlusSpec",
    "LoadedQATDeploymentParent",
    "Map50AccuracyGate",
    "Map50AccuracyGateReport",
    "Map50CheckpointSelectors",
    "Map50SelectionResult",
    "PaperTWNWeightSpec",
    "PreparedBBAT5SearchView",
    "PreparedFoldedQATTrainingGraph",
    "ProgressiveQuantizationSchedule",
    "QATActivationCalibrationReport",
    "QATArm",
    "QATConv2d",
    "QATDataContract",
    "QATDeploymentView",
    "QATGateSpec",
    "QATJointValidatorAdapter",
    "QATLinear",
    "QATMonitoringSpec",
    "QATOptimizerSpec",
    "QATTrainabilityController",
    "QATTrainabilityState",
    "QATTrainingSpec",
    "QSiLUFixedPointConfig",
    "QSiLUPQ",
    "QSiLUPQProfile",
    "QuantizationBalanceSelector",
    "QuantizationEpochController",
    "QuantizationEpochState",
    "QuantizerOptimizerGroupReport",
    "RequantizationResult",
    "SD4Encoding",
    "TrainableWeightFakeQuantizer",
    "UniformWeightSpec",
    "WeightAnalysisPlan",
    "WeightExperimentCell",
    "WeightFormatAnalysis",
    "WeightFormatAnalyzer",
    "WeightFormatMeasurement",
    "WeightParent",
    "WeightQuantizationAdapter",
    "WeightRegionAssignment",
    "WeightSensitivityCell",
    "WeightSensitivityStudy",
    "WeightSite",
    "WeightViewParityManifest",
    "build_qat_deployment_view",
    "calibrate_activation_outputs",
    "emulate_qsilu_pq_legacy",
    "fold_batch_norm_only",
    "grad_scale",
    "is_quantizer_parameter",
    "materialize_qat_deployment_graph",
    "optimal_scaled_codebook_scales",
    "prepare_bbat5_search_view",
    "prepare_folded_qat_training_graph",
    "round_to_nearest_even_ste",
    "split_quantizer_parameter_groups",
]
