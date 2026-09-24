"""
BharatSR — Pydantic Request & Response Schemas
Defines strictly validated data models for all API interactions.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ==========================================
# Common / Sub-models
# ==========================================

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    models_loaded: int = Field(..., example=2)
    device: str = Field(..., example="cpu")
    version: str = Field(..., example="0.2.0")
    active_jobs: int = Field(0, example=0)


class ModelInfo(BaseModel):
    id: str = Field(..., example="rcan")
    name: str = Field(..., example="Residual Channel Attention Network (RCAN)")
    scale_factor: int = Field(4, example=4)
    device: str = Field("cpu", example="cpu")
    architecture: Optional[str] = None
    n_bands: Optional[int] = 4
    epoch: Optional[Any] = None
    val_loss: Optional[float] = None
    has_uncertainty: bool = False
    supports_uncertainty: bool = False
    parameters: Optional[int] = None
    description: Optional[str] = None
    status: Optional[str] = "loaded"

    class Config:
        extra = "allow"


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]


class SampleMetadata(BaseModel):
    scene_id: str
    title: str
    source_dataset: str
    product_id: Optional[str] = None
    sensor: Optional[str] = None
    acquisition_date: Optional[str] = None
    has_geo: bool = False
    crs: Optional[str] = None
    transform: Optional[List[float]] = None
    bounds: Optional[List[float]] = None
    bands: List[str] = Field(default_factory=list)
    gsd: Optional[str] = None
    preprocessing_level: Optional[str] = None
    cloud_percentage: Optional[float] = None
    hr_source: Optional[str] = None
    hr_acquisition_metadata: Optional[str] = None
    alignment_quality: Optional[str] = None


class SampleInfo(BaseModel):
    id: str
    filename: str
    title: str
    region: str
    description: str
    tactical_category: str
    coordinates: str
    bands: int
    lr_size: str
    has_ground_truth: bool
    has_geo: bool
    crs: Optional[str] = None
    sensor: Optional[str] = None
    thumbnail: str
    views: Dict[str, str]


class SamplesListResponse(BaseModel):
    samples: List[SampleInfo]


class MetricItem(BaseModel):
    value: Optional[float] = None
    unit: str = ""
    description: str = ""


class BandProfile(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    band: str
    wavelength: str
    lr_reflectance: float
    bicubic_reflectance: Optional[float] = None
    sr_reflectance: float
    hr_reflectance: Optional[float] = None


class PixelProfileResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    sample_id: Optional[str] = None
    run_id: Optional[str] = None
    model_id: str
    hr_coordinates: Dict[str, int]
    lr_coordinates: Dict[str, int]
    bands_data: List[BandProfile]
    ndvi: Dict[str, Optional[float]]
    spectral_angle_deg: Optional[float] = None
    surface_classification: str
    signature_analysis: str
    interpretation_disclaimer: str = "Rule-based spectral interpretation (heuristic, not ground truth)"


class UncertaintySummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    mean_uncertainty: float
    max_uncertainty: float
    min_uncertainty: float
    high_uncertainty_pixel_pct: float
    calibration_status: str = "predicted uncertainty (uncalibrated until empirical validation)"


class UncertaintyOutput(BaseModel):
    model_config = ConfigDict(extra="allow")
    image: str
    summary: UncertaintySummary


class SuperResolveResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    model_id: str
    run_id: Optional[str] = None
    inference_time_s: float
    input: Dict[str, Any]
    output: Dict[str, Any]
    bicubic: Optional[Dict[str, Any]] = None
    error_map: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any]
    uncertainty: Optional[UncertaintyOutput] = None
    ground_truth: Optional[Dict[str, Any]] = None
    geospatial_metadata: Optional[Dict[str, Any]] = None


class DownstreamTaskMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")
    f1: float
    iou: float
    precision: float
    recall: float


class DownstreamTaskItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_name: str
    description: str
    bicubic: DownstreamTaskMetrics
    rcan: DownstreamTaskMetrics
    ground_truth_pixel_count: int
    masks: Dict[str, str]


class DownstreamMasksResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    tasks: Dict[str, DownstreamTaskItem]


class CompareResponse(BaseModel):
    status: str
    input: Dict[str, Any]
    models: Dict[str, Any]
    comparison_table: List[Dict[str, Any]]
    ground_truth: Optional[Dict[str, Any]] = None


class AsyncJobSubmitResponse(BaseModel):
    status: str
    job_id: str
    model_id: str
    status_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    model_id: str
    progress_pct: int = 0
    is_cancelled: bool = False
    created_at: str
    completed_at: Optional[str] = None
    inference_time_s: Optional[float] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobListResponse(BaseModel):
    jobs: List[Dict[str, Any]]


class ReportResponse(BaseModel):
    title: str
    problem_statement: str
    target_organization: str
    sample_id: str
    model_id: str
    scale_factor: str
    input_dimension: List[int]
    output_dimension: List[int]
    latency_seconds: float
    metrics: Dict[str, Any]
    uncertainty_summary: Optional[Dict[str, Any]] = None
    spectral_integrity_compliance: Dict[str, Any]


class SpectralIndicesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    indices: Dict[str, Any]


class CropHealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    disclaimer: str
    model_id: str
    classification_map: str
    area_statistics: Dict[str, Any]
    health_score: float
    mean_ndvi: float
    mean_evi: float
    sr_vs_lr_ndvi_uplift: float
    recommendations: List[str]
    class_legend: Dict[str, str]


class FieldBoundaryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    disclaimer: str
    model_id: str
    sr_edge_overlay: str
    lr_edge_overlay: str
    sr_edge_density: float
    lr_edge_density: float
    boundary_improvement_ratio: float
    sr_edge_pixel_count: int
    lr_edge_pixel_count: int
    method: str


class ChangeDetectionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    run_id_t1: str
    run_id_t2: str
    method: str
    ndvi_difference_map: str
    spectral_difference_map: str
    change_magnitude_map: str
    statistics: Dict[str, Any]
    interpretation: str
    disclaimer: str


class BatchSubmitResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    batch_id: str
    job_ids: List[str]
    total: int
    status_url: str


class BatchStatusResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    batch_id: str
    overall_status: str
    completed: int
    failed: int
    total: int
    jobs: List[Dict[str, Any]]


class ModelReloadResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    model_id: str
    checkpoint: str
    metadata: Optional[Dict[str, Any]] = None
