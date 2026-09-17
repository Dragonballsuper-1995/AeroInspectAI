export type Severity = "Low" | "Medium" | "High" | "Critical" | "Not assessed";

export interface Detection {
  id: string;
  confidence: number;
  area: string;
  severity: Severity;
  region: string;
  x: number;
  y: number;
  width: number;
  height: number;
  polygon: number[][];
}

export interface Inspection {
  id: string;
  name: string;
  date: string;
  location: string;
  images: number;
  cracks: number;
  severity: Severity;
  status: "Completed" | "Processing" | "Failed";
  imageName: string;
  imageSize: string;
  detections: Detection[];
  originalUrl?: string;
  annotatedUrl?: string;
  processingTimeMs?: number;
  inferenceTimeMs?: number;
  threshold?: number;
  affectedArea?: string;
  model?: string;
  source: "live" | "simulation";
}

export interface InspectionRecord {
  inspection_id: string;
  timestamp: string;
  status: string;
  filename: string;
  detection_count: number;
  average_confidence: number;
  max_confidence: number;
  processing_time_ms: number;
}

export interface ApiInspectionResult {
  inspection_id: string;
  status: string;
  image: { original_url: string; annotated_url: string };
  summary: {
    detections: number;
    cracks_detected: number;
    average_confidence: number;
    max_confidence: number;
    min_confidence: number | null;
    total_mask_area_pixels: number;
    image_width: number;
    image_height: number;
  };
  detections: Array<{
    id: number;
    class_id: number;
    class_name: string;
    confidence: number;
    bounding_box: { x1: number; y1: number; x2: number; y2: number };
    mask: { polygon: number[][]; area_pixels: number; area_ratio: number } | null;
  }>;
  inference: {
    model: string;
    device: string;
    image_size: number;
    confidence_threshold: number;
    iou_threshold: number;
    inference_time_ms: number;
  };
}

export interface SystemStatus {
  backend: { status: string };
  model: { loaded: boolean; name: string; path: string; classes: Record<string, string> };
  compute: {
    device: string;
    cuda_available: boolean;
    gpu_name: string;
    torch_version: string;
    ultralytics_version: string;
    vram_gb: number;
  };
}

export interface SimulationTelemetry {
  timestamp: string;
  latitude?: number | null;
  longitude?: number | null;
  relative_altitude_m?: number | null;
  north_m?: number | null;
  east_m?: number | null;
  down_m?: number | null;
  roll_deg?: number | null;
  pitch_deg?: number | null;
  yaw_deg?: number | null;
  ground_speed_m_s?: number | null;
  battery_percent?: number | null;
  flight_mode: string;
  armed: boolean;
}

export interface SimulationStatus {
  connection_state: "SIMULATION_UNAVAILABLE" | "AWAITING_SIMULATOR" | "READY" | "RUNNING" | "ERROR";
  mission_state: "IDLE" | "QUEUED" | "ARMING" | "TAKEOFF" | "TRANSIT" | "INSPECTING" | "RETURNING" | "LANDING" | "COMPLETED" | "ABORTED" | "ERROR";
  mission_id?: string | null;
  waypoint_id?: string | null;
  detail: string;
  telemetry?: SimulationTelemetry | null;
  latest_frame_url?: string | null;
  latest_inspection_id?: string | null;
  captures_completed: number;
  updated_at: string;
}

export interface SimulationEvent {
  type: string;
  timestamp: string;
  data: SimulationStatus | { inspection_id?: string; frame_url?: string };
}

export interface Experiment {
  id: string;
  label: string;
  model: string;
  imageSize: number;
  batch: number;
  epochs: number;
  learningRate: string;
  augmentation: string;
  weightDecay: string;
  status: string;
  f1: number | null;
  precision: number | null;
  recall: number | null;
  map50: number | null;
  map5095: number | null;
  bestEpoch: number | null;
}
