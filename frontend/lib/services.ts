import type {
  ApiInspectionResult,
  Inspection,
  InspectionRecord,
  SimulationEvent,
  SimulationStatus,
  SystemStatus,
} from "./types";

export const API_BASE_URL = (
  // The bundled launcher binds FastAPI to IPv4 loopback. Using the explicit
  // address avoids browsers preferring an unavailable IPv6 localhost socket.
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail ?? body.error?.message ?? message;
    } catch {
      // Keep the HTTP fallback message.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function absoluteUrl(path: string): string {
  return path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function mapApiResult(
  result: ApiInspectionResult,
  metadata?: { name?: string; date?: string },
): Inspection {
  const width = result.summary.image_width || 1;
  const height = result.summary.image_height || 1;
  const detections = result.detections.map((detection, index) => {
    const box = detection.bounding_box;
    return {
      id: `Crack #${String(index + 1).padStart(2, "0")}`,
      confidence: detection.confidence,
      area: detection.mask
        ? `${(detection.mask.area_ratio * 100).toFixed(3)}% image area`
        : "Mask unavailable",
      severity: "Not assessed" as const,
      region: `Region ${String.fromCharCode(65 + (index % 26))}`,
      x: (box.x1 / width) * 100,
      y: (box.y1 / height) * 100,
      width: ((box.x2 - box.x1) / width) * 100,
      height: ((box.y2 - box.y1) / height) * 100,
      polygon: detection.mask?.polygon ?? [],
    };
  });
  const affectedRatio = result.detections.reduce(
    (total, detection) => total + (detection.mask?.area_ratio ?? 0),
    0,
  );

  return {
    id: result.inspection_id,
    name: metadata?.name || result.inspection_id,
    date: metadata?.date || "Saved inspection",
    location: "Uploaded inspection image",
    images: 1,
    cracks: result.summary.cracks_detected,
    severity: "Not assessed",
    status: result.status === "completed" ? "Completed" : "Processing",
    imageName: metadata?.name || "Uploaded inspection image",
    imageSize: `${result.summary.image_width.toLocaleString()} × ${result.summary.image_height.toLocaleString()} px`,
    detections,
    originalUrl: absoluteUrl(result.image.original_url),
    annotatedUrl: absoluteUrl(result.image.annotated_url),
    inferenceTimeMs: result.inference.inference_time_ms,
    threshold: result.inference.confidence_threshold,
    affectedArea: `${(affectedRatio * 100).toFixed(3)}%`,
    model: result.inference.model,
    source: "live",
  };
}

function mapRecord(record: InspectionRecord): Inspection {
  return {
    id: record.inspection_id,
    name: record.filename,
    date: formatDate(record.timestamp),
    location: "Uploaded inspection image",
    images: 1,
    cracks: record.detection_count,
    severity: "Not assessed",
    status: record.status === "completed" ? "Completed" : "Failed",
    imageName: record.filename,
    imageSize: "Open result for dimensions",
    detections: [],
    processingTimeMs: record.processing_time_ms,
    source: "live",
  };
}

export const inferenceService = {
  async run(file: File, threshold: number): Promise<Inspection> {
    const form = new FormData();
    form.append("file", file);
    form.append("confidence", String(threshold));
    const result = await apiFetch<ApiInspectionResult>("/api/v1/inspect", {
      method: "POST",
      body: form,
    });
    return mapApiResult(result, { name: file.name });
  },
};

export const inspectionService = {
  async list(): Promise<Inspection[]> {
    const records = await apiFetch<InspectionRecord[]>("/api/v1/inspections?limit=100");
    return records.map(mapRecord);
  },
  async get(id: string): Promise<Inspection> {
    const [result, records] = await Promise.all([
      apiFetch<ApiInspectionResult>(`/api/v1/inspections/${encodeURIComponent(id)}`),
      apiFetch<InspectionRecord[]>("/api/v1/inspections?limit=500").catch(() => []),
    ]);
    const record = records.find((item) => item.inspection_id === id);
    return mapApiResult(result, record ? {
      name: record.filename,
      date: formatDate(record.timestamp),
    } : undefined);
  },
};

export const systemService = {
  getStatus: () => apiFetch<SystemStatus>("/api/v1/system/status"),
  health: () => apiFetch<{ status: string }>("/api/v1/health"),
};

export const simulationService = {
  getStatus: () => apiFetch<SimulationStatus>("/api/v1/simulation/status"),
  startMission: () => apiFetch<SimulationStatus>("/api/v1/simulation/mission/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  }),
  stopMission: () => apiFetch<SimulationStatus>("/api/v1/simulation/mission/stop", { method: "POST" }),
  frameUrl: (path?: string | null) => path ? absoluteUrl(path) : undefined,
  subscribe(listener: (event: SimulationEvent) => void, onError: () => void) {
    const url = `${API_BASE_URL.replace(/^http/, "ws")}/api/v1/simulation/ws`;
    const socket = new WebSocket(url);
    socket.onmessage = (message) => {
      try { listener(JSON.parse(message.data) as SimulationEvent); } catch { /* Ignore malformed events. */ }
    };
    socket.onerror = onError;
    return () => socket.close();
  },
};
