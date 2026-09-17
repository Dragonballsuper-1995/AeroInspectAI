"use client";

import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  ArrowDown,
  ArrowUp,
  BarChart3,
  Bot,
  Box,
  Camera,
  Check,
  Clock3,
  Cpu,
  Database,
  Gauge,
  Home,
  Image as ImageIcon,
  Layers3,
  Menu,
  Moon,
  Navigation,
  Play,
  Plus,
  Search,
  Settings,
  ShieldCheck,
  Sun,
  Target,
  Upload,
  Wifi,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { datasetStats, experiments } from "@/lib/mock/demo-data";
import {
  inferenceService,
  inspectionService,
  simulationService,
  systemService,
} from "@/lib/services";
import type { Inspection, SimulationStatus, SystemStatus } from "@/lib/types";

type View =
  | "dashboard"
  | "new"
  | "inspection"
  | "inspections"
  | "analytics"
  | "experiments"
  | "system"
  | "settings";

const navItems: Array<{ label: string; href: string; icon: LucideIcon }> = [
  { label: "Overview", href: "/dashboard", icon: Home },
  { label: "New inspection", href: "/inspection/new", icon: Plus },
  { label: "Inspections", href: "/inspections", icon: ImageIcon },
  { label: "Analytics", href: "/analytics", icon: BarChart3 },
  { label: "Experiments", href: "/experiments", icon: Box },
  { label: "System", href: "/system", icon: Activity },
];

export function Workspace({ view, inspectionId }: { view: View; inspectionId?: string }) {
  const pathname = usePathname();
  const router = useRouter();
  const [mobileNav, setMobileNav] = useState(false);
  const [dark, setDark] = useState(true);
  const [system, setSystem] = useState<SystemStatus | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  }, [dark]);

  useEffect(() => {
    systemService.getStatus().then(setSystem).catch(() => setSystem(null));
  }, []);

  const backendReady = Boolean(system?.model.loaded);
  const title = view === "new" ? "New inspection" : view === "inspection" ? "Inspection result" : view[0].toUpperCase() + view.slice(1);

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Navigation size={18} /></div>
          <div><strong>AeroInspect</strong><span>AI CONSOLE</span></div>
          <button className="icon-button sidebar-close" aria-label="Close navigation" onClick={() => setMobileNav(false)}><X size={18} /></button>
        </div>
        <div className="sidebar-label">Workspace</div>
        <nav className="main-nav" aria-label="Main navigation">
          {navItems.map(({ label, href, icon: Icon }) => (
            <button key={href} className={`nav-item ${pathname.startsWith(href) ? "active" : ""}`} onClick={() => { router.push(href); setMobileNav(false); }}>
              <Icon size={17} /><span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-label lower-label">Live status</div>
        <div className="status-mini">
          <span className={`status-dot ${backendReady ? "ready" : "offline"}`} />
          <div><strong>{backendReady ? "Inference ready" : "Backend offline"}</strong><small>{backendReady ? system?.compute.device : "Start FastAPI on :8000"}</small></div>
        </div>
        <div className="status-mini"><span className="status-dot warning" /><div><strong>Simulation optional</strong><small>WSL Gazebo controller</small></div></div>
        <div className="sidebar-bottom">
          <button className="nav-item" onClick={() => router.push("/settings")}><Settings size={17} /><span>Settings</span></button>
          <button className="nav-item" onClick={() => setDark(!dark)}>{dark ? <Sun size={17} /> : <Moon size={17} />}<span>{dark ? "Light mode" : "Dark mode"}</span></button>
        </div>
      </aside>
      {mobileNav && <button className="scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
      <main className="main-area">
        <header className="topbar">
          <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileNav(true)}><Menu size={20} /></button>
          <div className="breadcrumbs"><span>AeroInspect AI</span><span>/</span><strong>{title}</strong></div>
          <div className="topbar-actions"><span className={backendReady ? "soft-badge" : "demo-pill"}><span className={`status-dot ${backendReady ? "ready" : "offline"}`} />{backendReady ? "LIVE MODEL" : "BACKEND OFFLINE"}</span></div>
        </header>
        <div className="page-content">
          <PageHeader view={view} backendReady={backendReady} onNew={() => router.push("/inspection/new")} />
          {view === "dashboard" && <Dashboard system={system} router={router} />}
          {view === "new" && <NewInspection router={router} backendReady={backendReady} />}
          {view === "inspection" && <InspectionResult inspectionId={inspectionId} router={router} />}
          {view === "inspections" && <InspectionHistory router={router} />}
          {view === "analytics" && <Analytics />}
          {view === "experiments" && <Experiments />}
          {view === "system" && <SystemPage system={system} />}
          {view === "settings" && <SettingsPage dark={dark} setDark={setDark} />}
        </div>
      </main>
    </div>
  );
}

function PageHeader({ view, backendReady, onNew }: { view: View; backendReady: boolean; onNew: () => void }) {
  const copy: Record<View, [string, string]> = {
    dashboard: ["AeroInspect AI", "Real crack-segmentation inference for structural inspection images."],
    new: ["Start an inspection", "Upload an image and run the trained YOLOv8n-Seg checkpoint."],
    inspection: ["Inspection result", "Model-derived boxes, segmentation masks, confidence, and image coverage."],
    inspections: ["Inspection history", "Results persisted by the local FastAPI backend."],
    analytics: ["Analytics", "Live inspection totals and verified experiment context."],
    experiments: ["Experiments", "EXP001 baseline and EXP002 screening evidence."],
    system: ["System", "Live simulator state, MAVLink telemetry, and captured inspection frames."],
    settings: ["Settings", "Local review configuration and connection details."],
  };
  return (
    <div className="page-header">
      <div><div className="eyebrow"><span className={`status-dot ${backendReady ? "ready" : "offline"}`} />{backendReady ? "MODEL CONNECTED" : "START BACKEND TO INSPECT"}</div><h1>{copy[view][0]}</h1><p>{copy[view][1]}</p></div>
      {(view === "dashboard" || view === "inspections") && <button className="button button-primary" onClick={onNew}><Plus size={17} />New inspection</button>}
    </div>
  );
}

function Dashboard({ system, router }: { system: SystemStatus | null; router: ReturnType<typeof useRouter> }) {
  const [records, setRecords] = useState<Inspection[]>([]);
  const [error, setError] = useState("");
  useEffect(() => { inspectionService.list().then(setRecords).catch((reason: Error) => setError(reason.message)); }, []);
  const cracks = records.reduce((sum, record) => sum + record.cracks, 0);
  return <>
    {error && <ErrorBanner message={`Live history unavailable: ${error}`} />}
    <div className="metric-grid overview-metrics">
      <MetricCard label="Saved inspections" value={String(records.length)} note="Live backend records" icon={Layers3} />
      <MetricCard label="Cracks detected" value={String(cracks)} note="Model detections" icon={Target} accent="amber" />
      <MetricCard label="Model status" value={system?.model.loaded ? "Ready" : "Offline"} note={system?.model.name ?? "FastAPI unavailable"} icon={Bot} accent="blue" />
      <MetricCard label="Compute" value={system?.compute.cuda_available ? "CUDA" : "CPU"} note={system?.compute.gpu_name ?? "Not connected"} icon={Cpu} accent="blue" />
    </div>
    <div className="dashboard-grid overview-grid">
      <section className="panel hero-panel"><div className="panel-heading"><div><span className="section-kicker">Review workflow</span><h2>From simulated capture to model review</h2></div><span className="tag tag-green">Real inference</span></div><div className="hero-visual architecture-visual"><img className="architecture-diagram" src="/architecture/aeroinspect-review-workflow.svg" alt="AeroInspect architecture: simulated drone camera sends frames through FastAPI and YOLOv8n-Seg to an annotated review result." /></div><div className="hero-footer"><div><span className="tiny-label">Checkpoint</span><strong>EXP001 best.pt</strong></div><div><span className="tiny-label">Task</span><strong>Crack instance segmentation</strong></div><button className="button button-primary" onClick={() => router.push("/inspection/new")}>Run inspection <ArrowUpRight size={15} /></button></div></section>
      <section className="panel model-card"><div className="panel-heading"><div><span className="section-kicker">Live deployment</span><h2>Current model</h2></div><span className={`tag ${system?.model.loaded ? "tag-green" : "tag-muted"}`}>{system?.model.loaded ? "Loaded" : "Offline"}</span></div><div className="model-name"><div className="model-symbol"><Cpu size={21} /></div><div><strong>{system?.model.name ?? "YOLOv8n-Seg"}</strong><span>EXP001 baseline checkpoint</span></div></div><div className="detail-list"><Detail label="Device" value={system?.compute.device ?? "Not connected"} /><Detail label="GPU" value={system?.compute.gpu_name ?? "Not connected"} /><Detail label="VRAM" value={system ? `${system.compute.vram_gb} GB` : "N/A"} /><Detail label="Classes" value={system ? Object.values(system.model.classes).join(", ") : "crack"} /></div></section>
    </div>
    <RecentInspections inspections={records.slice(0, 5)} router={router} />
  </>;
}

function NewInspection({ router, backendReady }: { router: ReturnType<typeof useRouter>; backendReady: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.25);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);
  const choose = (candidate?: File) => {
    if (!candidate) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(candidate.type)) return setError("Choose a JPG, PNG, or WEBP image.");
    if (candidate.size > 20 * 1024 * 1024) return setError("Images must be 20 MB or smaller.");
    setError(""); setFile(candidate); setPreviewUrl(URL.createObjectURL(candidate));
  };
  const run = async () => {
    if (!file) return;
    setBusy(true); setError("");
    try {
      const result = await inferenceService.run(file, threshold);
      sessionStorage.setItem(`inspection:${result.id}`, JSON.stringify(result));
      router.push(`/inspection/${result.id}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Inspection failed");
      setBusy(false);
    }
  };
  return <div className="new-inspection-grid">
    <section className="panel upload-panel"><div className="panel-heading"><div><span className="section-kicker">Step 01</span><h2>Select an inspection image</h2></div><span className="tag tag-blue">Local processing</span></div><button type="button" className={`dropzone ${file ? "has-file" : ""}`} onClick={() => inputRef.current?.click()}>{file && previewUrl ? <div className="file-preview"><img className="upload-preview" src={previewUrl} alt={`Preview of ${file.name}`} /><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · ready</span><small>Choose another image to replace this preview</small></div> : <><div className="upload-icon"><Upload size={21} /></div><strong>Choose a structural inspection image</strong><span>JPG, PNG, or WEBP</span><small>Maximum 20 MB</small></>}</button><input className="upload-input" ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(event) => choose(event.target.files?.[0])} />{error && <ErrorBanner message={error} />}</section>
    <section className="panel config-panel"><div className="panel-heading"><div><span className="section-kicker">Step 02</span><h2>Inference configuration</h2></div><Gauge size={17} /></div><label>Confidence threshold <output>{threshold.toFixed(2)}</output><input className="range" type="range" min="0" max="0.9" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label><div className="detail-list"><Detail label="Model" value="YOLOv8n-Seg / EXP001" /><Detail label="Output" value="Boxes + segmentation masks" /><Detail label="Storage" value="Local filesystem" /></div><div className="notice notice-amber"><AlertTriangle size={15} /><span>Results are visual defect detections, not a structural safety certification.</span></div><div className="config-footer"><small>{backendReady ? "Backend and model ready" : "Backend is offline"}</small><button className="button button-primary" disabled={!file || busy || !backendReady} onClick={run}>{busy ? <><Activity size={16} />Inspecting…</> : <><Play size={16} />Run inspection</>}</button></div></section>
  </div>;
}

function InspectionResult({ inspectionId, router }: { inspectionId?: string; router: ReturnType<typeof useRouter> }) {
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!inspectionId) return;
    const cached = sessionStorage.getItem(`inspection:${inspectionId}`);
    if (cached) { setInspection(JSON.parse(cached)); return; }
    inspectionService.get(inspectionId).then(setInspection).catch((reason: Error) => setError(reason.message));
  }, [inspectionId]);
  if (error) return <ErrorBanner message={`Could not load inspection: ${error}`} />;
  if (!inspection) return <LoadingPanel label="Loading inspection result…" />;
  const peak = inspection.detections.length ? Math.max(...inspection.detections.map((item) => item.confidence)) : 0;
  return <>
    <div className="result-toolbar"><div><span className="tag tag-green"><span className="status-dot ready" />Completed</span><span className="result-id">{inspection.id}</span></div><button className="button button-quiet" onClick={() => router.push("/inspections")}>History <ArrowUpRight size={15} /></button></div>
    <div className="result-grid">
      <section className="panel viewer-panel"><div className="panel-heading"><div><span className="section-kicker">Real model output</span><h2>{inspection.imageName}</h2></div><span className="viewer-meta">{inspection.imageSize}</span></div><div className="viewer-grid"><LiveImage label="Original image" src={inspection.originalUrl} /><LiveImage label="Annotated segmentation" src={inspection.annotatedUrl} /></div><div className="viewer-caption"><span><span className="legend-swatch box" />Bounding boxes</span><span><span className="legend-swatch mask" />Segmentation masks</span></div></section>
      <section className="panel summary-panel"><div className="panel-heading"><div><span className="section-kicker">Result summary</span><h2>Detected cracks</h2></div><span className="tag tag-blue">Live result</span></div><div className="result-metrics"><div><strong>{inspection.cracks}</strong><span>Total cracks</span></div><div><strong>{peak ? `${(peak * 100).toFixed(1)}%` : "—"}</strong><span>Peak confidence</span></div><div title="Sum of instance-mask areas; overlapping masks may be counted more than once."><strong>{inspection.affectedArea ?? "0%"}</strong><span>Summed mask coverage</span></div><div><strong>Not assessed</strong><span>Severity</span></div></div><div className="detection-list">{inspection.detections.map((detection, index) => <div className="detection-row" key={detection.id}><span className="detection-number">{String(index + 1).padStart(2, "0")}</span><span><strong>{detection.id}</strong><small>{detection.area}</small></span><span className="detection-confidence">{(detection.confidence * 100).toFixed(1)}%</span></div>)}</div>{inspection.detections.length === 0 && <div className="empty-state"><Check size={22} /><strong>No cracks above threshold</strong><span>This does not certify that the structure is defect-free.</span></div>}<div className="notice notice-amber"><AlertTriangle size={15} /><span>AI-assisted visual review only. A qualified engineer must assess safety.</span></div></section>
    </div>
    <section className="panel"><div className="panel-heading"><div><span className="section-kicker">Traceability</span><h2>Inference metadata</h2></div><Database size={17} /></div><div className="metadata-grid"><Detail label="Inspection ID" value={inspection.id} /><Detail label="Model" value={inspection.model ?? "YOLOv8n-Seg"} /><Detail label="Threshold" value={inspection.threshold?.toFixed(2) ?? "N/A"} /><Detail label="Inference time" value={inspection.inferenceTimeMs ? `${inspection.inferenceTimeMs.toFixed(1)} ms` : "N/A"} /><Detail label="Source" value="FastAPI + trained checkpoint" /><Detail label="Physical dimensions" value="Not available without calibration" /></div></section>
  </>;
}

function InspectionHistory({ router }: { router: ReturnType<typeof useRouter> }) {
  const [records, setRecords] = useState<Inspection[]>([]);
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<InspectionSortKey>("date");
  const [sortDirection, setSortDirection] = useState<SortDirection>("descending");
  const [error, setError] = useState("");
  useEffect(() => { inspectionService.list().then(setRecords).catch((reason: Error) => setError(reason.message)); }, []);
  const filtered = useMemo(() => {
    const searched = records.filter((item) => `${item.id} ${item.name} ${item.imageName}`.toLowerCase().includes(query.toLowerCase()));
    return [...searched].sort((left, right) => compareInspections(left, right, sortKey, sortDirection));
  }, [records, query, sortDirection, sortKey]);
  const toggleSort = (key: InspectionSortKey) => {
    if (sortKey === key) setSortDirection((direction) => direction === "ascending" ? "descending" : "ascending");
    else { setSortKey(key); setSortDirection("ascending"); }
  };
  return <section className="panel history-panel">{error && <ErrorBanner message={error} />}<div className="filter-bar"><div className="search-input"><Search size={16} /><input placeholder="Search inspections" value={query} onChange={(event) => setQuery(event.target.value)} /></div></div><div className="table-wrap"><table className="history-table"><thead><tr><SortableHeader label="Inspection ID" column="id" active={sortKey} direction={sortDirection} onSort={toggleSort} /><SortableHeader label="Date" column="date" active={sortKey} direction={sortDirection} onSort={toggleSort} /><SortableHeader label="Image" column="imageName" active={sortKey} direction={sortDirection} onSort={toggleSort} /><SortableHeader label="Cracks" column="cracks" active={sortKey} direction={sortDirection} onSort={toggleSort} /><SortableHeader label="Processing" column="processingTimeMs" active={sortKey} direction={sortDirection} onSort={toggleSort} /><SortableHeader label="Status" column="status" active={sortKey} direction={sortDirection} onSort={toggleSort} /></tr></thead><tbody>{filtered.map((inspection) => <tr key={inspection.id} onClick={() => router.push(`/inspection/${inspection.id}`)}><td data-label="Inspection ID"><strong>{inspection.id}</strong></td><td data-label="Date">{inspection.date}</td><td data-label="Image">{inspection.imageName}</td><td data-label="Cracks">{inspection.cracks}</td><td data-label="Processing">{inspection.processingTimeMs ? `${inspection.processingTimeMs.toFixed(0)} ms` : "—"}</td><td data-label="Status"><span className="status-text"><span className="status-dot ready" />{inspection.status}</span></td></tr>)}</tbody></table></div>{!error && filtered.length === 0 && <div className="empty-state"><ImageIcon size={23} /><strong>No saved inspections</strong><span>Run the first real inspection to populate history.</span></div>}<div className="table-footer"><span>{filtered.length} live backend records</span></div></section>;
}

type InspectionSortKey = "id" | "date" | "imageName" | "cracks" | "processingTimeMs" | "status";
type SortDirection = "ascending" | "descending";

function compareInspections(left: Inspection, right: Inspection, key: InspectionSortKey, direction: SortDirection) {
  const leftValue = left[key];
  const rightValue = right[key];
  const leftMissing = leftValue === null || leftValue === undefined || leftValue === "";
  const rightMissing = rightValue === null || rightValue === undefined || rightValue === "";
  if (leftMissing || rightMissing) {
    if (leftMissing && rightMissing) return 0;
    return leftMissing ? 1 : -1;
  }
  let comparison = 0;
  if (key === "cracks" || key === "processingTimeMs") comparison = Number(leftValue) - Number(rightValue);
  else if (key === "date") comparison = parseInspectionDate(String(leftValue)) - parseInspectionDate(String(rightValue));
  else comparison = String(leftValue).localeCompare(String(rightValue), undefined, { sensitivity: "base" });
  return direction === "ascending" ? comparison : -comparison;
}

function parseInspectionDate(value: string) {
  const parsed = Date.parse(value);
  if (!Number.isNaN(parsed)) return parsed;
  const localized = value.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4}),?\s*(\d{1,2}):(\d{2})(?::(\d{2}))?/);
  if (!localized) return 0;
  const [, day, month, year, hour, minute, second = "0"] = localized;
  return new Date(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second)).getTime();
}

function SortableHeader({ label, column, active, direction, onSort }: { label: string; column: InspectionSortKey; active: InspectionSortKey; direction: SortDirection; onSort: (column: InspectionSortKey) => void }) {
  const selected = active === column;
  return <th aria-sort={selected ? direction : "none"}><button type="button" className={`sortable-header ${selected ? "active" : ""}`} onClick={() => onSort(column)}>{label}{selected ? direction === "ascending" ? <ArrowUp size={13} /> : <ArrowDown size={13} /> : <ArrowDown size={12} />}</button></th>;
}

function Analytics() {
  const [records, setRecords] = useState<Inspection[]>([]);
  useEffect(() => { inspectionService.list().then(setRecords).catch(() => setRecords([])); }, []);
  const cracks = records.reduce((sum, item) => sum + item.cracks, 0);
  const avgCracks = records.length ? cracks / records.length : 0;
  return <><div className="verified-strip"><ShieldCheck size={17} /><span>Inspection totals are live. Experiment metrics are verified repository artifacts.</span></div><div className="metric-grid analytics-metrics"><MetricCard label="Live inspections" value={String(records.length)} note="Backend history" icon={Layers3} /><MetricCard label="Detected cracks" value={String(cracks)} note="All saved inspections" icon={Target} accent="amber" /><MetricCard label="Average cracks" value={avgCracks.toFixed(1)} note="Per inspection" icon={Gauge} accent="blue" /><MetricCard label="EXP002 F1" value="71.98%" note="Validation, raw epoch 22" icon={BarChart3} accent="blue" /></div><section className="panel dataset-panel"><div className="panel-heading"><div><span className="section-kicker">Validated dataset</span><h2>AeroInspect crack v1</h2></div><span className="tag tag-green">Quality checked</span></div><div className="dataset-stats"><DatasetStat label="Total images" value={datasetStats.total.toLocaleString()} /><DatasetStat label="Training" value={datasetStats.train.toLocaleString()} /><DatasetStat label="Validation" value={datasetStats.val.toLocaleString()} /><DatasetStat label="Held-out test" value={datasetStats.test.toLocaleString()} /></div><div className="notice notice-blue"><Database size={15} /><span>Single class: crack. The test split is not used for hyperparameter tuning.</span></div></section></>;
}

function Experiments() {
  const [selected, setSelected] = useState(experiments[0]);
  return <div className="experiments-layout"><section className="panel experiments-table"><div className="panel-heading"><div><span className="section-kicker">Training evidence</span><h2>Experiment explorer</h2></div><span className="tag tag-blue">2 tracked</span></div>{experiments.map((experiment) => <button className={`experiment-row ${selected.id === experiment.id ? "selected" : ""}`} key={experiment.id} onClick={() => setSelected(experiment)}><div className="experiment-id">{experiment.id}</div><div><strong>{experiment.label}</strong><span>{experiment.imageSize}px · batch {experiment.batch}</span></div><ArrowUpRight size={15} /></button>)}</section><section className="panel experiment-detail"><div className="panel-heading"><div><span className="section-kicker">Verified configuration</span><h2>{selected.id}</h2></div><span className="tag tag-green">{selected.status}</span></div><div className="detail-grid"><Detail label="Model" value={selected.model} /><Detail label="Image size" value={`${selected.imageSize} × ${selected.imageSize}`} /><Detail label="Batch size" value={String(selected.batch)} /><Detail label="Epochs" value={String(selected.epochs)} /><Detail label="Learning rate" value={selected.learningRate} /><Detail label="Augmentation" value={selected.augmentation} /><Detail label="Weight decay" value={selected.weightDecay} /><Detail label="Best epoch" value={selected.bestEpoch ? String(selected.bestEpoch) : "Checkpoint validation"} /></div><h3 className="subheading">Segmentation validation metrics</h3><div className="experiment-metrics"><MiniMetric label="Precision" value={selected.precision} /><MiniMetric label="Recall" value={selected.recall} /><MiniMetric label="F1" value={selected.f1} /><MiniMetric label="mAP50" value={selected.map50} /><MiniMetric label="mAP50-95" value={selected.map5095} /></div>{selected.id === "EXP002" && <div className="notice notice-amber"><AlertTriangle size={15} /><span>EXP002 selected the highest validation F1. The exporter reported epoch 23, but the raw one-based CSV identifies epoch 22. No test evaluation was performed.</span></div>}</section></div>;
}

function SystemPage({ system }: { system: SystemStatus | null }) {
  const [simulation, setSimulation] = useState<SimulationStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const refresh = () => simulationService.getStatus()
      .then((status) => { if (active) { setSimulation(status); setError(""); } })
      .catch((reason: Error) => { if (active) setError(reason.message); });
    refresh();
    const unsubscribe = simulationService.subscribe((event) => {
      if (event.type === "heartbeat") return;
      if ("mission_state" in event.data && active) setSimulation(event.data as SimulationStatus);
      else refresh();
    }, () => { if (active) refresh(); });
    const poll = window.setInterval(refresh, 5000);
    return () => { active = false; unsubscribe(); window.clearInterval(poll); };
  }, []);

  const requestMission = async () => {
    setBusy(true); setError("");
    try { setSimulation(await simulationService.startMission()); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Unable to start simulation mission."); }
    finally { setBusy(false); }
  };
  const stopMission = async () => {
    setBusy(true); setError("");
    try { setSimulation(await simulationService.stopMission()); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Unable to stop simulation mission."); }
    finally { setBusy(false); }
  };
  const telemetry = simulation?.telemetry;
  const isRunning = simulation?.connection_state === "RUNNING";
  const isReady = simulation?.connection_state === "READY";
  const frameUrl = simulationService.frameUrl(simulation?.latest_frame_url);
  const simulatorStatus = simulation?.connection_state ?? "SIMULATION_UNAVAILABLE";
  const maybe = (value: number | null | undefined, digits = 1) => value === null || value === undefined ? "—" : value.toFixed(digits);

  return <>
    {error && <ErrorBanner message={error} />}
    <div className="system-banner">
      <div className="system-orbit"><div className="orbit-core"><Navigation size={26} /></div><span className="orbit-ring ring-one" /><span className="orbit-ring ring-two" /></div>
      <div><span className="section-kicker">Gazebo + ArduPilot SITL</span><h2>{simulation?.detail ?? "Simulator status is loading."}</h2><p>Frames and telemetry shown here originate from the WSL controller. No values are synthesized by the dashboard.</p></div>
      {isRunning ? <button className="button button-quiet" disabled={busy} onClick={stopMission}><X size={15} />{busy ? "Requesting…" : "Abort & RTL"}</button> : <button className="button button-primary" disabled={!isReady || busy} onClick={requestMission}><Play size={15} />{busy ? "Queueing…" : "Start inspection mission"}</button>}
    </div>
    {!isReady && !isRunning && <div className="notice notice-blue"><Wifi size={15} /><span>Start the WSL simulation stack first. The dashboard will remain unavailable until the controller registers its camera and MAVLink connection.</span></div>}
    <div className="system-layout">
      <section className="panel telemetry-panel"><div className="panel-heading"><div><span className="section-kicker">Flight telemetry</span><h2>Live vehicle state</h2></div><span className="tag tag-blue">SITL</span></div><div className="system-grid">
        <SystemCard name="Backend" status={system?.backend.status ?? "Offline"} value="FastAPI :8000" note={system?.model.loaded ? "EXP001 model loaded" : "Start backend"} icon={Wifi} />
        <SystemCard name="Simulator" status={simulatorStatus} value={simulation?.mission_state ?? "IDLE"} note={simulation?.waypoint_id ? `Waypoint ${simulation.waypoint_id}` : "Gazebo / SITL controller"} icon={Navigation} />
        <SystemCard name="Flight mode" status={isRunning ? "running" : "Standby"} value={telemetry?.flight_mode ?? "—"} note={telemetry?.armed ? "Armed" : "Not armed"} icon={Gauge} />
        <SystemCard name="Altitude" status={isRunning ? "running" : "Standby"} value={`${maybe(telemetry?.relative_altitude_m)} m`} note={`Speed ${maybe(telemetry?.ground_speed_m_s)} m/s`} icon={Activity} />
        <SystemCard name="Local NED" status={isRunning ? "running" : "Standby"} value={`N ${maybe(telemetry?.north_m)} · E ${maybe(telemetry?.east_m)}`} note={`D ${maybe(telemetry?.down_m)} m`} icon={Target} />
        <SystemCard name="Captured frames" status={simulation?.captures_completed ? "Ready" : "Standby"} value={String(simulation?.captures_completed ?? 0)} note={simulation?.latest_inspection_id ?? "No inspection record yet"} icon={Camera} />
      </div></section>
      <section className="panel simulation-frame-panel">
      <div className="panel-heading"><div><span className="section-kicker">Onboard camera</span><h2>Latest settled inspection frame</h2></div><span className={`tag ${frameUrl ? "tag-green" : "tag-muted"}`}>{frameUrl ? "Captured" : "Awaiting camera"}</span></div>
      <div className="simulation-frame-wrap">{frameUrl ? <img className="simulation-frame" src={`${frameUrl}?t=${encodeURIComponent(simulation?.updated_at ?? "")}`} alt="Latest simulated drone camera capture" /> : <div className="empty-state"><Camera size={23} /><strong>Camera frame unavailable</strong><span>Frames appear after a completed waypoint capture.</span></div>}</div>
      <p className="muted-copy">Frames originate from the simulated drone camera at each settled waypoint. The blank virtual wall carries five crack-photo panels as scene materials; the dashboard receives the complete rendered wall view, never a direct dataset upload.</p>
      {simulation?.latest_inspection_id && <button className="text-button" onClick={() => window.location.assign(`/inspection/${simulation.latest_inspection_id}`)}>Open linked inspection <ArrowUpRight size={14} /></button>}
    </section>
    </div>
  </>;
}

function SettingsPage({ dark, setDark }: { dark: boolean; setDark: (value: boolean) => void }) {
  return <div className="settings-grid"><section className="panel settings-section"><div className="panel-heading"><div><span className="section-kicker">Console</span><h2>Appearance</h2></div><Sun size={17} /></div><div className="theme-options"><button className={!dark ? "selected" : ""} onClick={() => setDark(false)}><Sun size={17} />Light</button><button className={dark ? "selected" : ""} onClick={() => setDark(true)}><Moon size={17} />Dark</button></div></section><section className="panel settings-section"><div className="panel-heading"><div><span className="section-kicker">Connection</span><h2>Local services</h2></div><Wifi size={17} /></div><div className="detail-list"><Detail label="API" value={process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"} /><Detail label="Telemetry" value="Frontend simulation only" /><Detail label="Hardware" value="Not connected" /></div></section></div>;
}

function RecentInspections({ inspections, router }: { inspections: Inspection[]; router: ReturnType<typeof useRouter> }) {
  return <section className="panel table-panel"><div className="panel-heading"><div><span className="section-kicker">Live history</span><h2>Recent inspections</h2></div><button className="text-button" onClick={() => router.push("/inspections")}>View all <ArrowUpRight size={14} /></button></div>{inspections.length ? <div className="table-wrap"><table><thead><tr><th>Inspection</th><th>Image</th><th>Cracks</th><th>Status</th></tr></thead><tbody>{inspections.map((item) => <tr key={item.id} onClick={() => router.push(`/inspection/${item.id}`)}><td><strong>{item.id}</strong></td><td>{item.imageName}</td><td>{item.cracks}</td><td><span className="status-text"><span className="status-dot ready" />{item.status}</span></td></tr>)}</tbody></table></div> : <div className="empty-state"><ImageIcon size={23} /><strong>No inspections yet</strong><span>Upload an image to create the first record.</span></div>}</section>;
}

function LiveImage({ label, src }: { label: string; src?: string }) {
  return <div className="image-viewport"><div className="viewport-label">{label}<span>Live</span></div><div className="live-image-wrap">{src ? <img className="live-image" src={src} alt={label} /> : <div className="empty-state">Image unavailable</div>}</div></div>;
}

function SystemCard({ name, status, value, note, icon: Icon }: { name: string; status: string; value: string; note: string; icon: LucideIcon }) {
  const active = status === "Ready" || status === "running";
  return <div className="panel system-card"><div className="system-card-head"><div className="component-icon"><Icon size={18} /></div><span className={`tag ${active ? "tag-green" : status === "Offline" ? "tag-muted" : "tag-amber"}`}>{status}</span></div><h3>{name}</h3><strong>{value}</strong><small>{note}</small><div className="system-update"><Clock3 size={13} />{status === "Simulated" ? "SIMULATED DATA" : "Live local state"}</div></div>;
}

function MetricCard({ label, value, note, icon: Icon, accent = "green" }: { label: string; value: string; note: string; icon: LucideIcon; accent?: string }) { return <div className="metric-card"><div className={`metric-icon ${accent}`}><Icon size={18} /></div><span className="metric-label">{label}</span><strong>{value}</strong><small>{note}</small></div>; }
function Detail({ label, value }: { label: string; value: string }) { return <div className="detail"><span>{label}</span><strong>{value}</strong></div>; }
function DatasetStat({ label, value }: { label: string; value: string }) { return <div><strong>{value}</strong><span>{label}</span></div>; }
function MiniMetric({ label, value }: { label: string; value: number | null }) { return <div><span>{label}</span><strong>{value === null ? "N/A" : `${(value * 100).toFixed(2)}%`}</strong></div>; }
function ErrorBanner({ message }: { message: string }) { return <div className="notice notice-amber error-banner"><AlertTriangle size={16} /><span>{message}</span></div>; }
function LoadingPanel({ label }: { label: string }) { return <section className="panel loading-panel"><Activity size={22} /><strong>{label}</strong></section>; }
