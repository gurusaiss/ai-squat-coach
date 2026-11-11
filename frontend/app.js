const el = (id) => document.getElementById(id);

const sections = {
  upload: el("upload-section"),
  progress: el("progress-section"),
  results: el("results-section"),
  error: el("error-section"),
};

function showOnly(name) {
  for (const key in sections) {
    sections[key].classList.toggle("hidden", key !== name);
  }
}

async function loadRequirements() {
  try {
    const res = await fetch("/api/interface-requirements");
    const data = await res.json();
    const list = el("reqs-list");
    list.innerHTML = "";
    const li1 = document.createElement("li");
    li1.textContent = `Supported formats: ${data.supported_formats.join(", ")}. Max length ${data.max_duration_sec}s, max size ${data.max_upload_mb}MB.`;
    list.appendChild(li1);
    data.recording_requirements.forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      list.appendChild(li);
    });
  } catch (e) {
    el("reqs-list").innerHTML = "<li>Could not load requirements from server.</li>";
  }
}

let skillCache = null;
async function loadSkillAbout() {
  try {
    const res = await fetch("/api/skill");
    skillCache = await res.json();
  } catch (e) {
    skillCache = null;
  }
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setupDropzone() {
  const dz = el("dropzone");
  const input = el("file-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files.length) uploadFile(input.files[0]);
  });
}

async function uploadFile(file) {
  el("upload-error").classList.add("hidden");
  const chip = el("file-chip");
  chip.textContent = `${file.name} · ${formatBytes(file.size)}`;
  chip.classList.remove("hidden");

  const form = new FormData();
  form.append("file", file);

  showOnly("progress");
  setStepperStage("upload");
  setProgress(0, "Uploading video...");

  let jobId;
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errBody.detail || "Upload failed");
    }
    const data = await res.json();
    jobId = data.job_id;
  } catch (e) {
    showOnly("upload");
    el("upload-error").textContent = e.message;
    el("upload-error").classList.remove("hidden");
    return;
  }

  pollJob(jobId);
}

function setProgress(pct, message) {
  el("progress-fill").style.width = `${pct}%`;
  el("progress-message").textContent = message;
}

const STAGE_MATCHERS = [
  { key: "upload", re: /upload/i },
  { key: "pose", re: /pose|landmark/i },
  { key: "bar", re: /barbell|bar position/i },
  { key: "reps", re: /repetition/i },
  { key: "measure", re: /measurement|document/i },
  { key: "ai", re: /explanation|LLM|write-?up/i },
  { key: "render", re: /render|video/i },
];

function setStepperStage(activeKey) {
  const items = document.querySelectorAll("#stepper li");
  const order = STAGE_MATCHERS.map((s) => s.key);
  const activeIdx = order.indexOf(activeKey);
  items.forEach((li) => {
    const idx = order.indexOf(li.dataset.key);
    li.classList.remove("active", "done");
    if (idx < activeIdx) li.classList.add("done");
    else if (idx === activeIdx) li.classList.add("active");
  });
}

function stageFromMessage(message) {
  for (const { key, re } of STAGE_MATCHERS) {
    if (re.test(message)) return key;
  }
  return null;
}

async function pollJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    const data = await res.json();

    if (data.status === "error") {
      showOnly("error");
      el("error-message").textContent = data.error || "Processing failed.";
      return;
    }

    setProgress(data.progress || 0, data.message || "Working...");
    const stage = stageFromMessage(data.message || "");
    if (stage) setStepperStage(stage);

    if (data.status === "done") {
      setStepperStage("render");
      document.querySelectorAll("#stepper li").forEach((li) => { li.classList.add("done"); li.classList.remove("active"); });
      renderResults(data.result, jobId);
      return;
    }
    setTimeout(() => pollJob(jobId), 1200);
  } catch (e) {
    showOnly("error");
    el("error-message").textContent = "Lost connection while checking job status.";
  }
}

const CRITERION_LABELS = {
  depth: "Depth",
  back_angle_bottom: "Back angle (bottom)",
  hip_drive_ascent: "Hip drive / ascent mechanics",
  knee_travel: "Knee position",
  bar_path: "Bar path",
  head_neck_position: "Head / eye-gaze (low-confidence proxy)",
};

const PROVIDER_LABELS = {
  grok: "Explanations by Grok",
  gemini: "Explanations by Gemini (Grok fallback)",
  template_fallback: "Template fallback (no AI write-up this run)",
};

function verdictLabel(v) {
  return { meets: "Meets standard", does_not_meet: "Does not meet standard", cannot_assess: "Cannot assess" }[v] || v;
}

function verdictIcon(v) {
  return { meets: "✓", does_not_meet: "✕", cannot_assess: "?" }[v] || "";
}

function renderScorecard(findings, repCount) {
  const counts = { meets: 0, does_not_meet: 0, cannot_assess: 0 };
  findings.forEach((f) => { counts[f.verdict] = (counts[f.verdict] || 0) + 1; });
  const card = el("scorecard");
  card.innerHTML = "";
  const items = [
    { key: "reps", num: repCount, label: "Repetitions detected", cls: "" },
    { key: "meets", num: counts.meets, label: "Criteria met", cls: "meets" },
    { key: "does_not_meet", num: counts.does_not_meet, label: "Below standard", cls: "does_not_meet" },
    { key: "cannot_assess", num: counts.cannot_assess, label: "Could not assess", cls: "cannot_assess" },
  ];
  items.forEach((it) => {
    const div = document.createElement("div");
    div.className = `score-chip ${it.cls}`;
    div.innerHTML = `<div class="num">${it.num}</div><div class="label">${it.label}</div>`;
    card.appendChild(div);
  });
}

function renderResults(result, jobId) {
  showOnly("results");

  const video = el("result-video");
  video.src = `/api/jobs/${jobId}/video`;

  el("provider-badge").textContent = PROVIDER_LABELS[result.writeup_method] || result.writeup_method;

  renderScorecard(result.findings, result.rep_count);

  el("overall-summary").textContent = result.overall_summary || "(no summary)";
  el("confidence-caveats").textContent = result.confidence_caveats || "";
  el("meta-line").textContent =
    `${result.rep_count} rep(s) detected · tracked side: ${result.side_used} · ${result.fps.toFixed(1)} fps`;

  const existingWarn = document.getElementById("video-warning");
  if (existingWarn) existingWarn.remove();
  if (result.warnings && result.warnings.length) {
    const warnEl = document.createElement("p");
    warnEl.id = "video-warning";
    warnEl.className = "error";
    warnEl.textContent = result.warnings.join(" ");
    el("confidence-caveats").after(warnEl);
  }

  const container = el("findings-container");
  container.innerHTML = "";

  const byRep = {};
  result.findings.forEach((f) => {
    byRep[f.rep_index] = byRep[f.rep_index] || [];
    byRep[f.rep_index].push(f);
  });

  Object.keys(byRep).sort((a, b) => a - b).forEach((repIdx) => {
    const repHeader = document.createElement("h4");
    repHeader.textContent = `Rep ${repIdx}`;
    container.appendChild(repHeader);

    byRep[repIdx].forEach((f) => {
      const card = document.createElement("div");
      card.className = `finding-card ${f.verdict}`;
      card.tabIndex = 0;
      card.setAttribute("role", "button");

      const measurementStr = Object.entries(f.measurement || {})
        .filter(([, v]) => v !== null && v !== undefined)
        .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(3) : v}`)
        .join(", ");

      card.innerHTML = `
        <div class="finding-header">
          <span>${verdictIcon(f.verdict)} ${CRITERION_LABELS[f.criterion_id] || f.criterion_id}</span>
          <span class="finding-badge ${f.verdict}">${verdictLabel(f.verdict)}</span>
        </div>
        <div class="finding-meta">
          t=${f.timestamp} (frame ${f.frame})${measurementStr ? " · " + measurementStr : ""}${f.confidence_note ? " · " + f.confidence_note : ""}
          <span class="jump-hint">▶ jump to this moment</span>
        </div>
        <div class="finding-explanation">${f.explanation || ""}</div>
        <div class="finding-feedback">${f.feedback || ""}</div>
        <div class="finding-citation">Reference pages ${f.document_ref_pages.join(", ")} — ${f.document_sections.join("; ")}</div>
      `;
      const jump = () => {
        video.currentTime = f.timestamp_sec;
        video.play().catch(() => {});
        video.scrollIntoView({ behavior: "smooth", block: "center" });
      };
      card.addEventListener("click", jump);
      card.addEventListener("keypress", (e) => { if (e.key === "Enter") jump(); });
      container.appendChild(card);
    });
  });

  const naList = el("not-assessable-list");
  naList.innerHTML = "";
  (result.not_assessable || []).forEach((item) => {
    const div = document.createElement("div");
    div.className = "na-item";
    div.innerHTML = `<b>${item.name}</b> — ${item.reason} (ref. pages ${item.source.ref_pages.join(", ") || "n/a"})`;
    naList.appendChild(div);
  });

  renderAboutAnalysis(result);
}

function renderAboutAnalysis(result) {
  const about = el("about-analysis");
  const camera = skillCache?.camera_assumptions;
  const policy = skillCache?.insufficient_evidence_policy;
  about.innerHTML = `
    <div class="about-block">
      <h5>Write-up source</h5>
      <p>${PROVIDER_LABELS[result.writeup_method] || result.writeup_method}. The AI model never decides whether a rep meets a standard — that verdict is computed deterministically from the measurements and the reference document's thresholds before any model sees it; the model only writes the plain-language explanation.</p>
    </div>
    ${camera ? `<div class="about-block"><h5>Camera assumptions</h5><p>${camera.required_view}. ${camera.notes || ""}</p></div>` : ""}
    ${policy ? `<div class="about-block"><h5>When evidence is insufficient</h5><p>${policy}</p></div>` : ""}
    <div class="about-block">
      <h5>No mocked steps</h5>
      <p>Every run — including this one — goes through the real upload, pose-tracking, measurement, verdict, and video-rendering pipeline on whatever you uploaded. The only thing that can fall back to a static template is the wording of the explanation text, never the verdict itself, and the badge above tells you when that happened.</p>
    </div>
  `;
}

function reset() {
  el("file-input").value = "";
  el("file-chip").classList.add("hidden");
  showOnly("upload");
}

el("reset-btn").addEventListener("click", reset);
el("error-reset-btn").addEventListener("click", reset);

loadRequirements();
loadSkillAbout();
setupDropzone();
