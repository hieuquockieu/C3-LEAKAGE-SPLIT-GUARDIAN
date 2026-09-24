// app.js — SplitGuardian Interactive Visualizer Logic

const mockReport = {
  primary_metrics: {
    perceptual_hash: { precision: 0.4323, recall: 0.1412, f1: 0.2128, runtime_sec: 0.24 },
    deep_embeddings: { precision: 0.4246, recall: 0.4881, f1: 0.4541, runtime_sec: 0.91 },
    splitguardian_hybrid: { precision: 0.8855, recall: 1.0000, f1: 0.9393, runtime_sec: 1.50 }
  },
  leak_pairs_demo: [
    {
      type: "temporal_burst",
      type_label: "TYPE 1: TEMPORAL BURST LEAK",
      sampleA: "img_0004 (Scene 00)",
      camA: "cam_front_left (t=0.00s)",
      sampleB: "img_0005 (Scene 00)",
      camB: "cam_front_left (t=0.15s)",
      sim: "0.985",
      hamming: "2 bits",
      dTime: "0.15s",
      colorA: "#2563eb",
      colorB: "#2563eb"
    },
    {
      type: "multi_camera_cross_view",
      type_label: "TYPE 2: MULTI-CAMERA VIEW LEAK",
      sampleA: "img_0004 (Scene 00)",
      camA: "cam_front_left (angle -1.2°)",
      sampleB: "img_0008 (Scene 00)",
      camB: "cam_front_right (angle +1.2°)",
      sim: "0.952",
      hamming: "7 bits",
      dTime: "0.00s (Synchronous)",
      colorA: "#2563eb",
      colorB: "#1d4ed8"
    },
    {
      type: "temporal_burst",
      type_label: "TYPE 1: TEMPORAL BURST LEAK",
      sampleA: "img_0024 (Scene 01)",
      camA: "cam_front_left (Truck, t=30.0s)",
      sampleB: "img_0025 (Scene 01)",
      camB: "cam_front_left (Truck, t=30.15s)",
      sim: "0.991",
      hamming: "3 bits",
      dTime: "0.15s",
      colorA: "#d97706",
      colorB: "#d97706"
    },
    {
      type: "multi_camera_cross_view",
      type_label: "TYPE 2: MULTI-CAMERA VIEW LEAK",
      sampleA: "img_0048 (Scene 02)",
      camA: "cam_front_left (Motorcycle)",
      sampleB: "img_0052 (Scene 02)",
      camB: "cam_front_right (Motorcycle)",
      sim: "0.944",
      hamming: "8 bits",
      dTime: "0.00s (Simultaneous)",
      colorA: "#059669",
      colorB: "#047857"
    }
  ]
};

let currentPairIdx = 0;
let activeFilter = "all";

document.addEventListener("DOMContentLoaded", async () => {
  let reportData = mockReport;
  try {
    const res = await fetch("data_report.json");
    if (res.ok) {
      reportData = await res.json();
    }
  } catch (e) {
    console.log("Using cached dashboard report data");
  }

  initNavigation();
  populateTables(reportData);
  initDetectorChart(reportData);
  initThresholdChart();
  initScalingChart();
  initInspector();
});

function initNavigation() {
  document.querySelectorAll(".nav-item").forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      document.querySelectorAll(".nav-item").forEach(l => l.classList.remove("active"));
      link.classList.add("active");
      const targetId = link.getAttribute("data-target");
      const el = document.getElementById(targetId);
      if (el) {
        el.scrollIntoView({ behavior: "smooth" });
      }
    });
  });
}

function populateTables(data) {
  const p = data.primary_metrics;
  const tbody = document.getElementById("detectorTableBody");
  if (!tbody) return;

  const rows = [
    { name: "Baseline: Perceptual Hash (pHash/dHash)", ...p.perceptual_hash },
    { name: "Deep Visual Embeddings (CNN)", ...p.deep_embeddings },
    { name: "⭐ SplitGuardian (Hybrid Multimodal Graph)", ...p.splitguardian_hybrid, highlight: true }
  ];

  tbody.innerHTML = rows.map(r => `
    <tr style="${r.highlight ? 'background: rgba(99, 102, 241, 0.12); font-weight: 600;' : ''}">
      <td>${r.name}</td>
      <td>${(r.precision * 100).toFixed(1)}%</td>
      <td>${(r.recall * 100).toFixed(1)}%</td>
      <td><span class="badge ${r.f1 > 0.8 ? 'badge-success' : 'badge-warning'}">${r.f1.toFixed(4)}</span></td>
      <td><code>${r.runtime_sec}s</code></td>
    </tr>
  `).join('');
}

function initDetectorChart(data) {
  const ctx = document.getElementById("detectorMetricsChart");
  if (!ctx) return;

  const p = data.primary_metrics;

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Perceptual Hash", "Deep Embeddings", "SplitGuardian (Hybrid)"],
      datasets: [
        {
          label: "Precision",
          data: [p.perceptual_hash.precision, p.deep_embeddings.precision, p.splitguardian_hybrid.precision],
          backgroundColor: "rgba(56, 189, 248, 0.75)",
          borderColor: "#38bdf8",
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: "Recall",
          data: [p.perceptual_hash.recall, p.deep_embeddings.recall, p.splitguardian_hybrid.recall],
          backgroundColor: "rgba(16, 185, 129, 0.75)",
          borderColor: "#10b981",
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: "F1 Score (Primary)",
          data: [p.perceptual_hash.f1, p.deep_embeddings.f1, p.splitguardian_hybrid.f1],
          backgroundColor: "rgba(99, 102, 241, 0.85)",
          borderColor: "#6366f1",
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#9ca3af", font: { family: "Plus Jakarta Sans" } } }
      },
      scales: {
        y: {
          min: 0,
          max: 1.05,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#9ca3af" }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#e5e7eb" }
        }
      }
    }
  });
}

function initThresholdChart() {
  const ctx = document.getElementById("thresholdChart");
  if (!ctx) return;

  new Chart(ctx, {
    type: "line",
    data: {
      labels: ["0.35", "0.45", "0.55", "0.65", "0.75", "0.85"],
      datasets: [
        {
          label: "Precision",
          data: [0.50, 0.65, 0.85, 0.89, 0.89, 0.89],
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.1)",
          fill: true,
          tension: 0.3
        },
        {
          label: "Recall",
          data: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          borderColor: "#10b981",
          borderDash: [5, 5],
          tension: 0.1
        },
        {
          label: "F1 Score",
          data: [0.67, 0.79, 0.92, 0.94, 0.94, 0.94],
          borderColor: "#6366f1",
          borderWidth: 3,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#9ca3af" } }
      },
      scales: {
        y: { min: 0.4, max: 1.05, grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } },
        x: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } }
      }
    }
  });
}

function initScalingChart() {
  const ctx = document.getElementById("scalingChart");
  if (!ctx) return;

  new Chart(ctx, {
    type: "line",
    data: {
      labels: ["50", "100", "200", "336"],
      datasets: [
        {
          label: "Throughput (imgs/sec)",
          data: [388.3, 365.7, 303.8, 249.4],
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.1)",
          yAxisID: "yThroughput",
          tension: 0.3
        },
        {
          label: "Total Runtime (s)",
          data: [0.129, 0.273, 0.658, 1.347],
          borderColor: "#a855f7",
          borderDash: [4, 4],
          yAxisID: "yTime",
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#9ca3af" } }
      },
      scales: {
        yThroughput: {
          type: "linear",
          position: "left",
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#f59e0b" },
          title: { display: true, text: "Throughput (FPS)", color: "#f59e0b" }
        },
        yTime: {
          type: "linear",
          position: "right",
          grid: { display: false },
          ticks: { color: "#a855f7" },
          title: { display: true, text: "Runtime (sec)", color: "#a855f7" }
        },
        x: { grid: { display: false }, ticks: { color: "#9ca3af" }, title: { display: true, text: "Sample Count (N)", color: "#9ca3af" } }
      }
    }
  });
}

function initInspector() {
  const pairs = mockReport.leak_pairs_demo;
  renderCurrentPair(pairs[0]);

  document.getElementById("prevPairBtn")?.addEventListener("click", () => {
    currentPairIdx = (currentPairIdx - 1 + pairs.length) % pairs.length;
    renderCurrentPair(pairs[currentPairIdx]);
  });

  document.getElementById("nextPairBtn")?.addEventListener("click", () => {
    currentPairIdx = (currentPairIdx + 1) % pairs.length;
    renderCurrentPair(pairs[currentPairIdx]);
  });

  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeFilter = btn.getAttribute("data-filter");
      const filtered = activeFilter === "all" ? pairs : pairs.filter(p => p.type === activeFilter);
      if (filtered.length > 0) {
        currentPairIdx = 0;
        renderCurrentPair(filtered[0], filtered.length);
      }
    });
  });
}

function renderCurrentPair(item, total = 4) {
  if (!item) return;

  document.getElementById("pairCounter").innerText = `Showing Leakage Pair ${currentPairIdx + 1} of ${total}`;
  document.getElementById("relationBadge").innerText = item.type_label;
  document.getElementById("sampleAMeta").innerText = `Sample A: ${item.sampleA}`;
  document.getElementById("sampleACam").innerText = item.camA;
  document.getElementById("sampleBMeta").innerText = `Sample B: ${item.sampleB}`;
  document.getElementById("sampleBCam").innerText = item.camB;

  document.getElementById("pairSim").innerText = item.sim;
  document.getElementById("pairHamming").innerText = item.hamming;
  document.getElementById("pairTime").innerText = item.dTime;

  const va = document.getElementById("vehicleA");
  const vb = document.getElementById("vehicleB");
  if (va) va.style.background = item.colorA;
  if (vb) vb.style.background = item.colorB;
}
