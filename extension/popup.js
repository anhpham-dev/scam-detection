const $ = id => document.getElementById(id);

const colors = {
  benign: "#34d399",
  malware: "#f87171",
  phishing: "#fb923c",
  defacement: "#c084fc",
  spam: "#60a5fa"
};

const radius = 52;
const circumference = 2 * Math.PI * radius;

function colorFor(label) {
  return colors[label] || "#71717a";
}

function prettyLabel(label) {
  if (!label) return "—";
  const text = String(label);
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function rgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);

  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function renderUrl(raw) {
  const el = $("url");
  el.replaceChildren();

  try {
    const parsed = new URL(raw);
    const host = document.createElement("span");
    host.className = "url-host";
    host.textContent = parsed.hostname;

    const rest = document.createElement("span");
    rest.textContent =
      parsed.pathname + parsed.search;

    el.append(host, rest);
  } catch {
    el.textContent = raw;
  }
}

function renderGauge(score, color) {
  const gauge = $("gauge-fill");

  gauge.style.strokeDasharray = circumference;
  gauge.style.strokeDashoffset =
    circumference * (1 - score);

  $("grad-stop-1").setAttribute("stop-color", color);
  $("grad-stop-2").setAttribute("stop-color", color);
  $("grad-stop-2").setAttribute("stop-opacity", "0.45");

  $("gauge-text").textContent =
    Math.round(score * 100) + "%";
}

function renderProbabilities(probabilities) {
  const container = $("prob-bars");
  container.replaceChildren();

  Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, value]) => {
      const percent = Math.round(value * 100);
      const color = colorFor(label);

      const row = document.createElement("div");
      row.className = "prob-row";

      const name = document.createElement("span");
      name.className = "prob-label";
      name.textContent = prettyLabel(label);

      const track = document.createElement("div");
      track.className = "prob-bar-track";

      const fill = document.createElement("div");
      fill.className = "prob-bar-fill";
      fill.style.width = percent + "%";
      fill.style.background = color;

      track.appendChild(fill);

      const num = document.createElement("span");
      num.className = "prob-value";
      num.textContent = percent + "%";

      row.append(name, track, num);
      container.appendChild(row);
    });
}

function renderResult(data) {
  $("loading").classList.add("hidden");
  $("error").classList.add("hidden");
  $("result").classList.remove("hidden");
  setBusy(false);

  renderUrl(data.url);
  $("prediction").textContent =
    prettyLabel(data.prediction);
  $("confidence").textContent =
    Math.round((data.confidence || 0) * 100) + "%";

  renderGauge(
    data.riskScore,
    data.riskLevel.color
  );

  const badge = document.createElement("span");

  badge.textContent = data.riskLevel.label;
  badge.style.color = data.riskLevel.color;
  badge.style.background =
    rgba(data.riskLevel.color, 0.1);

  $("risk-badge").replaceChildren(badge);
  $("risk-mini-label").textContent =
    data.riskLevel.label;

  if (
    data.probabilities &&
    Object.keys(data.probabilities).length
  ) {
    renderProbabilities(data.probabilities);
  } else {
    $("prob-bars").replaceChildren();
  }
}

function showError(message, apiUrl) {
  $("loading").classList.add("hidden");
  $("result").classList.add("hidden");
  $("error").classList.remove("hidden");
  setBusy(false);

  $("error-msg").textContent = message;

  if (apiUrl) {
    $("error-api-url").textContent =
      apiUrl.replace(/^https?:\/\//, "");
  }
}

function loading() {
  $("loading").classList.remove("hidden");
  $("result").classList.add("hidden");
  $("error").classList.add("hidden");
}

function setBusy(busy) {
  const app = document.querySelector(".app");
  const btn = $("reanalyze");

  app.classList.toggle("busy", busy);
  btn.disabled = busy;
  $("reanalyze-icon").classList.toggle(
    "spinning",
    busy
  );
}

function analyze(soft) {
  setBusy(true);

  if (!soft) loading();

  chrome.runtime.sendMessage(
    { type: "analyze" },
    result => {
      setBusy(false);

      if (chrome.runtime.lastError) {
        showError(chrome.runtime.lastError.message);
        return;
      }

      if (result && !result.error) {
        renderResult(result);
      } else if (result) {
        showError(
          result.error || "Analysis failed.",
          result.apiUrl
        );
      } else {
        showError("Analysis failed.");
      }
    }
  );
}

function load() {
  chrome.runtime.sendMessage(
    { type: "getResult" },
    result => {
      if (result && !result.error) {
        renderResult(result);
      } else {
        analyze(false);
      }
    }
  );
}

function checkStatus() {
  chrome.runtime.sendMessage(
    { type: "ping" },
    state => {
      const online = state && state.online;

      $("status-text").textContent = online
        ? "Connected"
        : "Offline";

      $("status-dot").classList.toggle(
        "offline",
        !online
      );
      $("status").title = online
        ? `Local API at ${state.url}`
        : `No response from ${state ? state.url : "the local API"}`;
    }
  );
}

$("reanalyze").addEventListener(
  "click",
  () => analyze(true)
);

$("open-settings").addEventListener(
  "click",
  () => chrome.runtime.openOptionsPage()
);

$("version").textContent =
  "v" + chrome.runtime.getManifest().version;

checkStatus();
load();
