const $ = (id) => document.getElementById(id);

const CLASS_COLORS = {
  benign: "#34d399",
  malware: "#f87171",
  phishing: "#fb923c",
  defacement: "#c084fc",
  spam: "#60a5fa",
};

const GAUGE_RADIUS = 52;
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * GAUGE_RADIUS;

function getClassColor(label) {
  return CLASS_COLORS[label] || "#71717a";
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function lightenColor(hex, amount) {
  amount = amount || 0.35;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const nr = Math.round(r + (255 - r) * amount);
  const ng = Math.round(g + (255 - g) * amount);
  const nb = Math.round(b + (255 - b) * amount);
  return (
    "#" +
    nr.toString(16).padStart(2, "0") +
    ng.toString(16).padStart(2, "0") +
    nb.toString(16).padStart(2, "0")
  );
}

function renderGauge(riskScore, color) {
  const gauge = $("gauge-fill");

  // Update SVG gradient stops
  const stop1 = $("grad-stop-1");
  const stop2 = $("grad-stop-2");
  if (stop1 && stop2) {
    stop1.setAttribute("stop-color", color);
    stop2.setAttribute("stop-color", lightenColor(color));
  }

  // Update glow filter color
  gauge.style.filter =
    "drop-shadow(0 0 5px " + hexToRgba(color, 0.35) + ")";

  const offset = GAUGE_CIRCUMFERENCE * (1 - riskScore);

  gauge.style.strokeDasharray = GAUGE_CIRCUMFERENCE;
  gauge.style.strokeDashoffset = offset;

  $("gauge-text").textContent =
    Math.round(riskScore * 100) + "%";
}

function renderBadge(riskLevel) {
  const badge = $("risk-badge");

  badge.innerHTML = "";

  const span = document.createElement("span");

  span.textContent = riskLevel.label;

  span.style.background = hexToRgba(riskLevel.color, 0.1);
  span.style.color = riskLevel.color;
  span.style.border =
    "1px solid " + hexToRgba(riskLevel.color, 0.2);

  badge.appendChild(span);

  $("risk-mini-label").textContent = riskLevel.label;
}

function renderProbabilities(probabilities) {
  const container = $("prob-bars");

  container.innerHTML = "";

  const sorted = Object.entries(probabilities)
    .sort(function (a, b) { return b[1] - a[1]; });

  for (const [label, value] of sorted) {
    const row = document.createElement("div");
    row.className = "prob-row";

    const labelElement = document.createElement("span");
    labelElement.className = "prob-label";
    labelElement.textContent = label;

    const track = document.createElement("div");
    track.className = "prob-bar-track";

    const fill = document.createElement("div");
    fill.className = "prob-bar-fill";

    const percentage = Math.round(value * 100);

    fill.style.width = percentage + "%";
    fill.style.background = getClassColor(label);

    track.appendChild(fill);

    const percentageElement = document.createElement("span");
    percentageElement.className = "prob-value";
    percentageElement.textContent = percentage + "%";

    row.appendChild(labelElement);
    row.appendChild(track);
    row.appendChild(percentageElement);

    container.appendChild(row);
  }
}

function showError(message) {
  $("loading").classList.add("hidden");
  $("result").classList.add("hidden");
  $("error").classList.remove("hidden");

  $("error-msg").textContent = message;
}

function showResult(result) {
  $("loading").classList.add("hidden");
  $("error").classList.add("hidden");
  $("result").classList.remove("hidden");

  // Re-trigger entrance animation
  const resultEl = $("result");
  resultEl.style.animation = "none";
  resultEl.offsetHeight;
  resultEl.style.animation = "";

  $("url").textContent = result.url;

  $("prediction").textContent = result.prediction;

  $("confidence").textContent =
    Math.round(result.confidence * 100) + "%";

  renderGauge(result.riskScore, result.riskLevel.color);
  renderBadge(result.riskLevel);

  if (
    result.probabilities &&
    Object.keys(result.probabilities).length
  ) {
    renderProbabilities(result.probabilities);
  }
}

function setLoading() {
  $("loading").classList.remove("hidden");
  $("result").classList.add("hidden");
  $("error").classList.add("hidden");
}

function load() {
  setLoading();

  chrome.runtime.sendMessage(
    { type: "getResult" },
    function (result) {
      if (chrome.runtime.lastError) {
        showError(chrome.runtime.lastError.message);
        return;
      }

      if (result) {
        showResult(result);
        return;
      }

      chrome.runtime.sendMessage(
        { type: "analyze" },
        function (analysisResult) {
          if (chrome.runtime.lastError) {
            showError(chrome.runtime.lastError.message);
            return;
          }

          if (analysisResult) {
            showResult(analysisResult);
          } else {
            showError(
              "No result. Is the API server running?"
            );
          }
        }
      );
    }
  );
}

$("reanalyze").addEventListener(
  "click",
  function () {
    setLoading();

    chrome.runtime.sendMessage(
      { type: "analyze" },
      function (result) {
        if (chrome.runtime.lastError) {
          showError(chrome.runtime.lastError.message);
          return;
        }

        if (result) {
          showResult(result);
        } else {
          showError(
            "Analysis failed. Check the API server."
          );
        }
      }
    );
  }
);

load();
