const $ = (id) => document.getElementById(id);

const CLASS_COLORS = {
  benign: "#35D07F",
  malware: "#FF5C5C",
  phishing: "#FFAD42",
  defacement: "#A78BFA",
  spam: "#5EA7FF",
};

function getClassColor(label) {
  return CLASS_COLORS[label] || "#607D8B";
}

function renderGauge(riskScore, color) {
  const radius = 62;
  const circumference = 2 * Math.PI * radius;

  const gauge = $("gauge-fill");

  gauge.style.stroke = color;

  const offset =
    circumference - (riskScore * circumference);

  gauge.style.strokeDasharray = circumference;
  gauge.style.strokeDashoffset = offset;

  $("gauge-text").textContent =
    `${Math.round(riskScore * 100)}%`;
}

function renderBadge(riskLevel) {
  const badge = $("risk-badge");

  badge.innerHTML = "";

  const span = document.createElement("span");

  span.textContent = riskLevel.label;

  span.style.background = `${riskLevel.color}18`;
  span.style.color = riskLevel.color;
  span.style.border = `1px solid ${riskLevel.color}35`;

  badge.appendChild(span);

  $("risk-mini-label").textContent =
    riskLevel.label;
}

function renderProbabilities(probabilities) {
  const container = $("prob-bars");

  container.innerHTML = "";

  const sorted = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1]);

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

    fill.style.width = `${percentage}%`;
    fill.style.background = getClassColor(label);

    track.appendChild(fill);

    const percentageElement = document.createElement("span");
    percentageElement.className = "prob-value";
    percentageElement.textContent = `${percentage}%`;

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

  $("url").textContent = result.url;

  $("prediction").textContent =
    result.prediction;

  $("confidence").textContent =
    `${Math.round(result.confidence * 100)}%`;

  renderGauge(
    result.riskScore,
    result.riskLevel.color
  );

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
    (result) => {
      if (chrome.runtime.lastError) {
        showError(
          chrome.runtime.lastError.message
        );
        return;
      }

      if (result) {
        showResult(result);
        return;
      }

      chrome.runtime.sendMessage(
        { type: "analyze" },
        (analysisResult) => {
          if (chrome.runtime.lastError) {
            showError(
              chrome.runtime.lastError.message
            );
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
  () => {
    setLoading();

    chrome.runtime.sendMessage(
      { type: "analyze" },
      (result) => {
        if (chrome.runtime.lastError) {
          showError(
            chrome.runtime.lastError.message
          );
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