const DEFAULT_SETTINGS = {
  apiBaseUrl: "http://localhost:8000",
  highRiskThreshold: 0.7,
  showHighRiskBanner: true,
  autoAnalyze: true
};

const form = document.getElementById("settings-form");
const apiBaseUrl = document.getElementById("api-base-url");
const threshold = document.getElementById("high-risk-threshold");
const thresholdValue = document.getElementById("threshold-value");
const showBanner = document.getElementById("show-high-risk-banner");
const autoAnalyze = document.getElementById("auto-analyze");
const urlError = document.getElementById("url-error");
const saveStatus = document.getElementById("save-status");
const testBtn = document.getElementById("test-connection");
const connectionStatus =
  document.getElementById("connection-status");
const resetBtn = document.getElementById("reset-defaults");

function updateThresholdLabel() {
  thresholdValue.value = `${Math.round(Number(threshold.value) * 100)}%`;
  thresholdValue.textContent = thresholdValue.value;
}

function isLocalApiUrl(value) {
  try {
    const parsed = new URL(value);
    return ["http:", "https:"].includes(parsed.protocol) &&
      ["localhost", "127.0.0.1"].includes(parsed.hostname);
  } catch {
    return false;
  }
}

function applySettings(settings) {
  apiBaseUrl.value = settings.apiBaseUrl;
  threshold.value = settings.highRiskThreshold;
  showBanner.checked = settings.showHighRiskBanner;
  autoAnalyze.checked = settings.autoAnalyze;
  updateThresholdLabel();
  setConnectionStatus("");
}

async function loadSettings() {
  const settings = await chrome.storage.local.get(DEFAULT_SETTINGS);
  applySettings(settings);
}

threshold.addEventListener("input", updateThresholdLabel);

function setConnectionStatus(text, state) {
  connectionStatus.textContent = text;
  connectionStatus.classList.remove("ok", "fail");
  if (state) connectionStatus.classList.add(state);
}

async function testConnection() {
  const value = apiBaseUrl.value.trim().replace(/\/$/, "");

  urlError.textContent = "";
  if (!isLocalApiUrl(value)) {
    setConnectionStatus("Fix the URL first, then test again.", "fail");
    apiBaseUrl.focus();
    return;
  }

  testBtn.disabled = true;
  setConnectionStatus("Checking…");

  try {
    const result = await chrome.runtime.sendMessage({
      type: "testConnection",
      baseUrl: value
    });

    setConnectionStatus(
      result && result.online
        ? `Server is up at ${result.url}`
        : `No response from ${value}`,
      result && result.online ? "ok" : "fail"
    );
  } catch {
    setConnectionStatus("Test failed — try again.", "fail");
  } finally {
    testBtn.disabled = false;
  }
}

testBtn.addEventListener("click", testConnection);

resetBtn.addEventListener("click", async () => {
  await chrome.storage.local.set({ ...DEFAULT_SETTINGS });
  applySettings(DEFAULT_SETTINGS);
  saveStatus.textContent = "Defaults restored";
  setTimeout(() => { saveStatus.textContent = ""; }, 2000);
});

form.addEventListener("submit", async event => {
  event.preventDefault();
  urlError.textContent = "";
  saveStatus.textContent = "";

  const value = apiBaseUrl.value.trim().replace(/\/$/, "");
  if (!isLocalApiUrl(value)) {
    urlError.textContent = "Use a localhost or 127.0.0.1 HTTP(S) URL.";
    apiBaseUrl.focus();
    return;
  }

  await chrome.storage.local.set({
    apiBaseUrl: value,
    highRiskThreshold: Number(threshold.value),
    showHighRiskBanner: showBanner.checked,
    autoAnalyze: autoAnalyze.checked
  });

  saveStatus.textContent = "Saved";
  setTimeout(() => { saveStatus.textContent = ""; }, 2000);
});

loadSettings();
