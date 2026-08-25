const DEFAULT_SETTINGS = {
  apiBaseUrl: "http://localhost:8000",
  highRiskThreshold: 0.7,
  showHighRiskBanner: true,
  autoAnalyze: true
};

async function getSettings() {
  const stored = await chrome.storage.local.get(DEFAULT_SETTINGS);
  return {
    ...DEFAULT_SETTINGS,
    ...stored,
    highRiskThreshold: Math.min(
      1,
      Math.max(0, Number(stored.highRiskThreshold))
    )
  };
}

async function pingApi(baseOverride) {
  const settings = await getSettings();
  const base = (baseOverride || settings.apiBaseUrl).replace(/\/$/, "");

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4000);

    await fetch(`${base}/health`, { signal: controller.signal });
    clearTimeout(timer);

    return { online: true, url: base };
  } catch {
    return { online: false, url: base };
  }
}

async function analyzeUrl(url) {
  const settings = await getSettings();

  try {
    const response = await fetch(
      `${settings.apiBaseUrl.replace(/\/$/, "")}/predict`,
      {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
        },
        body: JSON.stringify({ url })
      }
    );

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    const riskScore =
      1 - (data.probabilities.benign || 0);

    return {
      url,
      prediction: data.prediction,
      confidence: data.confidence,
      probabilities: data.probabilities,
      riskScore: Math.round(riskScore * 100) / 100,
      riskLevel: getRiskLevel(riskScore, settings.highRiskThreshold),
      time: Date.now()
    };
  } catch (error) {
    return {
      url,
      prediction: "unknown",
      confidence: 0,
      probabilities: {},
      riskScore: 0,
      riskLevel: {
        label: "Error",
        color: "#71717a",
        badge: "?"
      },
      apiUrl: settings.apiBaseUrl,
      error: error.message
    };
  }
}

function getRiskLevel(score, highRiskThreshold) {
  if (score < 0.1) {
    return {
      label: "Safe",
      color: "#34d399",
      badge: "OK"
    };
  }

  if (score < 0.4) {
    return {
      label: "Low Risk",
      color: "#fb923c",
      badge: "!"
    };
  }

  if (score < highRiskThreshold) {
    return {
      label: "Medium Risk",
      color: "#f97316",
      badge: "!!"
    };
  }

  return {
    label: "High Risk",
    color: "#f87171",
    badge: "!"
  };
}

async function checkTab(tabId, url, settings) {
  if (!url) return;

  if (
    url.startsWith("chrome://") ||
    url.startsWith("chrome-extension://")
  ) {
    return;
  }

  const result = await analyzeUrl(url);

  await chrome.storage.local.set({
    ["tab_" + tabId]: result
  });

  await chrome.action.setBadgeText({
    tabId,
    text: result.riskLevel.badge
  });

  await chrome.action.setBadgeBackgroundColor({
    tabId,
    color: result.riskLevel.color
  });

  if (settings.showHighRiskBanner && result.riskScore >= settings.highRiskThreshold) {
    try {
      await chrome.tabs.sendMessage(tabId, {
        type: "showWarning",
        riskLevel: result.riskLevel.label,
        prediction: result.prediction
      });
    } catch {}
  }

  return result;
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url) {
    getSettings().then(settings => {
      if (settings.autoAnalyze) checkTab(tabId, tab.url, settings);
    });
  }
});

chrome.runtime.onMessage.addListener(
  (message, sender, sendResponse) => {

    if (message.type === "analyze") {
      chrome.tabs.query(
        { active: true, currentWindow: true },
        async tabs => {
          const tab = tabs[0];

          if (!tab) {
            sendResponse(null);
            return;
          }

          const settings = await getSettings();
          const result = await checkTab(tab.id, message.url || tab.url, settings);

          sendResponse(result);
        }
      );

      return true;
    }

    if (message.type === "ping") {
      pingApi().then(sendResponse);
      return true;
    }

    if (message.type === "testConnection") {
      pingApi(message.baseUrl).then(sendResponse);
      return true;
    }

    if (message.type === "getResult") {
      chrome.tabs.query(
        { active: true, currentWindow: true },
        async tabs => {
          const tab = tabs[0];

          if (!tab) {
            sendResponse(null);
            return;
          }

          const data = await chrome.storage.local.get(
            "tab_" + tab.id
          );

          sendResponse(data["tab_" + tab.id] || null);
        }
      );

      return true;
    }
  }
);