const API_URL = "http://localhost:8000/predict";

async function analyzeUrl(url) {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url }),
    });

    if (!response.ok) {
      throw new Error("API returned " + response.status);
    }

    const data = await response.json();
    const riskScore = 1 - (data.probabilities.benign || 0);

    return {
      url: url,
      prediction: data.prediction,
      confidence: data.confidence,
      probabilities: data.probabilities,
      riskScore: Math.round(riskScore * 100) / 100,
      riskLevel: getRiskLevel(riskScore),
      timestamp: Date.now(),
    };
  } catch (err) {
    return {
      url: url,
      prediction: "unknown",
      confidence: 0,
      probabilities: {},
      riskScore: 0,
      riskLevel: { label: "Error", color: "#71717a", badge: "?" },
      error: err.message,
      timestamp: Date.now(),
    };
  }
}

function getRiskLevel(score) {
  if (score < 0.1)
    return { label: "Safe", color: "#34d399", badge: "Y" };
  if (score < 0.4)
    return { label: "Low Risk", color: "#fb923c", badge: "!" };
  if (score < 0.7)
    return { label: "Medium Risk", color: "#f97316", badge: "!!" };
  return { label: "High Risk", color: "#f87171", badge: "X" };
}

async function setBadge(tabId, riskLevel) {
  await chrome.action.setBadgeText({
    text: riskLevel.badge,
    tabId: tabId,
  });
  await chrome.action.setBadgeBackgroundColor({
    color: riskLevel.color,
    tabId: tabId,
  });
}

async function processTab(tabId, url) {
  if (
    !url ||
    url.startsWith("chrome://") ||
    url.startsWith("chrome-extension://")
  ) {
    return;
  }

  const result = await analyzeUrl(url);

  await chrome.storage.local.set({ ["tab_" + tabId]: result });
  await setBadge(tabId, result.riskLevel);

  if (result.riskScore >= 0.7) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ["content.js"],
      });
    } catch (_) {}
  }

  return result;
}

chrome.tabs.onUpdated.addListener(function (tabId, changeInfo, tab) {
  if (changeInfo.status === "complete" && tab.url) {
    processTab(tabId, tab.url);
  }
});

chrome.runtime.onMessage.addListener(function (msg, sender, sendResponse) {
  if (msg.type === "analyze") {
    chrome.tabs.query(
      { active: true, currentWindow: true },
      async function (tabs) {
        const tab = tabs[0];
        if (tab) {
          const result = await processTab(
            tab.id,
            msg.url || tab.url
          );
          sendResponse(result);
        } else {
          sendResponse(null);
        }
      }
    );
    return true;
  }

  if (msg.type === "getResult") {
    chrome.tabs.query(
      { active: true, currentWindow: true },
      async function (tabs) {
        const tab = tabs[0];
        if (tab) {
          const data = await chrome.storage.local.get(
            "tab_" + tab.id
          );
          sendResponse(data["tab_" + tab.id] || null);
        } else {
          sendResponse(null);
        }
      }
    );
    return true;
  }

  if (msg.type === "contentLoaded") {
    const tabId = sender.tab ? sender.tab.id : null;

    if (!tabId) {
      return;
    }

    chrome.storage.local
      .get("tab_" + tabId)
      .then(function (data) {
        const result = data["tab_" + tabId];

        if (result && result.riskScore >= 0.7) {
          chrome.tabs.sendMessage(tabId, {
            type: "showWarning",
            riskLevel: result.riskLevel.label,
            prediction: result.prediction,
          });
        }
      });

    return;
  }
});
