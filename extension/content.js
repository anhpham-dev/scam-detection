(() => {
  const BANNER_ID = "scam-detector-warning";

  function showWarning(riskLevel, prediction) {
    if (document.getElementById(BANNER_ID)) return;

    const banner = document.createElement("div");
    banner.id = BANNER_ID;
    banner.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 2147483647;
      background: linear-gradient(135deg, #b71c1c, #c62828);
      color: #fff;
      padding: 10px 16px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    `;

    banner.innerHTML = `
      <span>⚠ <strong>Warning:</strong> This page is flagged as <strong>${riskLevel}</strong> (${prediction})</span>
      <button id="${BANNER_ID}-close" style="
        background: transparent;
        border: 1px solid rgba(255,255,255,0.4);
        color: #fff;
        padding: 2px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
      ">Dismiss</button>
    `;

    document.body.prepend(banner);

    document.getElementById(`${BANNER_ID}-close`).addEventListener("click", () => {
      banner.remove();
    });
  }

  chrome.runtime.sendMessage({
    type: "contentLoaded",
    tabId: chrome.runtime.id ? null : undefined,
  });

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "showWarning") {
      showWarning(msg.riskLevel, msg.prediction);
    }
  });
})();