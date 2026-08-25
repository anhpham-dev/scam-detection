(() => {
  const BANNER_ID = "scam-detector-warning";

  function showWarning(riskLevel, prediction) {
    if (document.getElementById(BANNER_ID)) return;

    const banner = document.createElement("div");
    banner.id = BANNER_ID;
    banner.style.cssText =
      "position:fixed;top:0;left:0;right:0;z-index:2147483647;" +
      "background:#18181b;color:#ededf0;padding:10px 16px;" +
      "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,sans-serif;" +
      "font-size:13px;display:flex;justify-content:space-between;align-items:center;" +
      "border-bottom:1px solid #f87171;";

    banner.innerHTML =
      '<span><strong style="color:#f87171">Warning:</strong> This page is flagged as <strong>' +
      riskLevel +
      '</strong> (' +
      prediction +
      ')</span>' +
      '<button id="' +
      BANNER_ID +
      '-close" style="' +
      'background:transparent;border:1px solid #3f3f46;color:#a1a1aa;' +
      'padding:3px 10px;border-radius:5px;cursor:pointer;font-size:11px;' +
      '">Dismiss</button>';

    document.body.prepend(banner);

    document
      .getElementById(BANNER_ID + "-close")
      .addEventListener("click", function () {
        banner.remove();
      });
  }

  chrome.runtime.sendMessage({
    type: "contentLoaded",
    tabId: chrome.runtime.id ? null : undefined,
  });

  chrome.runtime.onMessage.addListener(function (msg) {
    if (msg.type === "showWarning") {
      showWarning(msg.riskLevel, msg.prediction);
    }
  });
})();
