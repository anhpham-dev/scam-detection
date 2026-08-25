const ID = "scam-buzzer-warning";

function showWarning(level, prediction) {
  if (document.getElementById(ID)) return;

  const box = document.createElement("div");
  box.id = ID;
  box.setAttribute("role", "alert");

  const text = document.createElement("span");
  text.className = `${ID}-text`;

  const strong = document.createElement("strong");
  strong.textContent = "Warning: ";
  strong.style.color = "#f87171";

  text.appendChild(strong);
  text.append(
    "This page was flagged as ",
    makeBold(level),
    ` (${prediction}). Be careful with links and forms here.`
  );

  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Dismiss";

  close.addEventListener("click", () => box.remove());

  box.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 2147483647;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    background: #18181b;
    color: #ededf0;
    border-bottom: 1px solid rgba(248, 113, 113, .55);
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 13px;
    line-height: 1.5;
  `;

  close.style.cssText = `
    flex: 0 0 auto;
    padding: 4px 12px;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    background: transparent;
    color: #d4d4d8;
    font: inherit;
    font-size: 12px;
    cursor: pointer;
  `;

  close.addEventListener("mouseover", () => {
    close.style.background = "#27272a";
    close.style.color = "#fff";
  });

  close.addEventListener("mouseout", () => {
    close.style.background = "transparent";
    close.style.color = "#d4d4d8";
  });

  box.append(text, close);
  document.body.prepend(box);

  if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
    box.animate(
      [
        { transform: "translateY(-100%)" },
        { transform: "translateY(0)" }
      ],
      { duration: 220, easing: "ease-out" }
    );
  }
}

function makeBold(value) {
  const el = document.createElement("strong");
  el.textContent = value || "";
  return el;
}

chrome.runtime.onMessage.addListener(message => {
  if (message.type === "showWarning") {
    showWarning(
      message.riskLevel,
      message.prediction
    );
  }
});
