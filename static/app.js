const predictBtn = document.getElementById("predict-btn");
const input = document.getElementById("review-input");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const result = document.getElementById("result");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const probsEl = document.getElementById("probs");

function setLoading(isLoading) {
  loading.classList.toggle("hidden", !isLoading);
  predictBtn.disabled = isLoading;
}

function setError(message) {
  if (!message) {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
    return;
  }
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function renderProbabilities(probabilities) {
  probsEl.innerHTML = "";
  Object.entries(probabilities).forEach(([name, value]) => {
    const pct = (value * 100).toFixed(2);
    const row = document.createElement("div");
    row.className = "prob";
    row.innerHTML = `
      <span>${name}</span>
      <div class="bar"><div class="fill" style="width: ${pct}%"></div></div>
      <span>${pct}%</span>
    `;
    probsEl.appendChild(row);
  });
}

predictBtn.addEventListener("click", async () => {
  const text = input.value.trim();

  result.classList.add("hidden");
  setError("");

  if (!text) {
    setError("Please enter some text.");
    return;
  }

  setLoading(true);

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || "Request failed");
    }

    labelEl.textContent = data.label;
    confidenceEl.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    renderProbabilities(data.probabilities);
    result.classList.remove("hidden");
  } catch (err) {
    setError(err.message || "Something went wrong.");
  } finally {
    setLoading(false);
  }
});
