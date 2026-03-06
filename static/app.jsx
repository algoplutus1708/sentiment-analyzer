const { useMemo, useState } = React;
const API_BASE =
  window.location.origin.includes("127.0.0.1:8000") ||
  window.location.origin.includes("localhost:8000")
    ? ""
    : "http://127.0.0.1:8000";

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function ProbabilityBars({ probabilities }) {
  const entries = useMemo(() => Object.entries(probabilities || {}), [probabilities]);

  if (!entries.length) return null;

  return (
    <div id="probs">
      {entries.map(([name, value]) => {
        const pct = value * 100;
        return (
          <div className="prob" key={name}>
            <span>{name}</span>
            <div className="bar">
              <div className="fill" style={{ width: `${pct.toFixed(2)}%` }}></div>
            </div>
            <span>{pct.toFixed(2)}%</span>
          </div>
        );
      })}
    </div>
  );
}

function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function handlePredict() {
    const input = text.trim();
    setError("");
    setResult(null);

    if (!input) {
      setError("Please enter some text.");
      return;
    }

    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      });

      let data = null;
      try {
        data = await resp.json();
      } catch {
        data = null;
      }

      if (!resp.ok) {
        throw new Error((data && data.detail) || `Request failed (${resp.status})`);
      }

      setResult(data);
    } catch (err) {
      const msg = err && err.message ? err.message : "Something went wrong.";
      if (msg === "Load failed" || msg === "Failed to fetch") {
        setError("Cannot reach backend API. Ensure `python -m app.server` is running.");
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <section className="card">
        <h1>Sentiment Analyzer</h1>
        <p className="subtitle">Enter text and get sentiment prediction from your TinyLLM model.</p>

        <label htmlFor="review-input">Input text</label>
        <textarea
          id="review-input"
          rows="8"
          placeholder="Type or paste text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        ></textarea>

        <button type="button" onClick={handlePredict} disabled={loading}>
          {loading ? "Predicting..." : "Predict Sentiment"}
        </button>

        {error ? <div className="error">{error}</div> : null}

        {result ? (
          <section className="result">
            <h2>Prediction</h2>
            <p>
              <strong>Label:</strong> {result.label}
            </p>
            <p>
              <strong>Confidence:</strong> {formatPercent(result.confidence)}
            </p>
            <ProbabilityBars probabilities={result.probabilities} />
            {result.positive_reply || result.negative_reply ? (
              <div className="replies">
                {result.positive_reply ? (
                  <div className="reply positive">
                    <h3>Positive Reply</h3>
                    <p>{result.positive_reply}</p>
                  </div>
                ) : null}
                {result.negative_reply ? (
                  <div className="reply negative">
                    <h3>Negative Reply</h3>
                    <p>{result.negative_reply}</p>
                  </div>
                ) : null}
              </div>
            ) : null}
            {result.inference_version ? (
              <p className="subtitle">Inference: {result.inference_version}</p>
            ) : null}
          </section>
        ) : null}
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
