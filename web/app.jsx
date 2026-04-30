const { useEffect, useMemo, useRef, useState } = React;

const TEST_MODES = {
  MANUAL: "manual",
  BULK: "bulk",
};

function percent(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

const RESULT_EXPORT_COLUMNS = [
  { key: "row", label: "Sira" },
  { key: "message", label: "Kullanici Mesaji" },
  { key: "rewrittenText", label: "Rewrite" },
  { key: "prediction", label: "Tahmin" },
  { key: "confidence", label: "Guven" },
  { key: "autoReply", label: "Otomatik Cevap" },
  { key: "requiresHumanReview", label: "Insan Kontrolu" },
  { key: "top3Text", label: "Top 3" },
];

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data = {};

  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      const shortText = text.length > 180 ? `${text.slice(0, 180)}...` : text;
      throw new Error(shortText || "Sunucudan JSON olmayan cevap geldi.");
    }
  }

  if (!response.ok) {
    throw new Error(data.error || data.detail || `Istek tamamlanamadi. HTTP ${response.status}`);
  }
  return data;
}

function splitMessages(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function createJobId() {
  return `job-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function buildExportRows(results) {
  return results.map((item) => ({
    row: item.row,
    message: item.message || "",
    rewrittenText: item.rewrittenText || "",
    prediction: item.prediction || "",
    confidence: item.confidence ?? "",
    autoReply: item.autoReply || "",
    requiresHumanReview: item.requiresHumanReview ? "Evet" : "Hayir",
    top3Text: (item.top3 || [])
      .map((candidate) => `${candidate.label} (${percent(candidate.confidence)})`)
      .join(" | "),
  }));
}

function csvEscape(value) {
  const text = String(value ?? "");
  return `"${text.replace(/"/g, '""')}"`;
}

function buildDelimitedExport(results, delimiter = "\t") {
  const rows = buildExportRows(results);
  const header = RESULT_EXPORT_COLUMNS.map((column) => column.label).join(delimiter);
  const body = rows.map((row) =>
    RESULT_EXPORT_COLUMNS.map((column) => csvEscape(row[column.key])).join(delimiter)
  );
  return [header, ...body].join("\n");
}

function downloadBlob(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function buildWorkbookRows(results) {
  const rows = buildExportRows(results);
  return rows.map((row) => {
    const output = {};
    RESULT_EXPORT_COLUMNS.forEach((column) => {
      output[column.label] = row[column.key];
    });
    return output;
  });
}

function App() {
  const [status, setStatus] = useState(null);
  const [activeMode, setActiveMode] = useState(TEST_MODES.MANUAL);
  const [singleMessage, setSingleMessage] = useState("");
  const [bulkText, setBulkText] = useState("");
  const [rewriteEnabled, setRewriteEnabled] = useState(true);
  const [sampleCount, setSampleCount] = useState(10);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileInfo, setFileInfo] = useState("");
  const [results, setResults] = useState([]);
  const [summary, setSummary] = useState(null);
  const [correctionForms, setCorrectionForms] = useState({});
  const [savedCorrections, setSavedCorrections] = useState({});
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const abortRef = useRef(null);
  const jobRef = useRef(null);

  useEffect(() => {
    loadStatus();
  }, []);

  const predictionItems = useMemo(() => {
    if (activeMode === TEST_MODES.MANUAL) {
      return singleMessage.trim()
        ? [{ message: singleMessage.trim() }]
        : [];
    }
    const messages = splitMessages(bulkText);
    return messages.map((message, index) => ({
      message,
    }));
  }, [activeMode, bulkText, singleMessage]);

  async function loadStatus() {
    try {
      setStatus(await getJson("/api/status"));
    } catch (err) {
      setError(err.message);
    }
  }

  async function runPredict() {
    if (!predictionItems.length) {
      setError("Tahmin icin en az bir mesaj gir.");
      return;
    }

    const jobId = createJobId();
    const controller = new AbortController();
    startPrediction(jobId, controller);

    try {
      const data = await getJson("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jobId, items: predictionItems, rewriteEnabled }),
        signal: controller.signal,
      });
      setResults(data.results || []);
      setSummary(data);
      setCorrectionForms({});
      setSavedCorrections({});
      setNotice(data.cancelled ? "Tahmin iptal edildi." : "Tahmin tamamlandi.");
    } catch (err) {
      if (err.name === "AbortError") {
        setNotice("Tahmin iptal edildi.");
      } else {
        setError(err.message);
      }
    } finally {
      finishPrediction();
    }
  }

  function startPrediction(jobId, controller) {
    abortRef.current = controller;
    jobRef.current = jobId;
    setBusy(true);
    setError("");
    setNotice("Tahmin basladi. Yeni tahmin oncesi eski sonuclar temizlendi.");
    setResults([]);
    setSummary(null);
    setCorrectionForms({});
    setSavedCorrections({});
  }

  function finishPrediction() {
    setBusy(false);
    abortRef.current = null;
    jobRef.current = null;
  }

  async function cancelPredict() {
    const jobId = jobRef.current;
    if (jobId) {
      fetch("/api/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jobId }),
      }).catch(() => {});
    }
    abortRef.current?.abort();
    setBusy(false);
    setNotice("Iptal istegi gonderildi.");
  }

  function selectSampleFile(event) {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setFileInfo(file ? `${file.name} secildi.` : "");
  }

  async function pullRandomMessages() {
    if (!selectedFile) {
      setError("Once CSV, XLSX veya XLS dosyasi sec.");
      return;
    }

    setError("");
    setNotice("Dosyadan rastgele mesaj cekiliyor.");

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const data = await getJson(`/api/sample-upload?count=${sampleCount}`, {
        method: "POST",
        body: form,
      });
      setBulkText(data.messages.join("\n"));
      setFileInfo(buildFileInfo(data));
      setActiveMode(TEST_MODES.BULK);
      setNotice("Rastgele mesajlar toplu test alanina eklendi.");
    } catch (err) {
      setError(err.message);
    }
  }

  function clearAll() {
    setSingleMessage("");
    setBulkText("");
    setResults([]);
    setSummary(null);
    setCorrectionForms({});
    setSavedCorrections({});
    setNotice("");
    setError("");
    setSelectedFile(null);
    setFileInfo("");
  }

  async function saveCorrection(item) {
    const form = correctionForms[item.row] || {};
    const correctLabel = String(form.correctLabel || "").trim() || item.prediction;
    const correctReply = String(form.correctReply || "").trim();
    if (!correctLabel && !correctReply) {
      setError("Kaydetmek icin dogru kategori veya dogru cevap gir.");
      return;
    }

    try {
      const data = await getJson("/api/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          record: {
            message: item.message,
            rewrittenText: item.rewrittenText || "",
            predictedLabel: item.prediction,
            correctLabel,
            confidence: item.confidence,
            autoReply: item.autoReply || "",
            correctReply,
          },
        }),
      });
      setSavedCorrections((previous) => ({ ...previous, [item.row]: true }));
      setNotice(`Duzeltme kaydedildi: ${data.feedbackPath}`);
      loadStatus();
    } catch (err) {
      setError(err.message);
    }
  }

  async function copyBulkResults() {
    if (!results.length) {
      setError("Kopyalanacak sonuc yok.");
      return;
    }

    try {
      await navigator.clipboard.writeText(buildDelimitedExport(results));
      setNotice("Toplu sonuclar panoya kopyalandi.");
    } catch (err) {
      setError(`Kopyalama basarisiz: ${err.message}`);
    }
  }

  function exportBulkCsv() {
    if (!results.length) {
      setError("Indirilecek sonuc yok.");
      return;
    }
    downloadBlob("support-bot-toplu-sonuclar.csv", `\uFEFF${buildDelimitedExport(results, ",")}`, "text/csv;charset=utf-8");
  }

  function exportBulkXlsx() {
    if (!results.length) {
      setError("Indirilecek sonuc yok.");
      return;
    }
    if (!window.XLSX) {
      setError("XLSX kutuphanesi yuklenemedi. Internet baglantisini kontrol et.");
      return;
    }

    const worksheet = XLSX.utils.json_to_sheet(buildWorkbookRows(results));
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Sonuclar");
    const content = XLSX.write(workbook, { bookType: "xlsx", type: "array" });
    downloadBlob(
      "support-bot-toplu-sonuclar.xlsx",
      content,
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    );
  }

  return (
    <main className="shell">
      <Header status={status} />
      <section className="layout layout-stacked">
        <InputPanel
          activeMode={activeMode}
          busy={busy}
          fileInfo={fileInfo}
          messageCount={predictionItems.length}
          rewriteAvailable={Boolean(status?.rewrite?.enabled)}
          rewriteEnabled={rewriteEnabled}
          rewriteModel={status?.rewrite?.model || "gemma4:e4b"}
          sampleCount={sampleCount}
          selectedFile={selectedFile}
          singleMessage={singleMessage}
          bulkText={bulkText}
          onModeChange={setActiveMode}
          onSingleMessageChange={setSingleMessage}
          onBulkTextChange={setBulkText}
          onRewriteEnabledChange={setRewriteEnabled}
          onSampleCountChange={setSampleCount}
          onFileChange={selectSampleFile}
          onPullRandomMessages={pullRandomMessages}
          onPredict={runPredict}
          onCancel={cancelPredict}
          onClear={clearAll}
        />
        <ResultsPanel
          correctionForms={correctionForms}
          error={error}
          isBulkMode={activeMode === TEST_MODES.BULK}
          labels={status?.labels || []}
          notice={notice}
          results={results}
          savedCorrections={savedCorrections}
          summary={summary}
          onCorrectionChange={setCorrectionForms}
          onCopyResults={copyBulkResults}
          onExportCsv={exportBulkCsv}
          onExportXls={exportBulkXlsx}
          onSaveCorrection={saveCorrection}
        />
      </section>
    </main>
  );
}

function Header({ status }) {
  return (
    <header className="topbar">
      <h1>Support Bot v1</h1>
      <div className="status">
        <div><strong>{status?.modelName || "Model okunuyor"}</strong></div>
        <div>Macro F1: <strong>{status?.macroF1 ? percent(status.macroF1) : "Okunuyor"}</strong></div>
        <div>{status?.loaded ? "Model bellekte hazir" : "Ilk tahminde model yuklenecek"}</div>
        <div>{status?.feedback ? `${status.feedback.feedbackCount} duzeltme kaydi` : ""}</div>
      </div>
    </header>
  );
}
function InputPanel(props) {
  return (
    <div className="panel">
      <h2>Test Girisi</h2>
      <ModeTabs activeMode={props.activeMode} busy={props.busy} onModeChange={props.onModeChange} />

      {props.activeMode === TEST_MODES.MANUAL && (
        <ManualTest
          message={props.singleMessage}
          messageCount={props.messageCount}
          onChange={props.onSingleMessageChange}
        />
      )}

      {props.activeMode === TEST_MODES.BULK && (
        <BulkTest
          bulkText={props.bulkText}
          fileInfo={props.fileInfo}
          messageCount={props.messageCount}
          sampleCount={props.sampleCount}
          selectedFile={props.selectedFile}
          onBulkTextChange={props.onBulkTextChange}
          onFileChange={props.onFileChange}
          onPullRandomMessages={props.onPullRandomMessages}
          onSampleCountChange={props.onSampleCountChange}
        />
      )}

      <ActionButtons
        busy={props.busy}
        onCancel={props.onCancel}
        onClear={props.onClear}
        onPredict={props.onPredict}
        rewriteAvailable={props.rewriteAvailable}
        rewriteEnabled={props.rewriteEnabled}
        rewriteModel={props.rewriteModel}
        onRewriteEnabledChange={props.onRewriteEnabledChange}
      />
    </div>
  );
}

function ModeTabs({ activeMode, busy, onModeChange }) {
  return (
    <div className="tabs" role="tablist" aria-label="Test modu">
      <button
        className={`tab ${activeMode === TEST_MODES.MANUAL ? "active" : ""}`}
        disabled={busy}
        onClick={() => onModeChange(TEST_MODES.MANUAL)}
      >
        Manuel Test
      </button>
      <button
        className={`tab ${activeMode === TEST_MODES.BULK ? "active" : ""}`}
        disabled={busy}
        onClick={() => onModeChange(TEST_MODES.BULK)}
      >
        Toplu Test
      </button>
    </div>
  );
}

function buildFileInfo(data) {
  const rewriteInfo = data.rewriteColumn ? ` - ${data.rewriteColumn} rewrite kolonu` : " - rewrite kolonu yok";
  return `${data.filename} - ${data.column} mesaj kolonu${rewriteInfo} - ${data.sampleCount}/${data.rowCount} satir`;
}

function ManualTest({ message, messageCount, onChange }) {
  return (
    <>
      <div className="field">
        <label>Kullanici mesaji</label>
        <textarea
          value={message}
          onChange={(event) => onChange(event.target.value)}
          placeholder="Oyuna giremiyorum, hesabim acilmiyor"
        />
        <div className="hint">{messageCount} mesaj tahmine hazir.</div>
      </div>
    </>
  );
}

function BulkTest(props) {
  return (
    <>
      <div className="field">
        <label>Satir satir mesaj listesi</label>
        <textarea
          className="bulk"
          value={props.bulkText}
          onChange={(event) => props.onBulkTextChange(event.target.value)}
          placeholder={"Her satira bir kullanici mesaji yaz\nSatir satir tahmin alinir"}
        />
        <div className="hint">{props.messageCount} mesaj tahmine hazir</div>
      </div>

      <div className="field">
        <label>Dosyadan rastgele mesaj cek</label>
        <div className="row">
          <input type="file" accept=".csv,.xlsx,.xls" onChange={props.onFileChange} />
          <input
            type="number"
            min="1"
            max="200"
            value={props.sampleCount}
            onChange={(event) => props.onSampleCountChange(event.target.value)}
            style={{ width: 110 }}
          />
          <button className="btn secondary" disabled={!props.selectedFile} onClick={props.onPullRandomMessages}>
            Rastgele cek
          </button>
        </div>
        <div className="hint">{props.fileInfo || "Once dosya sec, sonra kac mesaj cekilecegini gir."}</div>
      </div>
    </>
  );
}

function ActionButtons({
  busy,
  onCancel,
  onClear,
  onPredict,
  rewriteAvailable,
  rewriteEnabled,
  rewriteModel,
  onRewriteEnabledChange,
}) {
  return (
    <div className="actions">
      <label className={`toggle ${rewriteEnabled ? "on" : ""} ${!rewriteAvailable ? "unavailable" : ""}`}>
        <input
          type="checkbox"
          checked={rewriteEnabled}
          disabled={busy || !rewriteAvailable}
          onChange={(event) => onRewriteEnabledChange(event.target.checked)}
        />
        <span>Rewrite {rewriteEnabled && rewriteAvailable ? "aktif" : "pasif"}</span>
        <small>{rewriteAvailable ? rewriteModel : "Gemma kapali"}</small>
      </label>
      <div className="row">
        <button className="btn primary" disabled={busy} onClick={onPredict}>Tahmin et</button>
        <button className="btn danger" disabled={!busy} onClick={onCancel}>Iptal et</button>
      </div>
      <button className="btn secondary" disabled={busy} onClick={onClear}>Temizle</button>
    </div>
  );
}

function ResultsPanel({
  correctionForms,
  error,
  isBulkMode,
  labels,
  notice,
  results,
  savedCorrections,
  summary,
  onCorrectionChange,
  onCopyResults,
  onExportCsv,
  onExportXls,
  onSaveCorrection,
}) {
  return (
    <div className="panel">
      <div className="panel-title-row">
        <h2>Sonuclar</h2>
        {isBulkMode && results.length > 0 && (
          <div className="export-actions">
            <button className="btn secondary compact" onClick={onCopyResults}>Kopyala</button>
            <button className="btn secondary compact" onClick={onExportCsv}>CSV</button>
            <button className="btn secondary compact" onClick={onExportXls}>XLSX</button>
          </div>
        )}
      </div>
      {error && <div className="message error">{error}</div>}
      {notice && <div className="message info">{notice}</div>}
      {summary && <PredictionSummary summary={summary} />}
      <datalist id="category-options">
        {labels.map((label) => (
          <option value={label} key={label} />
        ))}
      </datalist>
      <ResultsTable
        correctionForms={correctionForms}
        labels={labels}
        results={results}
        savedCorrections={savedCorrections}
        onCorrectionChange={onCorrectionChange}
        onSaveCorrection={onSaveCorrection}
      />
    </div>
  );
}

function PredictionSummary({ summary }) {
  return (
    <div className="summary">
      <div className="metric"><span>Tamamlanan</span><strong>{summary.completed}/{summary.total}</strong></div>
      <div className="metric"><span>Sure</span><strong>{summary.elapsedSeconds}s</strong></div>
      <div className="metric"><span>Durum</span><strong>{summary.cancelled ? "Iptal" : "Bitti"}</strong></div>
    </div>
  );
}

function ResultsTable({
  correctionForms,
  labels,
  results,
  savedCorrections,
  onCorrectionChange,
  onSaveCorrection,
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Mesaj</th>
            <th>Tahmin</th>
            <th>Guven</th>
            <th>Otomatik Cevap</th>
            <th>Duzelt</th>
            <th>Top 3</th>
            <th>Rewrite</th>
          </tr>
        </thead>
        <tbody>
          {results.length === 0 && (
            <tr>
              <td colSpan="8" className="hint">Henuz sonuc yok.</td>
            </tr>
          )}
          {results.map((item) => (
            <ResultRow
              correctionForm={correctionForms[item.row] || {}}
              isSaved={Boolean(savedCorrections[item.row])}
              item={item}
              key={`${item.row}-${item.message}`}
              labels={labels}
              onCorrectionChange={(patch) =>
                onCorrectionChange((previous) => ({
                  ...previous,
                  [item.row]: {
                    ...(previous[item.row] || {}),
                    ...patch,
                  },
                }))
              }
              onSaveCorrection={() => onSaveCorrection(item)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultRow({
  correctionForm,
  isSaved,
  item,
  labels,
  onCorrectionChange,
  onSaveCorrection,
}) {
  const correctLabel = correctionForm.correctLabel || "";
  const correctReply = correctionForm.correctReply || "";
  const canSave = Boolean(correctLabel.trim() || correctReply.trim());

  return (
    <tr>
      <td>{item.row}</td>
      <td>{item.message}</td>
      <td><span className="pill">{item.prediction}</span></td>
      <td>{percent(item.confidence)}</td>
      <td>
        <div className="reply-text">{item.autoReply}</div>
        {item.requiresHumanReview && <div className="hint">Insan kontrolu onerilir.</div>}
      </td>
      <td>
        <div className="correction">
          <input
            type="text"
            list="category-options"
            value={correctLabel}
            onChange={(event) => onCorrectionChange({ correctLabel: event.target.value })}
            placeholder={`Kategori: ${item.prediction}`}
          />
          <textarea
            className="reply-correction"
            value={correctReply}
            onChange={(event) => onCorrectionChange({ correctReply: event.target.value })}
            placeholder="Dogru cevap yaz"
          />
          <button
            className="btn secondary compact"
            disabled={!canSave || isSaved}
            onClick={onSaveCorrection}
          >
            {isSaved ? "Kaydedildi" : "Veriye ekle"}
          </button>
        </div>
      </td>
      <td><TopCandidates candidates={item.top3} /></td>
      <td>
        {item.rewriteUsed ? item.rewrittenText : <span className="hint">Yok</span>}
        {item.rewriteError && <div className="hint">{item.rewriteError}</div>}
      </td>
    </tr>
  );
}

function TopCandidates({ candidates }) {
  return (
    <div className="top3">
      {candidates.map((candidate) => (
        <div className="bar" key={candidate.label}>
          <div>
            <div className="hint">{candidate.label}</div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: percent(candidate.confidence) }} />
            </div>
          </div>
          <strong>{percent(candidate.confidence)}</strong>
        </div>
      ))}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

