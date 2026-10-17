// Chat with your PDF: everything about the document stays in this browser tab.
pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

const CHUNK_SIZE = 1000;     // characters per passage
const CHUNK_OVERLAP = 200;   // repeated between passages so sentences aren't lost
const MAX_CHUNKS = 300;      // keeps the free AI quota safe (~150 dense pages)
const EMBED_BATCH = 50;
const TOP_K = 6;             // passages sent to the AI per question
const MAX_FILE_MB = 20;

const $ = (id) => document.getElementById(id);
const state = { chunks: [], vectors: null, keyword: null, history: [], busy: false };

// ---------------- reading the PDF ----------------

async function readPdf(data) {
  const pdf = await pdfjsLib.getDocument({ data }).promise;
  const pages = [];
  for (let n = 1; n <= pdf.numPages; n++) {
    const page = await pdf.getPage(n);
    const content = await page.getTextContent();
    const text = content.items.map((it) => it.str + (it.hasEOL ? "\n" : " ")).join("");
    pages.push({ page: n, text: text.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim() });
    setProgress((n / pdf.numPages) * 30, `Reading page ${n} of ${pdf.numPages}…`);
  }
  return pages;
}

function splitIntoChunks(pages) {
  const chunks = [];
  for (const { page, text } of pages) {
    let start = 0;
    while (start < text.length) {
      let end = Math.min(start + CHUNK_SIZE, text.length);
      if (end < text.length) {
        // End at a sentence or line break when there is one near the limit.
        const cut = Math.max(text.lastIndexOf(". ", end), text.lastIndexOf("\n", end));
        if (cut > start + CHUNK_SIZE * 0.6) end = cut + 1;
      }
      const piece = text.slice(start, end).trim();
      if (piece.length > 30) chunks.push({ page, text: piece });
      if (end >= text.length) break;
      start = Math.max(end - CHUNK_OVERLAP, start + 1);
    }
  }
  return chunks;
}

// ---------------- search: vectors + keywords (hybrid) ----------------

const STOP = new Set(("a an and are as at be but by can do does for from has have how i in is it its " +
  "me my of on or our so that the their them then there these they this to was we were what when " +
  "where which who why will with you your about into than also any all").split(" "));

const tokenize = (text) =>
  (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter((w) => w.length > 1 && !STOP.has(w));

function buildKeywordIndex(chunks) {
  const docs = chunks.map((c) => tokenize(c.text));
  const df = new Map();
  docs.forEach((words) => new Set(words).forEach((w) => df.set(w, (df.get(w) || 0) + 1)));
  const avgLen = docs.reduce((s, d) => s + d.length, 0) / Math.max(docs.length, 1);
  return { docs, df, avgLen };
}

// BM25: the classic keyword-ranking formula used by search engines.
function keywordScores(question) {
  const { docs, df, avgLen } = state.keyword;
  const terms = [...new Set(tokenize(question))];
  const N = docs.length, k1 = 1.4, b = 0.75;
  return docs.map((words) => {
    let score = 0;
    for (const t of terms) {
      const n = df.get(t);
      if (!n) continue;
      const tf = words.filter((w) => w === t).length;
      if (!tf) continue;
      const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
      score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * words.length / avgLen));
    }
    return score;
  });
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na * nb) || 1);
}

const ranks = (scores) => {
  const order = scores.map((s, i) => [s, i]).sort((x, y) => y[0] - x[0]);
  const r = new Array(scores.length);
  order.forEach(([, i], pos) => (r[i] = pos));
  return r;
};

async function findPassages(question) {
  const n = state.chunks.length;

  // "Summarise this document" needs passages from the whole document, not one spot.
  if (/summar|overview|what is this (document|pdf|file)|main (points|ideas)|key points/i.test(question)) {
    const count = Math.min(8, n);
    return Array.from({ length: count }, (_, i) => Math.floor((i * n) / count));
  }

  const kw = keywordScores(question);
  let vec = null;
  if (state.vectors) {
    try {
      const [q] = (await api("/api/embed", { texts: [question], task: "query" })).vectors;
      vec = state.vectors.map((v) => cosine(q, v));
    } catch { /* fall back to keyword search only */ }
  }

  // Reciprocal Rank Fusion: combine both rankings into one.
  const kwRank = ranks(kw);
  const vecRank = vec ? ranks(vec) : null;
  const fused = Array.from({ length: n }, (_, i) =>
    (kw[i] > 0 ? 1 / (60 + kwRank[i]) : 0) + (vecRank ? 1 / (60 + vecRank[i]) : 0));
  return ranks(fused)
    .map((r, i) => [r, i]).sort((a, b) => a[0] - b[0])
    .slice(0, TOP_K).map(([, i]) => i)
    .filter((i) => fused[i] > 0);
}

// ---------------- talking to the server ----------------

async function api(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail?.[0]?.msg || data.detail || "Something went wrong. Please try again.");
  return data;
}

async function embedChunks(chunks) {
  const vectors = [];
  for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
    const batch = chunks.slice(i, i + EMBED_BATCH).map((c) => c.text);
    const { vectors: v } = await api("/api/embed", { texts: batch, task: "document" });
    vectors.push(...v);
    const done = Math.min(i + EMBED_BATCH, chunks.length);
    setProgress(30 + (done / chunks.length) * 70, `Understanding passages: ${done} of ${chunks.length}…`);
  }
  return vectors;
}

// ---------------- loading a document ----------------

async function loadPdf(name, data) {
  if (state.busy) return;
  state.busy = true;
  resetChat();
  $("doc-info").hidden = true;
  $("progress").hidden = false;
  setProgress(0, "Opening the PDF…");
  try {
    const pages = await readPdf(data);
    let chunks = splitIntoChunks(pages);
    if (!chunks.length) {
      throw new Error("No text found. This PDF is probably scanned images, which can't be read.");
    }
    let notice = "";
    if (chunks.length > MAX_CHUNKS) {
      chunks = chunks.slice(0, MAX_CHUNKS);
      notice = `This is a long document, so only the first ${chunks.at(-1).page} pages are used in this free demo.`;
    }
    state.chunks = chunks;
    state.keyword = buildKeywordIndex(chunks);
    try {
      state.vectors = await embedChunks(chunks);
    } catch (err) {
      state.vectors = null;
      notice = (notice ? notice + " " : "") +
        `Smart search is unavailable right now (${err.message}), so keyword search is used instead.`;
    }

    $("doc-name").textContent = name;
    $("doc-meta").textContent = `${pages.length} page${pages.length === 1 ? "" : "s"} · ${chunks.length} passages`;
    $("doc-notice").textContent = notice;
    $("doc-notice").hidden = !notice;
    $("doc-info").hidden = false;
    $("drop").hidden = true;
    $("sample").hidden = true;
    enableChat(name);
  } catch (err) {
    addMessage("bot error", escapeHtml(err.message || "Couldn't read this PDF."));
  } finally {
    $("progress").hidden = true;
    state.busy = false;
  }
}

function handleFile(file) {
  if (!file) return;
  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    return addMessage("bot error", "Please choose a PDF file.");
  }
  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    return addMessage("bot error", `This file is over ${MAX_FILE_MB} MB. Please choose a smaller PDF.`);
  }
  file.arrayBuffer().then((buf) => loadPdf(file.name, new Uint8Array(buf)));
}

// ---------------- chat ----------------

async function ask(question) {
  question = question.trim();
  if (!question || state.busy || !state.chunks.length) return;
  state.busy = true;
  $("send").disabled = true;
  $("suggestions").hidden = true;
  $("question").value = "";
  addMessage("user", escapeHtml(question));
  const thinking = addMessage("bot typing", "Searching the document…");

  try {
    const picked = await findPassages(question);
    const sources = picked.map((i) => state.chunks[i]);
    const { answer } = await api("/api/ask", { question, sources, history: state.history.slice(-6) });
    thinking.remove();
    showAnswer(answer, sources);
    state.history.push({ role: "user", content: question }, { role: "assistant", content: answer.slice(0, 4000) });
  } catch (err) {
    thinking.remove();
    addMessage("bot error", escapeHtml(err.message));
  } finally {
    state.busy = false;
    $("send").disabled = false;
    $("question").focus();
  }
}

function showAnswer(answer, sources) {
  const msg = addMessage("bot", formatAnswer(answer));
  const cited = new Set([...answer.matchAll(/\[p\.\s*([\d,\s]+)\]/gi)]
    .flatMap((m) => m[1].split(",").map((p) => parseInt(p, 10))));
  const used = sources.filter((s) => cited.has(s.page));
  if (!used.length) return;

  const box = document.createElement("details");
  box.className = "sources";
  box.innerHTML = `<summary>Show the passages this answer used (${used.length})</summary>` +
    used.map((s) => `<div class="source" data-page="${s.page}"><b>Page ${s.page}</b>${escapeHtml(s.text)}</div>`).join("");
  msg.appendChild(box);

  // Clicking a page chip opens the passages and highlights that page.
  msg.querySelectorAll(".cite").forEach((btn) => btn.addEventListener("click", () => {
    box.open = true;
    const target = box.querySelector(`.source[data-page="${btn.dataset.page}"]`);
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    target.classList.add("flash");
    setTimeout(() => target.classList.remove("flash"), 1200);
  }));
}

// Turns the AI's simple Markdown into safe HTML, with clickable page chips.
function formatAnswer(text) {
  const html = [];
  let list = false;
  for (const raw of escapeHtml(text).split("\n")) {
    const line = raw.trim();
    const item = line.match(/^[-*•]\s+(.*)/);
    if (item && !list) { html.push("<ul>"); list = true; }
    if (!item && list) { html.push("</ul>"); list = false; }
    if (!line) continue;
    const body = (item ? item[1] : line)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\[p\.\s*([\d,\s]+)\]/gi, (_, pages) => pages.split(",").map((p) => p.trim()).filter(Boolean)
        .map((p) => `<button type="button" class="cite" data-page="${p}" title="Show the passage">p. ${p}</button>`).join(""));
    html.push(item ? `<li>${body}</li>` : `<p>${body}</p>`);
  }
  if (list) html.push("</ul>");
  return html.join("");
}

function addMessage(kind, html) {
  $("empty")?.remove();
  const div = document.createElement("div");
  div.className = "msg " + kind;
  div.innerHTML = html;
  $("messages").appendChild(div);
  $("messages").scrollTop = $("messages").scrollHeight;
  return div;
}

function enableChat(name) {
  $("question").disabled = false;
  $("send").disabled = false;
  $("question").placeholder = "Ask anything about this document…";
  addMessage("bot", `<p>I've read <strong>${escapeHtml(name)}</strong>. Ask me anything about it, and I'll show the page each answer comes from.</p>`);
  const box = $("suggestions");
  box.innerHTML = "";
  ["Summarise this document", "What are the key points?", "List any dates, deadlines or numbers"].forEach((q) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = q;
    b.onclick = () => ask(q);
    box.appendChild(b);
  });
  box.hidden = false;
  $("question").focus();
}

function resetChat() {
  state.history = [];
  $("messages").innerHTML = "";
  $("suggestions").hidden = true;
  $("question").disabled = true;
  $("send").disabled = true;
  $("question").placeholder = "Upload a PDF first";
}

function setProgress(pct, text) {
  $("bar-fill").style.width = pct + "%";
  $("progress-text").textContent = text;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------- page events ----------------

$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

$("sample").addEventListener("click", async () => {
  const resp = await fetch("/static/sample-handbook.pdf");
  loadPdf("Sample: Brightside Coffee employee handbook.pdf", new Uint8Array(await resp.arrayBuffer()));
});

$("reset").addEventListener("click", () => {
  state.chunks = []; state.vectors = null; state.keyword = null;
  resetChat();
  $("messages").innerHTML = '<div class="empty" id="empty">Upload a PDF to start chatting.</div>';
  $("doc-info").hidden = true;
  $("drop").hidden = false;
  $("sample").hidden = false;
  $("file").value = "";
});

$("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("question").value); });
