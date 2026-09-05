  "use strict";

  /* ── Utilities ─────────────────────────────────────────────────────────── */
  const $ = (id) => document.getElementById(id);
  const fmt = {
    bytes:    (b) => b < 1024 ? `${b} B` : b < 1048576 ? `${(b/1024).toFixed(1)} KB` : `${(b/1048576).toFixed(2)} MB`,
    score:    (s) => (s * 100).toFixed(1) + "%",
    ms:       (ms) => ms < 1000 ? `${Math.round(ms)} ms` : `${(ms/1000).toFixed(1)} s`,
    truncate: (s, n=220) => s && s.length > n ? s.slice(0, n) + "…" : (s || ""),
    escape:   (s) => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"),
  };

  /* Highlight query keywords in a snippet */
  function highlight(text, query) {
    const escaped = fmt.escape(text);
    if (!query) return escaped;
    const words = query.trim().split(/\s+/).filter(w => w.length > 2).map(w =>
      w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    );
    if (!words.length) return escaped;
    const re = new RegExp(`(${words.join("|")})`, "gi");
    return escaped.replace(re, "<mark>$1</mark>");
  }

  /* ── Theme ──────────────────────────────────────────────────────────────── */
  const themeToggle = $("theme-toggle");
  function setTheme(dark) {
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
    themeToggle.textContent = dark ? "☀️" : "🌙";
    localStorage.setItem("se-theme", dark ? "dark" : "light");
  }
  const _saved = localStorage.getItem("se-theme");
  if (_saved) setTheme(_saved === "dark");
  else setTheme(window.matchMedia("(prefers-color-scheme: dark)").matches);
  themeToggle.addEventListener("click", () =>
    setTheme(document.documentElement.getAttribute("data-theme") !== "dark")
  );

  /* ── Toast ──────────────────────────────────────────────────────────────── */
  const toastContainer = $("toast-container");
  function toast(type, message, duration = 3500) {
    const icons = { success: "✅", error: "❌", info: "ℹ️" };
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.innerHTML = `<span>${icons[type] || "ℹ️"}</span><span>${message}</span>`;
    el.addEventListener("click", () => el.remove());
    toastContainer.appendChild(el);
    setTimeout(() => el.style.opacity = "0", duration);
    setTimeout(() => el.remove(), duration + 400);
  }

  /* ── Sidebar Resizer ────────────────────────────────────────────────────── */
  const sidebarResizer = $("sidebar-resizer");
  const shell = document.querySelector(".shell");
  let isResizing = false;

  const savedSidebarWidth = localStorage.getItem("se-sidebar-width");
  if (savedSidebarWidth) {
    shell.style.setProperty("--sidebar-width", savedSidebarWidth + "px");
  }

  sidebarResizer.addEventListener("mousedown", (e) => {
    isResizing = true;
    sidebarResizer.classList.add("is-resizing");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  document.addEventListener("mousemove", (e) => {
    if (!isResizing) return;
    let newWidth = e.clientX;
    if (newWidth < 180) newWidth = 180;
    if (newWidth > 600) newWidth = 600;
    shell.style.setProperty("--sidebar-width", newWidth + "px");
  });

  document.addEventListener("mouseup", () => {
    if (isResizing) {
      isResizing = false;
      sidebarResizer.classList.remove("is-resizing");
      document.body.style.cursor = "default";
      document.body.style.userSelect = "auto";
      const currentWidth = shell.style.getPropertyValue("--sidebar-width");
      if (currentWidth) {
        localStorage.setItem("se-sidebar-width", parseInt(currentWidth));
      }
    }
  });

  /* ── Navigation ─────────────────────────────────────────────────────────── */
  let currentPage = "search";
  document.querySelectorAll(".nav-item[data-page]").forEach(btn => {
    btn.addEventListener("click", () => {
      const page = btn.dataset.page;
      if (page === currentPage) return;
      currentPage = page;
      document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
      $(`page-${page}`).classList.add("active");
      if (page === "documents") loadDocuments();
    });
  });

  /* ── Health check ───────────────────────────────────────────────────────── */
  const healthBadge = $("health-badge");
  const healthText  = $("health-text");
  async function checkHealth() {
    try {
      const d = await fetch("/health").then(r => r.json());
      healthBadge.className = `health-badge ${d.status === "ok" ? "ok" : "degraded"}`;
      const model = d.ollama_reachable ? "Ollama ✓" : "BM25 only";
      healthText.textContent = `${d.docs_indexed} docs · ${model}`;
      $("doc-count-badge").textContent = d.docs_indexed;
    } catch {
      healthBadge.className = "health-badge error";
      healthText.textContent = "Offline";
    }
  }
  checkHealth();
  setInterval(checkHealth, 30_000);

  /* ════════════════════════════════════════════════════════════════════════ */
  /*  SEARCH PAGE                                                             */
  /* ════════════════════════════════════════════════════════════════════════ */
  const searchInput  = $("search-input");
  const searchBtn    = $("search-btn");
  const searchClear  = $("search-clear");
  const acList       = $("autocomplete-list");
  const resultsWrap  = $("search-results-wrap");
  let searchMode     = "hybrid";
  let acTimer        = null;
  let acFocusIdx     = -1;

  /* Mode pills */
  document.querySelectorAll(".mode-pill").forEach(p => {
    p.addEventListener("click", () => {
      document.querySelectorAll(".mode-pill").forEach(x => x.classList.remove("active"));
      p.classList.add("active");
      searchMode = p.dataset.mode;
    });
  });

  /* Clear button */
  searchInput.addEventListener("input", () => {
    searchClear.classList.toggle("visible", searchInput.value.length > 0);
    triggerAutocomplete();
  });

  searchClear.addEventListener("click", () => {
    searchInput.value = "";
    searchClear.classList.remove("visible");
    hideAutocomplete();
  });

  /* Autocomplete */
  function triggerAutocomplete() {
    clearTimeout(acTimer);
    const q = searchInput.value.trim();
    if (q.length < 2) { hideAutocomplete(); return; }
    acTimer = setTimeout(() => fetchSuggestions(q), 180);
  }

  async function fetchSuggestions(prefix) {
    try {
      const d = await fetch(`/suggest?prefix=${encodeURIComponent(prefix)}&limit=7`).then(r => r.json());
      renderAutocomplete(d.suggestions);
    } catch { hideAutocomplete(); }
  }

  function renderAutocomplete(suggestions) {
    if (!suggestions.length) { hideAutocomplete(); return; }
    acFocusIdx = -1;
    acList.innerHTML = suggestions.map((s, i) =>
      `<div class="autocomplete-item" data-term="${fmt.escape(s.term)}" role="option" tabindex="-1">
        <span class="ac-icon">🔤</span>${fmt.escape(s.term)}
      </div>`
    ).join("");
    acList.querySelectorAll(".autocomplete-item").forEach(item => {
      item.addEventListener("mousedown", (e) => {
        e.preventDefault();
        searchInput.value = item.dataset.term;
        hideAutocomplete();
        runSearch();
      });
    });
    acList.classList.add("visible");
  }

  function hideAutocomplete() {
    acList.classList.remove("visible");
    acList.innerHTML = "";
    acFocusIdx = -1;
  }

  /* Keyboard navigation in autocomplete */
  searchInput.addEventListener("keydown", (e) => {
    const items = acList.querySelectorAll(".autocomplete-item");
    if (e.key === "ArrowDown") {
      e.preventDefault();
      acFocusIdx = Math.min(acFocusIdx + 1, items.length - 1);
      items.forEach((it, i) => it.classList.toggle("focused", i === acFocusIdx));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      acFocusIdx = Math.max(acFocusIdx - 1, -1);
      items.forEach((it, i) => it.classList.toggle("focused", i === acFocusIdx));
    } else if (e.key === "Enter") {
      if (acFocusIdx >= 0 && items[acFocusIdx]) {
        searchInput.value = items[acFocusIdx].dataset.term;
        hideAutocomplete();
      }
      runSearch();
    } else if (e.key === "Escape") {
      hideAutocomplete();
    }
  });

  searchInput.addEventListener("blur", () => setTimeout(hideAutocomplete, 150));

  /* Run search */
  searchBtn.addEventListener("click", runSearch);

  function showSearchSkeleton() {
    resultsWrap.innerHTML = [1,2,3].map(() => `
      <div class="skeleton-card" style="margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;">
          <div class="skeleton" style="width:45%;height:14px;"></div>
          <div class="skeleton" style="width:12%;height:14px;"></div>
        </div>
        <div class="skeleton" style="width:100%;height:11px;"></div>
        <div class="skeleton" style="width:80%;height:11px;"></div>
        <div class="skeleton" style="width:25%;height:10px;margin-top:4px;"></div>
      </div>
    `).join("");
  }

  async function runSearch() {
    const query = searchInput.value.trim();
    if (!query) return;
    hideAutocomplete();
    searchBtn.disabled = true;
    searchBtn.textContent = "…";
    showSearchSkeleton();

    try {
      const res = await fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          mode:  searchMode,
          top_k: 8,
          alpha: 0.3,
          beta:  0.7,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderSearchResults(data, query);
    } catch (err) {
      resultsWrap.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚠️</div>
          <h3>Search failed</h3>
          <p>${fmt.escape(err.message)}</p>
        </div>`;
    } finally {
      searchBtn.disabled = false;
      searchBtn.textContent = "Search";
    }
  }

  function renderSearchResults(data, query) {
    if (!data.results || data.results.length === 0) {
      resultsWrap.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🔍</div>
          <h3>No results found</h3>
          <p>Try different keywords or switch to Semantic mode for meaning-based search.</p>
        </div>`;
      return;
    }

    const modeLabel = { bm25: "BM25 Keyword", semantic: "Semantic", hybrid: "Hybrid" }[searchMode] || searchMode;
    resultsWrap.innerHTML = `
      <div class="results-meta">${data.total_results} result${data.total_results !== 1 ? "s" : ""} · ${modeLabel} · ${data.doc_count} docs indexed</div>
      ${data.results.map(r => `
        <div class="result-card">
          <div class="result-card-header">
            <span class="result-title">📄 ${fmt.escape(r.title)}</span>
            <span class="score-pill">${fmt.score(r.score)}</span>
          </div>
          <p class="result-snippet">${highlight(r.content, query)}</p>
          <div class="result-meta-row">
            <span>Doc #${r.doc_id}</span>
            <span>${modeLabel}</span>
          </div>
        </div>
      `).join("")}
    `;
  }

  /* ════════════════════════════════════════════════════════════════════════ */
  /*  ASK AI PAGE                                                             */
  /* ════════════════════════════════════════════════════════════════════════ */
  const askInput  = $("ask-input");
  const askBtn    = $("ask-btn");
  const askWrap   = $("ask-results-wrap");

  askBtn.addEventListener("click", runAsk);

  askInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runAsk(); }
  });

  /* Auto-resize textarea */
  askInput.addEventListener("input", function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
  });

  /* Suggestion pills */
  document.querySelectorAll(".ask-suggestion-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      askInput.value = pill.textContent;
      runAsk();
    });
  });

  function showAskSkeleton() {
    askWrap.innerHTML = `
      <div class="answer-card" style="border-color:var(--border);">
        <div class="answer-label">Generating answer…</div>
        <div style="display:flex;flex-direction:column;gap:10px;">
          <div class="skeleton" style="width:100%;height:13px;"></div>
          <div class="skeleton" style="width:88%;height:13px;"></div>
          <div class="skeleton" style="width:60%;height:13px;"></div>
        </div>
      </div>`;
  }

  async function runAsk() {
    const question = askInput.value.trim();
    if (!question) return;
    askBtn.disabled = true;
    askBtn.textContent = "…";
    showAskSkeleton();

    try {
      const res = await fetch("/rag/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          mode:   "hybrid",
          top_k:  3,
          window: 0,
          alpha:  0.3,
          beta:   0.7,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderAskResults(data, question);
    } catch (err) {
      askWrap.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚠️</div>
          <h3>Request failed</h3>
          <p>${fmt.escape(err.message)}</p>
        </div>`;
    } finally {
      askBtn.disabled = false;
      askBtn.textContent = "↑";
    }
  }

  function renderAskResults(data, question) {
    const sourcesHtml = data.sources && data.sources.length ? `
      <div class="sources-label" style="margin-top:20px;">Sources (${data.sources.length})</div>
      ${data.sources.map((s, i) => `
        <div class="source-card">
          <div class="source-header" onclick="toggleSource(this)">
            <div class="source-title-wrap">
              <span class="source-icon">📄</span>
              <span class="source-name">${fmt.escape(s.title)}</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;">
              <span class="source-meta">Chunk ${s.chunk_index} · ${fmt.score(s.score)}</span>
              <span class="source-expand">▼</span>
            </div>
          </div>
          <div class="source-body">
            ${fmt.escape(s.chunk_text)}
          </div>
        </div>
      `).join("")}
    ` : `<div style="font-size:0.82rem;color:var(--text-3);margin-top:16px;">No source chunks retrieved.</div>`;

    askWrap.innerHTML = `
      <div class="answer-card">
        <div class="answer-label">✨ AI Answer</div>
        <div class="answer-text">${fmt.escape(data.answer)}</div>
        <div class="answer-footer-row">
          <span>Model: ${data.model}</span>
          <span>Latency: ${fmt.ms(data.latency_ms)}</span>
        </div>
      </div>
      ${sourcesHtml}
    `;
  }

  function toggleSource(header) {
    const body   = header.nextElementSibling;
    const arrow  = header.querySelector(".source-expand");
    const isOpen = body.classList.toggle("open");
    arrow.textContent = isOpen ? "▲" : "▼";
  }

  /* ════════════════════════════════════════════════════════════════════════ */
  /*  DOCUMENTS PAGE                                                          */
  /* ════════════════════════════════════════════════════════════════════════ */
  const dropZone   = $("drop-zone");
  const fileInput  = $("file-input");
  const browseBtn  = $("browse-btn");
  const fileRemove = $("file-remove");
  const uploadBtn  = $("upload-btn");
  const progWrap   = $("progress-wrap");
  const progFill   = $("progress-fill");
  const progLabel  = $("progress-label");
  const progPct    = $("progress-pct");
  const uploadStat = $("upload-status");
  const docListWrap = $("doc-list-wrap");
  let selectedFile = null;

  /* File selection */
  browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });

  fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

  ["dragenter","dragover"].forEach(ev => dropZone.addEventListener(ev, (e) => {
    e.preventDefault(); dropZone.classList.add("drag-over");
  }));
  ["dragleave","drop"].forEach(ev => dropZone.addEventListener(ev, (e) => {
    e.preventDefault(); dropZone.classList.remove("drag-over");
  }));
  dropZone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  fileRemove.addEventListener("click", clearFile);

  function setFile(file) {
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["txt","pdf"].includes(ext)) {
      toast("error", `Unsupported file type .${ext}. Use .txt or .pdf.`);
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      toast("error", `File too large (${fmt.bytes(file.size)}). Max is 5 MB.`);
      return;
    }
    if (file.size === 0) { toast("error", "File is empty."); return; }

    selectedFile = file;
    $("file-type-icon").textContent = ext === "pdf" ? "📕" : "📄";
    $("file-name-text").textContent = file.name;
    $("file-size-text").textContent = `${fmt.bytes(file.size)} · ${ext.toUpperCase()}`;
    $("file-selected").classList.add("visible");
    uploadBtn.style.display = "block";
    uploadStat.classList.remove("visible");
    progWrap.classList.remove("visible");
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    $("file-selected").classList.remove("visible");
    uploadBtn.style.display = "none";
    progWrap.classList.remove("visible");
    uploadStat.classList.remove("visible");
  }

  /* Upload */
  uploadBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    uploadBtn.disabled = true;
    uploadStat.classList.remove("visible");
    progWrap.classList.add("visible");
    progFill.style.width = "0%";
    progFill.classList.remove("indeterminate");
    progLabel.textContent = "Uploading…";
    progPct.textContent = "0%";

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const result = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener("progress", (e) => {
          if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            progFill.style.width = pct + "%";
            progPct.textContent = pct + "%";
            if (pct >= 100) {
              progLabel.textContent = "Indexing…";
              progPct.textContent = "";
              progFill.classList.add("indeterminate");
            }
          }
        });
        xhr.addEventListener("load", () => {
          if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
          else {
            let msg = "Upload failed.";
            try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
            reject(new Error(msg));
          }
        });
        xhr.addEventListener("error", () => reject(new Error("Network error.")));
        xhr.open("POST", "/ingest");
        xhr.send(form);
      });

      progFill.classList.remove("indeterminate");
      progFill.style.width = "100%";
      progLabel.textContent = "Complete";
      progPct.textContent = "✓";

      uploadStat.className = "upload-status visible success";
      $("upload-status-icon").textContent = "✅";
      $("upload-status-text").textContent =
        `"${result.title}" indexed as Doc #${result.doc_id}. ${result.doc_count} docs · ${result.chunk_count} chunks.`;

      toast("success", `"${result.title}" indexed successfully!`);
      clearFile();
      checkHealth();
      loadDocuments();
    } catch (err) {
      progFill.classList.remove("indeterminate");
      progWrap.classList.remove("visible");
      uploadStat.className = "upload-status visible error";
      $("upload-status-icon").textContent = "❌";
      $("upload-status-text").textContent = err.message;
      toast("error", err.message);
    } finally {
      uploadBtn.disabled = false;
    }
  });

  /* Document list */
  $("refresh-docs-btn").addEventListener("click", loadDocuments);

  async function loadDocuments() {
    docListWrap.innerHTML = `
      ${[1,2,3].map(() => `
        <div class="skeleton-card" style="margin-bottom:8px;flex-direction:row;gap:14px;align-items:center;">
          <div class="skeleton" style="width:28px;height:28px;border-radius:6px;flex-shrink:0;"></div>
          <div style="flex:1;display:flex;flex-direction:column;gap:8px;">
            <div class="skeleton" style="width:50%;height:12px;"></div>
            <div class="skeleton" style="width:30%;height:10px;"></div>
          </div>
        </div>
      `).join("")}
    `;
    try {
      const data = await fetch("/documents").then(r => r.json());
      renderDocList(data.documents);
    } catch {
      docListWrap.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚠️</div>
          <h3>Failed to load documents</h3>
          <p>Check that the server is running.</p>
        </div>`;
    }
  }

  function renderDocList(docs) {
    if (!docs || docs.length === 0) {
      docListWrap.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">📭</div>
          <h3>No documents yet</h3>
          <p>Upload a .txt or .pdf file above to get started.</p>
        </div>`;
      return;
    }
    docListWrap.innerHTML = `<div class="doc-list">${docs.map(doc => `
      <div class="doc-card">
        <span class="doc-icon">📄</span>
        <div class="doc-info">
          <div class="doc-title">${fmt.escape(doc.title)}</div>
          <div class="doc-meta">Doc #${doc.doc_id} · ${fmt.bytes(doc.char_count)} · ${doc.token_count.toLocaleString()} tokens</div>
        </div>
        <span class="doc-badge indexed">Indexed ✓</span>
      </div>
    `).join("")}</div>`;
  }

  /* Load docs when first navigating to Documents tab */
  /* (also called after upload) */

