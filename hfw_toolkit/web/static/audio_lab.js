/**
 * Interactive Audio, Music, SFX & Voicelines Studio for HFWToolkit.
 * Supports browsing Guerrilla's reconstructed developer directory tree, live HTML5 audio preview
 * (both beat-synced Music Cue mixes & individual stems/SFX/voicelines), and batch extraction
 * with optional developer directory structure preservation.
 */
window.HFWAudioLab = (function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function fmtDur(sec) {
    const s = Number(sec || 0);
    if (!isFinite(s) || s <= 0) return "0:00";
    const mins = Math.floor(s / 60);
    const rem = s - mins * 60;
    return mins > 0 ? `${mins}:${rem < 10 ? "0" : ""}${rem.toFixed(1)}` : `${rem.toFixed(2)}s`;
  }

  async function api(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`${res.status}: ${txt}`);
    }
    return res.json();
  }

  let initialized = false;
  let audioMeta = null;
  let audioKind = "music"; // "music" | "sfx" | "voicelines"
  let selectedCategory = "";
  let selectedCue = null;
  let selectedGroup = null;
  let currentGroupItems = [];
  let searchTimer = null;
  let activeJobPoll = null;

  async function init() {
    if (!initialized) {
      initialized = true;
      wireControls();
    }
    await refreshStats();
    await loadCategories();
    await loadSidebarList();
  }

  function wireControls() {
    $$(".au-kind-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        audioKind = btn.dataset.aukind;
        selectedCategory = "";
        selectedCue = null;
        selectedGroup = null;
        $$(".au-kind-btn").forEach((b) => b.classList.toggle("active", b === btn));
        updateOptionVisibility();
        await loadCategories();
        await loadSidebarList();
        renderDetailPlaceholder();
      });
    });

    $("#au-search").addEventListener("input", () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        loadSidebarList();
      }, 220);
    });

    $("#au-cat-filter").addEventListener("change", (e) => {
      selectedCategory = e.target.value;
      loadSidebarList();
    });

    $("#au-btn-extract-matching").addEventListener("click", () => startExtraction({ scope: "matching" }));
    $("#au-btn-extract-all-music").addEventListener("click", () => {
      if (!confirm("Extract ALL indexed Horizon Forbidden West music cues to disk? (Tip: MP3/FLAC/OGG saves disk space vs WAV!)")) {
        return;
      }
      startExtraction({ scope: "all_music" });
    });
    $("#au-btn-extract-selected").addEventListener("click", () => startExtraction({ scope: "selected" }));
    $("#au-btn-deep-index").addEventListener("click", startDeepIndex);
    $("#au-btn-open-folder").addEventListener("click", () => {
      const customOut = ($("#au-out-dir").value || "").trim() || null;
      api("/api/audio/open-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: customOut }),
      }).catch((e) => alert(e.message));
    });

    $("#au-detail-filter").addEventListener("input", () => {
      if (selectedGroup) renderGroupItemsTable();
    });
  }

  function updateOptionVisibility() {
    const isMusic = audioKind === "music";
    const isVo = audioKind === "voicelines";
    $("#au-music-opts").classList.toggle("hidden", !isMusic);
    $("#au-vo-opts").classList.toggle("hidden", !isVo);
    $("#au-btn-extract-all-music").classList.toggle("hidden", !isMusic);
    $("#au-btn-deep-index").classList.toggle("hidden", isMusic);
    $("#au-search").placeholder = isMusic
      ? "Search songs & cues (PlasmaSpine, MQ11, Boss, Calm, EndCredits...)"
      : isVo
      ? "Search quests, characters, speakers, or indexed subtitles (Aloy, SQ07, GAIA...)"
      : "Search machines, weapons, UI, or SFX folders (Slaughterspine, Bilegut, Bow...)";
  }

  async function refreshStats() {
    try {
      audioMeta = await api("/api/audio/stats");
      const st = audioMeta.stats;
      const langs = audioMeta.installed_languages || [];
      const langSel = $("#au-lang");
      if (langSel) {
        langSel.innerHTML = "";
        (langs.length ? langs : [{ code: "en", label: "English" }]).forEach((l) => {
          const opt = document.createElement("option");
          opt.value = l.code;
          opt.textContent = `${l.code} — ${l.label}`;
          langSel.appendChild(opt);
        });
        if (langs.length > 1) {
          const optAll = document.createElement("option");
          optAll.value = "all";
          optAll.textContent = `All Installed (${langs.map((x) => x.code).join(", ")})`;
          langSel.appendChild(optAll);
        }
      }
      if ($("#au-out-dir") && !$("#au-out-dir").value) {
        $("#au-out-dir").placeholder = audioMeta.default_out_dir || "exports/Audio";
      }
      $("#au-kpi-bar").innerHTML = `
        <div class="kpi"><div class="kpi-val">${(st.music_cues || 0).toLocaleString()}</div><div class="kpi-lbl">Music Cues (${(st.music_stems || 0).toLocaleString()} stems)</div></div>
        <div class="kpi"><div class="kpi-val">${(st.sfx_waves_total || 0).toLocaleString()}</div><div class="kpi-lbl">SFX Waves (${(st.sfx_groups || 0).toLocaleString()} folders)</div></div>
        <div class="kpi"><div class="kpi-val">${(st.voicelines_total || 0).toLocaleString()}</div><div class="kpi-lbl">Voicelines (${(st.voices || 0).toLocaleString()} voices)</div></div>
        <div class="kpi"><div class="kpi-val">${langs.map((x) => x.code.toUpperCase()).join(", ") || "EN"}</div><div class="kpi-lbl">Installed Spoken Audio • ${audioMeta.pyav_available ? "ATRAC9 Ready" : "No PyAV"}</div></div>
      `;
      updateOptionVisibility();
    } catch (e) {
      $("#au-kpi-bar").innerHTML = `<div style="color:#f87171">${esc(e.message)}</div>`;
    }
  }

  async function loadCategories() {
    const sel = $("#au-cat-filter");
    sel.innerHTML = `<option value="">All Categories</option>`;
    try {
      const data = await api(`/api/audio/categories?kind=${encodeURIComponent(audioKind)}`);
      (data.items || []).forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c.category;
        const cnt = c.cue_count != null ? `${c.cue_count} cues (${c.item_count} stems)` : `${c.group_count} dirs (${c.item_count.toLocaleString()} clips)`;
        opt.textContent = `${c.category} — ${cnt}`;
        if (c.category === selectedCategory) opt.selected = true;
        sel.appendChild(opt);
      });
    } catch (_e) {
      // ignore
    }
  }

  async function loadSidebarList() {
    const list = $("#au-list");
    const q = ($("#au-search").value || "").trim();
    list.innerHTML = `<div class="empty-state" style="padding:18px;">Loading ${esc(audioKind)}...</div>`;

    try {
      if (audioKind === "music") {
        const data = await api(
          `/api/audio/music?q=${encodeURIComponent(q)}&category=${encodeURIComponent(selectedCategory)}&limit=300`
        );
        list.innerHTML = "";
        if (!data.items.length) {
          list.innerHTML = `<div class="empty-state" style="padding:18px;">No matching music cues.</div>`;
          return;
        }
        data.items.forEach((cue) => {
          const el = document.createElement("div");
          el.className = "list-item";
          if (selectedCue && selectedCue.cue_id === cue.cue_id) el.classList.add("active");
          el.innerHTML = `
            <div class="list-item-title">${esc(cue.cue_name)}</div>
            <div class="list-item-sub">
              <span>${esc(cue.category)} • ${cue.stem_count} stem${cue.stem_count === 1 ? "" : "s"}</span>
              <span>${fmtDur(cue.duration_sec)}</span>
            </div>
            <div class="au-path-sub">${esc(cue.dev_folder)}</div>
          `;
          el.addEventListener("click", () => selectMusicCue(cue, el));
          list.appendChild(el);
        });
      } else {
        const data = await api(
          `/api/audio/groups?kind=${encodeURIComponent(audioKind)}&q=${encodeURIComponent(q)}&category=${encodeURIComponent(selectedCategory)}&limit=300`
        );
        list.innerHTML = "";

        // If user typed a search query, also show matching deep-indexed clips (e.g. subtitles / speaker / SFX sound name)!
        let clipHits = [];
        if (q.length >= 2) {
          try {
            const sr = await api(
              `/api/audio/search?kind=${encodeURIComponent(audioKind)}&q=${encodeURIComponent(q)}&limit=60`
            );
            clipHits = sr.items || [];
          } catch (_e) {}
        }

        if (!data.items.length && !clipHits.length) {
          list.innerHTML = `<div class="empty-state" style="padding:18px;">No matching ${esc(audioKind)} directories. Tip: click <strong>Deep-Index</strong> to search inside all spoken subtitles &amp; SFX graph names.</div>`;
          return;
        }

        data.items.forEach((g) => {
          const el = document.createElement("div");
          el.className = "list-item";
          if (selectedGroup && selectedGroup.group_id === g.group_id) el.classList.add("active");
          el.innerHTML = `
            <div class="list-item-title">${esc(g.owner_name)}</div>
            <div class="list-item-sub">
              <span>${esc(g.category)} • ${g.item_count.toLocaleString()} clips</span>
              <span>group ${g.group_id}</span>
            </div>
            <div class="au-path-sub">${esc(g.dev_folder)}</div>
          `;
          el.addEventListener("click", () => selectAudioGroup(g, el));
          list.appendChild(el);
        });

        if (clipHits.length) {
          const hdr = document.createElement("div");
          hdr.className = "au-section-divider";
          hdr.textContent = `Direct Clip / Subtitle Matches (${clipHits.length})`;
          list.appendChild(hdr);
          clipHits.forEach((it) => {
            const el = document.createElement("div");
            el.className = "list-item";
            el.innerHTML = `
              <div class="list-item-title">${esc(it.speaker ? `${it.speaker}: ${it.subtitle || it.name}` : it.name)}</div>
              <div class="list-item-sub">
                <span>${esc(it.owner_name || it.sound_graph_name)}</span>
                <span>${fmtDur(it.duration_sec)}</span>
              </div>
              <div class="au-path-sub">${esc(it.dev_folder)}</div>
            `;
            el.addEventListener("click", () => {
              playAudioObject(it.object_id, it.name, it.dev_folder, it.subtitle, it.filename_stem);
            });
            list.appendChild(el);
          });
        }
      }
    } catch (e) {
      list.innerHTML = `<div class="empty-state" style="color:#f87171">${esc(e.message)}</div>`;
    }
  }

  function renderDetailPlaceholder() {
    $("#au-detail-head").innerHTML = `
      <div>
        <h3 style="margin:0;">Select a ${audioKind === "music" ? "Music Cue" : audioKind === "sfx" ? "SFX Directory" : "Voiceline Directory"} on the left</h3>
        <div class="hint" style="margin-top:4px;">Or use the extraction bar above to export all songs or matching folders in one click — with Guerrilla's developer directory structure preserved!</div>
      </div>
    `;
    $("#au-detail-filter-wrap").classList.add("hidden");
    $("#au-detail-body").innerHTML = "";
    $("#au-btn-extract-selected").disabled = true;
  }

  function setNowPlaying(srcUrl, title, devPath, subtitle, downloadUrl) {
    const bar = $("#au-player-bar");
    bar.classList.remove("hidden");
    $("#au-now-title").textContent = title;
    $("#au-now-path").textContent = devPath || "";
    const subEl = $("#au-now-sub");
    if (subtitle) {
      subEl.textContent = `"${subtitle}"`;
      subEl.classList.remove("hidden");
    } else {
      subEl.textContent = "";
      subEl.classList.add("hidden");
    }
    const dl = $("#au-now-dl");
    dl.href = downloadUrl;
    const player = $("#au-player");
    player.src = srcUrl;
    player.play().catch(() => {});
  }

  function getSelectedLang() {
    const v = ($("#au-lang")?.value || "en").trim();
    return v === "all" ? "en" : v;
  }

  function getSelectedFmt() {
    return ($("#au-format")?.value || "wav").trim();
  }

  function playMusicCueMix(cue) {
    const src = `/api/audio/stream/cue/${encodeURIComponent(cue.cue_id)}?fmt=wav`;
    const dl = `/api/audio/stream/cue/${encodeURIComponent(cue.cue_id)}?fmt=${encodeURIComponent(getSelectedFmt())}&download=1`;
    setNowPlaying(src, `${cue.cue_name} (Full Beat-Synced Mix)`, cue.dev_folder, "", dl);
  }

  function playAudioObject(oid, title, devPath, subtitle, filenameStem) {
    const lang = getSelectedLang();
    const fmt = getSelectedFmt();
    const src = `/api/audio/stream/object/${encodeURIComponent(oid)}?lang=${encodeURIComponent(lang)}&fmt=wav`;
    const dl = `/api/audio/stream/object/${encodeURIComponent(oid)}?lang=${encodeURIComponent(lang)}&fmt=${encodeURIComponent(fmt)}&download=1&filename=${encodeURIComponent(filenameStem || oid.replace(":", "_"))}`;
    setNowPlaying(src, title, devPath, subtitle || "", dl);
  }

  function selectMusicCue(cue, el) {
    selectedCue = cue;
    selectedGroup = null;
    $$("#au-list .list-item").forEach((x) => x.classList.remove("active"));
    if (el) el.classList.add("active");
    $("#au-btn-extract-selected").disabled = false;
    $("#au-btn-extract-selected").textContent = "Extract This Song";
    $("#au-detail-filter-wrap").classList.add("hidden");

    const stems = cue.stems || [];
    $("#au-detail-head").innerHTML = `
      <div>
        <h3 style="margin:0;">${esc(cue.cue_name)}</h3>
        <div class="au-path-badge">${esc(cue.dev_folder)}</div>
        <div class="tag-row">
          <span class="badge">${esc(cue.project_name)}</span>
          <span class="badge violet">${esc(cue.category)}</span>
          <span class="badge green">${fmtDur(cue.duration_sec)}</span>
          <span class="badge amber">${stems.length} stem${stems.length === 1 ? "" : "s"} • ${Math.round(cue.initial_tempo || 120)} BPM</span>
          <span class="badge">${esc(cue.cue_id)}</span>
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <button class="btn-primary" id="au-play-cue-mix">▶ Play Full Song Mix</button>
        <a class="btn-ghost" id="au-dl-cue-mix" href="/api/audio/stream/cue/${encodeURIComponent(cue.cue_id)}?fmt=${encodeURIComponent(getSelectedFmt())}&download=1" style="text-decoration:none;">⬇ Download Song</a>
      </div>
    `;
    $("#au-play-cue-mix").addEventListener("click", () => playMusicCueMix(cue));

    const body = $("#au-detail-body");
    body.innerHTML = `
      <table class="au-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Wave ID</th>
            <th>Track</th>
            <th>Start Beat</th>
            <th>Duration</th>
            <th>Format</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${stems
            .map(
              (st, idx) => `
            <tr>
              <td>${idx + 1}</td>
              <td><code>${esc(st.wave_id)}</code></td>
              <td>${st.track_index >= 0 ? `Track ${st.track_index}` : "—"}</td>
              <td>${Number(st.start_beats || 0).toFixed(2)}</td>
              <td>${fmtDur(st.duration_sec)}</td>
              <td>${st.sample_rate} Hz • ${st.channels}ch</td>
              <td class="au-actions-cell">
                <button class="btn-ghost au-sm-btn" data-play-stem="${esc(st.wave_id)}" data-stem-idx="${idx + 1}">▶ Play</button>
                <a class="btn-ghost au-sm-btn" style="text-decoration:none;" href="/api/audio/stream/object/${encodeURIComponent(st.wave_id)}?fmt=${encodeURIComponent(getSelectedFmt())}&download=1&filename=${encodeURIComponent(`${cue.cue_name}_stem${String(idx + 1).padStart(2, "0")}`)}">⬇</a>
              </td>
            </tr>
          `
            )
            .join("")}
        </tbody>
      </table>
    `;

    body.querySelectorAll("[data-play-stem]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const wid = btn.dataset.playStem;
        const sidx = btn.dataset.stemIdx;
        playAudioObject(wid, `${cue.cue_name} — Stem #${sidx} (${wid})`, `${cue.dev_folder}/Stems`, "", `${cue.cue_name}_stem${sidx}`);
      });
    });
  }

  async function selectAudioGroup(group, el) {
    selectedGroup = group;
    selectedCue = null;
    $$("#au-list .list-item").forEach((x) => x.classList.remove("active"));
    if (el) el.classList.add("active");
    $("#au-btn-extract-selected").disabled = false;
    $("#au-btn-extract-selected").textContent = "Extract This Folder";
    $("#au-detail-filter").value = "";
    $("#au-detail-filter-wrap").classList.remove("hidden");

    $("#au-detail-head").innerHTML = `
      <div>
        <h3 style="margin:0;">${esc(group.owner_name)}</h3>
        <div class="au-path-badge">${esc(group.dev_folder)}</div>
        <div class="tag-row">
          <span class="badge">${esc(group.category)}</span>
          <span class="badge green">${group.item_count.toLocaleString()} clips</span>
          <span class="badge">Group ${group.group_id}</span>
        </div>
      </div>
    `;
    $("#au-detail-body").innerHTML = `<div class="empty-state">Parsing &amp; indexing Group ${group.group_id} developer sound graph / subtitles...</div>`;

    try {
      const data = await api(`/api/audio/group/${group.group_id}?kind=${encodeURIComponent(audioKind)}&limit=1500`);
      currentGroupItems = data.items || [];
      renderGroupItemsTable();
    } catch (e) {
      $("#au-detail-body").innerHTML = `<div class="empty-state" style="color:#f87171">${esc(e.message)}</div>`;
    }
  }

  function renderGroupItemsTable() {
    const flt = ($("#au-detail-filter").value || "").trim().toLowerCase();
    const rows = flt
      ? currentGroupItems.filter(
          (it) =>
            it.name.toLowerCase().includes(flt) ||
            it.dev_folder.toLowerCase().includes(flt) ||
            (it.sound_graph_name || "").toLowerCase().includes(flt) ||
            (it.speaker || "").toLowerCase().includes(flt) ||
            (it.subtitle || "").toLowerCase().includes(flt) ||
            it.object_id.includes(flt)
        )
      : currentGroupItems;

    const isVo = audioKind === "voicelines";
    const body = $("#au-detail-body");
    if (!rows.length) {
      body.innerHTML = `<div class="empty-state">No clips match "${esc(flt)}".</div>`;
      return;
    }

    const shown = rows.slice(0, 400);
    body.innerHTML = `
      <div class="hint" style="margin-bottom:8px;">Showing ${shown.length} of ${rows.length} clips in this developer folder.</div>
      <table class="au-table">
        <thead>
          <tr>
            <th>${isVo ? "Speaker" : "Sound Name"}</th>
            <th>${isVo ? "Spoken Subtitle / Transcript" : "Developer Subfolder"}</th>
            <th>Duration</th>
            <th>ID</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${shown
            .map(
              (it, idx) => `
            <tr>
              <td><strong>${esc(isVo ? it.speaker : it.name)}</strong></td>
              <td>
                ${isVo && it.subtitle ? `<div class="au-sub-quote">"${esc(it.subtitle)}"</div>` : ""}
                <div class="au-path-sub">${esc(it.dev_folder)}</div>
              </td>
              <td>${fmtDur(it.duration_sec)}</td>
              <td><code>${esc(it.object_id)}</code></td>
              <td class="au-actions-cell">
                <button class="btn-ghost au-sm-btn" data-play-idx="${idx}">▶ Play</button>
                <a class="btn-ghost au-sm-btn" style="text-decoration:none;" href="/api/audio/stream/object/${encodeURIComponent(it.object_id)}?lang=${encodeURIComponent(getSelectedLang())}&fmt=${encodeURIComponent(getSelectedFmt())}&download=1&filename=${encodeURIComponent(it.filename_stem)}">⬇</a>
              </td>
            </tr>
          `
            )
            .join("")}
        </tbody>
      </table>
    `;

    body.querySelectorAll("[data-play-idx]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const it = shown[Number(btn.dataset.playIdx)];
        playAudioObject(it.object_id, it.name, it.dev_folder, it.subtitle, it.filename_stem);
      });
    });
  }

  async function startExtraction({ scope }) {
    const stEl = $("#au-job-status");
    stEl.className = "export-status";
    stEl.textContent = "Starting extraction job...";
    if (activeJobPoll) clearInterval(activeJobPoll);

    const payload = {
      category: scope === "all_music" ? "music" : audioKind,
      subcat: scope === "all_music" ? "" : selectedCategory,
      query: scope === "all_music" ? "" : ($("#au-search").value || "").trim(),
      format: getSelectedFmt(),
      preserve_structure: $("#au-preserve-structure").checked,
      mix_cues: $("#au-mix-cues").checked,
      export_stems: $("#au-export-stems").checked,
      include_ost: $("#au-include-ost").checked,
      lang: ($("#au-lang")?.value || "en").trim(),
      group_by: ($("#au-group-by")?.value || "quest").trim(),
      write_transcript: $("#au-write-transcript").checked,
      out_dir: ($("#au-out-dir").value || "").trim() || null,
    };

    if (scope === "selected") {
      if (selectedCue) {
        payload.cue_id = selectedCue.cue_id;
      } else if (selectedGroup) {
        payload.group_id = selectedGroup.group_id;
        payload.query = ($("#au-detail-filter").value || "").trim();
      } else {
        stEl.className = "export-status err";
        stEl.textContent = "Select a music cue or developer folder first.";
        return;
      }
    }

    try {
      const resp = await api("/api/audio/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      pollAudioJob(resp.job_id);
    } catch (e) {
      stEl.className = "export-status err";
      stEl.textContent = e.message;
    }
  }

  async function startDeepIndex() {
    if (!confirm("Deep-index all ~168,000 SFX clips & spoken voiceline subtitles into SQLite? This runs in the background so you can full-text search every spoken line and sound graph name across HFW.")) {
      return;
    }
    const stEl = $("#au-job-status");
    stEl.className = "export-status";
    stEl.textContent = "Starting deep audio & subtitle indexing...";
    if (activeJobPoll) clearInterval(activeJobPoll);
    try {
      const resp = await api("/api/audio/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rebuild: false, deep: true }),
      });
      pollAudioJob(resp.job_id, () => refreshStats());
    } catch (e) {
      stEl.className = "export-status err";
      stEl.textContent = e.message;
    }
  }

  function pollAudioJob(jobId, onDone) {
    const stEl = $("#au-job-status");
    activeJobPoll = setInterval(async () => {
      try {
        const js = await api(`/api/audio/job/${encodeURIComponent(jobId)}`);
        const pct = js.total > 0 ? ` (${Math.min(100, Math.round((js.current / js.total) * 100))}%)` : "";
        stEl.textContent = `${js.message}${pct}`;
        if (js.status === "done") {
          clearInterval(activeJobPoll);
          activeJobPoll = null;
          stEl.className = "export-status done";
          if (onDone) onDone();
        } else if (js.status === "error") {
          clearInterval(activeJobPoll);
          activeJobPoll = null;
          stEl.className = "export-status err";
        }
      } catch (err) {
        clearInterval(activeJobPoll);
        activeJobPoll = null;
        stEl.className = "export-status err";
        stEl.textContent = err.message;
      }
    }, 900);
  }

  return { init };
})();
