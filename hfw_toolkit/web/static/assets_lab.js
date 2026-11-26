/**
 * HFWToolkit Web Studio — Characters Prompt Studio & Guerrilla Developer-Directory Asset Catalog.
 */
(function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  async function api(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json();
  }

  let charsInitialized = false;
  let assetsInitialized = false;
  let chKindMode = "main";
  let selectedChar = null;
  let selectedAsset = null;
  let currentTreeDir = "";
  let charAnimsData = null;
  let selectedCharAnimIds = new Set();

  async function pollExportJob(jobId, statusEl) {
    statusEl.className = "export-status";
    statusEl.classList.remove("hidden");
    const iv = setInterval(async () => {
      try {
        const js = await api(`/api/export/status/${jobId}`);
        statusEl.textContent = js.message || js.status;
        if (js.status === "done") {
          clearInterval(iv);
          statusEl.className = "export-status done";
          statusEl.innerHTML = `
            <div>${esc(js.message)}</div>
            <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;">
              ${js.output_blend ? `<button class="btn-primary" data-open-blend="${esc(js.output_blend)}">Open in Blender</button>` : ""}
              <button class="btn-ghost" data-open-folder="1">Open Exports Folder</button>
            </div>
          `;
          const bOpen = statusEl.querySelector("[data-open-blend]");
          if (bOpen) {
            bOpen.addEventListener("click", () => {
              api("/api/open-blend", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ blend_path: bOpen.dataset.openBlend }),
              }).catch((e) => alert(e.message));
            });
          }
          const bDir = statusEl.querySelector("[data-open-folder]");
          if (bDir) {
            bDir.addEventListener("click", () => {
              api("/api/open-blend", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ open_folder: true }),
              }).catch((e) => alert(e.message));
            });
          }
        } else if (js.status === "error") {
          clearInterval(iv);
          statusEl.className = "export-status err";
        }
      } catch (err) {
        clearInterval(iv);
        statusEl.className = "export-status err";
        statusEl.textContent = err.message;
      }
    }, 1200);
  }

  // =========================================================================
  // CHARACTERS & OUTFITS PROMPT STUDIO
  // =========================================================================
  async function loadCharacterList() {
    const q = ($("#ch-search")?.value || "").trim();
    const kindParam = chKindMode === "npc" ? "npc" : "";
    const data = await api(`/api/characters?q=${encodeURIComponent(q)}&kind=${encodeURIComponent(kindParam)}&limit=180`);
    const counts = data.counts || {};
    $("#ch-kpi-bar").innerHTML = `
      <div class="kpi"><div class="kpi-val">${(counts.characters || 0).toLocaleString()}</div><div class="kpi-lbl">Indexed Characters</div></div>
      <div class="kpi"><div class="kpi-val">${(counts.outfits || 0).toLocaleString()}</div><div class="kpi-lbl">Outfit Variations</div></div>
      <div class="kpi"><div class="kpi-val">${(counts.heads || 0).toLocaleString()}</div><div class="kpi-lbl">Modular Heads</div></div>
      <div class="kpi"><div class="kpi-val">Unified Rig</div><div class="kpi-lbl">Body + Face + Hair</div></div>
    `;
    const list = $("#ch-list");
    list.innerHTML = "";
    const items = chKindMode === "main"
      ? data.items.filter((c) => c.character_kind !== "npc")
      : data.items;

    items.forEach((c) => {
      const el = document.createElement("div");
      el.className = "list-item" + (selectedChar && selectedChar.char_id === c.char_id ? " active" : "");
      const badge = c.character_kind === "hero" ? "★ HERO" : (c.character_kind === "companion" ? "COMPANION" : "NPC");
      el.innerHTML = `
        <div class="list-item-title">${esc(c.display_name)}</div>
        <div class="list-item-sub">
          <span>${badge} • ${esc(c.tribe)}</span>
          <span>${c.outfit_count > 1 ? `${c.outfit_count} outfits` : esc(c.gender)}</span>
        </div>
      `;
      el.addEventListener("click", () => selectCharacter(c));
      list.appendChild(el);
    });
  }

  async function populateHeadsDropdown() {
    const sel = $("#ch-head-select");
    if (!sel || sel.options.length > 2) return;
    const data = await api("/api/characters/heads?limit=250");
    sel.innerHTML = `<option value="">Default Character Head</option>`;
    data.items.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h.facial_id;
      opt.textContent = `${h.display_name} (${h.gender}) [${h.facial_id}]`;
      sel.appendChild(opt);
    });
  }

  async function selectCharacter(c, preferredOutfitId = "") {
    selectedChar = c;
    $$("#ch-list .list-item").forEach((el) => el.classList.remove("active"));
    $("#ch-detail-title").textContent = `${c.display_name} (${c.tribe})`;
    $("#ch-detail-meta").innerHTML = `ID: <code>${esc(c.char_id)}</code> • Kind: <strong>${esc(c.character_kind)}</strong> • Gender: <strong>${esc(c.gender)}</strong> • Dev Folder: <code>${esc(c.dev_folder)}</code>`;

    const sel = $("#ch-outfit-select");
    sel.innerHTML = `<option value="">Default Appearance</option>`;
    try {
      const odata = await api(`/api/characters/${encodeURIComponent(c.char_id)}/outfits`);
      odata.items.forEach((o) => {
        const opt = document.createElement("option");
        opt.value = o.outfit_id;
        opt.textContent = `${o.display_name} (${o.tribe})${o.is_dlc ? " [DLC]" : ""}${o.has_headgear_toggle ? " [HG Toggle]" : ""}`;
        if (preferredOutfitId && o.outfit_id === preferredOutfitId) opt.selected = true;
        sel.appendChild(opt);
      });
    } catch (_) {}
  }

  async function resolvePromptNow(promptText) {
    const p = (promptText || $("#ch-prompt-input").value || "").trim();
    if (!p) return null;
    $("#ch-prompt-input").value = p;
    const box = $("#ch-resolve-box");
    box.className = "export-status";
    box.classList.remove("hidden");
    box.textContent = `Resolving "${p}"...`;
    try {
      const r = await api(`/api/characters/resolve?prompt=${encodeURIComponent(p)}`);
      box.className = "export-status done";
      const altsHtml = (r.alternatives || []).slice(0, 4).map((a) =>
        `<button class="pill-btn" style="padding:3px 9px;font-size:11.5px;" data-alt-prompt="${esc(a.display_name + (a.outfit_name ? " " + a.outfit_name : ""))}">${esc(a.display_name)}${a.outfit_name ? " — " + esc(a.outfit_name) : ""}</button>`
      ).join(" ");
      box.innerHTML = `
        <div><strong>Matched:</strong> ${esc(r.summary)} <span style="opacity:.75;">(confidence ${Math.round((r.confidence || 0) * 100)}%, headgear=${r.headgear ? "ON" : "OFF"}, hair=${r.include_hair ? "ON" : "OFF"})</span></div>
        ${altsHtml ? `<div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;"><span style="font-size:11.5px;opacity:.75;">Alternatives:</span>${altsHtml}</div>` : ""}
      `;
      box.querySelectorAll("[data-alt-prompt]").forEach((btn) => {
        btn.addEventListener("click", () => resolvePromptNow(btn.dataset.altPrompt));
      });
      if (r.character) {
        $("#ch-chk-headgear").checked = !!r.headgear;
        $("#ch-chk-hair").checked = !!r.include_hair;
        $("#ch-hair-lod").value = String(r.hair_lod ?? 2);
        await selectCharacter(r.character, r.outfit ? r.outfit.outfit_id : "");
      }
      return r;
    } catch (err) {
      box.className = "export-status err";
      box.textContent = err.message;
      return null;
    }
  }

  function resolveCharAnimSpec() {
    const mode = ($("#ch-anim-mode")?.value || "").trim();
    if (!mode) return null;
    if (mode === "selected") {
      return selectedCharAnimIds.size ? Array.from(selectedCharAnimIds).join(",") : null;
    }
    return mode;
  }

  function updateCharSelectedModeOption() {
    const opt = $("#ch-anim-mode")?.querySelector('option[value="selected"]');
    if (opt) opt.textContent = `Anims: Checked Clips (${selectedCharAnimIds.size})`;
  }

  async function loadCharacterAnimations() {
    const target = selectedChar ? `${selectedChar.display_name} ${selectedChar.gender || ""}` : "Aloy female";
    const sumEl = $("#ch-anim-summary");
    if (sumEl) sumEl.textContent = "Scanning HFW humanoid animations...";
    try {
      charAnimsData = await api(`/api/animations/character/${encodeURIComponent(target)}?limit=0`);
      const sel = $("#ch-anim-cat-filter");
      if (sel) {
        const counts = charAnimsData.category_counts || {};
        sel.innerHTML = `<option value="">All Categories (${counts.all || 0})</option>`;
        (charAnimsData.categories_meta || []).forEach((cm) => {
          const n = counts[cm.key] || 0;
          if (n > 0) {
            const opt = document.createElement("option");
            opt.value = cm.key;
            opt.textContent = `${cm.label} (${n})`;
            sel.appendChild(opt);
          }
        });
      }
      renderCharAnimList();
    } catch (err) {
      if (sumEl) sumEl.textContent = "Failed to load animations: " + err.message;
    }
  }

  function getVisibleCharAnims() {
    if (!charAnimsData) return [];
    const cat = ($("#ch-anim-cat-filter")?.value || "").trim();
    const q = ($("#ch-anim-search")?.value || "").trim().toLowerCase();
    return (charAnimsData.items || []).filter((c) => {
      if (cat && c.category !== cat) return false;
      if (q) {
        const hay = `${c.name} ${c.description} ${c.anim_id} ${(c.events || []).join(" ")}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }

  function renderCharAnimList() {
    const list = $("#ch-anim-list");
    const sumEl = $("#ch-anim-summary");
    if (!list || !charAnimsData) return;
    const vis = getVisibleCharAnims();
    const counts = charAnimsData.category_counts || {};
    if (sumEl) {
      sumEl.textContent = `${counts.all || 0} humanoid clips discovered • Showing ${vis.length} • Checked: ${selectedCharAnimIds.size}`;
    }
    list.innerHTML = "";
    vis.slice(0, 300).forEach((c) => {
      const row = document.createElement("div");
      row.className = "list-item" + (selectedCharAnimIds.has(c.anim_id) ? " active" : "");
      row.style.padding = "6px 8px";
      const chk = selectedCharAnimIds.has(c.anim_id) ? "checked" : "";
      row.innerHTML = `
        <div style="display:flex;align-items:flex-start;gap:7px;">
          <input type="checkbox" ${chk} style="margin-top:3px;" />
          <div style="flex:1;min-width:0;">
            <div style="display:flex;justify-content:space-between;gap:8px;font-size:12.5px;font-weight:600;">
              <span>${esc(c.name)}</span>
              <span style="color:#94a3b8;font-weight:400;">[${esc(c.category)}] ${c.duration.toFixed(2)}s (${c.num_frames}f)</span>
            </div>
            <div style="font-size:11px;color:#94a3b8;margin-top:2px;">${esc(c.description || "")}</div>
          </div>
        </div>
      `;
      row.addEventListener("click", (ev) => {
        const cb = row.querySelector("input[type=checkbox]");
        if (ev.target !== cb) cb.checked = !cb.checked;
        if (cb.checked) {
          selectedCharAnimIds.add(c.anim_id);
          if ($("#ch-anim-mode")) $("#ch-anim-mode").value = "selected";
        } else {
          selectedCharAnimIds.delete(c.anim_id);
        }
        row.classList.toggle("active", cb.checked);
        updateCharSelectedModeOption();
        if (sumEl) {
          sumEl.textContent = `${counts.all || 0} humanoid clips discovered • Showing ${vis.length} • Checked: ${selectedCharAnimIds.size}`;
        }
      });
      list.appendChild(row);
    });
  }

  async function triggerCharacterExport(usePrompt, exportFbx) {
    const st = $("#ch-export-status");
    const ctrlRig = $("#ch-chk-ctrl-rig") ? $("#ch-chk-ctrl-rig").checked : true;
    const animSpec = resolveCharAnimSpec();
    const body = usePrompt
      ? {
          prompt: ($("#ch-prompt-input").value || "").trim(),
          hair_lod: Number($("#ch-hair-lod").value || 2),
          pack_textures: $("#ch-chk-pack").checked,
          export_fbx: exportFbx,
          build_control_rig: ctrlRig,
          animations: animSpec,
        }
      : {
          char_id: selectedChar ? selectedChar.char_id : "aloy",
          outfit_id: $("#ch-outfit-select").value || "",
          facial_override: $("#ch-head-select").value || "",
          headgear: $("#ch-chk-headgear").checked,
          include_hair: $("#ch-chk-hair").checked,
          hair_lod: Number($("#ch-hair-lod").value || 2),
          pack_textures: $("#ch-chk-pack").checked,
          export_fbx: exportFbx,
          build_control_rig: ctrlRig,
          animations: animSpec,
        };
    if (usePrompt && !body.prompt) {
      alert("Enter a character prompt first (e.g. 'Aloy Carja Blazon no headgear').");
      return;
    }
    st.className = "export-status";
    st.classList.remove("hidden");
    st.textContent = "Starting HFW character assembly...";
    try {
      const res = await api("/api/characters/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      pollExportJob(res.job_id, st);
    } catch (err) {
      st.className = "export-status err";
      st.textContent = err.message;
    }
  }

  // =========================================================================
  // GUERRILLA DEVELOPER-DIRECTORY ASSET CATALOG
  // =========================================================================
  async function loadAssetCategories() {
    const data = await api("/api/assets/categories");
    const sel = $("#as-cat-filter");
    sel.innerHTML = `<option value="">All Categories (${(data.total || 0).toLocaleString()} assets)</option>`;
    (data.categories || []).forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.category;
      opt.textContent = `${c.category} (${c.asset_count} | ${c.rigged_count} rigged)`;
      sel.appendChild(opt);
    });
    $("#as-kpi-bar").innerHTML = (data.categories || []).slice(0, 6).map((c) => `
      <div class="kpi" style="cursor:pointer;" data-as-kpi="${esc(c.category)}">
        <div class="kpi-val">${c.asset_count}</div>
        <div class="kpi-lbl">${esc(c.category)}</div>
      </div>
    `).join("");
    $("#as-kpi-bar").querySelectorAll("[data-as-kpi]").forEach((k) => {
      k.addEventListener("click", () => {
        $("#as-cat-filter").value = k.dataset.asKpi;
        loadAssetSidebarSearch();
      });
    });
  }

  async function loadAssetSidebarSearch() {
    const q = ($("#as-search")?.value || "").trim();
    const cat = $("#as-cat-filter")?.value || "";
    const data = await api(`/api/assets/search?q=${encodeURIComponent(q)}&category=${encodeURIComponent(cat)}&limit=160`);
    const list = $("#as-list");
    list.innerHTML = "";
    data.items.forEach((a) => {
      const el = document.createElement("div");
      el.className = "list-item" + (selectedAsset && selectedAsset.dev_path === a.dev_path ? " active" : "");
      el.innerHTML = `
        <div class="list-item-title">${esc(a.display_name)}</div>
        <div class="list-item-sub">
          <span>${esc(a.category)} • ${a.is_skinned ? "Rigged" : "Static"}</span>
          <span>${esc(a.resource_id)}</span>
        </div>
      `;
      el.addEventListener("click", () => selectDevAsset(a));
      list.appendChild(el);
    });
  }

  async function loadAssetTree(dirPath) {
    currentTreeDir = dirPath || "";
    const data = await api(`/api/assets/tree?dir=${encodeURIComponent(currentTreeDir)}`);
    const bc = $("#as-breadcrumbs");
    const segs = currentTreeDir ? currentTreeDir.split("/") : [];
    let acc = "";
    bc.innerHTML = `<button class="pill-btn" style="padding:3px 10px;font-size:12px;" data-bc="">work:/</button>` + segs.map((s) => {
      acc = acc ? `${acc}/${s}` : s;
      return `<span style="opacity:.5;">/</span><button class="pill-btn" style="padding:3px 10px;font-size:12px;" data-bc="${esc(acc)}">${esc(s)}</button>`;
    }).join("");
    bc.querySelectorAll("[data-bc]").forEach((b) => {
      b.addEventListener("click", () => loadAssetTree(b.dataset.bc));
    });

    const grid = $("#as-tree-grid");
    grid.innerHTML = "";
    (data.folders || []).forEach((f) => {
      const card = document.createElement("div");
      card.className = "list-item";
      card.style.border = "1px solid #1e293b";
      card.style.borderRadius = "8px";
      card.style.padding = "10px 12px";
      card.innerHTML = `
        <div class="list-item-title">📁 ${esc(f.name)}</div>
        <div class="list-item-sub"><span>${f.asset_count} assets</span><span>${esc((f.categories || []).slice(0, 2).join(", "))}</span></div>
      `;
      card.addEventListener("click", () => loadAssetTree(f.path));
      grid.appendChild(card);
    });
    (data.assets || []).forEach((a) => {
      const card = document.createElement("div");
      card.className = "list-item";
      card.style.border = "1px solid #334155";
      card.style.borderRadius = "8px";
      card.style.padding = "10px 12px";
      card.innerHTML = `
        <div class="list-item-title">${a.is_skinned ? "🦴" : "🧊"} ${esc(a.display_name)}</div>
        <div class="list-item-sub"><span>${esc(a.resource_type)}</span><span>${esc(a.resource_id)}</span></div>
      `;
      card.addEventListener("click", () => selectDevAsset(a));
      grid.appendChild(card);
    });
  }

  function selectDevAsset(a) {
    selectedAsset = a;
    $("#as-detail-card").classList.remove("hidden");
    $("#as-detail-title").textContent = a.display_name;
    $("#as-detail-meta").innerHTML = `
      Guerrilla Path: <code>work:${esc(a.dev_path)}</code><br/>
      Category: <strong>${esc(a.category)}</strong> • Type: <code>${esc(a.resource_type)}</code> (<code>${esc(a.resource_id)}</code>) •
      <strong>${a.is_skinned ? "Rigged Skeletal Asset" : "Static Mesh Asset"}</strong> •
      ${a.tset_count} TextureSets • ${a.shader_count} Shaders
    `;
    $("#as-export-status").classList.add("hidden");
  }

  async function triggerDevAssetExport(exportFbx) {
    if (!selectedAsset) return;
    const st = $("#as-export-status");
    st.className = "export-status";
    st.classList.remove("hidden");
    st.textContent = `Assembling ${selectedAsset.dev_path}...`;
    try {
      const res = await api("/api/assets/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset: selectedAsset.dev_path,
          pack_textures: $("#as-chk-pack").checked,
          export_fbx: exportFbx,
        }),
      });
      pollExportJob(res.job_id, st);
    } catch (err) {
      st.className = "export-status err";
      st.textContent = err.message;
    }
  }

  window.HFWAssetsLab = {
    async initCharacters() {
      if (!charsInitialized) {
        charsInitialized = true;
        $$(".ch-kind-btn").forEach((b) => {
          b.addEventListener("click", () => {
            chKindMode = b.dataset.chkind;
            $$(".ch-kind-btn").forEach((x) => x.classList.toggle("active", x === b));
            loadCharacterList();
          });
        });
        $("#ch-search")?.addEventListener("input", () => loadCharacterList());
        $("#ch-btn-resolve")?.addEventListener("click", () => resolvePromptNow());
        $("#ch-prompt-input")?.addEventListener("keydown", (e) => { if (e.key === "Enter") resolvePromptNow(); });
        $$(".ch-chip").forEach((c) => c.addEventListener("click", () => resolvePromptNow(c.dataset.chprompt)));
        $("#ch-btn-export-blend")?.addEventListener("click", () => triggerCharacterExport(true, false));
        $("#ch-btn-export-ue5")?.addEventListener("click", () => triggerCharacterExport(true, true));
        $("#ch-btn-export-sel-blend")?.addEventListener("click", () => triggerCharacterExport(false, false));
        $("#ch-btn-export-sel-ue5")?.addEventListener("click", () => triggerCharacterExport(false, true));
        $("#ch-btn-load-anims")?.addEventListener("click", () => loadCharacterAnimations());
        $("#ch-anim-cat-filter")?.addEventListener("change", () => renderCharAnimList());
        $("#ch-anim-search")?.addEventListener("input", () => renderCharAnimList());
        $("#ch-anim-sel-vis")?.addEventListener("click", () => {
          getVisibleCharAnims().forEach((c) => selectedCharAnimIds.add(c.anim_id));
          updateCharSelectedModeOption();
          if ($("#ch-anim-mode")) $("#ch-anim-mode").value = "selected";
          renderCharAnimList();
        });
        $("#ch-anim-clr")?.addEventListener("click", () => {
          selectedCharAnimIds.clear();
          updateCharSelectedModeOption();
          renderCharAnimList();
        });
      }
      await Promise.all([loadCharacterList(), populateHeadsDropdown()]);
    },
    async initAssets() {
      if (!assetsInitialized) {
        assetsInitialized = true;
        $("#as-search")?.addEventListener("input", () => loadAssetSidebarSearch());
        $("#as-cat-filter")?.addEventListener("change", () => loadAssetSidebarSearch());
        $("#as-btn-export-blend")?.addEventListener("click", () => triggerDevAssetExport(false));
        $("#as-btn-export-ue5")?.addEventListener("click", () => triggerDevAssetExport(true));
      }
      await Promise.all([loadAssetCategories(), loadAssetTree(currentTreeDir), loadAssetSidebarSearch()]);
    },
  };
})();
