/**
 * Unified Characters & Guerrilla Developer-Directory Asset Catalog navigator.
 * Integrates seamlessly into the consistent 3-pane Studio Shell (Pane 1 Navigator ->
 * Pane 2 Unified Inspector & Export Midbar -> Pane 3 6-Tab Material & Shader Workspace).
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
  let asViewMode = "tree"; // "tree" | "list"
  let selectedChar = null;
  let selectedAsset = null;
  let currentTreeDir = "";

  // =========================================================================
  // CHARACTERS & OUTFITS (Pane 1 Navigator -> Pane 2 Unified Inspector!)
  // =========================================================================
  async function loadCharacterList() {
    const q = ($("#ch-search")?.value || "").trim();
    const kindParam = chKindMode === "npc" ? "npc" : "";
    const data = await api(`/api/characters?q=${encodeURIComponent(q)}&kind=${encodeURIComponent(kindParam)}&limit=180`);
    const list = $("#ch-list");
    if (!list) return;
    list.innerHTML = "";
    const items = chKindMode === "main"
      ? data.items.filter((c) => c.character_kind !== "npc")
      : data.items;

    if (!items.length) {
      list.innerHTML = `<div class="empty-state" style="padding:16px;">No characters match your filter.</div>`;
      return;
    }

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
      el.addEventListener("click", () => selectCharacter(c, ""));
      list.appendChild(el);
    });
  }

  async function populateHeadsDropdown() {
    const sel = $("#ch-head-select");
    if (!sel || sel.options.length > 2) return;
    try {
      const data = await api("/api/characters/heads?limit=250");
      sel.innerHTML = `<option value="">Default Head</option>`;
      data.items.forEach((h) => {
        const opt = document.createElement("option");
        opt.value = h.facial_id;
        opt.textContent = `${h.display_name} (${h.gender})`;
        sel.appendChild(opt);
      });
    } catch (_) {}
  }

  async function selectCharacter(c, preferredOutfitId = "") {
    selectedChar = c;
    $$("#ch-list .list-item").forEach((el) => {
      const t = el.querySelector(".list-item-title")?.textContent || "";
      el.classList.toggle("active", t === c.display_name);
    });

    // Populate outfits dropdown first
    const sel = $("#ch-outfit-select");
    if (sel) {
      sel.innerHTML = `<option value="">Default Appearance</option>`;
      try {
        const odata = await api(`/api/characters/${encodeURIComponent(c.char_id)}/outfits`);
        odata.items.forEach((o) => {
          const opt = document.createElement("option");
          opt.value = o.outfit_id;
          opt.textContent = `${o.display_name} (${o.tribe})${o.is_dlc ? " [DLC]" : ""}`;
          if (preferredOutfitId && o.outfit_id === preferredOutfitId) opt.selected = true;
          sel.appendChild(opt);
        });
      } catch (_) {}
    }

    if (window.HFWStudio) {
      await window.HFWStudio.inspectCharacter(c, {
        outfit_id: preferredOutfitId || ($("#ch-outfit-select")?.value || ""),
        facial_override: $("#ch-head-select")?.value || "",
        headgear: $("#ch-chk-headgear") ? $("#ch-chk-headgear").checked : true,
        include_hair: $("#ch-chk-hair") ? $("#ch-chk-hair").checked : true,
        hair_lod: Number($("#ch-hair-lod")?.value || 2),
      });
    }
  }

  async function resolvePromptNow(promptText) {
    const p = (promptText || $("#ch-prompt-input")?.value || "").trim();
    if (!p) return null;
    if ($("#ch-prompt-input")) $("#ch-prompt-input").value = p;
    const box = $("#ch-resolve-box");
    if (box) {
      box.className = "export-status";
      box.classList.remove("hidden");
      box.textContent = `Resolving "${p}"...`;
    }
    try {
      const r = await api(`/api/characters/resolve?prompt=${encodeURIComponent(p)}`);
      if (box) {
        box.className = "export-status done";
        box.innerHTML = `<strong>Matched:</strong> ${esc(r.summary)} (${Math.round((r.confidence || 0) * 100)}%)`;
      }
      if (r.character) {
        if ($("#ch-chk-headgear")) $("#ch-chk-headgear").checked = !!r.headgear;
        if ($("#ch-chk-hair")) $("#ch-chk-hair").checked = !!r.include_hair;
        if ($("#ch-hair-lod")) $("#ch-hair-lod").value = String(r.hair_lod ?? 2);
        await selectCharacter(r.character, r.outfit ? r.outfit.outfit_id : "");
      }
      return r;
    } catch (err) {
      if (box) {
        box.className = "export-status err";
        box.textContent = err.message;
      }
      return null;
    }
  }

  // =========================================================================
  // GUERRILLA DEV-DIRECTORY ASSET CATALOG (Unified Sidebar Tree / List!)
  // =========================================================================
  async function loadAssetCategories() {
    const sel = $("#as-cat-filter");
    if (!sel || sel.options.length > 2) return;
    const data = await api("/api/assets/categories");
    sel.innerHTML = `<option value="">All Categories (${(data.total || 0).toLocaleString()} assets)</option>`;
    (data.categories || []).forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.category;
      opt.textContent = `${c.category} (${c.asset_count} | ${c.rigged_count} rigged)`;
      sel.appendChild(opt);
    });
  }

  function setAssetViewMode(mode) {
    asViewMode = mode;
    $("#as-view-tree-btn")?.classList.toggle("active", mode === "tree");
    $("#as-view-list-btn")?.classList.toggle("active", mode === "list");
    $("#as-tree-bar")?.classList.toggle("hidden", mode !== "tree");
    $("#as-list-filters")?.classList.toggle("hidden", mode !== "list");
    if (mode === "tree") {
      loadAssetTree(currentTreeDir);
    } else {
      loadAssetSidebarSearch();
    }
  }

  async function loadAssetSidebarSearch() {
    const q = ($("#as-search")?.value || "").trim();
    const cat = $("#as-cat-filter")?.value || "";
    const data = await api(`/api/assets/search?q=${encodeURIComponent(q)}&category=${encodeURIComponent(cat)}&limit=180`);
    const list = $("#as-list");
    if (!list) return;
    list.innerHTML = "";
    if (!data.items.length) {
      list.innerHTML = `<div class="empty-state" style="padding:16px;">No assets match your search.</div>`;
      return;
    }
    data.items.forEach((a) => {
      const el = document.createElement("div");
      el.className = "list-item" + (selectedAsset && selectedAsset.dev_path === a.dev_path ? " active" : "");
      el.innerHTML = `
        <div class="list-item-title">${a.is_skinned ? "🦴" : "🧊"} ${esc(a.display_name)}</div>
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
    const bc = $("#as-tree-bar");
    if (bc) {
      const segs = currentTreeDir ? currentTreeDir.split("/") : [];
      let acc = "";
      bc.innerHTML = `<button class="tree-bc-btn" data-bc="">work:/</button>` + segs.map((s) => {
        acc = acc ? `${acc}/${s}` : s;
        return `<span style="opacity:.45;">/</span><button class="tree-bc-btn" data-bc="${esc(acc)}">${esc(s)}</button>`;
      }).join("");
      bc.querySelectorAll("[data-bc]").forEach((b) => {
        b.addEventListener("click", () => loadAssetTree(b.dataset.bc));
      });
    }

    const list = $("#as-list");
    if (!list) return;
    list.innerHTML = "";

    if (currentTreeDir) {
      const parentPath = currentTreeDir.includes("/") ? currentTreeDir.slice(0, currentTreeDir.lastIndexOf("/")) : "";
      const upEl = document.createElement("div");
      upEl.className = "list-item";
      upEl.innerHTML = `
        <div class="list-item-title" style="color:#7dd3fc;">⬆ .. (Up One Folder)</div>
        <div class="list-item-sub"><span>Back to ${esc(parentPath || "work:/")}</span></div>
      `;
      upEl.addEventListener("click", () => loadAssetTree(parentPath));
      list.appendChild(upEl);
    }

    (data.folders || []).forEach((f) => {
      const el = document.createElement("div");
      el.className = "list-item";
      el.innerHTML = `
        <div class="list-item-title">📁 ${esc(f.name)}</div>
        <div class="list-item-sub">
          <span>${f.asset_count} assets</span>
          <span>${esc((f.categories || []).slice(0, 2).join(", "))}</span>
        </div>
      `;
      el.addEventListener("click", () => loadAssetTree(f.path));
      list.appendChild(el);
    });

    (data.assets || []).forEach((a) => {
      const el = document.createElement("div");
      el.className = "list-item" + (selectedAsset && selectedAsset.dev_path === a.dev_path ? " active" : "");
      el.innerHTML = `
        <div class="list-item-title">${a.is_skinned ? "🦴" : "🧊"} ${esc(a.display_name)}</div>
        <div class="list-item-sub">
          <span>${esc(a.resource_type)}</span>
          <span>${esc(a.resource_id)}</span>
        </div>
      `;
      el.addEventListener("click", () => selectDevAsset(a));
      list.appendChild(el);
    });

    if (!list.children.length) {
      list.innerHTML = `<div class="empty-state" style="padding:16px;">Empty developer folder.</div>`;
    }
  }

  async function selectDevAsset(a) {
    selectedAsset = a;
    $$("#as-list .list-item").forEach((el) => {
      el.classList.toggle("active", (el.textContent || "").includes(a.resource_id));
    });
    if (window.HFWStudio) {
      await window.HFWStudio.inspectAsset(a);
    }
  }

  window.HFWAssetsLab = {
    getSelectedChar() { return selectedChar; },
    getSelectedAsset() { return selectedAsset; },
    resolvePromptNow,
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
        ["#ch-outfit-select", "#ch-head-select", "#ch-chk-headgear", "#ch-chk-hair"].forEach((sel) => {
          $(sel)?.addEventListener("change", () => {
            if (selectedChar && window.HFWStudio) {
              window.HFWStudio.inspectCharacter(selectedChar, {
                outfit_id: $("#ch-outfit-select")?.value || "",
                facial_override: $("#ch-head-select")?.value || "",
                headgear: $("#ch-chk-headgear") ? $("#ch-chk-headgear").checked : true,
                include_hair: $("#ch-chk-hair") ? $("#ch-chk-hair").checked : true,
                hair_lod: Number($("#ch-hair-lod")?.value || 2),
              });
            }
          });
        });
      }
      await Promise.all([loadCharacterList(), populateHeadsDropdown()]);
    },
    async initAssets() {
      if (!assetsInitialized) {
        assetsInitialized = true;
        $("#as-view-tree-btn")?.addEventListener("click", () => setAssetViewMode("tree"));
        $("#as-view-list-btn")?.addEventListener("click", () => setAssetViewMode("list"));
        $("#as-search")?.addEventListener("input", () => {
          const q = ($("#as-search")?.value || "").trim();
          if (q && asViewMode === "tree") {
            setAssetViewMode("list");
          } else if (asViewMode === "list") {
            loadAssetSidebarSearch();
          }
        });
        $("#as-cat-filter")?.addEventListener("change", () => loadAssetSidebarSearch());
      }
      await loadAssetCategories();
      if (asViewMode === "tree") {
        await loadAssetTree(currentTreeDir);
      } else {
        await loadAssetSidebarSearch();
      }
    },
  };
})();
