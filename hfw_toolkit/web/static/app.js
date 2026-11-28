/**
 * Unified 3-Pane Studio Controller for HFWToolkit (Machines, Characters, Dev Assets, Audio, TextureSets & Diff).
 */
(function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  let activeMode = "machines";
  let currentMachineFilter = "";
  let cachedMachinesList = [];

  // Unified Inspector state (serves Machine, Character, AND Dev Asset identically!)
  let currentEntityKind = null; // "machine" | "character" | "asset"
  let currentEntityData = null;
  let currentAnims = null;
  let currentAnimCategory = "";
  let selectedAnimIds = new Set();
  let currentSubTab = "materials";
  let currentReport = null;
  let activeWsTab = "explain";

  async function api(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`${res.status}: ${txt}`);
    }
    return res.json();
  }

  // --- Topbar stats & non-blocking GitHub update check ---
  async function refreshTopbarStats() {
    try {
      const st = await api("/api/stats");
      const verLbl = st.game_version && st.game_version !== "not found" ? `v${esc(st.game_version)}` : "Not Found";
      const pyVer = st.python_runtime?.current_version || "";
      $("#topbar-stats").innerHTML = `
        <span>HFW: <strong>${verLbl}</strong></span>
        <span>Machines: <strong>${st.catalog.machines}</strong></span>
        <span>Chars: <strong>${(st.catalog.characters || 0).toLocaleString()}</strong></span>
        <span>Assets: <strong>${(st.catalog.dev_assets || 0).toLocaleString()}</strong></span>
        <span>Shaders: <strong>${st.catalog.shaders.toLocaleString()}</strong></span>
        ${pyVer ? `<span>Python: <strong>${esc(pyVer)}</strong></span>` : ""}
        <span>Blender: <strong>${st.blender_available ? "Ready" : "Missing"}</strong></span>
      `;
      if (st.diagnose && !st.diagnose.ready) {
        openSetupModal();
      }
      if (st.check_updates_on_startup) {
        checkUpdatesBackground();
      }
    } catch (e) {
      $("#topbar-stats").textContent = "Error loading stats: " + e.message;
    }
  }

  async function checkUpdatesBackground() {
    try {
      const up = await api("/api/updater/status");
      const pill = $("#topbar-update-pill");
      if (pill && up.update_available && up.remote) {
        pill.textContent = `⬆ Update Available (${up.remote.short_sha})`;
        pill.classList.remove("hidden");
      } else if (pill) {
        pill.classList.add("hidden");
      }
    } catch (_) {}
  }

  async function init() {
    await refreshTopbarStats();
    await loadMachines("");
    handleHashRoute();
  }

  // --- Machine Navigator ---
  const FLYER_CODES = new Set(["glider", "swooper", "stormbird", "sunwing", "waterwing", "dreadwing", "bat"]);
  const HEAVY_CODES = new Set(["horus", "titan", "slaughterspine", "thunderjaw", "tremortusk", "slitherfang", "tideripper", "shellsnapper", "fireclaw", "frostclaw", "behemoth", "rockbreaker", "bilegut"]);
  const SAC_CODES = new Set(["bilegut", "tideripper", "frostclaw", "fireclaw", "snapmaw", "bellowback", "widemaw", "spikesnout", "burrower"]);

  function renderMachinesList() {
    const list = $("#machine-list");
    if (!list) return;
    list.innerHTML = "";
    const filtered = cachedMachinesList.filter((m) => {
      const hay = `${m.display_name} ${m.codename}`.toLowerCase();
      if (currentMachineFilter === "flyer") return [...FLYER_CODES].some((k) => hay.includes(k));
      if (currentMachineFilter === "heavy") return [...HEAVY_CODES].some((k) => hay.includes(k));
      if (currentMachineFilter === "sac") return [...SAC_CODES].some((k) => hay.includes(k));
      return true;
    });

    filtered.forEach((m) => {
      const oid = `${m.group_id}:${m.object_index}`;
      const el = document.createElement("div");
      el.className = "list-item";
      if (currentEntityKind === "machine" && currentEntityData && currentEntityData.entity_id === oid) {
        el.classList.add("active");
      }
      el.innerHTML = `
        <div class="list-item-title">${esc(m.display_name)}</div>
        <div class="list-item-sub">
          <span>${esc(m.codename)}</span>
          <span>${oid} • ${m.part_count} parts</span>
        </div>
      `;
      el.addEventListener("click", () => selectMachine(oid, 0));
      list.appendChild(el);
    });
  }

  async function loadMachines(q) {
    const data = await api(`/api/machines?q=${encodeURIComponent(q)}`);
    cachedMachinesList = data.items || [];
    renderMachinesList();
  }

  async function loadTextureSets(q) {
    const data = await api(`/api/texturesets?q=${encodeURIComponent(q)}&limit=80`);
    const list = $("#ts-list");
    list.innerHTML = "";
    data.items.forEach((ts) => {
      const oid = `${ts.group_id}:${ts.object_index}`;
      const el = document.createElement("div");
      el.className = "list-item";
      el.innerHTML = `
        <div class="list-item-title">${esc(ts.name)}</div>
        <div class="list-item-sub">
          <span>${esc(ts.codename || ts.category)}</span>
          <span>${oid}</span>
        </div>
      `;
      el.addEventListener("click", () => openTextureSet(oid));
      list.appendChild(el);
    });
  }

  // --- Unified Pane 2 Inspector Activation ---
  function setMidInspectorKind(kind, titleText) {
    currentEntityKind = kind;
    $("#mid-empty").classList.add("hidden");
    $("#mid-inspector").classList.remove("hidden");
    $("#ctrl-machine").classList.toggle("hidden", kind !== "machine");
    $("#ctrl-character").classList.toggle("hidden", kind !== "character");
    $("#ctrl-asset").classList.toggle("hidden", kind !== "asset");
    const badge = $("#m-kind-badge");
    if (badge) {
      badge.textContent = kind.toUpperCase();
      badge.className = "badge " + (kind === "character" ? "violet" : (kind === "asset" ? "amber" : ""));
    }
    const animBtn = $("#subtab-anims-btn");
    if (animBtn) animBtn.classList.toggle("hidden", kind === "asset");
    if (kind === "asset" && currentSubTab === "animations") {
      currentSubTab = "materials";
      $$(".subtab-btn").forEach((x) => x.classList.toggle("active", x.dataset.subtab === "materials"));
    }
    $("#m-title").textContent = titleText;
    $("#export-status-wrap")?.classList.add("hidden");
    $("#mid-list").innerHTML = `<div class="empty-state">Assembling skeleton, meshes &amp; materials...</div>`;
  }

  async function selectMachine(entityId, variantIdx = 0) {
    setMidInspectorKind("machine", "Loading machine...");
    try {
      const data = await api(`/api/machine/${encodeURIComponent(entityId)}?variant=${variantIdx}`);
      currentEntityData = data;
      currentAnims = null;
      currentAnimCategory = "";
      selectedAnimIds = new Set();
      $("#m-title").textContent = `${data.display_name} (${data.codename})`;
      $("#m-meta").textContent = `${data.entity_id} • ${data.skeleton.joint_count} joints • ${data.skeleton.helper_count} helpers`;
      $("#m-mat-count").textContent = data.materials.length;
      $("#m-part-count").textContent = data.parts.length;
      if ($("#m-anim-total")) $("#m-anim-total").textContent = "…";

      const sel = $("#m-variant");
      sel.innerHTML = "";
      (data.variants.length ? data.variants : [{ variant_index: 0, label: "Base Model" }]).forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v.variant_index;
        opt.textContent = v.label;
        if (v.variant_index === variantIdx) opt.selected = true;
        sel.appendChild(opt);
      });

      renderMachinesList();
      renderMidList();
      loadEntityAnimations("machine", data.entity_id);
      if (data.materials.length) {
        selectMaterial(data.materials[0].material_id);
      }
    } catch (e) {
      $("#mid-list").innerHTML = `<div class="empty-state" style="color:#f87171">${esc(e.message)}</div>`;
    }
  }

  async function inspectCharacter(charObj, opts = {}) {
    setMidInspectorKind("character", `Loading ${charObj.display_name}...`);
    try {
      const qs = new URLSearchParams({
        outfit_id: opts.outfit_id || "",
        facial_override: opts.facial_override || "",
        headgear: String(opts.headgear ?? true),
        include_hair: String(opts.include_hair ?? true),
        hair_lod: String(opts.hair_lod ?? 2),
      });
      const data = await api(`/api/characters/${encodeURIComponent(charObj.char_id)}/inspect?${qs.toString()}`);
      currentEntityData = { ...data, charObj };
      currentAnims = null;
      currentAnimCategory = "";
      selectedAnimIds = new Set();
      $("#m-title").textContent = `${data.display_name}`;
      $("#m-meta").textContent = `${data.variant_label || charObj.tribe} • ${data.skeleton.joint_count} joints • ${data.parts.length} meshes`;
      $("#m-mat-count").textContent = data.materials.length;
      $("#m-part-count").textContent = data.parts.length;
      if ($("#m-anim-total")) $("#m-anim-total").textContent = "…";

      renderMidList();
      loadEntityAnimations("character", `${charObj.display_name} ${charObj.gender || ""}`);
      if (data.materials.length) {
        selectMaterial(data.materials[0].material_id);
      }
    } catch (e) {
      $("#mid-list").innerHTML = `<div class="empty-state" style="color:#f87171">${esc(e.message)}</div>`;
    }
  }

  async function inspectAsset(assetObj) {
    setMidInspectorKind("asset", `Loading ${assetObj.display_name}...`);
    if ($("#as-dev-path-badge")) {
      $("#as-dev-path-badge").textContent = `work:${assetObj.dev_path || assetObj.resource_id}`;
    }
    try {
      const data = await api(`/api/assets/inspect?asset=${encodeURIComponent(assetObj.dev_path || assetObj.resource_id)}`);
      currentEntityData = { ...data, assetObj };
      currentAnims = null;
      $("#m-title").textContent = data.display_name;
      $("#m-meta").textContent = `${data.category} • ${data.skeleton.joint_count ? `${data.skeleton.joint_count} joints (Rigged)` : "Static Mesh"} • ${data.resource_id}`;
      $("#m-mat-count").textContent = data.materials.length;
      $("#m-part-count").textContent = data.parts.length;

      renderMidList();
      if (data.materials.length) {
        selectMaterial(data.materials[0].material_id);
      }
    } catch (e) {
      $("#mid-list").innerHTML = `<div class="empty-state" style="color:#f87171">${esc(e.message)}</div>`;
    }
  }

  async function loadEntityAnimations(kind, target) {
    try {
      const adata = await api(`/api/animations/${encodeURIComponent(kind)}/${encodeURIComponent(target)}?limit=0`);
      if (currentEntityKind !== kind) return;
      currentAnims = adata;
      if ($("#m-anim-total")) $("#m-anim-total").textContent = adata.total;
      if (currentSubTab === "animations") renderMidList();
    } catch (_) {
      if ($("#m-anim-total")) $("#m-anim-total").textContent = "0";
    }
  }

  function updateSelectedAnimModeOption() {
    const sel = currentEntityKind === "character" ? $("#ch-anim-mode") : $("#m-anim-mode");
    const opt = sel?.querySelector('option[value="selected"]');
    if (opt) opt.textContent = `Anims: Checked (${selectedAnimIds.size})`;
  }

  function renderMidList() {
    if (!currentEntityData) return;
    const filter = ($("#mid-filter").value || "").trim().toLowerCase();
    const list = $("#mid-list");
    const animBar = $("#mid-anim-bar");
    list.innerHTML = "";
    if (animBar) animBar.classList.toggle("hidden", currentSubTab !== "animations");

    if (currentSubTab === "materials") {
      (currentEntityData.materials || [])
        .filter((m) => !filter || m.name.toLowerCase().includes(filter) || (m.tags || []).join(" ").toLowerCase().includes(filter) || m.material_id.includes(filter))
        .forEach((m) => {
          const el = document.createElement("div");
          el.className = "list-item";
          if (currentReport && currentReport.material_id === m.material_id) el.classList.add("active");
          el.innerHTML = `
            <div class="list-item-title">${esc(m.name)}</div>
            <div class="list-item-sub">
              <span>${m.texture_count} tex • ${m.variable_count} vars • ${m.used_by_count} parts</span>
              <span>${m.material_id}</span>
            </div>
            ${m.tags && m.tags.length ? `<div class="tag-row" style="margin-top:4px;">${m.tags.slice(0, 3).map((t) => `<span class="badge">${esc(t)}</span>`).join("")}</div>` : ""}
          `;
          el.addEventListener("click", () => selectMaterial(m.material_id));
          list.appendChild(el);
        });
    } else if (currentSubTab === "parts") {
      (currentEntityData.parts || [])
        .filter((p) => !filter || p.name.toLowerCase().includes(filter) || (p.collection || "").toLowerCase().includes(filter) || (p.parent_joint || "").toLowerCase().includes(filter))
        .forEach((p) => {
          const el = document.createElement("div");
          el.className = "list-item";
          el.innerHTML = `
            <div class="list-item-title">${esc(p.name)}</div>
            <div class="list-item-sub">
              <span>${esc(p.collection || "Mesh")}</span>
              <span>${p.is_skinned ? "Skinned" : "Bone: " + esc(p.parent_joint || "None")}</span>
            </div>
            <div class="list-item-sub"><span>${(p.triangle_count || 0).toLocaleString()} tris</span><span>${(p.material_ids || []).length} mat(s)</span></div>
          `;
          el.addEventListener("click", () => {
            if (p.material_ids && p.material_ids.length) selectMaterial(p.material_ids[0]);
          });
          list.appendChild(el);
        });
    } else {
      if (!currentAnims) {
        list.innerHTML = `<div class="empty-state">Scanning HFW in-game animations...</div>`;
        return;
      }
      const counts = currentAnims.category_counts || {};
      const catsMeta = currentAnims.categories_meta || [];
      const visible = (currentAnims.items || []).filter((c) => {
        if (currentAnimCategory && c.category !== currentAnimCategory) return false;
        if (filter) {
          const hay = `${c.name} ${c.description} ${c.anim_id} ${(c.events || []).join(" ")}`.toLowerCase();
          if (!hay.includes(filter)) return false;
        }
        return true;
      });
      if (animBar) {
        const pills = [`<button class="ch-btn ${!currentAnimCategory ? "active" : ""}" data-acat="">All (${counts.all || 0})</button>`];
        catsMeta.forEach((cm) => {
          const n = counts[cm.key] || 0;
          if (n > 0) {
            const shortLbl = cm.label.split("&")[0].trim();
            pills.push(`<button class="ch-btn ${currentAnimCategory === cm.key ? "active" : ""}" data-acat="${esc(cm.key)}">${esc(shortLbl)} (${n})</button>`);
          }
        });
        animBar.innerHTML = `
          <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:5px;">${pills.join("")}</div>
          <div style="display:flex;gap:6px;align-items:center;justify-content:space-between;font-size:11px;color:#94a3b8;">
            <span>Showing ${visible.length} • Checked: <strong>${selectedAnimIds.size}</strong></span>
            <span style="display:flex;gap:4px;">
              <button class="ch-btn" id="btn-ma-sel-vis">Check Visible</button>
              <button class="ch-btn" id="btn-ma-clr">Clear</button>
            </span>
          </div>
        `;
        animBar.querySelectorAll("[data-acat]").forEach((b) => {
          b.addEventListener("click", () => { currentAnimCategory = b.dataset.acat; renderMidList(); });
        });
        animBar.querySelector("#btn-ma-sel-vis")?.addEventListener("click", () => {
          visible.forEach((c) => selectedAnimIds.add(c.anim_id));
          updateSelectedAnimModeOption();
          const sel = currentEntityKind === "character" ? $("#ch-anim-mode") : $("#m-anim-mode");
          if (sel) sel.value = "selected";
          renderMidList();
        });
        animBar.querySelector("#btn-ma-clr")?.addEventListener("click", () => {
          selectedAnimIds.clear();
          updateSelectedAnimModeOption();
          renderMidList();
        });
      }
      if (!visible.length) {
        list.innerHTML = `<div class="empty-state">No animations match this filter.</div>`;
        return;
      }
      visible.slice(0, 300).forEach((c) => {
        const el = document.createElement("div");
        el.className = "list-item" + (selectedAnimIds.has(c.anim_id) ? " active" : "");
        const chk = selectedAnimIds.has(c.anim_id) ? "checked" : "";
        el.innerHTML = `
          <div style="display:flex;align-items:flex-start;gap:7px;">
            <input type="checkbox" ${chk} style="margin-top:3px;flex-shrink:0;" />
            <div style="min-width:0;flex:1;">
              <div class="list-item-title" style="font-size:12.5px;">${esc(c.name)}</div>
              <div class="list-item-sub">
                <span><span class="badge">${esc(c.category)}</span> ${c.duration.toFixed(2)}s (${c.num_frames}f)</span>
                <span>${esc(c.anim_id)}</span>
              </div>
              <div style="font-size:11px;color:#94a3b8;margin-top:3px;line-height:1.3;">${esc(c.description || "")}</div>
            </div>
          </div>
        `;
        el.addEventListener("click", (ev) => {
          const cb = el.querySelector("input[type=checkbox]");
          if (ev.target !== cb) cb.checked = !cb.checked;
          if (cb.checked) {
            selectedAnimIds.add(c.anim_id);
            const sel = currentEntityKind === "character" ? $("#ch-anim-mode") : $("#m-anim-mode");
            if (sel) sel.value = "selected";
          } else {
            selectedAnimIds.delete(c.anim_id);
          }
          el.classList.toggle("active", cb.checked);
          updateSelectedAnimModeOption();
          const cntSpan = animBar?.querySelector("strong");
          if (cntSpan) cntSpan.textContent = String(selectedAnimIds.size);
        });
        list.appendChild(el);
      });
    }
  }

  // --- Material Deep-Dive Workspace (Pane 3) ---
  async function selectMaterial(matId, externalDecompile = false) {
    showWorkspacePane("ws-material");
    $("#mat-title").textContent = `Analyzing ${matId}...`;
    $("#mat-tags").innerHTML = "";
    $$(".tab-panel").forEach((p) => (p.innerHTML = `<div class="empty-state">Disassembling DXIL &amp; analyzing textures...</div>`));

    try {
      const report = await api(`/api/material/${encodeURIComponent(matId)}?external_decompile=${externalDecompile}`);
      currentReport = report;
      $("#mat-title").textContent = `${report.name}  [${report.material_id}]`;
      $("#mat-tags").innerHTML = (report.tags || []).map((t) => `<span class="badge">${esc(t)}</span>`).join("");

      renderExplainTab(report);
      window.HFWTextureLab.render($("#tab-textures"), report, (eid) => {
        switchSidebarMode("machines");
        selectMachine(eid, 0);
      });
      renderDecimaGraphTab(report);
      renderUE5Tab(report);
      renderCodeTab(report);
      renderDXILTab(report);
      setWsTab(activeWsTab);
      renderMidList();
    } catch (e) {
      $("#tab-explain").innerHTML = `<div class="empty-state" style="color:#f87171">${esc(e.message)}</div>`;
      setWsTab("explain");
    }
  }

  function renderExplainTab(report) {
    const exp = report.explainer;
    const g = exp.gpu_profile;
    let html = `
      <div class="kpi-bar">
        <div class="kpi"><div class="kpi-val">${esc(g.shader_model)}</div><div class="kpi-lbl">Shader Model</div></div>
        <div class="kpi"><div class="kpi-val">${g.ps_instructions.toLocaleString()}</div><div class="kpi-lbl">PS Instructions</div></div>
        <div class="kpi"><div class="kpi-val">${g.ps_texture_samples}</div><div class="kpi-lbl">Texture Samples</div></div>
        <div class="kpi"><div class="kpi-val">${g.bound_material_textures}</div><div class="kpi-lbl">Bound Textures</div></div>
        <div class="kpi"><div class="kpi-val">${g.bound_material_variables}</div><div class="kpi-lbl">Parameters</div></div>
        <div class="kpi"><div class="kpi-val">${g.gbuffer_targets}</div><div class="kpi-lbl">G-Buffer Targets</div></div>
      </div>
      <div class="card" style="margin-bottom:14px;border-color:#2b4268;">
        <div class="card-cat">Complexity Tier</div>
        <h3>${esc(g.complexity_tier)}</h3>
        <p style="margin:0;">${esc(exp.headline)} ${g.uses_bindless_heap ? "• Uses DirectX 12 SM 6.6 Bindless Resource Descriptor Heap." : ""}</p>
      </div>
      <div class="card-grid">
    `;
    (exp.cards || []).forEach((c) => {
      html += `
        <div class="card">
          <div class="card-cat">${esc(c.category)}</div>
          <h3>${esc(c.title)}</h3>
          <p>${esc(c.summary)}</p>
          <div class="why-box"><strong>Game-Dev Takeaway:</strong> ${esc(c.why_it_matters)}</div>
          ${c.items && c.items.length ? `<ul class="mini-list">${c.items.map((x) => `<li><code>${esc(x)}</code></li>`).join("")}</ul>` : ""}
          <div class="tag-row" style="margin-top:10px;">${(c.badges || []).map((b) => `<span class="badge violet">${esc(b)}</span>`).join("")}</div>
        </div>
      `;
    });
    html += `</div>`;
    html += `
      <h3 style="margin:22px 0 8px;">Render Techniques &amp; Passes (${report.techniques_summary.length})</h3>
      <table class="data-table">
        <thead><tr><th>Set</th><th>Passes</th><th>Layer ID</th><th>Shader</th><th>Textures</th><th>Vars</th></tr></thead>
        <tbody>
          ${report.techniques_summary.map((t) => `
            <tr style="${t.is_primary ? "background:#13243b;" : ""}">
              <td>${esc(t.set_type)}${t.is_primary ? ' <span class="badge green">Primary</span>' : ""}</td>
              <td>${t.passes.map((p) => `<span class="badge">${esc(p)}</span>`).join(" ")}</td>
              <td>${t.layer_id}</td>
              <td><code>${esc(t.shader_id || "—")}</code></td>
              <td>${t.tex_count}</td>
              <td>${t.var_count}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    $("#tab-explain").innerHTML = html;
  }

  function renderDecimaGraphTab(report) {
    const el = $("#tab-decima-graph");
    const det = report.decima_detailed_graph;
    const hasDetailed = det && (det.nodes || []).length > 0;
    const stats = (det && det.stats) || {};
    let mode = hasDetailed ? "detailed" : "overview";

    el.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px;">
        <div>
          <h3 id="decima-graph-heading" style="margin:0;">${esc(hasDetailed ? det.title : report.decima_graph.title)}</h3>
          ${hasDetailed ? `
            <div class="tag-row" style="margin-top:5px;">
              <span class="badge violet">${stats.specialty_nodes || 0} Decima Specialty Nodes</span>
              <span class="badge">${stats.math_nodes || 0} Lifted DXIL Math Ops</span>
              <span class="badge green">${stats.texture_samplers || 0} Texture Samplers</span>
              <span class="badge amber">${stats.cbuffer_params || 0} CBuffer Params</span>
              <span class="badge red">${stats.outputs || 0} Render Targets</span>
            </div>
          ` : ""}
        </div>
        <div style="display:flex;gap:6px;">
          ${hasDetailed ? `<button class="btn-ghost active" data-dmode="detailed">Detailed DXIL Graph</button>` : ""}
          <button class="btn-ghost ${hasDetailed ? "" : "active"}" data-dmode="overview">High-Level Overview</button>
        </div>
      </div>
      <div id="decima-graph-mount"></div>
    `;

    const mount = el.querySelector("#decima-graph-mount");
    const heading = el.querySelector("#decima-graph-heading");
    function showMode(m) {
      mode = m;
      el.querySelectorAll("[data-dmode]").forEach((b) => b.classList.toggle("active", b.dataset.dmode === m));
      const gd = m === "detailed" && hasDetailed ? det : report.decima_graph;
      heading.textContent = gd.title || "";
      window.HFWShaderGraph.render(mount, gd);
    }
    el.querySelectorAll("[data-dmode]").forEach((b) => b.addEventListener("click", () => showMode(b.dataset.dmode)));
    showMode(mode);
  }

  function renderUE5Tab(report) {
    const el = $("#tab-ue5-graph");
    const fid = report.unreal_graph.fidelity || {};
    el.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px;">
        <div>
          <h3 style="margin:0;">${esc(report.unreal_graph.title)}</h3>
          <span style="color:#8ea3c1;font-size:12px;">In-Game Fidelity Estimate: <strong style="color:#34d399">${fid.score_percent || 85}%</strong> — ${esc(fid.summary || "")}</span>
        </div>
        <div style="display:flex;gap:8px;">
          <button class="btn-primary" id="btn-copy-ue5-t3d">Copy UE5 Nodes (Ctrl+V)</button>
          <button class="btn-ghost" id="btn-dl-ue5-py">Download UE5 Python (.py)</button>
        </div>
      </div>
      <div id="ue5-graph-mount"></div>
      <div class="fidelity-box">
        <div class="card-cat">Honest In-Game Fidelity Report (Decima -&gt; Unreal Engine 5)</div>
        <div class="fidelity-cols">
          <div>
            <h4 style="margin:6px 0;color:#34d399;">1:1 Accurate / Native UE5 Matches</h4>
            <ul class="mini-list">${(fid.exact_1to1 || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
          </div>
          <div>
            <h4 style="margin:6px 0;color:#f59e0b;">Runtime Effects Parameterized in UE5</h4>
            <ul class="mini-list">${(fid.parameterized_approximations || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
          </div>
        </div>
      </div>
      <details style="margin-top:14px;">
        <summary style="cursor:pointer;color:#22d3ee;font-weight:600;">Preview UE5 Python Import Script</summary>
        <pre class="code-box" style="margin-top:8px;max-height:360px;">${esc(report.ue5_python_script)}</pre>
      </details>
    `;
    window.HFWShaderGraph.render(el.querySelector("#ue5-graph-mount"), report.unreal_graph);
    el.querySelector("#btn-copy-ue5-t3d").addEventListener("click", async (ev) => {
      await navigator.clipboard.writeText(report.ue5_clipboard_t3d);
      ev.target.textContent = "Copied! Press Ctrl+V in UE5";
      setTimeout(() => { ev.target.textContent = "Copy UE5 Nodes (Ctrl+V)"; }, 2600);
    });
    el.querySelector("#btn-dl-ue5-py").addEventListener("click", () => {
      const blob = new Blob([report.ue5_python_script], { type: "text/x-python" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `ue5_import_${report.name.replace(/[^A-Za-z0-9_]/g, "_")}.py`;
      a.click();
    });
  }

  function highlightGlossaryTerms(codeText, glossary) {
    let html = esc(codeText);
    const terms = Object.keys(glossary).sort((a, b) => b.length - a.length);
    if (!terms.length) return html;
    const re = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`, "g");
    return html.replace(re, (m) => `<span class="gloss-term" data-term="${esc(m)}">${m}</span>`);
  }

  function renderCodeTab(report) {
    const el = $("#tab-code");
    const ext = report.external_hlsl || {};
    el.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px;">
        <div>
          <h3 style="margin:0;">Annotated Readable Pseudo-HLSL</h3>
          <span style="color:#8ea3c1;font-size:12px;">Hover over any <span class="gloss-term">cyan underlined term</span> for a beginner-friendly explanation!</span>
        </div>
        <button class="btn-ghost" id="btn-ext-decompile">Run Optional dxil-spirv + spirv-cross</button>
      </div>
      <pre class="code-box" id="pseudo-hlsl-box">${highlightGlossaryTerms(report.pseudo_hlsl, report.glossary || {})}</pre>
      ${ext.hlsl ? `<h3 style="margin:18px 0 8px;">Full Decompiled HLSL</h3><pre class="code-box">${esc(ext.hlsl)}</pre>` : (ext.note ? `<p class="hint" style="margin-top:10px;">${esc(ext.note)}</p>` : "")}
    `;
    const pop = $("#glossary-pop");
    el.querySelectorAll(".gloss-term").forEach((span) => {
      span.addEventListener("mouseenter", (e) => {
        const term = span.dataset.term;
        const def = (report.glossary || {})[term];
        if (!def) return;
        pop.innerHTML = `<strong>${esc(term)}</strong>${esc(def)}`;
        pop.style.left = Math.min(window.innerWidth - 360, e.clientX + 14) + "px";
        pop.style.top = Math.min(window.innerHeight - 140, e.clientY + 14) + "px";
        pop.classList.remove("hidden");
      });
      span.addEventListener("mouseleave", () => pop.classList.add("hidden"));
    });
    el.querySelector("#btn-ext-decompile").addEventListener("click", () => selectMaterial(report.material_id, true));
  }

  function renderDXILTab(report) {
    const el = $("#tab-dxil");
    const stages = report.stages || [];
    if (!stages.length) {
      el.innerHTML = `<div class="empty-state">No compiled DXIL shader stages attached to this technique.</div>`;
      return;
    }
    let html = `<div style="display:flex;gap:6px;margin-bottom:12px;">`;
    stages.forEach((st, i) => {
      html += `<button class="btn-ghost ${i === 0 ? "active" : ""}" data-stage-idx="${i}">${esc(st.stage)} (${esc(st.shader_model || "?")}) — ${st.instructions || 0} insts</button>`;
    });
    html += `</div><div id="dxil-stage-body"></div>`;
    el.innerHTML = html;

    function showStage(idx) {
      el.querySelectorAll("[data-stage-idx]").forEach((b) => b.classList.toggle("active", Number(b.dataset.stageIdx) === idx));
      const st = stages[idx];
      el.querySelector("#dxil-stage-body").innerHTML = `
        <div class="tag-row" style="margin-bottom:10px;">
          <span class="badge">${esc(st.debug_name || st.stage)}</span>
          <span class="badge violet">Blob: ${(st.blob_size || 0).toLocaleString()} bytes</span>
          ${(st.dxbc_chunks || []).map((c) => `<span class="badge">${esc(c)}</span>`).join("")}
        </div>
        <h4 style="margin:8px 0 4px;">Resource Bindings (${(st.bindings || []).length})</h4>
        <table class="data-table">
          <thead><tr><th>ID</th><th>Type</th><th>Dim</th><th>HLSL Register</th><th>Space</th><th>Decima Correlated Asset</th></tr></thead>
          <tbody>
            ${(st.bindings || []).map((b) => {
              const corr = report.binding_map[`tex:${b.space}:${b.register}`];
              const corrTxt = corr ? `<strong>${esc(corr.name)}</strong> (${esc(corr.semantic_role)})` : (b.space === 2 ? `<span style="color:#2dd4bf">Global Engine / Weather</span>` : "—");
              return `<tr><td><code>${esc(b.id)}</code></td><td>${esc(b.type)}</td><td>${esc(b.dim)}</td><td><code>${esc(b.hlsl_bind)}</code></td><td>space${b.space}</td><td>${corrTxt}</td></tr>`;
            }).join("")}
          </tbody>
        </table>
        <h4 style="margin:16px 0 6px;">Raw DXIL LLVM-IR Disassembly</h4>
        <pre class="code-box">${esc(st.disassembly || "(empty)")}</pre>
      `;
    }
    el.querySelectorAll("[data-stage-idx]").forEach((b) => b.addEventListener("click", () => showStage(Number(b.dataset.stageIdx))));
    showStage(0);
  }

  async function openTextureSet(tsId) {
    showWorkspacePane("ws-textureset");
    const el = $("#ws-textureset");
    el.innerHTML = `<div class="empty-state">Decoding TextureSet ${esc(tsId)}...</div>`;
    try {
      const ts = await api(`/api/textureset/${encodeURIComponent(tsId)}`);
      el.innerHTML = `
        <h2 style="margin:0 0 6px;">TextureSet: ${esc(ts.name)} [${esc(ts.object_id)}]</h2>
        <div class="card" style="margin-bottom:14px;">
          <div class="card-cat">Original Guerrilla Authoring Source Paths</div>
          <ul class="mini-list">${Object.entries(ts.source_paths).map(([k, v]) => `<li><strong>${esc(k)}:</strong> <code>${esc(v)}</code></li>`).join("")}</ul>
          ${ts.used_by.length ? `<div style="margin-top:10px;"><strong style="color:#22d3ee;">Used by (${ts.used_by.length}):</strong> ${ts.used_by.map((u) => `<button class="ch-btn" data-jump-entity="${u.group_id}:${u.object_index}">${esc(u.display_name)}</button>`).join(" ")}</div>` : ""}
        </div>
        <h3>Unpacked Semantic Maps (${ts.semantic_maps.length})</h3>
        <div class="tex-grid">
          ${ts.semantic_maps.map((m) => `
            <div class="tex-card">
              <div class="tex-preview-wrap"><img src="/api/textureset/${encodeURIComponent(ts.object_id)}/map/${encodeURIComponent(m)}?max_dim=512" /></div>
              <div class="tex-body"><div class="tex-name">${esc(m)}</div></div>
            </div>
          `).join("")}
        </div>
      `;
      el.querySelectorAll("[data-jump-entity]").forEach((b) => b.addEventListener("click", () => {
        switchSidebarMode("machines");
        selectMachine(b.dataset.jumpEntity, 0);
      }));
    } catch (e) {
      el.innerHTML = `<div class="empty-state" style="color:#f87171">${esc(e.message)}</div>`;
    }
  }

  async function runMaterialDiff() {
    const a = $("#diff-a").value.trim();
    const b = $("#diff-b").value.trim();
    if (!a || !b) return;
    showWorkspacePane("ws-diff");
    const el = $("#ws-diff");
    el.innerHTML = `<div class="empty-state">Comparing ${esc(a)} vs ${esc(b)}...</div>`;
    try {
      const d = await api(`/api/diff?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
      el.innerHTML = `
        <h2 style="margin:0 0 12px;">Material Diff: ${esc(d.a.name)} [${esc(a)}] vs ${esc(d.b.name)} [${esc(b)}]</h2>
        <h3>Texture Bindings Comparison</h3>
        <table class="data-table">
          <thead><tr><th>Binding Hash</th><th>Status</th><th>Material A</th><th>Material B</th></tr></thead>
          <tbody>
            ${d.textures.map((t) => {
              const cls = t.status === "changed" ? "row-changed" : (t.status === "added_in_b" ? "row-added" : (t.status === "removed_in_b" ? "row-removed" : ""));
              return `<tr class="${cls}">
                <td><code>${esc(t.binding_hash)}</code></td>
                <td><span class="badge ${t.status === "changed" ? "amber" : (t.status === "same" ? "" : "violet")}">${esc(t.status)}</span></td>
                <td>${t.a ? `${esc(t.a.name)} (${esc(t.a.resource_id)})` : "—"}</td>
                <td>${t.b ? `${esc(t.b.name)} (${esc(t.b.resource_id)})` : "—"}</td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
        <h3 style="margin-top:20px;">Shader Variable Parameters Comparison</h3>
        <table class="data-table">
          <thead><tr><th>Binding Hash</th><th>Status</th><th>Material A</th><th>Material B</th></tr></thead>
          <tbody>
            ${d.variables.map((v) => {
              const cls = v.status === "changed" ? "row-changed" : (v.status === "added_in_b" ? "row-added" : (v.status === "removed_in_b" ? "row-removed" : ""));
              return `<tr class="${cls}">
                <td><code>${esc(v.binding_hash)}</code></td>
                <td><span class="badge ${v.status === "changed" ? "amber" : ""}">${esc(v.status)}</span></td>
                <td>${v.a ? `<strong>${esc(v.a.name)}</strong> <code>(${v.a.float4.map((x) => x.toFixed(3)).join(", ")})</code>` : "—"}</td>
                <td>${v.b ? `<strong>${esc(v.b.name)}</strong> <code>(${v.b.float4.map((x) => x.toFixed(3)).join(", ")})</code>` : "—"}</td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
      `;
    } catch (e) {
      el.innerHTML = `<div class="empty-state" style="color:#f87171">${esc(e.message)}</div>`;
    }
  }

  function showWorkspacePane(id) {
    ["ws-empty", "ws-material", "ws-textureset", "ws-diff", "ws-audio"].forEach((pid) => {
      const el = $("#" + pid);
      if (el) el.classList.toggle("hidden", pid !== id);
    });
  }

  function setWsTab(tabName) {
    activeWsTab = tabName;
    $$(".ws-tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tabName));
    $$(".tab-panel").forEach((p) => p.classList.toggle("hidden", p.id !== "tab-" + tabName));
  }

  function switchSidebarMode(mode) {
    activeMode = mode;
    $$(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    ["machines", "characters", "assets", "audio", "texturesets", "diff"].forEach((m) => {
      const el = $("#pane-" + m);
      if (el) el.classList.toggle("hidden", m !== mode);
    });
    // UNIFIED 3-PANE LAYOUT across Machines, Characters, Assets, TextureSets & Diff!
    // Only Audio expands to wide table layout for full-width tracklists & sticky player.
    $("#studio-layout").classList.toggle("audio-wide", mode === "audio");
    if (mode === "audio") {
      showWorkspacePane("ws-audio");
      if (window.HFWAudioLab) window.HFWAudioLab.init();
    } else {
      showWorkspacePane(currentReport ? "ws-material" : "ws-empty");
      if (mode === "characters" && window.HFWAssetsLab) window.HFWAssetsLab.initCharacters();
      if (mode === "assets" && window.HFWAssetsLab) window.HFWAssetsLab.initAssets();
      if (mode === "texturesets" && !$("#ts-list").children.length) loadTextureSets("");
    }
  }

  // --- Unified Export Runner (Machine, Character, AND Dev Asset!) ---
  function pollUnifiedExport(jobId) {
    const wrap = $("#export-status-wrap");
    const stEl = $("#export-status");
    wrap?.classList.remove("hidden");
    stEl.className = "export-status";
    const poll = setInterval(async () => {
      try {
        const js = await api(`/api/export/status/${jobId}`);
        stEl.textContent = js.message;
        if (js.status === "done") {
          clearInterval(poll);
          stEl.className = "export-status done";
          stEl.innerHTML = `
            <div>${esc(js.message)}</div>
            <div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;">
              ${js.output_blend ? `<button class="btn-ghost" id="btn-launch-exported-blend" style="padding:4px 8px;font-size:11px;">Open in Blender</button>` : ""}
              <button class="btn-ghost" id="btn-reveal-exported-blend" style="padding:4px 8px;font-size:11px;">Open Exports Folder</button>
            </div>
          `;
          stEl.querySelector("#btn-launch-exported-blend")?.addEventListener("click", () => {
            api("/api/open-blend", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ blend_path: js.output_blend }) }).catch((e) => alert(e.message));
          });
          stEl.querySelector("#btn-reveal-exported-blend")?.addEventListener("click", () => {
            api("/api/open-blend", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ open_folder: true }) }).catch((e) => alert(e.message));
          });
        } else if (js.status === "error") {
          clearInterval(poll);
          stEl.className = "export-status err";
        }
      } catch (err) {
        clearInterval(poll);
        stEl.className = "export-status err";
        stEl.textContent = err.message;
      }
    }, 1200);
  }

  function resolveAnimSpec(selId) {
    const mode = ($(selId)?.value || "").trim();
    if (!mode) return null;
    if (mode === "selected") return selectedAnimIds.size ? Array.from(selectedAnimIds).join(",") : null;
    return mode;
  }

  async function runMachineExport(exportFbx) {
    if (currentEntityKind !== "machine" || !currentEntityData) return;
    $("#export-status-wrap")?.classList.remove("hidden");
    const stEl = $("#export-status");
    stEl.className = "export-status";
    stEl.textContent = exportFbx ? "Exporting UE5 rigged FBX + animated shaders..." : "Building Blender 5.3 scene...";
    try {
      const resp = await api("/api/export/blend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          machine: currentEntityData.entity_id,
          variant_index: Number($("#m-variant").value || 0),
          lod_index: Number($("#m-lod") ? $("#m-lod").value : -1),
          unpack_textures: $("#chk-unpack") ? $("#chk-unpack").checked : false,
          pack_textures: $("#chk-pack").checked,
          export_fbx: Boolean(exportFbx),
          build_control_rig: $("#chk-ctrl-rig") ? $("#chk-ctrl-rig").checked : true,
          animations: resolveAnimSpec("#m-anim-mode"),
        }),
      });
      pollUnifiedExport(resp.job_id);
    } catch (e) {
      stEl.className = "export-status err";
      stEl.textContent = e.message;
    }
  }

  async function runCharacterExport(exportFbx) {
    if (currentEntityKind !== "character" || !currentEntityData?.charObj) return;
    $("#export-status-wrap")?.classList.remove("hidden");
    const stEl = $("#export-status");
    stEl.className = "export-status";
    stEl.textContent = exportFbx ? "Exporting UE5 rigged character..." : "Building Blender 5.3 character...";
    try {
      const resp = await api("/api/characters/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          char_id: currentEntityData.charObj.char_id,
          outfit_id: $("#ch-outfit-select")?.value || "",
          facial_override: $("#ch-head-select")?.value || "",
          headgear: $("#ch-chk-headgear") ? $("#ch-chk-headgear").checked : true,
          include_hair: $("#ch-chk-hair") ? $("#ch-chk-hair").checked : true,
          hair_lod: Number($("#ch-hair-lod")?.value || 2),
          pack_textures: $("#ch-chk-pack") ? $("#ch-chk-pack").checked : false,
          export_fbx: Boolean(exportFbx),
          build_control_rig: $("#ch-chk-ctrl-rig") ? $("#ch-chk-ctrl-rig").checked : true,
          animations: resolveAnimSpec("#ch-anim-mode"),
        }),
      });
      pollUnifiedExport(resp.job_id);
    } catch (e) {
      stEl.className = "export-status err";
      stEl.textContent = e.message;
    }
  }

  async function runAssetExport(exportFbx) {
    if (currentEntityKind !== "asset" || !currentEntityData?.assetObj) return;
    $("#export-status-wrap")?.classList.remove("hidden");
    const stEl = $("#export-status");
    stEl.className = "export-status";
    stEl.textContent = exportFbx ? "Exporting UE5 dev asset..." : "Building Blender 5.3 dev asset...";
    try {
      const resp = await api("/api/assets/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset: currentEntityData.assetObj.dev_path || currentEntityData.assetObj.resource_id,
          pack_textures: $("#as-chk-pack") ? $("#as-chk-pack").checked : false,
          export_fbx: Boolean(exportFbx),
        }),
      });
      pollUnifiedExport(resp.job_id);
    } catch (e) {
      stEl.className = "export-status err";
      stEl.textContent = e.message;
    }
  }

  // --- Setup, Updates & Python Runtime Modal ---
  function renderDoctorGrid(diag) {
    const okColor = "color:#34d399";
    const warnColor = "color:#fcd34d";
    const errColor = "color:#f87171";
    const py = diag.python_runtime || {};
    const newestPy = py.newest_system_python;
    const pyText = py.current_version
      ? `v${py.current_version}${py.upgrade_available && newestPy ? ` (v${newestPy.version} available)` : " (Newest)"}`
      : "Unknown";
    const lv = diag.version_info || {};
    $("#setup-health-box").innerHTML = `
      <div class="health-item">
        <div class="h-lbl">HFW Game Installation</div>
        <div class="h-val" style="${diag.game_dir_valid ? okColor : errColor}">
          ${diag.game_dir_valid ? `Ready (v${esc(diag.game_version)} • ${diag.core_archive_count} .core)` : "Not Found"}
        </div>
      </div>
      <div class="health-item">
        <div class="h-lbl">Decima RTTI Schema</div>
        <div class="h-val" style="${diag.rtti_ok ? okColor : errColor}">
          ${diag.rtti_ok ? `${(diag.rtti_type_count || 0).toLocaleString()} types` : "Missing"}
        </div>
      </div>
      <div class="health-item">
        <div class="h-lbl">Python Runtime</div>
        <div class="h-val" style="${py.upgrade_available ? warnColor : okColor}">${esc(pyText)}</div>
      </div>
      <div class="health-item">
        <div class="h-lbl">Toolkit Build</div>
        <div class="h-val" style="color:#22d3ee">${esc(lv.short_sha || "local")} (${esc(lv.install_mode || "zip")} • ${esc(lv.branch || "main")})</div>
      </div>
      <div class="health-item">
        <div class="h-lbl">Blender &amp; DXC</div>
        <div class="h-val" style="${diag.blender_found && diag.dxc_found ? okColor : warnColor}">
          Blender: ${diag.blender_found ? "Ready" : "Missing"} • DXC: ${diag.dxc_found ? "Ready" : "Missing"}
        </div>
      </div>
      <div class="health-item">
        <div class="h-lbl">Exports Output Folder</div>
        <div class="h-val" style="color:#22d3ee;font-size:11.5px;">${esc(diag.exports_dir)}</div>
      </div>
    `;
    $("#cfg-game-dir").value = diag.game_dir || "";
    $("#cfg-blender-exe").value = diag.blender_exe || "";
    if ($("#cfg-auto-update")) $("#cfg-auto-update").checked = Boolean(diag.auto_update);
    if ($("#cfg-check-startup")) $("#cfg-check-startup").checked = diag.check_updates_on_startup !== false;
  }

  async function refreshUpdaterPanel(force = false) {
    const line = $("#updater-status-line");
    if (line) line.textContent = "Checking GitHub main branch...";
    try {
      const up = await api(`/api/updater/status?force=${force ? "true" : "false"}`);
      const loc = up.local || {};
      const rem = up.remote;
      if (up.error) {
        line.innerHTML = `<span style="color:#fcd34d;">${esc(up.error)}</span>`;
      } else if (up.update_available && rem) {
        line.innerHTML = `<strong style="color:#34d399;">Update available!</strong> Local <code>${esc(loc.short_sha)}</code> → Remote <code>${esc(rem.short_sha)}</code> — ${esc(rem.commit_message)}`;
      } else if (up.ahead_of_remote && rem) {
        line.innerHTML = `<span style="color:#22d3ee;">Ahead of <code>main</code></span> (local <code>${esc(loc.short_sha)}</code> on <code>${esc(loc.branch)}</code> includes latest main <code>${esc(rem.short_sha)}</code>).`;
      } else if (rem) {
        line.innerHTML = `<span style="color:#34d399;">Up to date!</span> Running <code>${esc(loc.short_sha)}</code> — ${esc(rem.commit_message)}`;
      }
      const pill = $("#topbar-update-pill");
      if (pill) pill.classList.toggle("hidden", !up.update_available);
    } catch (e) {
      if (line) line.innerHTML = `<span style="color:#f87171;">${esc(e.message)}</span>`;
    }
  }

  async function openSetupModal() {
    $("#setup-modal").classList.remove("hidden");
    $("#setup-msg").classList.add("hidden");
    $("#updater-msg")?.classList.add("hidden");
    try {
      const diag = await api("/api/doctor");
      renderDoctorGrid(diag);
      refreshUpdaterPanel(false);
    } catch (e) {
      $("#setup-health-box").innerHTML = `<div style="color:#f87171">${esc(e.message)}</div>`;
    }
  }

  function handleHashRoute() {
    const h = location.hash || "";
    if (h.startsWith("#/material/")) {
      selectMaterial(decodeURIComponent(h.slice("#/material/".length)));
    } else if (h.startsWith("#/machine/")) {
      selectMachine(decodeURIComponent(h.slice("#/machine/".length)), 0);
    }
  }

  // --- Event wiring ---
  $$(".mode-btn").forEach((b) => b.addEventListener("click", () => switchSidebarMode(b.dataset.mode)));
  $$(".subtab-btn").forEach((b) => b.addEventListener("click", () => {
    currentSubTab = b.dataset.subtab;
    $$(".subtab-btn").forEach((x) => x.classList.toggle("active", x === b));
    renderMidList();
  }));
  $$(".ws-tab").forEach((b) => b.addEventListener("click", () => setWsTab(b.dataset.tab)));
  $("#machine-search").addEventListener("input", (e) => loadMachines(e.target.value));
  $$("#machine-filter-chips [data-mfilter]").forEach((b) => b.addEventListener("click", () => {
    currentMachineFilter = b.dataset.mfilter;
    $$("#machine-filter-chips [data-mfilter]").forEach((x) => x.classList.toggle("active", x === b));
    renderMachinesList();
  }));
  $("#ts-search").addEventListener("input", (e) => loadTextureSets(e.target.value));
  $("#mid-filter").addEventListener("input", renderMidList);
  $("#m-variant").addEventListener("change", (e) => {
    if (currentEntityKind === "machine" && currentEntityData) selectMachine(currentEntityData.entity_id, Number(e.target.value));
  });
  $("#btn-run-diff").addEventListener("click", runMaterialDiff);
  $("#btn-set-diff-a").addEventListener("click", () => { if (currentReport) { $("#diff-a").value = currentReport.material_id; switchSidebarMode("diff"); } });
  $("#btn-set-diff-b").addEventListener("click", () => { if (currentReport) { $("#diff-b").value = currentReport.material_id; switchSidebarMode("diff"); } });

  $$("[data-quick]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      switchSidebarMode("machines");
      const q = btn.dataset.quick;
      $("#machine-search").value = q;
      await loadMachines(q);
      if (cachedMachinesList.length) {
        const best = cachedMachinesList[0];
        selectMachine(`${best.group_id}:${best.object_index}`, 0);
      }
    });
  });
  $$(".ch-chip").forEach((btn) => {
    btn.addEventListener("click", async () => {
      switchSidebarMode("characters");
      if (window.HFWAssetsLab) {
        await window.HFWAssetsLab.initCharacters();
        await window.HFWAssetsLab.resolvePromptNow(btn.dataset.chprompt);
      }
    });
  });
  $("#btn-quick-audio")?.addEventListener("click", () => switchSidebarMode("audio"));
  $("#btn-quick-assets")?.addEventListener("click", () => switchSidebarMode("assets"));

  $("#btn-export-blend")?.addEventListener("click", () => runMachineExport(false));
  $("#btn-export-ue5")?.addEventListener("click", () => runMachineExport(true));
  $("#ch-btn-export-sel-blend")?.addEventListener("click", () => runCharacterExport(false));
  $("#ch-btn-export-sel-ue5")?.addEventListener("click", () => runCharacterExport(true));
  $("#as-btn-export-blend")?.addEventListener("click", () => runAssetExport(false));
  $("#as-btn-export-ue5")?.addEventListener("click", () => runAssetExport(true));

  // Setup & Updater Modal controls
  $("#btn-open-setup").addEventListener("click", openSetupModal);
  $("#topbar-update-pill")?.addEventListener("click", openSetupModal);
  $("#btn-close-setup").addEventListener("click", () => $("#setup-modal").classList.add("hidden"));
  $("#btn-check-updates")?.addEventListener("click", () => refreshUpdaterPanel(true));
  $("#btn-apply-update")?.addEventListener("click", async () => {
    const msg = $("#updater-msg");
    msg.className = "export-status";
    msg.classList.remove("hidden");
    msg.textContent = "Pulling latest HFWToolkit update from GitHub main...";
    try {
      const res = await api("/api/updater/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      msg.className = res.status === "updated" ? "export-status done" : "export-status err";
      msg.textContent = res.message + ((res.steps || []).length ? ` (${res.steps.join(" • ")})` : "");
      refreshUpdaterPanel(true);
    } catch (e) {
      msg.className = "export-status err";
      msg.textContent = e.message;
    }
  });
  ["#cfg-auto-update", "#cfg-check-startup"].forEach((sel) => {
    $(sel)?.addEventListener("change", async () => {
      try {
        await api("/api/updater/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            auto_update: $("#cfg-auto-update")?.checked,
            check_updates_on_startup: $("#cfg-check-startup")?.checked,
          }),
        });
      } catch (_) {}
    });
  });

  $("#btn-save-cfg").addEventListener("click", async () => {
    const msg = $("#setup-msg");
    msg.className = "export-status";
    msg.classList.remove("hidden");
    msg.textContent = "Saving configuration & verifying game files...";
    try {
      const diag = await api("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          game_dir: $("#cfg-game-dir").value.trim(),
          blender_exe: $("#cfg-blender-exe").value.trim(),
          auto_update: $("#cfg-auto-update")?.checked,
          check_updates_on_startup: $("#cfg-check-startup")?.checked,
        }),
      });
      renderDoctorGrid(diag);
      await refreshTopbarStats();
      await loadMachines($("#machine-search").value);
      msg.className = "export-status done";
      msg.textContent = "Saved configuration to hfw_toolkit_config.json!";
    } catch (e) {
      msg.className = "export-status err";
      msg.textContent = e.message;
    }
  });

  $("#btn-autodetect-cfg").addEventListener("click", async () => {
    const msg = $("#setup-msg");
    msg.className = "export-status";
    msg.classList.remove("hidden");
    msg.textContent = "Scanning Steam Registry, libraryfolders.vdf, Epic manifests & drives...";
    try {
      const diag = await api("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ auto_detect: true }) });
      renderDoctorGrid(diag);
      await refreshTopbarStats();
      msg.className = "export-status done";
      msg.textContent = `Auto-detected HFW (${diag.game_dir}) and Blender (${diag.blender_exe})!`;
    } catch (e) {
      msg.className = "export-status err";
      msg.textContent = e.message;
    }
  });

  $("#btn-install-addon").addEventListener("click", async () => {
    const msg = $("#setup-msg");
    msg.className = "export-status";
    msg.classList.remove("hidden");
    msg.textContent = "Installing Blender N-Panel Add-on...";
    try {
      const res = await api("/api/install-addon", { method: "POST" });
      msg.className = "export-status done";
      msg.textContent = `Installed Blender Add-on to: ${(res.installed_paths || []).join(", ")}`;
    } catch (e) {
      msg.className = "export-status err";
      msg.textContent = e.message;
    }
  });

  $("#btn-open-exports").addEventListener("click", async () => {
    await api("/api/open-blend", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ open_folder: true }) });
  });

  // Global keyboard shortcuts: `/` focuses active navigator search, `Esc` closes modal
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      $("#setup-modal")?.classList.add("hidden");
      return;
    }
    if (e.key === "/" && !/INPUT|TEXTAREA|SELECT/.test((document.activeElement?.tagName || "").toUpperCase())) {
      e.preventDefault();
      const map = { machines: "#machine-search", characters: "#ch-search", assets: "#as-search", audio: "#au-search", texturesets: "#ts-search" };
      const inp = $(map[activeMode] || "#machine-search");
      inp?.focus();
      inp?.select();
    }
  });

  window.HFWStudio = {
    selectMachine,
    inspectCharacter,
    inspectAsset,
    selectMaterial,
    switchSidebarMode,
  };

  window.addEventListener("hashchange", handleHashRoute);
  init();
})();
