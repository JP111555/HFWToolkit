/**
 * Interactive Texture & RGBA Channel Lab + Material Parameters viewer.
 */
(function () {
  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function renderTextureLab(container, report, onOpenMachine) {
    const textures = report.primary_textures || [];
    const variables = report.primary_variables || [];
    const xrefs = report.texture_set_xrefs || {};

    let html = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div>
          <h3 style="margin:0;">Bound Textures &amp; Channel Packing (${textures.length})</h3>
          <span style="color:#8ea3c1;font-size:12px;">Click R / G / B / A / Normal buttons on any card to isolate individual packed channels in real time.</span>
        </div>
      </div>
      <div class="tex-grid" id="tex-lab-grid"></div>
      <h3 style="margin:22px 0 8px;">Material Shader Variables / Parameters (${variables.length})</h3>
      <table class="data-table">
        <thead><tr><th>#</th><th>Parameter Name / Hint</th><th>Type</th><th>Value (XYZW)</th><th>Color Swatch</th><th>Binding Hash</th></tr></thead>
        <tbody>
    `;

    variables.forEach((v) => {
      const f = v.float4 || [0, 0, 0, 0];
      const isColor = f.slice(0, 3).every((x) => x >= 0 && x <= 1.001) && (f[0] !== f[1] || f[1] !== f[2]);
      const swatch = isColor
        ? `<span class="swatch" style="background:rgb(${Math.round(f[0]*255)},${Math.round(f[1]*255)},${Math.round(f[2]*255)})"></span>`
        : `<span style="color:#64748b">—</span>`;
      html += `
        <tr>
          <td>${v.index}</td>
          <td><strong>${esc(v.name)}</strong></td>
          <td><span class="badge">${esc(v.type)}</span></td>
          <td><code>(${f.map((x) => Number(x).toFixed(3)).join(", ")})</code></td>
          <td>${swatch}</td>
          <td><code>${esc(v.binding_hash)}</code></td>
        </tr>
      `;
    });
    html += `</tbody></table>`;
    container.innerHTML = html;

    const grid = container.querySelector("#tex-lab-grid");
    textures.forEach((tb) => {
      const card = document.createElement("div");
      card.className = "tex-card";
      const rid = tb.resource_id;
      const isTS = tb.resource_type === "TextureSet";
      const chMap = tb.target_channels || {};
      const chBadges = Object.entries(chMap)
        .map(([lane, spec]) => `<span class="badge green">${esc(lane)}: ${esc(spec)}</span>`)
        .join(" ");

      const initialUrl = rid ? `/api/texture/${encodeURIComponent(rid)}/image?mode=rgba&max_dim=512` : "";
      const pathsHtml = Object.entries(tb.source_paths || {})
        .map(([k, p]) => `<div class="tex-path"><strong style="color:#94a3b8">${esc(k)}:</strong> ${esc(p)}</div>`)
        .join("");

      const users = (rid && xrefs[rid]) || [];
      const xrefHtml = users.length
        ? `<div style="margin-top:8px;padding-top:6px;border-top:1px solid #25324a;font-size:11.5px;">
             <span style="color:#8ea3c1;">Shared by ${users.length} entity/machines:</span>
             <div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px;">
               ${users.slice(0, 8).map((u) => `<button class="ch-btn" data-jump-entity="${u.group_id}:${u.object_index}">${esc(u.display_name)}</button>`).join("")}
             </div>
           </div>`
        : "";

      card.innerHTML = `
        <div class="tex-preview-wrap">
          ${initialUrl ? `<img loading="lazy" src="${initialUrl}" alt="${esc(tb.name)}" />` : `<span style="color:#64748b">Unresolved texture</span>`}
        </div>
        <div class="tex-channels">
          <button class="ch-btn active" data-mode="rgba">RGBA</button>
          <button class="ch-btn" data-mode="rgb">RGB</button>
          <button class="ch-btn" data-mode="r">R${chMap.R ? " (" + esc(chMap.R.split(".")[0]) + ")" : ""}</button>
          <button class="ch-btn" data-mode="g">G${chMap.G ? " (" + esc(chMap.G.split(".")[0]) + ")" : ""}</button>
          <button class="ch-btn" data-mode="b">B${chMap.B ? " (" + esc(chMap.B.split(".")[0]) + ")" : ""}</button>
          <button class="ch-btn" data-mode="a">A${chMap.A ? " (" + esc(chMap.A.split(".")[0]) + ")" : ""}</button>
          <button class="ch-btn" data-mode="normal_dx">Normal DX</button>
          <button class="ch-btn" data-mode="normal_gl">Normal GL</button>
          <button class="ch-btn" data-tile="1">2×2 Tile</button>
        </div>
        <div class="tex-body">
          <div class="tex-name">${esc(tb.name)}</div>
          <div class="tag-row" style="margin:5px 0;">
            <span class="badge">${esc(tb.role)}</span>
            <span class="badge violet">${esc(tb.resource_type || "?")} ${esc(rid || "")}</span>
            ${tb.is_detail ? `<span class="badge amber">Detail / Micro-Surface</span>` : ""}
            ${chBadges}
          </div>
          ${pathsHtml || `<div class="tex-path">Standalone library texture • hash <code>${esc(tb.binding_hash)}</code></div>`}
          ${xrefHtml}
        </div>
      `;

      const wrap = card.querySelector(".tex-preview-wrap");
      const img = wrap.querySelector("img");
      let curUrl = initialUrl;

      card.querySelectorAll(".ch-btn[data-mode]").forEach((btn) => {
        btn.addEventListener("click", () => {
          if (!rid || !img) return;
          card.querySelectorAll(".ch-btn[data-mode]").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          const m = btn.dataset.mode;
          curUrl = `/api/texture/${encodeURIComponent(rid)}/image?mode=${m}&max_dim=512`;
          img.src = curUrl;
          if (wrap.classList.contains("tiled")) {
            wrap.style.backgroundImage = `url('${curUrl}')`;
          }
        });
      });

      const tileBtn = card.querySelector(".ch-btn[data-tile]");
      tileBtn.addEventListener("click", () => {
        const tiled = wrap.classList.toggle("tiled");
        tileBtn.classList.toggle("active", tiled);
        wrap.style.backgroundImage = tiled ? `url('${curUrl}')` : "";
      });

      card.querySelectorAll("[data-jump-entity]").forEach((b) => {
        b.addEventListener("click", () => onOpenMachine && onOpenMachine(b.dataset.jumpEntity));
      });

      grid.appendChild(card);
    });
  }

  window.HFWTextureLab = { render: renderTextureLab };
})();
