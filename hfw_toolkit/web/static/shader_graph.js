/**
 * Upgraded Interactive SVG Shader Node Graph Renderer (HFWToolkit v1.2).
 *
 * Features:
 *   - Translucent pipeline stage group backdrop frames
 *   - Embedded live texture thumbnails inside Texture Sampler nodes
 *   - Inline RGB color swatches inside Float4/CBuffer parameter nodes
 *   - Draggable nodes with real-time cubic Bezier wire rerouting + Reset Layout
 *   - Interactive corner Minimap with live draggable viewport rectangle
 *   - Live node search & jump (highlights matches and auto-centers camera)
 *   - Output channel filter pills (All / Albedo / Normal / Roughness / Emissive / Specular)
 *   - Floating Canvas Zoom HUD (-, 100%, +, Fit, Reset Layout, Export SVG, Fullscreen)
 *   - Rich Node Inspector Drawer with large texture preview, color swatch, upstream/downstream jump chips, & DXIL SSA lines
 */
(function () {
  const CAT_COLORS = {
    texture:   { header: "#0e7490", border: "#22d3ee", bg: "#0f1e2e" },
    param:     { header: "#92400e", border: "#f59e0b", bg: "#1f1912" },
    math:      { header: "#1d4ed8", border: "#60a5fa", bg: "#121b2e" },
    blend:     { header: "#6b21a8", border: "#c084fc", bg: "#1c1329" },
    specialty: { header: "#86198f", border: "#e879f9", bg: "#26112f" },
    global:    { header: "#0f766e", border: "#2dd4bf", bg: "#0f2222" },
    output:    { header: "#991b1b", border: "#f87171", bg: "#241316" },
  };

  function escHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function renderInteractiveNodeGraph(container, graphData) {
    container.innerHTML = "";
    const nodes = graphData.nodes || [];
    const edges = graphData.edges || [];
    const stageFrames = graphData.stage_frames || [];
    const isUE = graphData.mode === "unreal";
    const isDetailed = graphData.mode === "decima_detailed";
    const availTargets = graphData.available_targets || [];

    const rootWrap = document.createElement("div");
    rootWrap.className = "graph-studio-shell";
    container.appendChild(rootWrap);

    // 1. Top Toolbar: Legend, Node Search & Jump, Output Target Filter Pills
    const toolbar = document.createElement("div");
    toolbar.className = "graph-toolbar";
    const targetPillsHtml = availTargets.length
      ? `<div class="graph-filter-pills" style="display:flex;gap:4px;flex-wrap:wrap;align-items:center;">
           <span style="color:#8ea3c1;font-size:11px;margin-right:2px;">Target:</span>
           <button class="ch-btn active" data-tg="all">All</button>
           ${availTargets.map((tg) => `<button class="ch-btn" data-tg="${escHtml(tg)}">${escHtml(tg.charAt(0).toUpperCase() + tg.slice(1))}</button>`).join("")}
         </div>`
      : "";

    toolbar.innerHTML = `
      <div class="graph-legend">
        <span><i class="legend-dot" style="background:#22d3ee"></i>Texture</span>
        <span><i class="legend-dot" style="background:#f59e0b"></i>Param / UV</span>
        <span><i class="legend-dot" style="background:#e879f9"></i>Decima Specialty</span>
        <span><i class="legend-dot" style="background:#60a5fa"></i>DXIL Math</span>
        <span><i class="legend-dot" style="background:#c084fc"></i>Blend</span>
        <span><i class="legend-dot" style="background:#2dd4bf"></i>Global / Time</span>
        <span><i class="legend-dot" style="background:#f87171"></i>${isUE ? "UE5 Output" : "G-Buffer"}</span>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <div class="graph-search-wrap">
          <input type="text" class="graph-search-input" placeholder="Search nodes (detail, flow, f0, fresnel)..." />
          <span class="graph-search-count"></span>
          <button class="ch-btn" data-act="search-prev" title="Previous match">▲</button>
          <button class="ch-btn" data-act="search-next" title="Next match">▼</button>
        </div>
        ${targetPillsHtml}
        <button class="btn-ghost" data-act="clear" style="padding:4px 9px;font-size:11.5px;">Clear Selection</button>
      </div>
    `;
    rootWrap.appendChild(toolbar);

    // 2. Canvas Wrap + Floating Zoom HUD + Interactive Minimap
    const wrap = document.createElement("div");
    wrap.className = "graph-canvas-wrap";
    if (isDetailed) wrap.style.height = "620px";
    rootWrap.appendChild(wrap);

    const hud = document.createElement("div");
    hud.className = "graph-hud";
    hud.innerHTML = `
      <button class="graph-hud-btn" data-act="zoom-out" title="Zoom Out">−</button>
      <span class="graph-zoom-pct" title="Click to reset to 100%" data-act="zoom-100">100%</span>
      <button class="graph-hud-btn" data-act="zoom-in" title="Zoom In">+</button>
      <span class="graph-hud-sep"></span>
      <button class="graph-hud-btn" data-act="fit" title="Fit entire shader graph in view">Fit</button>
      <button class="graph-hud-btn" data-act="reset-layout" title="Restore auto-arranged node positions">Reset Layout</button>
      <button class="graph-hud-btn" data-act="export-svg" title="Download graph as scalable SVG">Export SVG</button>
      <button class="graph-hud-btn" data-act="fullscreen" title="Toggle Fullscreen Graph">⛶</button>
    `;
    wrap.appendChild(hud);

    const minimapWrap = document.createElement("div");
    minimapWrap.className = "graph-minimap";
    minimapWrap.title = "Click or drag to pan the shader graph";
    wrap.appendChild(minimapWrap);

    // 3. Rich Node Inspector Drawer
    const inspector = document.createElement("div");
    inspector.className = "node-inspector";
    inspector.innerHTML = `<div class="hint">Click any node to trace its upstream/downstream dataflow, preview bound textures, or inspect decoded DXIL math. Drag any node header to reorganize the graph.</div>`;
    rootWrap.appendChild(inspector);

    // 4. Compute clean initial column/row layout (with room for embedded texture thumbnails!)
    const NODE_W = isDetailed ? 274 : 258;
    const COL_GAP = isDetailed ? 356 : 336;
    const HEADER_H = 38;
    const PORT_STEP = 21;
    const THUMB_H = 74;

    const nodeById = {};
    nodes.forEach((n) => { nodeById[n.id] = n; });

    function nodeHasThumb(n) {
      return n.category === "texture" && Boolean(n.meta && n.meta.preview_url);
    }

    function computeAutoLayout() {
      const byCol = {};
      nodes.forEach((n) => {
        const c = n.col || 0;
        (byCol[c] = byCol[c] || []).push(n);
      });
      const outPos = {};
      const outDims = {};
      Object.keys(byCol).forEach((cKey) => {
        const colNodes = byCol[cKey].sort((a, b) => (a.row || 0) - (b.row || 0));
        let curY = 58; // leave top room for Stage Frame headers
        colNodes.forEach((n) => {
          const portCount = Math.max((n.inputs || []).length, (n.outputs || []).length, 1);
          const extraThumb = nodeHasThumb(n) ? THUMB_H : 0;
          const h = HEADER_H + 14 + portCount * PORT_STEP + extraThumb + 10;
          outPos[n.id] = { x: 48 + Number(cKey) * COL_GAP, y: curY };
          outDims[n.id] = { w: NODE_W, h, hasThumb: extraThumb > 0 };
          curY += h + 24;
        });
      });
      return { pos: outPos, dims: outDims };
    }

    const layout0 = computeAutoLayout();
    const pos = {};
    const dims = layout0.dims;
    Object.keys(layout0.pos).forEach((k) => { pos[k] = { ...layout0.pos[k] }; });

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("class", "hfw-shader-svg");

    const defs = document.createElementNS(svgNS, "defs");
    // Subtle checkerboard pattern for texture thumbnails
    const pat = document.createElementNS(svgNS, "pattern");
    pat.setAttribute("id", "hfw-checker-" + Math.random().toString(36).slice(2, 8));
    pat.setAttribute("width", "10");
    pat.setAttribute("height", "10");
    pat.setAttribute("patternUnits", "userSpaceOnUse");
    const pr1 = document.createElementNS(svgNS, "rect");
    pr1.setAttribute("width", "10"); pr1.setAttribute("height", "10"); pr1.setAttribute("fill", "#111827");
    const pr2 = document.createElementNS(svgNS, "rect");
    pr2.setAttribute("width", "5"); pr2.setAttribute("height", "5"); pr2.setAttribute("fill", "#1e293b");
    const pr3 = document.createElementNS(svgNS, "rect");
    pr3.setAttribute("x", "5"); pr3.setAttribute("y", "5"); pr3.setAttribute("width", "5"); pr3.setAttribute("height", "5"); pr3.setAttribute("fill", "#1e293b");
    pat.appendChild(pr1); pat.appendChild(pr2); pat.appendChild(pr3);
    defs.appendChild(pat);
    svg.appendChild(defs);

    const rootG = document.createElementNS(svgNS, "g");
    const frameG = document.createElementNS(svgNS, "g");
    const edgeG = document.createElementNS(svgNS, "g");
    const nodeG = document.createElementNS(svgNS, "g");
    rootG.appendChild(frameG);
    rootG.appendChild(edgeG);
    rootG.appendChild(nodeG);
    svg.appendChild(rootG);
    wrap.insertBefore(svg, hud);

    function getPortXY(nid, portName, side) {
      const n = nodeById[nid];
      if (!n || !pos[nid]) return { x: 0, y: 0 };
      const list = side === "in" ? (n.inputs || []) : (n.outputs || []);
      let idx = list.indexOf(portName);
      if (idx < 0) idx = 0;
      return {
        x: side === "in" ? pos[nid].x : pos[nid].x + dims[nid].w,
        y: pos[nid].y + HEADER_H + 18 + idx * PORT_STEP,
      };
    }

    function computeEdgePath(e) {
      const p1 = getPortXY(e.from, e.from_port, "out");
      const p2 = getPortXY(e.to, e.to_port, "in");
      const dx = Math.max(58, Math.abs(p2.x - p1.x) * 0.45);
      return `M ${p1.x} ${p1.y} C ${p1.x + dx} ${p1.y}, ${p2.x - dx} ${p2.y}, ${p2.x} ${p2.y}`;
    }

    // 5. Render Stage Backdrop Frames (updates cleanly when nodes move!)
    function renderStageFrames() {
      frameG.innerHTML = "";
      const framesToDraw = stageFrames.length
        ? stageFrames
        : (isUE
            ? [
                { cols: [0], title: "Coordinates & Parameters", color: "#f59e0b" },
                { cols: [1], title: "UE5 Texture Samplers", color: "#22d3ee" },
                { cols: [2, 3], title: "Material Expressions", color: "#c084fc" },
                { cols: [4], title: "Material Output", color: "#f87171" },
              ]
            : []);

      framesToDraw.forEach((fr) => {
        const colSet = new Set(fr.cols || []);
        const members = nodes.filter((n) => colSet.has(n.col || 0) && pos[n.id]);
        if (!members.length) return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        members.forEach((m) => {
          const p = pos[m.id];
          const d = dims[m.id];
          minX = Math.min(minX, p.x);
          minY = Math.min(minY, p.y);
          maxX = Math.max(maxX, p.x + d.w);
          maxY = Math.max(maxY, p.y + d.h);
        });
        const padX = 18, padTop = 34, padBot = 18;
        const g = document.createElementNS(svgNS, "g");
        const r = document.createElementNS(svgNS, "rect");
        r.setAttribute("x", String(minX - padX));
        r.setAttribute("y", String(minY - padTop));
        r.setAttribute("width", String(maxX - minX + padX * 2));
        r.setAttribute("height", String(maxY - minY + padTop + padBot));
        r.setAttribute("rx", "12");
        r.setAttribute("fill", fr.color || "#38bdf8");
        r.setAttribute("fill-opacity", "0.045");
        r.setAttribute("stroke", fr.color || "#38bdf8");
        r.setAttribute("stroke-opacity", "0.26");
        r.setAttribute("stroke-dasharray", "6 4");
        r.setAttribute("stroke-width", "1.2");
        g.appendChild(r);

        const lbl = document.createElementNS(svgNS, "text");
        lbl.setAttribute("x", String(minX - padX + 12));
        lbl.setAttribute("y", String(minY - padTop + 20));
        lbl.setAttribute("fill", fr.color || "#94a3b8");
        lbl.setAttribute("fill-opacity", "0.9");
        lbl.setAttribute("font-size", "11.5");
        lbl.setAttribute("font-weight", "700");
        lbl.setAttribute("letter-spacing", "0.4");
        lbl.textContent = (fr.title || "").toUpperCase();
        g.appendChild(lbl);
        frameG.appendChild(g);
      });
    }

    // 6. Draw Cubic Bezier Wires
    const edgeEls = [];
    edges.forEach((e) => {
      const path = document.createElementNS(svgNS, "path");
      path.setAttribute("d", computeEdgePath(e));
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", e.color || "#38bdf8");
      path.setAttribute("stroke-width", "2.1");
      path.setAttribute("stroke-opacity", "0.74");
      edgeG.appendChild(path);
      edgeEls.push({ edge: e, el: path });
    });

    function updateConnectedEdges(nid) {
      edgeEls.forEach(({ edge, el }) => {
        if (!nid || edge.from === nid || edge.to === nid) {
          el.setAttribute("d", computeEdgePath(edge));
        }
      });
    }

    // 7. Draw Interactive Nodes (with Thumbnails, Color Swatches & Drag Support!)
    const nodeEls = {};
    let dragNode = null;
    let didDragNode = false;

    nodes.forEach((n) => {
      const p = pos[n.id];
      const d = dims[n.id];
      const pal = CAT_COLORS[n.category] || CAT_COLORS.math;
      const m = n.meta || {};

      const g = document.createElementNS(svgNS, "g");
      g.style.cursor = "grab";
      g.setAttribute("transform", `translate(${p.x},${p.y})`);

      const rect = document.createElementNS(svgNS, "rect");
      rect.setAttribute("width", String(d.w));
      rect.setAttribute("height", String(d.h));
      rect.setAttribute("rx", "9");
      rect.setAttribute("fill", pal.bg);
      rect.setAttribute("stroke", pal.border);
      rect.setAttribute("stroke-width", n.category === "specialty" ? "2.0" : "1.4");
      g.appendChild(rect);

      const hdr = document.createElementNS(svgNS, "rect");
      hdr.setAttribute("width", String(d.w));
      hdr.setAttribute("height", String(HEADER_H));
      hdr.setAttribute("rx", "8");
      hdr.setAttribute("fill", pal.header);
      g.appendChild(hdr);

      // Optional inline RGB color swatch on CBuffer / Param nodes!
      let titleX = 10;
      if (m.color_swatch) {
        const sw = document.createElementNS(svgNS, "rect");
        sw.setAttribute("x", "9");
        sw.setAttribute("y", "9");
        sw.setAttribute("width", "18");
        sw.setAttribute("height", "18");
        sw.setAttribute("rx", "4");
        sw.setAttribute("fill", m.color_swatch);
        sw.setAttribute("stroke", "#ffffff");
        sw.setAttribute("stroke-opacity", "0.75");
        sw.setAttribute("stroke-width", "1.2");
        g.appendChild(sw);
        titleX = 34;
      }

      const title = document.createElementNS(svgNS, "text");
      title.setAttribute("x", String(titleX));
      title.setAttribute("y", "16");
      title.setAttribute("fill", "#ffffff");
      title.setAttribute("font-size", "12");
      title.setAttribute("font-weight", "700");
      title.textContent = (n.title || "").slice(0, m.color_swatch ? 29 : 34);
      g.appendChild(title);

      const sub = document.createElementNS(svgNS, "text");
      sub.setAttribute("x", String(titleX));
      sub.setAttribute("y", "31");
      sub.setAttribute("fill", "#e2e8f0");
      sub.setAttribute("font-size", "10");
      sub.textContent = (n.subtitle || "").slice(0, m.color_swatch ? 37 : 42);
      g.appendChild(sub);

      // Input ports
      (n.inputs || []).forEach((pname, idx) => {
        const cy = HEADER_H + 18 + idx * PORT_STEP;
        const c = document.createElementNS(svgNS, "circle");
        c.setAttribute("cx", "0");
        c.setAttribute("cy", String(cy));
        c.setAttribute("r", "4.3");
        c.setAttribute("fill", "#e2e8f0");
        c.setAttribute("stroke", pal.border);
        g.appendChild(c);

        const lbl = document.createElementNS(svgNS, "text");
        lbl.setAttribute("x", "10");
        lbl.setAttribute("y", String(cy + 4));
        lbl.setAttribute("fill", "#cbd5e1");
        lbl.setAttribute("font-size", "10.5");
        lbl.textContent = pname.slice(0, 22);
        g.appendChild(lbl);
      });

      // Output ports
      (n.outputs || []).forEach((pname, idx) => {
        const cy = HEADER_H + 18 + idx * PORT_STEP;
        const c = document.createElementNS(svgNS, "circle");
        c.setAttribute("cx", String(d.w));
        c.setAttribute("cy", String(cy));
        c.setAttribute("r", "4.3");
        c.setAttribute("fill", pal.border);
        g.appendChild(c);

        const lbl = document.createElementNS(svgNS, "text");
        lbl.setAttribute("x", String(d.w - 10));
        lbl.setAttribute("y", String(cy + 4));
        lbl.setAttribute("fill", "#e2e8f0");
        lbl.setAttribute("font-size", "10.5");
        lbl.setAttribute("text-anchor", "end");
        lbl.textContent = pname.slice(0, 22);
        g.appendChild(lbl);
      });

      // Embedded live texture thumbnail inside Texture Sampler nodes!
      if (d.hasThumb && m.preview_url) {
        const thumbY = d.h - THUMB_H - 6;
        const thumbW = 64;
        const thumbH = 64;
        const bgR = document.createElementNS(svgNS, "rect");
        bgR.setAttribute("x", "10");
        bgR.setAttribute("y", String(thumbY));
        bgR.setAttribute("width", String(thumbW));
        bgR.setAttribute("height", String(thumbH));
        bgR.setAttribute("rx", "5");
        bgR.setAttribute("fill", `url(#${pat.id})`);
        bgR.setAttribute("stroke", "#2c4366");
        g.appendChild(bgR);

        const img = document.createElementNS(svgNS, "image");
        img.setAttribute("x", "10");
        img.setAttribute("y", String(thumbY));
        img.setAttribute("width", String(thumbW));
        img.setAttribute("height", String(thumbH));
        img.setAttribute("preserveAspectRatio", "xMidYMid slice");
        img.setAttribute("href", m.preview_url);
        g.appendChild(img);

        const info1 = document.createElementNS(svgNS, "text");
        info1.setAttribute("x", "84");
        info1.setAttribute("y", String(thumbY + 20));
        info1.setAttribute("fill", "#7dd3fc");
        info1.setAttribute("font-size", "10.5");
        info1.setAttribute("font-weight", "600");
        info1.textContent = (m.role || "Texture").slice(0, 24);
        g.appendChild(info1);

        const info2 = document.createElementNS(svgNS, "text");
        info2.setAttribute("x", "84");
        info2.setAttribute("y", String(thumbY + 36));
        info2.setAttribute("fill", "#94a3b8");
        info2.setAttribute("font-size", "10");
        info2.textContent = m.resource_id ? `ID: ${m.resource_id}` : "";
        g.appendChild(info2);

        if (m.binding_hash) {
          const info3 = document.createElementNS(svgNS, "text");
          info3.setAttribute("x", "84");
          info3.setAttribute("y", String(thumbY + 51));
          info3.setAttribute("fill", "#64748b");
          info3.setAttribute("font-size", "9.5");
          info3.textContent = `Hash ${m.binding_hash}`;
          g.appendChild(info3);
        }
      }

      // Node drag + click selection
      g.addEventListener("mousedown", (ev) => {
        if (ev.button !== 0) return;
        ev.stopPropagation();
        didDragNode = false;
        dragNode = {
          id: n.id,
          startMouseX: ev.clientX,
          startMouseY: ev.clientY,
          origX: pos[n.id].x,
          origY: pos[n.id].y,
        };
        g.style.cursor = "grabbing";
      });

      g.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (didDragNode) {
          didDragNode = false;
          return;
        }
        highlightConnected(n.id);
      });

      nodeG.appendChild(g);
      nodeEls[n.id] = { node: n, g, rect };
    });

    renderStageFrames();

    // 8. Rich Inspector Drawer (Texture preview, Color swatch, Upstream/Downstream Jump Chips, DXIL lines)
    function renderNodeInspector(sel, upSet, downSet) {
      const m = sel.meta || {};
      const snips = Array.isArray(m.dxil_snippet) ? m.dxil_snippet : [];
      const consts = Array.isArray(m.constants) && m.constants.length ? m.constants : (Array.isArray(m.float4) ? m.float4 : []);
      const tgs = Array.isArray(m.targets) ? m.targets : [];
      const upIds = [...upSet].filter((id) => id !== sel.id);
      const downIds = [...downSet].filter((id) => id !== sel.id);
      const pal = CAT_COLORS[sel.category] || CAT_COLORS.math;

      const thumbHtml = m.preview_url
        ? `<div class="insp-thumb-box"><img src="${escHtml(m.preview_url.replace("max_dim=96", "max_dim=256"))}" alt="${escHtml(sel.title)}" /></div>`
        : "";
      const swatchHtml = m.color_swatch
        ? `<span class="swatch" style="background:${escHtml(m.color_swatch)};width:18px;height:18px;vertical-align:middle;"></span>`
        : "";

      inspector.innerHTML = `
        <div style="display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;">
          ${thumbHtml}
          <div style="flex:1;min-width:260px;">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
              <div>
                ${swatchHtml}
                <strong style="color:${pal.border};font-size:13.5px;">${escHtml(sel.title)}</strong>
                <span class="badge" style="margin-left:6px;">${escHtml(sel.kind)}</span>
                ${m.ssa_var ? `<span class="badge violet" style="margin-left:4px;">SSA ${escHtml(m.ssa_var)}</span>` : ""}
                ${m.resource_id ? `<span class="badge amber" style="margin-left:4px;">${escHtml(m.resource_id)}</span>` : ""}
                ${tgs.map((t) => `<span class="badge green" style="margin-left:4px;">${escHtml(t)}</span>`).join("")}
              </div>
              <span style="color:#8ea3c1;font-size:11.5px;">Trace: <strong>${upSet.size + downSet.size - 1}/${nodes.length}</strong> nodes (${upIds.length} upstream • ${downIds.length} downstream)</span>
            </div>
            ${m.formula ? `<div style="margin-top:6px;color:#dbeafe;font-size:12px;"><strong>Math / Behavior:</strong> <code>${escHtml(m.formula)}</code></div>` : ""}
            ${consts.length ? `<div style="margin-top:4px;color:#fcd34d;font-size:11.5px;"><strong>Decoded Constants:</strong> <code>[${consts.join(", ")}]</code></div>` : ""}
            ${(upIds.length || downIds.length) ? `
              <div style="margin-top:8px;display:flex;gap:10px;flex-wrap:wrap;font-size:11px;">
                ${upIds.length ? `<div><span style="color:#94a3b8;">Upstream:</span> ${upIds.slice(0, 6).map((id) => `<button class="ch-btn" data-jump-node="${escHtml(id)}">${escHtml((nodeById[id]?.title || id).slice(0, 22))}</button>`).join(" ")}</div>` : ""}
                ${downIds.length ? `<div><span style="color:#94a3b8;">Downstream:</span> ${downIds.slice(0, 6).map((id) => `<button class="ch-btn" data-jump-node="${escHtml(id)}">${escHtml((nodeById[id]?.title || id).slice(0, 22))}</button>`).join(" ")}</div>` : ""}
              </div>
            ` : ""}
            ${snips.length ? `<pre class="code-box" style="margin-top:8px;max-height:130px;padding:8px 10px;font-size:11.5px;">${escHtml(snips.join("\n"))}</pre>` : ""}
          </div>
        </div>
      `;
      inspector.querySelectorAll("[data-jump-node]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const jid = btn.dataset.jumpNode;
          highlightConnected(jid);
          centerOnNode(jid);
        });
      });
    }

    function highlightConnected(startId) {
      if (!nodeEls[startId]) return;
      const up = new Set([startId]);
      const down = new Set([startId]);
      let changed = true;
      while (changed) {
        changed = false;
        edges.forEach((e) => {
          if (up.has(e.to) && !up.has(e.from)) { up.add(e.from); changed = true; }
          if (down.has(e.from) && !down.has(e.to)) { down.add(e.to); changed = true; }
        });
      }
      const keep = new Set([...up, ...down]);
      Object.entries(nodeEls).forEach(([id, rec]) => {
        const on = keep.has(id);
        rec.g.setAttribute("opacity", on ? "1" : "0.16");
        rec.rect.setAttribute("stroke-width", id === startId ? "3.6" : (rec.node.category === "specialty" ? "2.0" : "1.4"));
      });
      edgeEls.forEach(({ edge, el }) => {
        const on = keep.has(edge.from) && keep.has(edge.to);
        el.setAttribute("stroke-opacity", on ? "1" : "0.07");
        el.setAttribute("stroke-width", on ? "3.1" : "1.4");
      });
      renderNodeInspector(nodeEls[startId].node, up, down);
    }

    function filterByTarget(tg) {
      toolbar.querySelectorAll("[data-tg]").forEach((b) => b.classList.toggle("active", b.dataset.tg === tg));
      if (tg === "all") {
        clearHighlight();
        return;
      }
      const keep = new Set();
      nodes.forEach((n) => {
        const tgs = (n.meta && n.meta.targets) || [];
        if (tgs.includes(tg)) keep.add(n.id);
      });
      Object.entries(nodeEls).forEach(([id, rec]) => {
        rec.g.setAttribute("opacity", keep.has(id) ? "1" : "0.15");
      });
      edgeEls.forEach(({ edge, el }) => {
        const on = keep.has(edge.from) && keep.has(edge.to);
        el.setAttribute("stroke-opacity", on ? "0.95" : "0.07");
        el.setAttribute("stroke-width", on ? "2.8" : "1.4");
      });
      inspector.innerHTML = `<div class="hint">Showing <strong>${keep.size}/${nodes.length}</strong> nodes contributing to <strong style="color:#22d3ee">${escHtml(tg.toUpperCase())}</strong>. Click any node for exact DXIL lines.</div>`;
    }

    function clearHighlight() {
      toolbar.querySelectorAll("[data-tg]").forEach((b) => b.classList.toggle("active", b.dataset.tg === "all"));
      Object.values(nodeEls).forEach((rec) => {
        rec.g.setAttribute("opacity", "1");
        rec.rect.setAttribute("stroke-width", rec.node.category === "specialty" ? "2.0" : "1.4");
      });
      edgeEls.forEach(({ el }) => {
        el.setAttribute("stroke-opacity", "0.74");
        el.setAttribute("stroke-width", "2.1");
      });
    }

    // 9. Pan, Zoom, Center-on-Node & Live Interactive Minimap
    let tx = 20, ty = 20, scale = 0.78;
    const zoomPctEl = hud.querySelector(".graph-zoom-pct");

    function getGraphBounds() {
      let minX = 0, minY = 0, maxX = 980, maxY = 540;
      nodes.forEach((n) => {
        const p = pos[n.id];
        const d = dims[n.id];
        if (!p || !d) return;
        minX = Math.min(minX, p.x - 30);
        minY = Math.min(minY, p.y - 45);
        maxX = Math.max(maxX, p.x + d.w + 60);
        maxY = Math.max(maxY, p.y + d.h + 60);
      });
      return { minX, minY, maxX, maxY, w: Math.max(1, maxX - minX), h: Math.max(1, maxY - minY) };
    }

    // Build Minimap SVG
    const MM_W = 186, MM_H = 112;
    const mmSvg = document.createElementNS(svgNS, "svg");
    mmSvg.setAttribute("width", String(MM_W));
    mmSvg.setAttribute("height", String(MM_H));
    const mmNodesG = document.createElementNS(svgNS, "g");
    const mmViewport = document.createElementNS(svgNS, "rect");
    mmViewport.setAttribute("fill", "rgba(34, 211, 238, 0.14)");
    mmViewport.setAttribute("stroke", "#22d3ee");
    mmViewport.setAttribute("stroke-width", "1.4");
    mmSvg.appendChild(mmNodesG);
    mmSvg.appendChild(mmViewport);
    minimapWrap.appendChild(mmSvg);

    let mmTransform = { s: 0.1, ox: 0, oy: 0, minX: 0, minY: 0 };

    function renderMinimap() {
      const b = getGraphBounds();
      const s = Math.min((MM_W - 10) / b.w, (MM_H - 10) / b.h);
      const ox = (MM_W - b.w * s) * 0.5;
      const oy = (MM_H - b.h * s) * 0.5;
      mmTransform = { s, ox, oy, minX: b.minX, minY: b.minY };

      mmNodesG.innerHTML = "";
      nodes.forEach((n) => {
        const p = pos[n.id], d = dims[n.id];
        if (!p || !d) return;
        const pal = CAT_COLORS[n.category] || CAT_COLORS.math;
        const r = document.createElementNS(svgNS, "rect");
        r.setAttribute("x", String(ox + (p.x - b.minX) * s));
        r.setAttribute("y", String(oy + (p.y - b.minY) * s));
        r.setAttribute("width", String(Math.max(3, d.w * s)));
        r.setAttribute("height", String(Math.max(2, d.h * s)));
        r.setAttribute("rx", "1.5");
        r.setAttribute("fill", pal.border);
        r.setAttribute("fill-opacity", "0.82");
        mmNodesG.appendChild(r);
      });
      updateMinimapViewport();
    }

    function updateMinimapViewport() {
      const cw = wrap.clientWidth || 1000;
      const ch = wrap.clientHeight || 580;
      const { s, ox, oy, minX, minY } = mmTransform;
      const vx = (-tx) / scale;
      const vy = (-ty) / scale;
      const vw = cw / scale;
      const vh = ch / scale;
      mmViewport.setAttribute("x", String(ox + (vx - minX) * s));
      mmViewport.setAttribute("y", String(oy + (vy - minY) * s));
      mmViewport.setAttribute("width", String(Math.max(8, vw * s)));
      mmViewport.setAttribute("height", String(Math.max(6, vh * s)));
    }

    function applyTransform() {
      rootG.setAttribute("transform", `translate(${tx},${ty}) scale(${scale})`);
      if (zoomPctEl) zoomPctEl.textContent = `${Math.round(scale * 100)}%`;
      updateMinimapViewport();
    }

    function fitView() {
      const b = getGraphBounds();
      const cw = wrap.clientWidth || 1080;
      const ch = wrap.clientHeight || 580;
      scale = Math.max(0.22, Math.min(0.96, Math.min((cw - 24) / b.w, (ch - 24) / b.h)));
      tx = 12 - b.minX * scale;
      ty = 12 - b.minY * scale;
      renderMinimap();
      applyTransform();
    }

    function zoomAroundCenter(factor) {
      const cw = wrap.clientWidth || 1000;
      const ch = wrap.clientHeight || 580;
      const mx = cw * 0.5, my = ch * 0.5;
      const ns = Math.max(0.18, Math.min(2.6, scale * factor));
      tx = mx - (mx - tx) * (ns / scale);
      ty = my - (my - ty) * (ns / scale);
      scale = ns;
      applyTransform();
    }

    function centerOnNode(nid) {
      const p = pos[nid], d = dims[nid];
      if (!p || !d) return;
      const cw = wrap.clientWidth || 1000;
      const ch = wrap.clientHeight || 580;
      if (scale < 0.72) scale = 0.85;
      tx = cw * 0.5 - (p.x + d.w * 0.5) * scale;
      ty = ch * 0.5 - (p.y + d.h * 0.5) * scale;
      applyTransform();
    }

    setTimeout(fitView, 30);

    // Minimap click/drag pan
    let mmDragging = false;
    function panFromMinimap(ev) {
      const r = minimapWrap.getBoundingClientRect();
      const mx = ev.clientX - r.left;
      const my = ev.clientY - r.top;
      const { s, ox, oy, minX, minY } = mmTransform;
      const gx = (mx - ox) / s + minX;
      const gy = (my - oy) / s + minY;
      const cw = wrap.clientWidth || 1000;
      const ch = wrap.clientHeight || 580;
      tx = cw * 0.5 - gx * scale;
      ty = ch * 0.5 - gy * scale;
      applyTransform();
    }
    minimapWrap.addEventListener("mousedown", (e) => { e.stopPropagation(); mmDragging = true; panFromMinimap(e); });

    // Canvas pan & node drag listeners
    let panDragging = false, lx = 0, ly = 0;
    wrap.addEventListener("mousedown", (e) => {
      if (e.target.closest(".graph-hud") || e.target.closest(".graph-minimap")) return;
      panDragging = true;
      lx = e.clientX;
      ly = e.clientY;
    });
    window.addEventListener("mouseup", () => {
      panDragging = false;
      mmDragging = false;
      if (dragNode && nodeEls[dragNode.id]) {
        nodeEls[dragNode.id].g.style.cursor = "grab";
        if (didDragNode) {
          renderStageFrames();
          renderMinimap();
        }
      }
      dragNode = null;
    });
    window.addEventListener("mousemove", (e) => {
      if (mmDragging) {
        panFromMinimap(e);
        return;
      }
      if (dragNode) {
        const dxScreen = e.clientX - dragNode.startMouseX;
        const dyScreen = e.clientY - dragNode.startMouseY;
        if (Math.abs(dxScreen) > 3 || Math.abs(dyScreen) > 3) didDragNode = true;
        pos[dragNode.id].x = dragNode.origX + dxScreen / scale;
        pos[dragNode.id].y = dragNode.origY + dyScreen / scale;
        nodeEls[dragNode.id].g.setAttribute("transform", `translate(${pos[dragNode.id].x},${pos[dragNode.id].y})`);
        updateConnectedEdges(dragNode.id);
        return;
      }
      if (!panDragging) return;
      tx += e.clientX - lx;
      ty += e.clientY - ly;
      lx = e.clientX;
      ly = e.clientY;
      applyTransform();
    });

    wrap.addEventListener("wheel", (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.13 : 0.88;
      const rect = wrap.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const ns = Math.max(0.18, Math.min(2.6, scale * factor));
      tx = mx - (mx - tx) * (ns / scale);
      ty = my - (my - ty) * (ns / scale);
      scale = ns;
      applyTransform();
    }, { passive: false });

    // 10. Node Search & Jump
    const searchInp = toolbar.querySelector(".graph-search-input");
    const searchCnt = toolbar.querySelector(".graph-search-count");
    let searchMatches = [];
    let searchIdx = -1;

    function runNodeSearch(jumpDelta = 0) {
      const q = (searchInp.value || "").trim().toLowerCase();
      if (!q) {
        searchMatches = [];
        searchIdx = -1;
        searchCnt.textContent = "";
        clearHighlight();
        return;
      }
      searchMatches = nodes.filter((n) => {
        const m = n.meta || {};
        const hay = `${n.title} ${n.subtitle} ${n.kind} ${m.formula || ""} ${m.role || ""} ${m.ssa_var || ""} ${m.resource_id || ""}`.toLowerCase();
        return hay.includes(q);
      });
      if (!searchMatches.length) {
        searchCnt.textContent = "0";
        return;
      }
      if (jumpDelta !== 0) {
        searchIdx = (searchIdx + jumpDelta + searchMatches.length) % searchMatches.length;
      } else {
        searchIdx = 0;
      }
      searchCnt.textContent = `${searchIdx + 1}/${searchMatches.length}`;
      const target = searchMatches[searchIdx];
      highlightConnected(target.id);
      centerOnNode(target.id);
    }

    searchInp.addEventListener("input", () => runNodeSearch(0));
    searchInp.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); runNodeSearch(e.shiftKey ? -1 : 1); }
      if (e.key === "Escape") { searchInp.value = ""; runNodeSearch(0); }
    });
    toolbar.querySelector('[data-act="search-prev"]').addEventListener("click", () => runNodeSearch(-1));
    toolbar.querySelector('[data-act="search-next"]').addEventListener("click", () => runNodeSearch(1));

    // 11. HUD Button Actions
    hud.querySelector('[data-act="zoom-in"]').addEventListener("click", (e) => { e.stopPropagation(); zoomAroundCenter(1.22); });
    hud.querySelector('[data-act="zoom-out"]').addEventListener("click", (e) => { e.stopPropagation(); zoomAroundCenter(0.82); });
    hud.querySelector('[data-act="zoom-100"]').addEventListener("click", (e) => { e.stopPropagation(); scale = 1.0; applyTransform(); });
    hud.querySelector('[data-act="fit"]').addEventListener("click", (e) => { e.stopPropagation(); fitView(); });
    hud.querySelector('[data-act="reset-layout"]').addEventListener("click", (e) => {
      e.stopPropagation();
      const fresh = computeAutoLayout();
      Object.keys(fresh.pos).forEach((k) => {
        pos[k] = { ...fresh.pos[k] };
        if (nodeEls[k]) nodeEls[k].g.setAttribute("transform", `translate(${pos[k].x},${pos[k].y})`);
      });
      updateConnectedEdges(null);
      renderStageFrames();
      fitView();
    });
    hud.querySelector('[data-act="export-svg"]').addEventListener("click", (e) => {
      e.stopPropagation();
      const b = getGraphBounds();
      const clone = svg.cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.setAttribute("viewBox", `${b.minX} ${b.minY} ${b.w} ${b.h}`);
      clone.setAttribute("width", String(Math.round(b.w)));
      clone.setAttribute("height", String(Math.round(b.h)));
      const innerG = clone.querySelector("g");
      if (innerG) innerG.setAttribute("transform", "");
      const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `hfw_shader_graph_${(graphData.mode || "decima")}.svg`;
      a.click();
    });
    hud.querySelector('[data-act="fullscreen"]').addEventListener("click", (e) => {
      e.stopPropagation();
      rootWrap.classList.toggle("graph-fullscreen");
      setTimeout(fitView, 60);
    });

    toolbar.querySelector('[data-act="clear"]').addEventListener("click", clearHighlight);
    toolbar.querySelectorAll("[data-tg]").forEach((b) => b.addEventListener("click", () => filterByTarget(b.dataset.tg)));
    wrap.addEventListener("click", (e) => {
      if (e.target.closest(".graph-hud") || e.target.closest(".graph-minimap")) return;
      clearHighlight();
    });
  }

  window.HFWShaderGraph = { render: renderInteractiveNodeGraph };
})();
