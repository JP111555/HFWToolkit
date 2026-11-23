/**
 * Interactive zoomable, pannable SVG Shader Node Graph renderer.
 * Supports:
 *   1. Detailed DXIL SSA Math & Decima Engine Specialty Node Graph (Multiply, Add, Lerp, Sin, Frac,
 *      23-Slice Detail Array, Dual-Phase Flow Map, POM Raymarch, Worn-Metal F0, Octahedral Normal, etc.)
 *   2. High-Level Decima Shader Flow Overview
 *   3. Unreal Engine 5 Animated Material Graph
 * Clicking any node highlights its full upstream & downstream chain and opens the DXIL Node Inspector!
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
    const isUE = graphData.mode === "unreal";
    const isDetailed = graphData.mode === "decima_detailed";
    const availTargets = graphData.available_targets || [];

    const toolbar = document.createElement("div");
    toolbar.className = "graph-toolbar";
    const targetPillsHtml = availTargets.length
      ? `<div class="graph-filter-pills" style="display:flex;gap:4px;flex-wrap:wrap;align-items:center;">
           <span style="color:#8ea3c1;font-size:11px;margin-right:2px;">Filter Output:</span>
           <button class="ch-btn active" data-tg="all">All</button>
           ${availTargets.map((tg) => `<button class="ch-btn" data-tg="${escHtml(tg)}">${escHtml(tg.charAt(0).toUpperCase() + tg.slice(1))}</button>`).join("")}
         </div>`
      : "";

    toolbar.innerHTML = `
      <div class="graph-legend">
        <span><i class="legend-dot" style="background:#22d3ee"></i>Texture Sampler</span>
        <span><i class="legend-dot" style="background:#f59e0b"></i>UV / Parameter</span>
        <span><i class="legend-dot" style="background:#e879f9"></i>Decima Specialty Node</span>
        <span><i class="legend-dot" style="background:#60a5fa"></i>DXIL Math Op</span>
        <span><i class="legend-dot" style="background:#c084fc"></i>Lerp / Branch</span>
        <span><i class="legend-dot" style="background:#2dd4bf"></i>Global Engine / Time</span>
        <span><i class="legend-dot" style="background:#f87171"></i>${isUE ? "UE5 Material Output" : "Decima G-Buffer Output"}</span>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        ${targetPillsHtml}
        <button class="btn-ghost" data-act="fit">Reset View</button>
        <button class="btn-ghost" data-act="clear">Clear Highlight</button>
      </div>
    `;
    container.appendChild(toolbar);

    const wrap = document.createElement("div");
    wrap.className = "graph-canvas-wrap";
    if (isDetailed) {
      wrap.style.height = "600px";
    }
    container.appendChild(wrap);

    const inspector = document.createElement("div");
    inspector.className = "node-inspector";
    inspector.innerHTML = `<div class="hint">Click any node in the graph to trace its full upstream/downstream chain and inspect its decoded math formula, constants, and compiled DXIL SSA lines.</div>`;
    container.appendChild(inspector);

    // Compute clean column/row layout
    const NODE_W = isDetailed ? 264 : 248;
    const COL_GAP = isDetailed ? 336 : 320;
    const byCol = {};
    nodes.forEach((n) => {
      const c = n.col || 0;
      (byCol[c] = byCol[c] || []).push(n);
    });

    const pos = {};
    const dims = {};
    Object.keys(byCol).forEach((cKey) => {
      const colNodes = byCol[cKey].sort((a, b) => (a.row || 0) - (b.row || 0));
      let curY = 36;
      colNodes.forEach((n) => {
        const portCount = Math.max((n.inputs || []).length, (n.outputs || []).length, 1);
        const h = 50 + portCount * 21 + 12;
        pos[n.id] = { x: 36 + Number(cKey) * COL_GAP, y: curY };
        dims[n.id] = { w: NODE_W, h };
        curY += h + 22;
      });
    });

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    const rootG = document.createElementNS(svgNS, "g");
    const edgeG = document.createElementNS(svgNS, "g");
    const nodeG = document.createElementNS(svgNS, "g");
    rootG.appendChild(edgeG);
    rootG.appendChild(nodeG);
    svg.appendChild(rootG);
    wrap.appendChild(svg);

    function getPortXY(nid, portName, side) {
      const n = nodes.find((x) => x.id === nid);
      if (!n || !pos[nid]) return { x: 0, y: 0 };
      const list = side === "in" ? (n.inputs || []) : (n.outputs || []);
      let idx = list.indexOf(portName);
      if (idx < 0) idx = 0;
      return {
        x: side === "in" ? pos[nid].x : pos[nid].x + dims[nid].w,
        y: pos[nid].y + 54 + idx * 21,
      };
    }

    // Draw cubic bezier wires
    const edgeEls = [];
    edges.forEach((e) => {
      const p1 = getPortXY(e.from, e.from_port, "out");
      const p2 = getPortXY(e.to, e.to_port, "in");
      const dx = Math.max(55, Math.abs(p2.x - p1.x) * 0.46);
      const d = `M ${p1.x} ${p1.y} C ${p1.x + dx} ${p1.y}, ${p2.x - dx} ${p2.y}, ${p2.x} ${p2.y}`;

      const path = document.createElementNS(svgNS, "path");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", e.color || "#38bdf8");
      path.setAttribute("stroke-width", "2.1");
      path.setAttribute("stroke-opacity", "0.74");
      edgeG.appendChild(path);
      edgeEls.push({ edge: e, el: path });
    });

    // Draw nodes
    const nodeEls = {};
    nodes.forEach((n) => {
      const p = pos[n.id];
      const d = dims[n.id];
      const pal = CAT_COLORS[n.category] || CAT_COLORS.math;

      const g = document.createElementNS(svgNS, "g");
      g.style.cursor = "pointer";
      g.setAttribute("transform", `translate(${p.x},${p.y})`);

      const rect = document.createElementNS(svgNS, "rect");
      rect.setAttribute("width", d.w);
      rect.setAttribute("height", d.h);
      rect.setAttribute("rx", "8");
      rect.setAttribute("fill", pal.bg);
      rect.setAttribute("stroke", pal.border);
      rect.setAttribute("stroke-width", n.category === "specialty" ? "2.0" : "1.4");
      g.appendChild(rect);

      const hdr = document.createElementNS(svgNS, "rect");
      hdr.setAttribute("width", d.w);
      hdr.setAttribute("height", "36");
      hdr.setAttribute("rx", "7");
      hdr.setAttribute("fill", pal.header);
      g.appendChild(hdr);

      const title = document.createElementNS(svgNS, "text");
      title.setAttribute("x", "10");
      title.setAttribute("y", "16");
      title.setAttribute("fill", "#ffffff");
      title.setAttribute("font-size", "12");
      title.setAttribute("font-weight", "700");
      title.textContent = (n.title || "").slice(0, 34);
      g.appendChild(title);

      const sub = document.createElementNS(svgNS, "text");
      sub.setAttribute("x", "10");
      sub.setAttribute("y", "30");
      sub.setAttribute("fill", "#e2e8f0");
      sub.setAttribute("font-size", "10");
      sub.textContent = (n.subtitle || "").slice(0, 42);
      g.appendChild(sub);

      // Input ports
      (n.inputs || []).forEach((pname, idx) => {
        const cy = 54 + idx * 21;
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
        const cy = 54 + idx * 21;
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

      g.addEventListener("click", (ev) => {
        ev.stopPropagation();
        highlightConnected(n.id);
      });
      nodeG.appendChild(g);
      nodeEls[n.id] = { node: n, g, rect };
    });

    function renderNodeInspector(sel, connectedCount) {
      const m = sel.meta || {};
      const snips = Array.isArray(m.dxil_snippet) ? m.dxil_snippet : [];
      const consts = Array.isArray(m.constants) && m.constants.length ? m.constants : (Array.isArray(m.float4) ? m.float4 : []);
      const tgs = Array.isArray(m.targets) ? m.targets : [];
      inspector.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
          <div>
            <strong style="color:${(CAT_COLORS[sel.category] || CAT_COLORS.math).border};font-size:13.5px;">${escHtml(sel.title)}</strong>
            <span class="badge" style="margin-left:6px;">${escHtml(sel.kind)}</span>
            ${m.ssa_var ? `<span class="badge violet" style="margin-left:4px;">SSA ${escHtml(m.ssa_var)}</span>` : ""}
            ${tgs.map((t) => `<span class="badge green" style="margin-left:4px;">${escHtml(t)}</span>`).join("")}
          </div>
          <span style="color:#8ea3c1;font-size:11.5px;">Connected path: <strong>${connectedCount}/${nodes.length}</strong> nodes</span>
        </div>
        ${m.formula ? `<div style="margin-top:6px;color:#dbeafe;font-size:12px;"><strong>Math / Behavior:</strong> <code>${escHtml(m.formula)}</code></div>` : ""}
        ${consts.length ? `<div style="margin-top:4px;color:#fcd34d;font-size:11.5px;"><strong>Decoded Constants:</strong> <code>[${consts.join(", ")}]</code></div>` : ""}
        ${snips.length ? `<pre class="code-box" style="margin-top:8px;max-height:130px;padding:8px 10px;font-size:11.5px;">${escHtml(snips.join("\n"))}</pre>` : ""}
      `;
    }

    function highlightConnected(startId) {
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
        rec.g.setAttribute("opacity", on ? "1" : "0.18");
        rec.rect.setAttribute("stroke-width", id === startId ? "3.4" : (rec.node.category === "specialty" ? "2.0" : "1.4"));
      });
      edgeEls.forEach(({ edge, el }) => {
        const on = keep.has(edge.from) && keep.has(edge.to);
        el.setAttribute("stroke-opacity", on ? "1" : "0.08");
        el.setAttribute("stroke-width", on ? "3.0" : "1.4");
      });
      renderNodeInspector(nodeEls[startId].node, keep.size);
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
        rec.g.setAttribute("opacity", keep.has(id) ? "1" : "0.16");
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

    // Pan & Zoom state
    let tx = 20, ty = 20, scale = 0.78;
    function applyTransform() {
      rootG.setAttribute("transform", `translate(${tx},${ty}) scale(${scale})`);
    }
    function fitView() {
      const maxX = Math.max(960, ...nodes.map((n) => (pos[n.id]?.x || 0) + NODE_W + 60));
      const maxY = Math.max(520, ...nodes.map((n) => (pos[n.id]?.y || 0) + (dims[n.id]?.h || 100) + 60));
      const cw = wrap.clientWidth || 1080;
      const ch = wrap.clientHeight || 580;
      scale = Math.min(0.98, Math.min(cw / maxX, ch / maxY));
      tx = 16;
      ty = 16;
      applyTransform();
    }
    setTimeout(fitView, 25);

    let dragging = false, lx = 0, ly = 0;
    wrap.addEventListener("mousedown", (e) => { dragging = true; lx = e.clientX; ly = e.clientY; });
    window.addEventListener("mouseup", () => { dragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      tx += e.clientX - lx;
      ty += e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      applyTransform();
    });
    wrap.addEventListener("wheel", (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 0.89;
      const rect = wrap.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const ns = Math.max(0.20, Math.min(2.4, scale * factor));
      tx = mx - (mx - tx) * (ns / scale);
      ty = my - (my - ty) * (ns / scale);
      scale = ns;
      applyTransform();
    }, { passive: false });

    toolbar.querySelector('[data-act="fit"]').addEventListener("click", fitView);
    toolbar.querySelector('[data-act="clear"]').addEventListener("click", clearHighlight);
    toolbar.querySelectorAll("[data-tg]").forEach((b) => b.addEventListener("click", () => filterByTarget(b.dataset.tg)));
    wrap.addEventListener("click", clearHighlight);
  }

  window.HFWShaderGraph = { render: renderInteractiveNodeGraph };
})();
