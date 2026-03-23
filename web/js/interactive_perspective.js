/**
 * Interactive Perspective Mixer – ComfyUI Frontend Extension
 *
 * Adds a full-screen editor to the InteractivePerspectiveMixer node.
 * The editor lets you drag four corner handles on top of the background
 * image to define a perspective quad.  The layer image is previewed with
 * a real-time CSS matrix3d perspective warp.
 *
 * Coordinate convention (stored in corners_input):
 *   Four tuples [(x,y), …] in TL → TR → BR → BL order.
 *   Values are RELATIVE (0-1) to the background image dimensions.
 *   Values outside [0,1] are allowed for vanishing-point support.
 */

import { app } from "../../../scripts/app.js";
console.log("The newest custom Js Loaded!")
// ─── Constants ───────────────────────────────────────────────────────────────
const NODE_TYPES     = new Set([
    "InteractivePerspectiveMixer",
]);
const HANDLE_RADIUS  = 9;          // px – drawn handle circle radius
const HANDLE_COLORS  = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]; // TL TR BR BL
const HANDLE_LABELS  = ["TL", "TR", "BR", "BL"];

// ─── Math helpers ────────────────────────────────────────────────────────────

/** Solve A x = b via Gauss-Jordan elimination.  A is n×n, b is length-n. */
function gaussJordan(A, b) {
    const n = A.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
        let maxRow = col;
        for (let r = col + 1; r < n; r++)
            if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;
        [M[col], M[maxRow]] = [M[maxRow], M[col]];
        const pivot = M[col][col];
        if (Math.abs(pivot) < 1e-12) continue;
        for (let r = 0; r < n; r++) {
            if (r === col) continue;
            const f = M[r][col] / pivot;
            for (let k = col; k <= n; k++) M[r][k] -= f * M[col][k];
        }
    }
    return M.map((row, i) => row[n] / row[i]);
}

/**
 * Compute 3×3 homography matrix H such that dst[i] ≈ H * src[i]
 * (4 point correspondences, homogeneous coordinates, h33 = 1).
 * Returns H as a flat row-major 9-element array.
 */
function computeHomography(src, dst) {
    const rows = [], rhs = [];
    for (let i = 0; i < 4; i++) {
        const [x, y] = src[i], [u, v] = dst[i];
        rows.push([-x, -y, -1,  0,  0,  0, u * x, u * y]); rhs.push(-u);
        rows.push([ 0,  0,  0, -x, -y, -1, v * x, v * y]); rhs.push(-v);
    }
    const h = gaussJordan(rows, rhs);
    return [h[0], h[1], h[2],
            h[3], h[4], h[5],
            h[6], h[7], 1.0];
}

/**
 * Convert a flat 3×3 homography (row-major) to a CSS `matrix3d(…)` string.
 * The mapping works for 2-D plane (z = 0) transforms.
 * Column-major order required by CSS.
 */
function homographyToCSSMatrix3d(H) {
    const [a, b, c,
           d, e, f,
           g, h, i] = H;
    // Build 4×4 column-major matrix (z row/col kept as identity pass-through)
    const m = [
        a,  d,  0,  g,
        b,  e,  0,  h,
        0,  0,  1,  0,
        c,  f,  0,  i,
    ];
    return `matrix3d(${m.join(",")})`;
}

// ─── Image URL helpers ───────────────────────────────────────────────────────

/**
 * Read a URL from node.imgs at the given index.
 * node.imgs entries may be HTMLImageElement, plain string, or {src/url} objects.
 */
function getNodeImgUrl(node, idx) {
    const img = node.imgs?.[idx];
    if (!img) return null;
    if (img instanceof HTMLImageElement) return img.src || null;
    if (typeof img === "string") return img || null;
    return img?.src ?? img?.url ?? null;
}

/**
 * Fallback: get a URL from a directly-connected LoadImage widget.
 * Only used on first open (before the node has ever been executed).
 */
function getLoadImageUrl(node, inputName) {
    const input = node.inputs?.find(i => i.name === inputName);
    const link  = input?.link;
    if (link == null) return null;
    const info = app.graph.links[link];
    if (!info) return null;
    const src = app.graph.getNodeById(info.origin_id);
    if (!src || src.type !== "LoadImage") return null;
    const w = src.widgets?.find(w => w.name === "image");
    if (w?.value) return `/view?filename=${encodeURIComponent(w.value)}&type=input`;
    return null;
}

// ─── Default corners ─────────────────────────────────────────────────────────

function defaultCorners() {
    // Centred 80 % rectangle – relative coords
    return [
        { x: 0.1, y: 0.1 }, // TL
        { x: 0.9, y: 0.1 }, // TR
        { x: 0.9, y: 0.9 }, // BR
        { x: 0.1, y: 0.9 }, // BL
    ];
}

function parseCorners(raw) {
    if (!raw || !raw.trim()) return null;
    // Accept [(x,y),(x,y),(x,y),(x,y)] format (written by this editor and corners_output)
    try {
        const s = raw.trim();
        // Use a simple regex-free approach: eval-safe parse via JSON after converting tuples
        const jsonStr = s
            .replace(/\(/g, "[")
            .replace(/\)/g, "]")
            .replace(/'/g, '"');
        const arr = JSON.parse(jsonStr);
        if (Array.isArray(arr) && arr.length === 4 &&
            arr.every(pt => Array.isArray(pt) && pt.length === 2 &&
                typeof pt[0] === "number" && typeof pt[1] === "number"))
            return arr.map(pt => ({ x: pt[0], y: pt[1] }));
    } catch (_) { /* ignore */ }
    return null;
}

// ─── Main editor ─────────────────────────────────────────────────────────────

function openEditor(node) {
    /* ── find corners_input widget ─────────────────────────────────────── */
    const cornersWidget = node.widgets?.find(w => w.name === "corners_input");
    let corners = parseCorners(cornersWidget?.value) ?? defaultCorners();

    /* ── overlay & dialog DOM ──────────────────────────────────────────── */
    const overlay = document.createElement("div");
    overlay.id = "ipm-overlay";
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", zIndex: "9999",
        background: "rgba(0,0,0,0.75)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontFamily: "sans-serif",
    });

    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#1a1a2e", borderRadius: "10px",
        padding: "16px", boxShadow: "0 8px 32px #000a",
        display: "flex", flexDirection: "column", gap: "10px",
        maxWidth: "92vw", maxHeight: "92vh",
        userSelect: "none",
    });

    /* ─ header ─ */
    const header = document.createElement("div");
    Object.assign(header.style, {
        color: "#eee", fontSize: "15px", fontWeight: "bold",
        display: "flex", justifyContent: "space-between", alignItems: "center",
    });
    header.innerHTML = `
        <span>🎯 Interactive Perspective Editor</span>
        <span style="font-size:11px;color:#aaa">
            Drag handle = reshape &nbsp;|&nbsp; Drag inside quad = move layer &nbsp;|&nbsp;
            Drag empty = pan &nbsp;|&nbsp; Scroll = zoom &nbsp;|&nbsp; Z = reset view
        </span>`;

    /* ─ canvas container ─ */
    const canvasContainer = document.createElement("div");
    Object.assign(canvasContainer.style, {
        position: "relative", overflow: "hidden",
        width: "800px", height: "600px",
        background: "#111", borderRadius: "6px",
        cursor: "default",
        flexShrink: "0",
    });

    const canvas = document.createElement("canvas");
    canvas.width  = 800;
    canvas.height = 600;
    Object.assign(canvas.style, {
        position: "absolute", left: "0", top: "0",
        width: "100%", height: "100%",
    });

    /* layer preview img – perspective-warped via CSS matrix3d */
    const layerPreview = document.createElement("img");
    Object.assign(layerPreview.style, {
        position: "absolute", left: "0", top: "0",
        transformOrigin: "0 0",
        opacity: "0.65",
        pointerEvents: "none",
    });

    canvasContainer.appendChild(canvas);
    canvasContainer.appendChild(layerPreview);

    /* ─ controls bar ─ */
    const controls = document.createElement("div");
    Object.assign(controls.style, {
        display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap",
    });

    function makeBtn(label, color) {
        const b = document.createElement("button");
        b.textContent = label;
        Object.assign(b.style, {
            padding: "7px 18px", borderRadius: "5px",
            border: "none", cursor: "pointer", fontWeight: "bold",
            background: color, color: "#fff", fontSize: "13px",
        });
        return b;
    }

    const btnReset  = makeBtn("Reset Corners", "#555");
    const btnCancel = makeBtn("Cancel",        "#c0392b");
    const btnApply  = makeBtn("✓ Apply",       "#27ae60");

    const hint = document.createElement("span");
    Object.assign(hint.style, { color: "#888", fontSize: "11px", flexGrow: "1" });
    hint.textContent = "Handles can be dragged outside the image for vanishing-point effects.";

    controls.append(hint, btnReset, btnCancel, btnApply);
    dialog.append(header, canvasContainer, controls);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    /* ── viewport state ──────────────────────────────────────────────────── */
    let viewScale = 1.0;     // current zoom level
    let viewTx    = 0.0;     // translate x (canvas pixels)
    let viewTy    = 0.0;     // translate y

    let bgImg     = null;    // HTMLImageElement for background
    let layerImg  = null;    // HTMLImageElement for layer
    let bgNatW    = 512, bgNatH = 512;
    let layerNatW = 256, layerNatH = 256;

    /* ── helper: image-space ↔ canvas-space ─────────────────────────────── */
    // rel   = corner's {x,y} in 0-1 bg-image coords
    // world = actual bg-image pixel coords (0..bgNatW, 0..bgNatH)
    // screen= canvas px after zoom+pan

    function worldToScreen(wx, wy) {
        return {
            x: wx * viewScale + viewTx,
            y: wy * viewScale + viewTy,
        };
    }
    function screenToWorld(sx, sy) {
        return {
            x: (sx - viewTx) / viewScale,
            y: (sy - viewTy) / viewScale,
        };
    }
    function cornerToScreen(c) {
        return worldToScreen(c.x * bgNatW, c.y * bgNatH);
    }
    function screenToCorner(sx, sy) {
        const w = screenToWorld(sx, sy);
        return { x: w.x / bgNatW, y: w.y / bgNatH };
    }

    /* ── fit background in the canvas view ──────────────────────────────── */
    function fitBackground() {
        const pad = 40;
        viewScale = Math.min(
            (canvas.width  - pad * 2) / bgNatW,
            (canvas.height - pad * 2) / bgNatH,
        );
        viewTx = (canvas.width  - bgNatW * viewScale) / 2;
        viewTy = (canvas.height - bgNatH * viewScale) / 2;
    }

    /* ── render ─────────────────────────────────────────────────────────── */
    function updateLayerPreview() {
        if (!layerImg || !layerImg.complete || !layerImg.naturalWidth) {
            layerPreview.style.display = "none";
            return;
        }

        // Source: corners of layer image in native pixels
        const src = [
            [0,          0          ],
            [layerNatW,  0          ],
            [layerNatW,  layerNatH  ],
            [0,          layerNatH  ],
        ];

        // Destination: screen positions of the 4 handles
        const dst = corners.map(c => {
            const s = cornerToScreen(c);
            return [s.x, s.y];
        });

        const H = computeHomography(src, dst);
        const cssM = homographyToCSSMatrix3d(H);

        layerPreview.src    = layerImg.src;
        layerPreview.style.display = "block";
        layerPreview.style.width  = layerNatW + "px";
        layerPreview.style.height = layerNatH + "px";
        layerPreview.style.transform = cssM;
    }

    function drawScene() {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        /* background */
        if (bgImg && bgImg.complete && bgImg.naturalWidth) {
            const s = worldToScreen(0, 0);
            ctx.drawImage(bgImg, s.x, s.y,
                bgNatW * viewScale, bgNatH * viewScale);
        } else {
            // Checkerboard placeholder
            const s  = worldToScreen(0, 0);
            const sz = 20 * viewScale;
            const W  = bgNatW * viewScale, H = bgNatH * viewScale;
            for (let yi = 0; yi * sz < H; yi++) {
                for (let xi = 0; xi * sz < W; xi++) {
                    ctx.fillStyle = (xi + yi) % 2 === 0 ? "#2a2a3a" : "#1a1a2a";
                    ctx.fillRect(
                        Math.round(s.x + xi * sz), Math.round(s.y + yi * sz),
                        Math.ceil(sz), Math.ceil(sz));
                }
            }
        }

        /* perspective quad fill (faint) */
        const sc = corners.map(c => cornerToScreen(c));
        ctx.beginPath();
        ctx.moveTo(sc[0].x, sc[0].y);
        for (let i = 1; i < 4; i++) ctx.lineTo(sc[i].x, sc[i].y);
        ctx.closePath();
        ctx.fillStyle = "rgba(255,255,255,0.04)";
        ctx.fill();

        /* quad outline */
        ctx.beginPath();
        ctx.moveTo(sc[0].x, sc[0].y);
        for (let i = 1; i < 4; i++) ctx.lineTo(sc[i].x, sc[i].y);
        ctx.closePath();
        ctx.strokeStyle = "rgba(255,255,255,0.7)";
        ctx.lineWidth   = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        /* diagonal cross for reference */
        ctx.beginPath();
        ctx.moveTo(sc[0].x, sc[0].y); ctx.lineTo(sc[2].x, sc[2].y);
        ctx.moveTo(sc[1].x, sc[1].y); ctx.lineTo(sc[3].x, sc[3].y);
        ctx.strokeStyle = "rgba(255,255,255,0.2)";
        ctx.lineWidth   = 1;
        ctx.stroke();

        /* handles */
        sc.forEach(({ x, y }, i) => {
            // shadow
            ctx.beginPath();
            ctx.arc(x, y, HANDLE_RADIUS + 3, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(0,0,0,0.5)";
            ctx.fill();
            // fill
            ctx.beginPath();
            ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = HANDLE_COLORS[i];
            ctx.fill();
            // border
            ctx.beginPath();
            ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2);
            ctx.strokeStyle = "#fff";
            ctx.lineWidth   = 1.5;
            ctx.stroke();
            // label
            ctx.fillStyle = "#fff";
            ctx.font = "bold 10px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(HANDLE_LABELS[i], x, y);
        });

        /* zoom indicator */
        ctx.fillStyle = "rgba(255,255,255,0.4)";
        ctx.font      = "11px monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(`zoom: ${(viewScale * 100).toFixed(0)}%`,
            8, canvas.height - 6);

        updateLayerPreview();
    }

    /* ── interactivity ───────────────────────────────────────────────────── */
    let dragging   = -1;   // handle index being dragged, or -1
    let moving     = false; // moving entire quad
    let moveStartWx = 0, moveStartWy = 0; // world-space anchor at move start
    let cornersAtMoveStart = null;
    let panning    = false;
    let panStartX  = 0, panStartY = 0;
    let panTxStart = 0, panTyStart = 0;

    /** Ray-casting: is screen point (sx,sy) inside the perspective quad? */
    function hitInsideQuad(sx, sy) {
        const sc = corners.map(c => cornerToScreen(c));
        let inside = false;
        for (let i = 0, j = sc.length - 1; i < sc.length; j = i++) {
            const xi = sc[i].x, yi = sc[i].y;
            const xj = sc[j].x, yj = sc[j].y;
            if (((yi > sy) !== (yj > sy)) &&
                (sx < (xj - xi) * (sy - yi) / (yj - yi) + xi))
                inside = !inside;
        }
        return inside;
    }

    function hitHandle(sx, sy) {
        for (let i = 0; i < 4; i++) {
            const s = cornerToScreen(corners[i]);
            const dx = sx - s.x, dy = sy - s.y;
            if (dx * dx + dy * dy < (HANDLE_RADIUS + 4) ** 2) return i;
        }
        return -1;
    }

    function startPan(clientX, clientY) {
        panning    = true;
        panStartX  = clientX;
        panStartY  = clientY;
        panTxStart = viewTx;
        panTyStart = viewTy;
        canvas.style.cursor = "grabbing";
    }

    function startMove(sx, sy) {
        moving = true;
        const w = screenToWorld(sx, sy);
        moveStartWx = w.x;
        moveStartWy = w.y;
        cornersAtMoveStart = corners.map(c => ({ x: c.x, y: c.y }));
        canvas.style.cursor = "move";
    }

    canvas.addEventListener("mousedown", e => {
        const rect = canvas.getBoundingClientRect();
        const sx   = (e.clientX - rect.left) * (canvas.width  / rect.width);
        const sy   = (e.clientY - rect.top)  * (canvas.height / rect.height);

        // Middle-button = pan (always)
        if (e.button === 1) {
            startPan(e.clientX, e.clientY);
            e.preventDefault();
            return;
        }

        if (e.button === 0) {
            const h = hitHandle(sx, sy);
            if (h >= 0) {
                // Hit a handle → drag it
                dragging = h;
                canvas.style.cursor = "crosshair";
            } else if (hitInsideQuad(sx, sy)) {
                // Inside quad → move entire layer
                startMove(sx, sy);
            } else {
                // Clicked empty canvas area → pan (PS-style hand)
                startPan(e.clientX, e.clientY);
            }
        }
    });

    canvas.addEventListener("mousemove", e => {
        const rect = canvas.getBoundingClientRect();
        const sx   = (e.clientX - rect.left) * (canvas.width  / rect.width);
        const sy   = (e.clientY - rect.top)  * (canvas.height / rect.height);

        if (panning) {
            viewTx = panTxStart + (e.clientX - panStartX);
            viewTy = panTyStart + (e.clientY - panStartY);
            drawScene();
            return;
        }

        if (moving) {
            const w  = screenToWorld(sx, sy);
            const dwx = w.x - moveStartWx;   // world-space delta
            const dwy = w.y - moveStartWy;
            corners = cornersAtMoveStart.map(c => ({
                x: c.x + dwx / bgNatW,
                y: c.y + dwy / bgNatH,
            }));
            drawScene();
            return;
        }

        if (dragging >= 0) {
            const c = screenToCorner(sx, sy);
            corners[dragging] = c;
            drawScene();
            return;
        }

        // Show grab/crosshair cursor on hover
        if (hitHandle(sx, sy) >= 0)
            canvas.style.cursor = "crosshair";
        else if (hitInsideQuad(sx, sy))
            canvas.style.cursor = "move";
        else
            canvas.style.cursor = "grab";
    });

    window.addEventListener("mouseup", e => {
        if (dragging >= 0 || panning || moving) {
            dragging = -1;
            panning  = false;
            moving   = false;
            cornersAtMoveStart = null;
            canvas.style.cursor = "default";
        }
    }, { once: false });

    /* scroll-wheel zoom (towards cursor) */
    canvas.addEventListener("wheel", e => {
        e.preventDefault();
        const rect   = canvas.getBoundingClientRect();
        const sx     = (e.clientX - rect.left) * (canvas.width  / rect.width);
        const sy     = (e.clientY - rect.top)  * (canvas.height / rect.height);
        const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        viewTx = sx - (sx - viewTx) * factor;
        viewTy = sy - (sy - viewTy) * factor;
        viewScale *= factor;
        viewScale = Math.max(0.05, Math.min(20, viewScale));
        drawScene();
    }, { passive: false });

    /* Z = reset / fit view */
    function onKey(e) {
        if (e.key === "z" || e.key === "Z") {
            fitBackground();
            drawScene();
        }
        if (e.key === "Escape") btnCancel.click();
    }
    window.addEventListener("keydown", onKey);

    /* ── buttons ─────────────────────────────────────────────────────────── */
    btnReset.addEventListener("click", () => {
        corners = defaultCorners();
        drawScene();
    });

    btnCancel.addEventListener("click", () => {
        window.removeEventListener("keydown", onKey);
        overlay.remove();
    });

    btnApply.addEventListener("click", () => {
        if (cornersWidget) {
            // Write [(x,y),(x,y),(x,y),(x,y)] format to corners_input
            const pts = corners.map(c =>
                `(${+c.x.toFixed(6)},${+c.y.toFixed(6)})`
            );
            cornersWidget.value = `[${pts.join(",")}]`;
            // Notify ComfyUI that widget value changed
            node.setDirtyCanvas(true, true);
            if (typeof app.graph?.change === "function") app.graph.change();
        }
        window.removeEventListener("keydown", onKey);
        overlay.remove();
    });

    /* close on overlay click (but not on dialog click) */
    overlay.addEventListener("click", e => {
        if (e.target === overlay) btnCancel.click();
    });

    /* ── load images and initialise ─────────────────────────────────────── */
    let loadCount = 0;
    const TOTAL   = 2;

    function onLoad() {
        loadCount++;
        if (loadCount >= TOTAL) {
            fitBackground();
            drawScene();
        }
    }

    // Prefer the original source file (LoadImage) for full fidelity and alpha.
    // Fall back to the temp PNG saved by the backend after execution.
    const bgUrl    = getLoadImageUrl(node, "background_image") ?? getNodeImgUrl(node, 1);
    const layerUrl = getLoadImageUrl(node, "layer_image")      ?? getNodeImgUrl(node, 2);

    bgImg = new Image();
    bgImg.crossOrigin = "anonymous";
    bgImg.onload = () => {
        bgNatW = bgImg.naturalWidth;
        bgNatH = bgImg.naturalHeight;
        onLoad();
    };
    bgImg.onerror = onLoad;
    bgImg.src = bgUrl ?? "";
    if (!bgUrl) onLoad();   // no background – fire immediately

    layerImg = new Image();
    layerImg.crossOrigin = "anonymous";
    layerImg.onload = () => {
        layerNatW = layerImg.naturalWidth;
        layerNatH = layerImg.naturalHeight;
        onLoad();
    };
    layerImg.onerror = onLoad;
    layerImg.src = layerUrl ?? "";
    if (!layerUrl) onLoad();

    // focus so keyboard events land immediately
    overlay.tabIndex = 0;
    overlay.focus();
}

// ─── ComfyUI extension registration ─────────────────────────────────────────

app.registerExtension({
    name: "aki.InteractivePerspectiveMixer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_TYPES.has(nodeData.name)) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated?.apply(this, arguments);


            /* ── add "Open Editor" button ─────────────────────────────── */
            this.addWidget("button", "✏  Open Perspective Editor", null, () => {
                openEditor(this);
            }, { serialize: false });

            return r;
        };
    },
});
