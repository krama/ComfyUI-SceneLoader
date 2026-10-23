// Pose Camera 3D view: the posed body from the last run, orbit with the mouse; the sliders follow.
// The view fills the node (resize the node to enlarge it); the ⤢ button opens a large window.
import { app } from "../../scripts/app.js";
import { COLORS, LIMBS, shoot, unpack } from "./pose_camera_math.js";

const MIN_VIEW_H = 220;
const MARGIN = 0.72; // share of the view the frame takes; the rest shows what is out of frame
const CAM_KEYS = ["framing", "yaw", "elevation", "fov", "zoom", "shift_x", "shift_y"];
const LIMITS = { yaw: [-180, 180], elevation: [-80, 89], zoom: [0.3, 4], shift_x: [-0.5, 0.5], shift_y: [-0.5, 0.5] };
const BTN_CSS = "position:absolute;top:4px;right:4px;width:26px;height:22px;padding:0;border:1px solid #567;"
    + "border-radius:4px;background:#252a30;color:#ddd;font:14px sans-serif;cursor:pointer;line-height:20px";

const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));
const widget = (node, name) => node.widgets?.find((w) => w.name === name);

// Current camera settings from the node widgets.
function camera(node) {
    const cam = {};
    for (const k of CAM_KEYS) cam[k] = widget(node, k)?.value;
    return cam;
}

// Set a camera widget like a user edit (value, callback, redraw).
function setValue(node, name, value) {
    const w = widget(node, name);
    if (!w) return;
    w.value = value;
    w.callback?.(value, app.canvas, node);
}

// Frame rectangle inside the view: {x, y, k} with k = view pixels per frame pixel.
function frameRect(view, body) {
    const k = Math.min((view.width * MARGIN) / body.width, (view.height * MARGIN) / body.height);
    return { x: (view.width - body.width * k) / 2, y: (view.height - body.height * k) / 2, k };
}

// Redraw every open view of the node (node widget and the large window).
function drawAll(node) {
    for (const cv of node.__poseCam.views) draw(node, cv);
}

// Draw the body, skeleton and frame on one canvas; its bitmap follows its on-screen size.
function draw(node, cv) {
    const st = node.__poseCam;
    const w = Math.max(50, Math.round(cv.clientWidth)), h = Math.max(50, Math.round(cv.clientHeight));
    if (cv.width !== w) cv.width = w;
    if (cv.height !== h) cv.height = h;
    const g = cv.getContext("2d");
    g.fillStyle = "#151515";
    g.fillRect(0, 0, w, h);
    g.font = "11px sans-serif";
    if (!st.body) {
        g.fillStyle = "#888";
        g.fillText("Run the graph once to see the pose here.", 10, 20);
        return;
    }
    const body = st.body;
    const f = frameRect(cv, body);
    const shot = shoot(body, camera(node));
    const toView = (p) => [f.x + p[0] * f.k, f.y + p[1] * f.k];
    g.fillStyle = "#000";
    g.fillRect(f.x, f.y, body.width * f.k, body.height * f.k);
    const dot = Math.max(2, Math.round(Math.min(w, h) / 220));
    // Far points first, so the near side covers them.
    const order = shot.verts.map((p, i) => i).sort((a, b) => shot.verts[b][2] - shot.verts[a][2]);
    for (const i of order) {
        const p = shot.verts[i];
        if (p[2] <= 0) continue;
        const [x, y] = toView(p);
        const inside = p[0] >= 0 && p[0] < body.width && p[1] >= 0 && p[1] < body.height;
        g.fillStyle = inside ? "#7d8a99" : "#3d434a";
        g.fillRect(x - dot / 2, y - dot / 2, dot, dot);
    }
    g.lineWidth = Math.max(3, dot + 1);
    LIMBS.forEach(([a, b], i) => {
        const pa = shot.points[a], pb = shot.points[b];
        if (!pa || !pb) return;
        g.strokeStyle = `rgb(${COLORS[i].join(",")})`;
        g.beginPath();
        g.moveTo(...toView(pa));
        g.lineTo(...toView(pb));
        g.stroke();
    });
    g.strokeStyle = "#ddd";
    g.lineWidth = 1;
    g.strokeRect(f.x + 0.5, f.y + 0.5, body.width * f.k, body.height * f.k);
    const c = camera(node);
    g.fillStyle = "#aaa";
    g.fillText(`yaw ${c.yaw}°  elevation ${c.elevation}°  zoom ${c.zoom}  ${body.width}x${body.height}`, 8, 14);
    g.fillText("drag: rotate · wheel: zoom · shift+drag: move", 8, h - 8);
    if (st.tags) g.fillText(`last run: ${st.tags}`, 8, 28);
    if (st.status) {
        g.font = "bold 13px sans-serif";
        g.fillStyle = "#ffd25a";
        g.fillText(st.status, 8, 46);
    }
}

// Mouse on one canvas: drag orbits (yaw / elevation), shift+drag moves the subject, wheel zooms.
// Returns a function that removes the window-level wheel listener.
function bindMouse(node, cv) {
    let last = null, move = false; // move: shift held at press -> shift the subject
    cv.addEventListener("pointerdown", (e) => {
        if (!node.__poseCam.body || e.button !== 0) return;
        last = [e.clientX, e.clientY];
        move = e.shiftKey;
        try {
            cv.setPointerCapture(e.pointerId); // keep the drag when the mouse leaves the view
        } catch { /* synthetic or already released pointer */ }
        e.preventDefault();
        e.stopPropagation();
    });
    cv.addEventListener("pointermove", (e) => {
        if (!last) return;
        const dx = e.clientX - last[0], dy = e.clientY - last[1];
        last = [e.clientX, e.clientY];
        const c = camera(node);
        if (move || e.shiftKey) {
            const f = frameRect(cv, node.__poseCam.body), b = node.__poseCam.body;
            setValue(node, "shift_x", +clamp(c.shift_x + dx / (b.width * f.k), LIMITS.shift_x).toFixed(3));
            setValue(node, "shift_y", +clamp(c.shift_y - dy / (b.height * f.k), LIMITS.shift_y).toFixed(3));
        } else {
            let yaw = c.yaw - dx * 0.5;
            yaw = ((((yaw + 180) % 360) + 360) % 360) - 180;
            setValue(node, "yaw", Math.round(yaw));
            setValue(node, "elevation", Math.round(clamp(c.elevation + dy * 0.5, LIMITS.elevation)));
        }
        drawAll(node);
        app.graph.setDirtyCanvas(true, false);
        e.stopPropagation();
    });
    const end = (e) => {
        last = null;
        if (cv.hasPointerCapture?.(e.pointerId)) cv.releasePointerCapture(e.pointerId);
    };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
    // Window capture phase: the frontend's own wheel handlers (canvas zoom) would see it first.
    const onWheel = (e) => {
        if (e.target !== cv || !cv.isConnected || !node.__poseCam.body) return;
        e.preventDefault();
        e.stopImmediatePropagation();
        const z = camera(node).zoom * Math.exp(-e.deltaY * 0.001);
        setValue(node, "zoom", +clamp(z, LIMITS.zoom).toFixed(2));
        drawAll(node);
        app.graph.setDirtyCanvas(true, false);
    };
    window.addEventListener("wheel", onWheel, { capture: true, passive: false });
    return () => window.removeEventListener("wheel", onWheel, { capture: true });
}

// A canvas view in a positioned box, with a corner button; returns {box, canvas, unbind}.
function makeView(node, label, onButton) {
    const box = document.createElement("div");
    box.style.cssText = "position:relative;width:100%;height:100%;min-height:" + MIN_VIEW_H + "px";
    const cv = document.createElement("canvas");
    cv.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab;touch-action:none";
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.style.cssText = BTN_CSS;
    btn.addEventListener("pointerdown", (e) => e.stopPropagation());
    btn.addEventListener("click", (e) => {
        e.stopPropagation();
        onButton();
    });
    box.append(cv, btn);
    node.__poseCam.views.push(cv);
    const unbind = bindMouse(node, cv);
    new ResizeObserver(() => draw(node, cv)).observe(cv);
    return { box, canvas: cv, unbind };
}

// Large view in a window over the canvas; closes with ✕, Esc or a click outside it.
function openLarge(node) {
    const st = node.__poseCam;
    if (st.large) return;
    const shade = document.createElement("div");
    shade.style.cssText = "position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.55);display:flex;"
        + "align-items:center;justify-content:center";
    const win = document.createElement("div");
    win.style.cssText = "width:min(1400px,92vw);height:86vh;background:#151515;border:1px solid #567;"
        + "border-radius:6px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,.6)";
    const close = () => {
        view.unbind();
        st.views = st.views.filter((v) => v !== view.canvas);
        window.removeEventListener("keydown", onKey, true);
        shade.remove();
        st.large = null;
    };
    const onKey = (e) => {
        if (e.key === "Escape") {
            e.stopPropagation();
            close();
        }
    };
    const view = makeView(node, "✕", close);
    win.append(view.box);
    shade.append(win);
    shade.addEventListener("pointerdown", (e) => e.target === shade && close());
    window.addEventListener("keydown", onKey, true);
    document.body.append(shade);
    st.large = { close };
    draw(node, view.canvas);
}

function setup(node) {
    if (node.__poseCam) return;
    node.__poseCam = { views: [], body: null, tags: "", large: null };
    const view = makeView(node, "⤢", () => openLarge(node));
    node.__poseCam.unbind = view.unbind;
    const w = node.addDOMWidget("view3d", "pose_camera_view", view.box, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => MIN_VIEW_H,
        getValue: () => undefined,
        setValue: () => {},
    });
    // Not saved in widgets_values: an extra value there shifts widgets the node gains later.
    if (w) w.serialize = false;
    for (const k of CAM_KEYS) {
        const w = widget(node, k);
        if (!w) continue;
        const cb = w.callback;
        w.callback = function () {
            const r = cb?.apply(this, arguments);
            drawAll(node);
            return r;
        };
    }
    node.setSize([Math.max(node.size[0], 420), Math.max(node.size[1], node.computeSize()[1] + 80)]);
    drawAll(node);
}

app.registerExtension({
    name: "SceneLoader.poseCamera",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PoseCamera") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            setup(this);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            // Saved graphs restore the old node size; keep the view wide enough to use.
            setTimeout(() => this.setSize([Math.max(this.size[0], 420), Math.max(this.size[1], this.computeSize()[1])]), 0);
            return r;
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.__poseCam?.large?.close();
            this.__poseCam?.unbind?.();
            return onRemoved?.apply(this, arguments);
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const payload = message?.pose_body?.[0];
            if (!payload) return;
            setup(this);
            this.__poseCam.body = unpack(payload);
            this.__poseCam.tags = message?.camera_tags?.[0] || "";
            this.__poseCam.status = message?.camera_status?.[0] || "";
            drawAll(this);
        };
    },
});
