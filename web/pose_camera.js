// Pose Camera 3D view: the posed body from the last run, orbit with the mouse; the sliders follow.
import { app } from "../../scripts/app.js";
import { COLORS, LIMBS, shoot, unpack } from "./pose_camera_math.js";

const VIEW_H = 300;
const MARGIN = 0.72; // share of the view the frame takes; the rest shows what is out of frame
const CAM_KEYS = ["framing", "yaw", "elevation", "fov", "zoom", "shift_x", "shift_y"];
const LIMITS = { yaw: [-180, 180], elevation: [-80, 89], zoom: [0.3, 4], shift_x: [-0.5, 0.5], shift_y: [-0.5, 0.5] };

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

// Draw the body, skeleton and frame for the current widget values.
function draw(node) {
    const st = node.__poseCam;
    const cv = st.canvas;
    const w = Math.max(50, Math.round(cv.clientWidth)), h = VIEW_H;
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
    // Far points first, so the near side covers them.
    const order = shot.verts.map((p, i) => i).sort((a, b) => shot.verts[b][2] - shot.verts[a][2]);
    for (const i of order) {
        const p = shot.verts[i];
        if (p[2] <= 0) continue;
        const [x, y] = toView(p);
        const inside = p[0] >= 0 && p[0] < body.width && p[1] >= 0 && p[1] < body.height;
        g.fillStyle = inside ? "#7d8a99" : "#3d434a";
        g.fillRect(x - 1, y - 1, 2, 2);
    }
    g.lineWidth = 3;
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
}

// Mouse: drag orbits (yaw / elevation), shift+drag moves the subject, wheel zooms.
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
        draw(node);
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
        draw(node);
        app.graph.setDirtyCanvas(true, false);
    };
    window.addEventListener("wheel", onWheel, { capture: true, passive: false });
    node.__poseCam.unbind = () => window.removeEventListener("wheel", onWheel, { capture: true });
}

function setup(node) {
    if (node.__poseCam) return;
    const cv = document.createElement("canvas");
    cv.style.cssText = `width:100%;height:${VIEW_H}px;display:block;cursor:grab;touch-action:none`;
    node.__poseCam = { canvas: cv, body: null, tags: "" };
    node.addDOMWidget("view3d", "pose_camera_view", cv, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => VIEW_H + 4,
        getValue: () => undefined,
        setValue: () => {},
    });
    for (const k of CAM_KEYS) {
        const w = widget(node, k);
        if (!w) continue;
        const cb = w.callback;
        w.callback = function () {
            const r = cb?.apply(this, arguments);
            draw(node);
            return r;
        };
    }
    bindMouse(node, cv);
    new ResizeObserver(() => draw(node)).observe(cv);
    node.setSize([Math.max(node.size[0], 420), node.computeSize()[1]]);
    draw(node);
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
            draw(this);
        };
    },
});
