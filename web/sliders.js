// Scene Sliders UI: slider LoRAs with previews; tick one to use it, set its strength (negative allowed).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ROW_H = 50;

// Create an element with properties and children.
function el(tag, props = {}, ...kids) {
    const e = Object.assign(document.createElement(tag), props);
    for (const k of kids) if (k) e.append(k);
    return e;
}

// Fetch [{file, w, min, max, preview, i}] for a base; null on error.
async function fetchSliders(base) {
    try {
        const r = await api.fetchApi(`/sceneloader/sliders?base=${encodeURIComponent(base)}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return (await r.json()).sliders;
    } catch (e) {
        console.warn("[SceneSliders] slider list unavailable", base, e);
        return null;
    }
}

// Parse the sliders widget ({file: strength}); invalid JSON gives {}.
function readValues(w) {
    try {
        const v = JSON.parse(w.value || "{}");
        return v && typeof v === "object" && !Array.isArray(v) ? v : {};
    } catch (e) {
        console.warn("[SceneSliders] bad sliders JSON, starting empty", e);
        return {};
    }
}

let zoom = null; // large preview that follows the hovered thumbnail

// Show the preview large next to the thumbnail; hide with null.
function showZoom(img) {
    zoom?.remove();
    zoom = null;
    if (!img) return;
    const r = img.getBoundingClientRect();
    zoom = el("img", { src: img.src, style: "position:fixed;z-index:10002;max-width:min(480px,45vw);max-height:70vh;"
        + "border:1px solid #567;background:#000;pointer-events:none;box-shadow:0 4px 16px rgba(0,0,0,.6)" });
    zoom.style.left = `${r.right + 8}px`;
    zoom.style.top = `${Math.max(8, Math.min(r.top - 100, innerHeight - 0.7 * innerHeight - 8))}px`;
    document.body.append(zoom);
}

// Render the slider list into the node's container.
async function render(node) {
    const st = node.__sliders;
    const base = node.widgets.find((w) => w.name === "base").value;
    const list = await fetchSliders(base);
    const box = st.box;
    box.replaceChildren();
    if (!list) {
        box.append(el("div", { textContent: "Scene Sliders: list unavailable (see console)", style: "color:#e66" }));
        return;
    }
    const values = readValues(st.w);
    const save = () => {
        st.w.value = JSON.stringify(values);
        node.setDirtyCanvas?.(true, true);
    };
    const known = new Set(list.map((s) => s.file));
    // A chosen slider whose file is gone stays visible (red) so the stack does not change silently.
    const rows = [...list, ...Object.keys(values).filter((f) => !known.has(f)).map((file) => ({ file, missing: true }))];
    for (const s of rows) {
        const on = s.file in values;
        const name = s.file.replace(/^slider\//, "").replace(/\.(safetensors|ckpt|pt)$/i, "");
        const cb = el("input", { type: "checkbox", checked: on, title: "use this slider" });
        const line = el("div", { style: "display:flex;gap:6px;align-items:center;height:46px;"
            + `opacity:${on ? 1 : 0.55};${s.missing ? "outline:1px solid #e66;" : ""}` });
        line.append(cb);
        if (s.preview) {
            const img = el("img", { src: api.apiURL(`/sceneloader/lora_preview?base=${encodeURIComponent(base)}`
                + `&file=${encodeURIComponent(s.file)}`), loading: "lazy",
            style: "width:44px;height:44px;object-fit:cover;border-radius:3px;flex:none;background:#000" });
            img.onmouseenter = () => showZoom(img);
            img.onmouseleave = () => showZoom(null);
            line.append(img);
        } else {
            line.append(el("div", { textContent: "no preview", style: "width:44px;flex:none;font-size:9px;opacity:.5" }));
        }
        line.append(el("span", { textContent: name, title: s.missing ? "file not found" : `${s.file}\n${s.i || ""}`,
            style: "flex:1;min-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" }));
        if (!s.missing) {
            const val = on ? values[s.file] : s.w;
            const range = el("input", { type: "range", min: s.min, max: s.max, step: 0.05, value: val,
                title: `strength ${s.min}..${s.max} (Civitai range)`, style: "width:110px" });
            const num = el("input", { type: "number", step: 0.05, value: val, style: "width:58px",
                title: "strength, may be negative or outside the range" });
            const reset = el("button", { textContent: "↺", title: `recommended strength ${s.w}` });
            // Moving any control ticks the slider.
            const set = (v) => {
                if (!Number.isFinite(v)) return;
                values[s.file] = v;
                range.value = v;
                num.value = v;
                cb.checked = true;
                line.style.opacity = 1;
                save();
            };
            range.oninput = () => set(Number(range.value));
            num.onchange = () => set(Number(num.value));
            reset.onclick = () => set(s.w);
            line.append(range, num, reset);
        }
        cb.onchange = () => {
            if (cb.checked) values[s.file] = s.missing ? values[s.file] : Number(line.querySelector("input[type=number]").value);
            else delete values[s.file];
            save();
            render(node);
        };
        box.append(line);
    }
    if (!rows.length) box.append(el("div", { textContent: `no LoRAs in loras/${base}/slider`, style: "opacity:.6" }));
    save();
    st.height = Math.max(1, rows.length) * ROW_H + 8;
    node.setSize([Math.max(node.size[0], 420), node.computeSize()[1]]);
}

// Attach the slider list and hide the raw JSON widget.
function setup(node) {
    if (node.__sliders) return;
    const w = node.widgets?.find((x) => x.name === "sliders");
    const baseW = node.widgets?.find((x) => x.name === "base");
    if (!w || !baseW) return;
    w.type = "hidden";
    w.computeSize = () => [0, -4];
    if (w.element) w.element.style.display = "none";
    const box = el("div", { style: "font:12px sans-serif;display:flex;flex-direction:column;gap:4px;overflow:auto" });
    box.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });
    node.__sliders = { w, box, height: ROW_H };
    const domW = node.addDOMWidget("slider_list", "slider_list", box, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => node.__sliders.height,
        getValue: () => undefined,
        setValue: () => {},
    });
    // Not saved in widgets_values: an extra value there shifts widgets the node gains later.
    if (domW) domW.serialize = false;
    const cb = baseW.callback;
    baseW.callback = function () {
        const r = cb?.apply(this, arguments);
        render(node);
        return r;
    };
    render(node);
}

app.registerExtension({
    name: "SceneLoader.sliders",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SceneSliders") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            setTimeout(() => setup(this), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            // Saved graphs: widget values are restored after configure, so re-render afterwards.
            setTimeout(() => (this.__sliders ? render(this) : setup(this)), 0);
            return r;
        };
    },
});
