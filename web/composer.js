// Scene Composer UI: slot rows (block or LoRA) edited in a DOM widget, stored as JSON in "composition".
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LORA_RE = /\.(safetensors|ckpt|pt)$/i;
const ROW_H = 28;
const cache = {};

// Fetch {slots, loras} for a base once per page load; null on error.
async function fetchOptions(base, force = false) {
    if (!force && cache[base]) return cache[base];
    try {
        const r = await api.fetchApi(`/sceneloader/composer/options?base=${encodeURIComponent(base)}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        cache[base] = await r.json();
    } catch (e) {
        console.warn("[SceneComposer] options unavailable", base, e);
        return null;
    }
    return cache[base];
}

// Parse the composition widget into a rows array; invalid JSON gives [].
function readRows(w) {
    try {
        const rows = JSON.parse(w.value || "[]");
        return Array.isArray(rows) ? rows : [];
    } catch (e) {
        console.warn("[SceneComposer] bad composition JSON, starting empty", e);
        return [];
    }
}

// Return the text shown in a row's item input.
function rowValue(row) {
    if (row.none) return "none";
    return row.lora || row.use || "";
}

// Build a row object from slot, item text and the previous row (keeps weight/trigger/extras for LoRAs).
function makeRow(slot, value, prev) {
    const v = value.trim();
    if (v === "none") return { slot, none: true };
    if (LORA_RE.test(v)) {
        const row = { slot, lora: v };
        if (prev?.lora === v) for (const k of ["weight", "trigger", "extras"]) if (prev[k] !== undefined) row[k] = prev[k];
        return row;
    }
    return v ? { slot, use: v } : { slot };
}

// Create an element with properties and children.
function el(tag, props = {}, ...kids) {
    const e = Object.assign(document.createElement(tag), props);
    for (const k of kids) if (k) e.append(k);
    return e;
}

let popup = null;

// Close the info popup if it is open.
function closePopup() {
    popup?.remove();
    popup = null;
    document.removeEventListener("pointerdown", onOutside, true);
    document.removeEventListener("keydown", onEsc, true);
}

// Close on a press outside the popup (inside it text stays selectable and links work).
function onOutside(e) {
    if (popup && !popup.contains(e.target)) closePopup();
}

// Close on Escape.
function onEsc(e) {
    if (e.key === "Escape") closePopup();
}

// Show text in a scrollable window near the click; URLs become links.
function showPopup(text, e) {
    e.stopPropagation();
    closePopup();
    const body = el("div", { style: "white-space:pre-wrap;word-break:break-word" });
    for (const part of text.split(/(https?:\/\/\S+)/)) {
        body.append(/^https?:\/\//.test(part)
            ? el("a", { href: part, textContent: part, target: "_blank", rel: "noopener", style: "color:#8cf" })
            : document.createTextNode(part));
    }
    popup = el("div", {
        id: "scene-composer-popup",
        style: "position:fixed;z-index:10000;max-width:min(720px,90vw);max-height:80vh;overflow:auto;padding:10px 12px;"
            + "background:#1e1e1e;color:#ddd;border:1px solid #567;border-radius:6px;font:12px monospace;"
            + "box-shadow:0 4px 16px rgba(0,0,0,.5);user-select:text",
    }, body);
    document.body.append(popup);
    const r = popup.getBoundingClientRect();
    popup.style.left = `${Math.max(8, Math.min(e.clientX + 12, innerWidth - r.width - 8))}px`;
    popup.style.top = `${Math.max(8, Math.min(e.clientY + 12, innerHeight - r.height - 8))}px`;
    document.addEventListener("pointerdown", onOutside, true);
    document.addEventListener("keydown", onEsc, true);
}

// Render the editor for a node into its container.
async function render(node) {
    const st = node.__composer;
    const baseW = node.widgets.find((w) => w.name === "base");
    const opts = await fetchOptions(baseW.value);
    const box = st.box;
    box.replaceChildren();
    if (!opts) {
        box.append(el("div", { textContent: "Scene Composer: options unavailable (see console)", style: "color:#e66" }));
        return;
    }
    const order = Object.keys(opts.slots);
    const rows = readRows(st.comp);
    // Keep rows in slot order; stable for rows of one slot.
    rows.sort((a, b) => order.indexOf(a.slot) - order.indexOf(b.slot));
    const save = () => {
        st.comp.value = JSON.stringify(rows);
        node.setDirtyCanvas?.(true, true);
    };

    for (const slot of order) {
        const o = opts.slots[slot];
        const list = el("datalist", { id: `sc-${node.id}-${slot}` });
        for (const v of ["none", ...o.blocks, ...o.loras]) list.append(el("option", { value: v }));
        box.append(list);
    }

    rows.forEach((row, i) => {
        if (row.slot === "detailer") return; // edited in the detailer section (later step)
        const o = opts.slots[row.slot];
        const valid = (v) => v === "none" || (o && (o.blocks.includes(v) || o.loras.includes(v)));
        const item = el("input", {
            value: rowValue(row), placeholder: "not chosen - type to search",
            style: "flex:1;min-width:120px",
        });
        item.setAttribute("list", `sc-${node.id}-${row.slot}`);
        const mark = () => (item.style.outline = item.value && !valid(item.value) ? "1px solid #e66" : "");
        mark();
        item.onchange = () => {
            rows[i] = makeRow(row.slot, item.value, row);
            save();
            render(node);
        };
        const line = el("div", { style: "display:flex;gap:4px;align-items:center;height:24px" },
            el("span", { textContent: row.slot, style: "width:80px;opacity:.7;overflow:hidden" }));
        // Hover info: block file text (+ all LoRAs incl. nested) or LoRA metadata summary.
        let tip = "";
        if (row.use && opts.block_text?.[row.use] !== undefined) {
            const bl = opts.blocks?.[row.use] || [];
            tip = opts.block_text[row.use].trim()
                + (bl.length ? "\n\nLoRAs (default weight):\n" + bl.map((x) => `${x.file} @${x.weight}`).join("\n") : "");
        } else if (row.lora && opts.loras[row.lora]) {
            tip = `${row.lora}\n${opts.loras[row.lora].i || ""}`;
        }
        const info = el("span", {
            textContent: "i", title: tip,
            style: `width:14px;text-align:center;border-radius:7px;font-weight:bold;cursor:pointer;`
                + `visibility:${tip ? "visible" : "hidden"};background:#456;color:#fff`,
        });
        info.onclick = async (e) => {
            e.stopPropagation();
            let text = tip;
            if (row.lora) {
                // Full description is loaded on demand; the hover tooltip keeps the short one.
                try {
                    const q = `base=${encodeURIComponent(baseW.value)}&file=${encodeURIComponent(row.lora)}`;
                    const r = await api.fetchApi(`/sceneloader/composer/lora_info?${q}`);
                    if (r.ok) text = `${row.lora}\n${(await r.json()).text}`;
                    else console.warn("[SceneComposer] lora_info HTTP", r.status);
                } catch (err) {
                    console.warn("[SceneComposer] lora_info failed, showing short info", err);
                }
            }
            showPopup(text, e);
        };
        line.append(info, item);
        // Strength of LoRAs inside a library block; identity (character) LoRAs are fixed by the block.
        for (const bl of (row.use && opts.blocks?.[row.use]) || []) {
            const fixed = bl.type === "character";
            const name = bl.file.split("/").pop().replace(LORA_RE, "");
            const wt = el("input", {
                type: "number", step: "0.05", style: "width:56px", disabled: fixed,
                value: row.weights?.[bl.file] ?? "", placeholder: String(bl.weight),
                title: `${name}\n${fixed ? "identity LoRA: weight is fixed in the block" : "empty = block weight"}`,
            });
            wt.onchange = () => {
                row.weights = row.weights || {};
                if (wt.value === "") delete row.weights[bl.file];
                else row.weights[bl.file] = Number(wt.value);
                if (!Object.keys(row.weights).length) delete row.weights;
                save();
            };
            line.append(wt);
        }
        if (row.lora) {
            const meta = opts.loras[row.lora] || { w: 1, t: [], x: false };
            const wt = el("input", {
                type: "number", step: "0.05", style: "width:56px",
                value: row.weight ?? "", placeholder: String(meta.w), title: "empty = recommended weight",
            });
            wt.onchange = () => {
                if (wt.value === "") delete row.weight;
                else row.weight = Number(wt.value);
                save();
            };
            line.append(wt);
            if (meta.t.length) {
                const sel = el("select", { style: "width:110px", title: "trigger words" });
                meta.t.forEach((t, k) => sel.append(el("option", { value: String(k), textContent: t })));
                if (meta.t.length > 1) sel.append(el("option", { value: "all", textContent: "(all sets)" }));
                sel.append(el("option", { value: "none", textContent: "(no triggers)" }));
                sel.value = String(row.trigger ?? 0);
                sel.onchange = () => {
                    row.trigger = sel.value === "all" || sel.value === "none" ? sel.value : Number(sel.value);
                    save();
                };
                line.append(sel);
            }
            if (meta.x) {
                const cb = el("input", { type: "checkbox", checked: !!row.extras, title: "author's extra positive/negative" });
                cb.onchange = () => {
                    row.extras = cb.checked;
                    save();
                };
                line.append(cb);
            }
        }
        const del = el("button", { textContent: "✕", title: "remove row" });
        del.onclick = () => {
            rows.splice(i, 1);
            save();
            render(node);
        };
        line.append(del);
        box.append(line);
    });

    const pick = el("select");
    for (const slot of order) pick.append(el("option", { value: slot, textContent: slot }));
    const add = el("button", { textContent: "+ row" });
    add.onclick = () => {
        rows.push({ slot: pick.value });
        save();
        render(node);
    };
    const reload = el("button", { textContent: "↻", title: "reload block and LoRA lists" });
    reload.onclick = async () => {
        await fetchOptions(baseW.value, true);
        render(node);
    };
    box.append(el("div", { style: "display:flex;gap:4px;margin-top:4px" }, pick, add, reload));
    save();
    st.height = (rows.filter((r) => r.slot !== "detailer").length + 1) * ROW_H + 12;
    node.setSize([Math.max(node.size[0], 440), node.computeSize()[1]]);
}

// Attach the editor widget and hide the raw JSON widget.
function setup(node) {
    if (node.__composer) return;
    const comp = node.widgets.find((w) => w.name === "composition");
    const baseW = node.widgets.find((w) => w.name === "base");
    if (!comp || !baseW) return;
    comp.type = "hidden";
    comp.computeSize = () => [0, -4];
    if (comp.element) comp.element.style.display = "none";
    const box = el("div", { style: "font:12px sans-serif;display:flex;flex-direction:column;gap:2px;overflow:auto" });
    node.__composer = { comp, box, height: ROW_H * 2 };
    node.addDOMWidget("composer", "composer", box, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => node.__composer.height,
        getValue: () => undefined,
        setValue: () => {},
    });
    const cb = baseW.callback;
    baseW.callback = function () {
        const r = cb?.apply(this, arguments);
        render(node);
        return r;
    };
    render(node);
}

app.registerExtension({
    name: "SceneLoader.composer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SceneComposer") return;
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
            setTimeout(() => (this.__composer ? render(this) : setup(this)), 0);
            return r;
        };
    },
});
