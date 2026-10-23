// Scene Composer UI: slot rows (block or LoRA) edited in a DOM widget, stored as JSON in "composition".
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LORA_RE = /\.(safetensors|ckpt|pt)$/i;
// Camera input names; camera_json/camera_prompt are legacy sockets in graphs saved before "camera".
const CAMERA_INPUTS = ["camera", "camera_json", "camera_prompt"];

// True when a camera input has a link; it then replaces the camera slot (composer.rows_to_scene).
function cameraLinked(node) {
    return !!node.inputs?.some((i) => CAMERA_INPUTS.includes(i.name) && i.link != null);
}
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
    return row.lora || row.use || row.tags || "";
}

// Build a row from slot and item text: block or LoRA from the slot lists, otherwise plain tags.
function makeRow(slot, value, prev, o) {
    const v = value.trim();
    if (v === "none") return { slot, none: true };
    if (LORA_RE.test(v)) {
        const row = { slot, lora: v };
        if (prev?.lora === v) for (const k of ["weight", "trigger", "extras"]) if (prev[k] !== undefined) row[k] = prev[k];
        return row;
    }
    if (o?.blocks.includes(v)) {
        const row = { slot, use: v };
        if (prev?.use === v) for (const k of ["weights", "tags"]) if (prev[k] !== undefined) row[k] = prev[k];
        return row;
    }
    return v ? { slot, tags: v } : { slot };
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

let picker = null; // {list, input, items, active}

// Close the dropdown list if it is open.
function closePicker() {
    picker?.list.remove();
    picker = null;
}

// Scrollable search list under an input: substring filter, arrows/Enter/Esc, click to pick.
// Picking sets input.value and fires "change"; free text still works (Enter without a highlight).
function attachPicker(input, values) {
    const MAX = 1000; // ponytail: plain DOM list; virtualize if a slot grows past a few thousand items
    const show = () => {
        const q = input.value.trim().toLowerCase();
        // Full list right after focus on a filled field; filter once the user types.
        const items = (input.dataset.typed ? values.filter((v) => v.toLowerCase().includes(q)) : values).slice(0, MAX);
        closePicker();
        if (!items.length) return;
        const r = input.getBoundingClientRect();
        const list = el("div", {
            id: "scene-composer-picker",
            style: `position:fixed;z-index:10001;left:${r.left}px;top:${r.bottom + 2}px;`
                + `min-width:${r.width}px;max-width:min(640px,90vw);max-height:min(320px,50vh);overflow:auto;`
                + "background:#1e1e1e;color:#ddd;border:1px solid #567;border-radius:4px;font:12px sans-serif;"
                + "box-shadow:0 4px 16px rgba(0,0,0,.5)",
        });
        list.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true }); // no canvas zoom
        items.forEach((v, k) => {
            const opt = el("div", { textContent: v, style: "padding:3px 8px;white-space:nowrap;cursor:pointer" });
            opt.onpointerdown = (e) => {
                e.preventDefault(); // keep focus in the input until the value is set
                pick(k);
            };
            opt.onmouseenter = () => highlight(k);
            list.append(opt);
        });
        document.body.append(list);
        picker = { list, input, items, active: -1 };
        const cur = items.indexOf(input.value);
        if (cur >= 0) highlight(cur);
    };
    const highlight = (k) => {
        if (!picker) return;
        [...picker.list.children].forEach((c, j) => (c.style.background = j === k ? "#345" : ""));
        picker.active = k;
        picker.list.children[k]?.scrollIntoView({ block: "nearest" });
    };
    const pick = (k) => {
        const v = picker?.items[k];
        closePicker();
        if (v === undefined) return;
        input.value = v;
        delete input.dataset.typed;
        input.dispatchEvent(new Event("change"));
    };
    input.addEventListener("focus", show);
    input.addEventListener("input", () => {
        input.dataset.typed = "1";
        show();
    });
    input.addEventListener("blur", () => setTimeout(() => picker?.input === input && closePicker(), 150));
    input.addEventListener("keydown", (e) => {
        if (!picker || picker.input !== input) {
            if (e.key === "ArrowDown") show();
            return;
        }
        const n = picker.items.length;
        if (e.key === "ArrowDown") highlight((picker.active + 1) % n);
        else if (e.key === "ArrowUp") highlight((picker.active - 1 + n) % n);
        else if (e.key === "Enter" && picker.active >= 0) pick(picker.active);
        else if (e.key === "Escape") closePicker();
        else return;
        e.preventDefault();
        e.stopPropagation(); // keep ComfyUI canvas shortcuts away from the list keys
    });
}

// "i" button: hover shows tip; click opens a popup (full LoRA description loaded when lora is set).
function infoButton(tip, lora, base) {
    const info = el("span", {
        textContent: "i", title: tip,
        style: "width:14px;flex:none;text-align:center;border-radius:7px;font-weight:bold;cursor:pointer;"
            + `visibility:${tip ? "visible" : "hidden"};background:#456;color:#fff`,
    });
    info.onclick = async (e) => {
        e.stopPropagation();
        let text = tip;
        if (lora) {
            try {
                const q = `base=${encodeURIComponent(base)}&file=${encodeURIComponent(lora)}`;
                const r = await api.fetchApi(`/sceneloader/composer/lora_info?${q}`);
                if (r.ok) text = `${lora}\n${(await r.json()).text}`;
                else console.warn("[SceneComposer] lora_info HTTP", r.status);
            } catch (err) {
                console.warn("[SceneComposer] lora_info failed, showing short info", err);
            }
        }
        showPopup(text, e);
    };
    return info;
}

// Fetch {scene: base|null} (shared with Scene Loader); {} on error.
async function fetchScenes() {
    try {
        const r = await api.fetchApi("/sceneloader/scenes");
        return r.ok ? (await r.json()).scenes || {} : {};
    } catch (e) {
        console.warn("[SceneComposer] scene list unavailable", e);
        return {};
    }
}

// Build the preset bar: load a scene into rows, or save rows as library/scenes/<name>.yaml.
async function presetBar(node, baseW) {
    const st = node.__composer;
    const scenes = await fetchScenes();
    const names = Object.keys(scenes).filter((s) => !scenes[s] || scenes[s] === baseW.value).sort();
    const pick = el("select", { style: "flex:1;min-width:90px", title: "scenes of this base (or without a base)" });
    for (const s of names) pick.append(el("option", { value: s, textContent: s }));
    if (node.properties.scene && names.includes(node.properties.scene)) pick.value = node.properties.scene;
    const status = el("div", { textContent: st.msg || "", style: `font-size:11px;color:${st.msgColor || "#aaa"}` });
    const setMsg = (text, color) => {
        st.msg = text;
        st.msgColor = color;
        status.textContent = text;
        status.style.color = color;
    };
    const load = el("button", { textContent: "load", title: "replace rows with the selected scene" });
    load.onclick = async () => {
        if (!pick.value) return;
        if (readRows(st.comp).length && !confirm(`Replace current rows with scene "${pick.value}"?`)) return;
        const r = await api.fetchApi(`/sceneloader/composer/scene?name=${encodeURIComponent(pick.value)}`);
        const data = await r.json();
        if (!r.ok) {
            console.warn("[SceneComposer] load scene failed", data);
            setMsg(`load failed: ${data.error}`, "#e66");
            return;
        }
        st.comp.value = JSON.stringify(data.rows);
        node.properties.scene = pick.value;
        name.value = pick.value;
        if (data.base && data.base !== baseW.value) baseW.value = data.base;
        const warn = data.warnings?.length ? ` (${data.warnings.length} warnings, see console)` : "";
        if (warn) console.warn("[SceneComposer] scene warnings", data.warnings);
        st.msg = `loaded ${pick.value}${warn}`;
        st.msgColor = warn ? "#eb6" : "#6c6";
        render(node);
    };
    const name = el("input", { value: node.properties.scene || "", placeholder: "scene name", style: "width:110px" });
    const saveBtn = el("button", { textContent: "save", title: "save rows to library/scenes/<name>.yaml" });
    saveBtn.onclick = async () => {
        const body = { name: name.value.trim(), base: baseW.value, rows: readRows(st.comp), overwrite: false };
        let r = await api.fetchApi("/sceneloader/composer/scene", { method: "POST", body: JSON.stringify(body) });
        if (r.status === 409) {
            if (!confirm(`Scene "${body.name}" exists. Overwrite?`)) return;
            body.overwrite = true;
            r = await api.fetchApi("/sceneloader/composer/scene", { method: "POST", body: JSON.stringify(body) });
        }
        const data = await r.json();
        if (!r.ok) {
            console.warn("[SceneComposer] save scene failed", data);
            setMsg(`save failed: ${data.error}`, "#e66");
            return;
        }
        node.properties.scene = body.name;
        st.msg = `saved ${data.path}`;
        st.msgColor = "#6c6";
        render(node);
    };
    return el("div", { style: "display:flex;flex-direction:column;gap:2px;margin-bottom:4px" },
        el("div", { style: "display:flex;gap:4px;align-items:center" },
            el("span", { textContent: "scene", style: "width:80px;opacity:.7" }), pick, load, name, saveBtn),
        status);
}

// Return LoRA types (first folder) present in the main stack: LoRA rows and LoRAs inside blocks.
function stackTypes(rows, opts) {
    const types = new Set();
    for (const r of rows) {
        if (r.none || r.slot === "detailer") continue;
        if (r.lora) types.add(r.lora.includes("/") ? r.lora.split("/")[0] : "other");
        for (const bl of (r.use && opts.blocks?.[r.use]) || []) types.add(bl.type);
    }
    return [...types].sort();
}

// Build the detailer delta editor (drop types, add LoRAs, override weights); returns {el, lines}.
function detailerSection(node, rows, opts, save) {
    let det = rows.find((r) => r.slot === "detailer");
    const ensure = () => {
        if (!det) {
            det = { slot: "detailer" };
            rows.push(det);
        }
        return det;
    };
    // Drop the detailer row when it holds nothing, so the composition stays clean.
    const commit = () => {
        for (const k of ["drop", "add", "override"]) if (det && !det[k]?.length) delete det[k];
        if (det && Object.keys(det).length === 1) rows.splice(rows.indexOf(det), 1);
        save();
        render(node);
    };
    const wrap = el("div", { style: "display:flex;flex-direction:column;gap:2px;margin-top:6px;"
        + "padding-top:4px;border-top:1px solid #444" });
    let lines = 1;

    const types = [...new Set([...stackTypes(rows, opts), ...(det?.drop || [])])].sort();
    const chips = el("div", { style: "display:flex;gap:3px;flex-wrap:wrap;align-items:center;min-height:24px" },
        el("span", { textContent: "detailer drop", style: "width:80px;opacity:.7",
            title: "LoRA types removed from the detailer stack" }));
    for (const t of types) {
        const on = det?.drop?.includes(t);
        const chip = el("button", { textContent: t, title: on ? "dropped in detailer" : "kept in detailer",
            style: `padding:0 6px;${on ? "background:#844;text-decoration:line-through" : ""}` });
        chip.onclick = () => {
            const d = ensure();
            d.drop = on ? (d.drop || []).filter((x) => x !== t) : [...(d.drop || []), t];
            commit();
        };
        chips.append(chip);
    }
    if (!types.length) chips.append(el("span", { textContent: "(no LoRAs in the stack)", style: "opacity:.5" }));
    wrap.append(chips);

    const allLoras = Object.keys(opts.loras).sort();
    (det?.add || []).forEach((a, k) => {
        const file = el("input", { value: a.file || "", placeholder: "detailer LoRA - type to search",
            style: `flex:1;min-width:120px;${a.file && !opts.loras[a.file] ? "outline:1px solid #e66" : ""}` });
        attachPicker(file, allLoras);
        file.onchange = () => {
            a.file = file.value.trim();
            if (a.weight === undefined && opts.loras[a.file]) a.weight = opts.loras[a.file].w;
            commit();
        };
        const wt = el("input", { type: "number", step: "0.05", value: a.weight ?? "", placeholder: "1.0",
            style: "width:56px" });
        wt.onchange = () => {
            a.weight = wt.value === "" ? 1.0 : Number(wt.value);
            save();
        };
        const del = el("button", { textContent: "✕", title: "remove" });
        del.onclick = () => {
            det.add.splice(k, 1);
            commit();
        };
        const known = a.file && opts.loras[a.file] ? a.file : "";
        const tip = known ? `${known}\n${opts.loras[known].i || ""}` : "";
        const base = node.widgets.find((w) => w.name === "base").value;
        wrap.append(el("div", { style: "display:flex;gap:4px;align-items:center;height:24px" },
            el("span", { textContent: "+ add", style: "width:80px;opacity:.7" }),
            infoButton(tip, known, base), file, wt, del));
        lines++;
    });
    (det?.override || []).forEach((ov, k) => {
        const sel = el("select", { style: "flex:1", title: "LoRA type whose weight changes in the detailer" });
        for (const t of [...new Set([...stackTypes(rows, opts), ov.type].filter(Boolean))]) {
            sel.append(el("option", { value: t, textContent: t }));
        }
        if (ov.file) sel.append(el("option", { value: `file:${ov.file}`, textContent: ov.file }));
        sel.value = ov.file ? `file:${ov.file}` : ov.type || "";
        sel.onchange = () => {
            delete ov.file;
            ov.type = sel.value;
            save();
        };
        const wt = el("input", { type: "number", step: "0.05", value: ov.weight ?? "", style: "width:56px" });
        wt.onchange = () => {
            ov.weight = wt.value === "" ? 1.0 : Number(wt.value);
            save();
        };
        const del = el("button", { textContent: "✕", title: "remove" });
        del.onclick = () => {
            det.override.splice(k, 1);
            commit();
        };
        wrap.append(el("div", { style: "display:flex;gap:4px;align-items:center;height:24px" },
            el("span", { textContent: "override", style: "width:80px;opacity:.7" }), sel, wt, del));
        lines++;
    });
    const addBtn = el("button", { textContent: "+ detailer LoRA" });
    addBtn.onclick = () => {
        const d = ensure();
        d.add = [...(d.add || []), { file: "" }]; // weight is taken from the sidecar when a file is picked
        save();
        render(node);
    };
    const ovBtn = el("button", { textContent: "+ override", disabled: !types.length });
    ovBtn.onclick = () => {
        const d = ensure();
        d.override = [...(d.override || []), { type: types[0], weight: 1.0 }];
        commit();
    };
    wrap.append(el("div", { style: "display:flex;gap:4px" }, addBtn, ovBtn));
    return { el: wrap, lines: lines + 1 }; // + the buttons line
}

// Render the editor for a node into its container.
async function render(node) {
    const st = node.__composer;
    const baseW = node.widgets.find((w) => w.name === "base");
    const opts = await fetchOptions(baseW.value);
    const box = st.box;
    const bar = opts && (await presetBar(node, baseW));
    // No awaits below this point: concurrent renders must not interleave DOM changes.
    box.replaceChildren();
    if (!opts) {
        box.append(el("div", { textContent: "Scene Composer: options unavailable (see console)", style: "color:#e66" }));
        return;
    }
    box.append(bar);
    const order = Object.keys(opts.slots);
    const rows = readRows(st.comp);
    // Keep rows in slot order; stable for rows of one slot.
    rows.sort((a, b) => order.indexOf(a.slot) - order.indexOf(b.slot));
    const save = () => {
        st.comp.value = JSON.stringify(rows);
        node.setDirtyCanvas?.(true, true);
    };

    rows.forEach((row, i) => {
        if (row.slot === "detailer") return; // edited in the detailer section (later step)
        const o = opts.slots[row.slot];
        // Unknown block or LoRA (deleted or renamed file) is outlined red; free text is plain tags.
        const missing = (row.use && !o?.blocks.includes(row.use))
            || (row.lora && !o?.loras.includes(row.lora));
        const item = el("input", {
            value: rowValue(row), placeholder: "not chosen - search, or type tags",
            title: row.tags && !row.use ? "plain tags" : "",
            style: `flex:1;min-width:120px;${row.tags && !row.use ? "font-style:italic;" : ""}`
                + (missing ? "outline:1px solid #e66" : ""),
        });
        attachPicker(item, o ? ["none", ...o.blocks, ...o.loras] : ["none"]);
        item.onchange = () => {
            rows[i] = makeRow(row.slot, item.value, row, o);
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
        line.append(infoButton(tip, row.lora, baseW.value), item);
        if (row.use) {
            // Extra tags appended to the block's own tags (e.g. "light smile" on a character).
            const extra = el("input", {
                value: row.tags || "", placeholder: "+ tags", title: "extra tags for this block",
                style: "width:80px",
            });
            extra.onchange = () => {
                if (extra.value.trim()) row.tags = extra.value.trim();
                else delete row.tags;
                save();
            };
            line.append(extra);
        }
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
        if (row.slot === "camera" && cameraLinked(node)) {
            line.style.opacity = "0.4"; // same rule as composer.rows_to_scene: the input replaces this row
            line.title = "replaced by the camera input (tags and LoRAs are not used)";
        }
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
    const linked = cameraLinked(node);
    const camState = el("span", {
        textContent: linked ? "camera: ● input" : "camera: rows",
        title: linked ? "camera input is connected and replaces the camera slot"
            : "camera input not connected: camera slot rows are used",
        style: `margin-left:auto;font-size:11px;align-self:center;color:${linked ? "#6c6" : "#aaa"}`,
    });
    box.append(el("div", { style: "display:flex;gap:4px;margin-top:4px" }, pick, add, reload, camState));
    const det = detailerSection(node, rows, opts, save);
    box.append(det.el);
    save();
    // Rows + preset bar (2) + add-row line (1) + detailer lines.
    st.height = (rows.filter((r) => r.slot !== "detailer").length + 3 + det.lines) * ROW_H + 12;
    node.setSize([Math.max(node.size[0], 440), node.computeSize()[1]]);
}

// Attach the editor widget and hide the raw JSON widget.
function setup(node) {
    if (node.__composer) return;
    const comp = node.widgets.find((w) => w.name === "composition");
    const baseW = node.widgets.find((w) => w.name === "base");
    if (!comp || !baseW) return;
    node.properties = node.properties || {}; // properties.scene: last loaded/saved scene, kept in the workflow
    comp.type = "hidden";
    comp.computeSize = () => [0, -4];
    if (comp.element) comp.element.style.display = "none";
    const box = el("div", { style: "font:12px sans-serif;display:flex;flex-direction:column;gap:2px;overflow:auto" });
    node.__composer = { comp, box, height: ROW_H * 2 };
    const domW = node.addDOMWidget("composer", "composer", box, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => node.__composer.height,
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
    name: "SceneLoader.composer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SceneComposer") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            setTimeout(() => setup(this), 0);
            return r;
        };
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange?.apply(this, arguments);
            if (this.__composer) render(this); // refresh "3d camera" linked status
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
