// Scene Loader UI: filter scenes by base; show Scene Save log after each run.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Fetch {scene: base|null} from the Scene Loader backend.
async function fetchSceneBases() {
    try {
        const r = await api.fetchApi("/sceneloader/scenes");
        return r.ok ? (await r.json()).scenes || {} : {};
    } catch (e) {
        console.warn("[SceneLoader] scene list unavailable", e);
        return {};
    }
}

// Limit the scene combo to scenes of the selected base or without a base.
async function applySceneFilter(node) {
    const sceneW = node.widgets?.find((w) => w.name === "scene");
    const baseW = node.widgets?.find((w) => w.name === "base");
    if (!sceneW || !baseW) return;
    const bases = await fetchSceneBases();
    const names = Object.keys(bases).sort();
    if (!names.length) return;
    const allowed = names.filter((s) => !bases[s] || bases[s] === baseW.value);
    sceneW.options.values = allowed.length ? allowed : ["(no scenes)"];
    if (!sceneW.options.values.includes(sceneW.value)) sceneW.value = sceneW.options.values[0];
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "SceneLoader.ui",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SceneLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                const baseW = this.widgets?.find((w) => w.name === "base");
                if (baseW) {
                    const cb = baseW.callback;
                    const node = this;
                    baseW.callback = function () {
                        const res = cb?.apply(this, arguments);
                        applySceneFilter(node);
                        return res;
                    };
                }
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure?.apply(this, arguments);
                setTimeout(() => applySceneFilter(this), 0);
                return r;
            };
        }
        if (nodeData.name === "SceneSave") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (!message || message.text === undefined) return;
                const t = Array.isArray(message.text) ? message.text.join("\n") : message.text;
                const w = this.widgets?.find((x) => x.name === "log_view");
                if (w) {
                    w.value = t;
                    if (w.inputEl) w.inputEl.readOnly = true;
                    this.setDirtyCanvas(true, true);
                }
            };
        }
    },
    async nodeCreated(node) {
        if (node.comfyClass === "SceneLoader") setTimeout(() => applySceneFilter(node), 0);
    },
});
