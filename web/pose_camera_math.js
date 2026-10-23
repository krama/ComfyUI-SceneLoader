// Pose Camera math for the browser view; mirrors sceneloader/pose_camera.py (render_pose).
// tests/test_pose_camera.py runs it under node and compares the points with Python.

export const FACE = [0, 14, 15, 16, 17];
export const LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
    [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]];
export const COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
    [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
    [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]];

const rad = (d) => (d * Math.PI) / 180;

// Camera rotation (rows of a 3x3): yaw about Y, then elevation about the camera X; same as Python.
export function cameraRotation(yaw, elevation) {
    const a = rad(yaw), e = rad(elevation);
    const ry = [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
    const rx = [[1, 0, 0], [0, Math.cos(e), Math.sin(e)], [0, -Math.sin(e), Math.cos(e)]];
    return ry.map((row) => [0, 1, 2].map((k) => row[0] * rx[0][k] + row[1] * rx[1][k] + row[2] * rx[2][k]));
}

// Project one point [x, y, z] -> [px, py, depth].
export function project(p, eye, rot, focal, width, height) {
    const d = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
    const cam = [0, 1, 2].map((k) => d[0] * rot[0][k] + d[1] * rot[1][k] + d[2] * rot[2][k]);
    const depth = -cam[2];
    const safe = depth > 1e-6 ? depth : 1e-6;
    return [width / 2 + (focal * cam[0]) / safe, height / 2 - (focal * cam[1]) / safe, depth];
}

// Bounding box [min, max] of projected points.
function bbox2(pts) {
    const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const p of pts) {
        lo[0] = Math.min(lo[0], p[0]); lo[1] = Math.min(lo[1], p[1]);
        hi[0] = Math.max(hi[0], p[0]); hi[1] = Math.max(hi[1], p[1]);
    }
    return [lo, hi];
}

// Place the camera so the framed points fill the frame: {eye, rot, focal, offset}.
export function solveCamera(framed, cam, width, height, fill) {
    const rot = cameraRotation(cam.yaw, cam.elevation);
    const focal = height / 2 / Math.tan(rad(cam.fov) / 2);
    const back = [rot[0][2], rot[1][2], rot[2][2]];
    const lo = [0, 1, 2].map((k) => Math.min(...framed.map((p) => p[k])));
    const hi = [0, 1, 2].map((k) => Math.max(...framed.map((p) => p[k])));
    const aim = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
    const eyeAt = (dist) => aim.map((v, k) => v + back[k] * dist);
    let dist = 3.0;
    for (let i = 0; i < 6; i++) {
        const [l, h] = bbox2(framed.map((p) => project(p, eyeAt(dist), rot, focal, width, height)));
        const sx = Math.max(h[0] - l[0], 1e-6), sy = Math.max(h[1] - l[1], 1e-6);
        dist *= Math.max(sx / (fill * width), sy / (fill * height));
    }
    dist /= Math.max(cam.zoom, 1e-3);
    const eye = eyeAt(dist);
    const [l, h] = bbox2(framed.map((p) => project(p, eye, rot, focal, width, height)));
    const offset = [width / 2 - (l[0] + h[0]) / 2 + cam.shift_x * width,
        height / 2 - (l[1] + h[1]) / 2 - cam.shift_y * height];
    return { eye, rot, focal, offset };
}

// Face points turned away from the camera are hidden (same rule as Python visible_face).
export function visibleFace(joints, eye) {
    const head = [0, 1, 2].map((k) => (joints[16][k] + joints[17][k]) / 2);
    const toCam = [0, 1, 2].map((k) => eye[k] - head[k]);
    const norm = (v) => Math.hypot(v[0], v[1], v[2]);
    const vis = new Array(18).fill(true);
    for (const i of FACE) {
        const v = [0, 1, 2].map((k) => joints[i][k] - head[k]);
        const n = norm(v) * norm(toCam);
        const dot = v[0] * toCam[0] + v[1] * toCam[1] + v[2] * toCam[2];
        vis[i] = n < 1e-9 || dot / n > (i === 16 || i === 17 ? -0.3 : 0.0);
    }
    return vis;
}

// Unpack the Python payload: verts / joints as [x, y, z] lists.
export function unpack(payload) {
    const tri = (flat) => Array.from({ length: flat.length / 3 }, (_, i) => flat.slice(3 * i, 3 * i + 3));
    return { ...payload, verts: tri(payload.verts), joints: tri(payload.joints) };
}

// Shoot the body: {points: 18 x ([x, y] | null), verts: [[x, y, depth]], camera} in frame pixels.
export function shoot(body, cam) {
    const { width, height } = body;
    const bit = 1 << Math.max(0, body.framings.indexOf(cam.framing));
    const framed = body.verts.filter((_, i) => body.bits[i] & bit);
    const c = solveCamera(framed.length ? framed : body.verts, cam, width, height, body.fill);
    const place = (p) => {
        const q = project(p, c.eye, c.rot, c.focal, width, height);
        return [q[0] + c.offset[0], q[1] + c.offset[1], q[2]];
    };
    const vis = visibleFace(body.joints, c.eye);
    const points = body.joints.map((j, i) => {
        const [x, y, depth] = place(j);
        const inside = x >= 0 && x < width && y >= 0 && y < height;
        return vis[i] && depth > 1e-3 && inside ? [x, y] : null;
    });
    return { points, verts: body.verts.map(place), camera: c };
}
