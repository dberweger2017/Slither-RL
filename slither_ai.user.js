// ==UserScript==
// @name         Slither.io RL Agent (ONNX)
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  Plays Slither.io using the trained PPO ONNX policy
// @author       Antigravity
// @match        *://slither.io/*
// @grant        none
// @require      https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js
// ==/UserScript==

(function () {
    'use strict';

    // To serve locally: python3 -m http.server 8080 (in the gravity folder)
    const ONNX_URL = "http://localhost:8080/slither_policy.onnx";

    let session = null;
    let isRunning = false;
    let aiEnabled = false;

    // AI Configuration Parameters (matched to Python env)
    const MAP_SIZE = 168;
    const VIEW_RADIUS = 500.0;
    const PIXELS_PER_UNIT = MAP_SIZE / (2 * VIEW_RADIUS);
    const CENTER = MAP_SIZE / 2;

    const canvas = document.createElement('canvas');
    canvas.width = MAP_SIZE;
    canvas.height = MAP_SIZE;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    async function loadModel() {
        console.log("Loading ONNX model from: " + ONNX_URL);
        try {
            // onnxruntime-web usage
            session = await ort.InferenceSession.create(ONNX_URL, { executionProviders: ['wasm'] });
            console.log("ONNX Model loaded successfully!");
            aiEnabled = true;
        } catch (e) {
            console.error("Failed to load ONNX model:", e);
        }
    }

    // Creates the 5x168x168 float32 observation map
    function generateMiniMap(player) {
        const obs = new Float32Array(5 * MAP_SIZE * MAP_SIZE);
        const hx = player.xx;
        const hy = player.yy;
        const heading = player.ang;

        // Transform formulas
        const cos_a = Math.cos(-heading + Math.PI / 2);
        const sin_a = Math.sin(-heading + Math.PI / 2);

        function worldToEgo(wx, wy) {
            const dx = wx - hx;
            const dy = wy - hy;
            const rx = dx * cos_a + dy * sin_a;
            const ry = -dx * sin_a + dy * cos_a;
            return {
                px: CENTER + rx * PIXELS_PER_UNIT,
                py: CENTER - ry * PIXELS_PER_UNIT
            };
        }

        function stampCircle(channel, px, py, radius, intensity) {
            const r = Math.max(1, Math.round(radius));
            const y0 = Math.max(0, Math.floor(py - r - 1));
            const y1 = Math.min(MAP_SIZE, Math.floor(py + r + 2));
            const x0 = Math.max(0, Math.floor(px - r - 1));
            const x1 = Math.min(MAP_SIZE, Math.floor(px + r + 2));

            const rSq = r * r;
            const offset = channel * MAP_SIZE * MAP_SIZE;

            for (let y = y0; y < y1; y++) {
                for (let x = x0; x < x1; x++) {
                    const dSq = (x - px) * (x - px) + (y - py) * (y - py);
                    if (dSq <= rSq) {
                        const idx = offset + y * MAP_SIZE + x;
                        if (intensity > obs[idx]) obs[idx] = intensity;
                    }
                }
            }
        }

        // 0: Self
        let playerRad = Math.max(15, player.sc * 20); // Estimation
        stampCircle(0, CENTER, CENTER, Math.max(1, playerRad * PIXELS_PER_UNIT), 1.0);

        if (player.pts) {
            for (let i = 0; i < player.pts.length; i += 2) {
                const pt = player.pts[i];
                if (!pt) continue;
                const d = Math.hypot(pt.xx - hx, pt.yy - hy);
                if (d > VIEW_RADIUS) continue;
                const ego = worldToEgo(pt.xx, pt.yy);
                const fade = 0.8 - 0.5 * (i / player.pts.length);
                stampCircle(0, ego.px, ego.py, Math.max(2, playerRad * 0.8 * PIXELS_PER_UNIT), Math.max(0.2, fade));
            }
        }

        // 1: Enemies & 4: Velocity streaks
        if (window.snakes) {
            for (let i = 0; i < window.snakes.length; i++) {
                const s = window.snakes[i];
                if (s === player || s.dead_amt > 0 || !s.pts) continue;

                const hd = Math.hypot(s.xx - hx, s.yy - hy);
                if (hd > VIEW_RADIUS + 200) continue;

                const eRad = Math.max(15, s.sc * 20);
                const massIntensity = Math.min(1.0, (s.sct * 30) / 2000.0);

                for (let j = 0; j < s.pts.length; j += 2) {
                    const pt = s.pts[j];
                    if (!pt) continue;
                    if (Math.hypot(pt.xx - hx, pt.yy - hy) > VIEW_RADIUS) continue;
                    const ego = worldToEgo(pt.xx, pt.yy);
                    stampCircle(1, ego.px, ego.py, Math.max(2, eRad * 0.8 * PIXELS_PER_UNIT), massIntensity * 0.7);
                }

                if (hd <= VIEW_RADIUS) {
                    const ego = worldToEgo(s.xx, s.yy);
                    stampCircle(1, ego.px, ego.py, Math.max(3, eRad * PIXELS_PER_UNIT), 1.0);

                    // Velocity streak (Channel 4)
                    const speed = s.sp || 5.0;
                    if (speed > 4.0) {
                        const streakLen = Math.min(30, speed * 2);
                        const endX = s.xx + Math.cos(s.ang) * streakLen;
                        const endY = s.yy + Math.sin(s.ang) * streakLen;
                        const egoEnd = worldToEgo(endX, endY);

                        // Simple DDA line drawing for JS
                        const dx = egoEnd.px - ego.px;
                        const dy = egoEnd.py - ego.py;
                        const steps = Math.max(Math.abs(dx), Math.abs(dy));
                        const val = Math.min(1.0, speed / 14.0);
                        for (let p = 0; p <= steps; p++) {
                            const x = ego.px + dx * (p / steps);
                            const y = ego.py + dy * (p / steps);
                            stampCircle(4, x, y, 2, val);
                        }
                    }
                }
            }
        }

        // 2: Food
        if (window.foods) {
            for (let i = 0; i < window.foods.length; i++) {
                const f = window.foods[i];
                if (!f) continue;
                const d = Math.hypot(f.rx - hx, f.ry - hy);
                if (d > VIEW_RADIUS) continue;

                const ego = worldToEgo(f.rx, f.ry);
                const val = Math.min(1.0, (f.sz * f.sz) / 5.0);
                stampCircle(2, ego.px, ego.py, Math.max(1, f.sz * PIXELS_PER_UNIT), Math.max(0.3, val));
            }
        }

        // 3: Boundary
        const grd = window.grd || 16384;
        const distFromCenter = Math.hypot(hx - grd, hy - grd);
        if (distFromCenter + VIEW_RADIUS > grd - 300) {
            const offset = 3 * MAP_SIZE * MAP_SIZE;
            for (let y = 0; y < MAP_SIZE; y++) {
                for (let x = 0; x < MAP_SIZE; x++) {
                    const relPx = (x - CENTER) / PIXELS_PER_UNIT;
                    const relPy = -(y - CENTER) / PIXELS_PER_UNIT;
                    const wx = hx + relPx * cos_a - relPy * sin_a;
                    const wy = hy + relPx * sin_a + relPy * cos_a;
                    const wDist = Math.hypot(wx - grd, wy - grd);
                    const dToEdge = grd - wDist;
                    obs[offset + y * MAP_SIZE + x] = Math.max(0, Math.min(1.0, 1.0 - dToEdge / 300.0));
                }
            }
        }

        return obs;
    }

    function generateProprioception(player) {
        const grd = window.grd || 16384;
        const distToWall = grd - Math.hypot(player.xx - grd, player.yy - grd);
        const massProxy = player.sct * 30 || 50;

        const state = new Float32Array(8);
        state[0] = Math.min(1.0, massProxy / 5000.0);
        state[1] = 1.0; // Turn rate proxy
        state[2] = Math.min(1.0, (player.sp || 5.0) / 14.0);
        state[3] = window.setAcceleration ? 1.0 : 0.0;
        state[4] = massProxy > 60 ? 1.0 : 0.0;
        state[5] = Math.min(1.0, Math.max(0.0, distToWall / grd));
        state[6] = Math.min(1.0, (player.pts ? player.pts.length : 10) / 500.0);
        state[7] = 0.0; // Kills unknown in web JS

        return state;
    }

    async function aiLoop() {
        if (!aiEnabled || !session || !window.snake) {
            requestAnimationFrame(aiLoop);
            return;
        }

        const player = window.snake;
        if (player.dead_amt > 0) {
            requestAnimationFrame(aiLoop);
            return;
        }

        // 1. Observe
        const mapData = generateMiniMap(player);
        const stateData = generateProprioception(player);

        // 2. Tensors
        const mapTensor = new ort.Tensor('float32', mapData, [1, 5, 168, 168]);
        const stateTensor = new ort.Tensor('float32', stateData, [1, 8]);

        // 3. Inference
        try {
            const results = await session.run({
                'map': mapTensor,
                'state': stateTensor
            });

            const actions = results['action'].data; // Float32Array [steering, boost]
            const steering = actions[0]; // [-1, 1]
            const boost = actions[1]; // [-1, 1] or [0, 1] depending on squashing

            // 4. Apply Actions
            const currentAngle = player.ang;
            const turnRate = 0.1; // Base turn rate constraint
            const targetAngle = currentAngle + steering * turnRate;

            // Slither.io UI Hook: Override mouse angles
            // xm/ym are mouse offsets from the center of the screen
            window.xm = Math.cos(targetAngle) * 100;
            window.ym = Math.sin(targetAngle) * 100;

            // Apply boost (if action[1] > 0 it means boost)
            if (boost > 0.0) {
                window.setAcceleration(1);
            } else {
                window.setAcceleration(0);
            }

        } catch (e) {
            console.error("ONNX inference failed", e);
        }

        requestAnimationFrame(aiLoop);
    }

    // UI Hook to toggle AI
    window.addEventListener('keydown', (e) => {
        if (e.key === 'q' || e.key === 'Q') {
            aiEnabled = !aiEnabled;
            console.log("AI Enabled:", aiEnabled);

            // Show HUD message directly via DOM
            let hud = document.getElementById('ai-hud');
            if (!hud) {
                hud = document.createElement('div');
                hud.id = 'ai-hud';
                hud.style.position = 'fixed';
                hud.style.top = '10px';
                hud.style.left = '10px';
                hud.style.padding = '10px';
                hud.style.background = 'rgba(0,0,0,0.7)';
                hud.style.color = '#fff';
                hud.style.zIndex = '999999';
                hud.style.fontSize = '24px';
                hud.style.fontFamily = 'monospace';
                hud.style.pointerEvents = 'none';
                document.body.appendChild(hud);
            }
            hud.innerText = "RL Agent: " + (aiEnabled ? "ON" : "OFF");
        }
    });

    // Start everything
    loadModel().then(() => {
        requestAnimationFrame(aiLoop);
    });

})();
