// Interactive grid for placing points + curve overlay

const GRID_MIN = -10;
const GRID_MAX = 10;
const GRID_RANGE = GRID_MAX - GRID_MIN;
const SNAP = 1; // Snap to integers
const SNAP_RADIUS = 0.4; // Tolerance for toggling off an existing point
const POINT_RADIUS = 0.5; // Visual radius of each point "circle"

export class Grid {
    constructor(canvas, onChange) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.points = []; // [{x, y}] — not snapped to integers
        this.onChange = onChange;
        this.curveFn = null;

        this._setupHiDPI();
        this._bindEvents();
        this.draw();
    }

    _setupHiDPI() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this.cssWidth = rect.width;
        this.cssHeight = rect.height;
    }

    _bindEvents() {
        this.canvas.addEventListener('pointerdown', (e) => this._handleClick(e));
        new ResizeObserver(() => { this._setupHiDPI(); this.draw(); }).observe(this.canvas);
    }

    _handleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;

        // Convert pixel to grid coordinates, snap to nearest quarter-integer
        let gx = GRID_MIN + (px / this.cssWidth) * GRID_RANGE;
        let gy = GRID_MAX - (py / this.cssHeight) * GRID_RANGE;
        gx = Math.round(gx / SNAP) * SNAP;
        gy = Math.round(gy / SNAP) * SNAP;

        if (gx < GRID_MIN || gx > GRID_MAX || gy < GRID_MIN || gy > GRID_MAX) return;

        // Toggle: remove if clicking near existing point
        const removeIdx = this.points.findIndex(p =>
            Math.abs(p.x - gx) < SNAP_RADIUS && Math.abs(p.y - gy) < SNAP_RADIUS
        );
        if (removeIdx >= 0) {
            this.points.splice(removeIdx, 1);
        } else {
            this.points.push({ x: gx, y: gy });
        }

        this.draw();
        this.onChange(this.getPoints());
    }

    getPoints() { return [...this.points]; }

    clear() {
        this.points = [];
        this.curveFn = null;
        this.draw();
        this.onChange(this.getPoints());
    }

    setCurve(diffFn) { this.curveFn = diffFn; this.draw(); }
    clearCurve() { this.curveFn = null; this.draw(); }

    // Convert grid coords to pixel coords
    _toPixel(gx, gy) {
        return [
            ((gx - GRID_MIN) / GRID_RANGE) * this.cssWidth,
            ((GRID_MAX - gy) / GRID_RANGE) * this.cssHeight,
        ];
    }

    draw() {
        const { ctx, cssWidth: w, cssHeight: h } = this;
        const cellW = w / GRID_RANGE;
        const cellH = h / GRID_RANGE;

        ctx.clearRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = '#f0f0f0';
        ctx.lineWidth = 1;
        for (let i = 0; i <= GRID_RANGE; i++) {
            const x = i * cellW, y = i * cellH;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }

        // Axes
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1.5;
        const [ox, oy] = this._toPixel(0, 0);
        ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(w, oy); ctx.stroke();

        // Axis labels
        ctx.fillStyle = '#ccc';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (let v = GRID_MIN; v <= GRID_MAX; v += 5) {
            if (v === 0) continue;
            const [lx] = this._toPixel(v, 0);
            ctx.fillText(v, lx, oy + 4);
        }
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let v = GRID_MIN; v <= GRID_MAX; v += 5) {
            if (v === 0) continue;
            const [, ly] = this._toPixel(0, v);
            ctx.fillText(v, ox - 4, ly);
        }

        // Curve (zero contour)
        if (this.curveFn) this._drawCurve(ctx, w, h);

        // Points with tolerance circles
        for (const { x: gx, y: gy } of this.points) {
            const [px, py] = this._toPixel(gx, gy);
            const r = (POINT_RADIUS / GRID_RANGE) * w;

            // Tolerance circle (faint)
            ctx.strokeStyle = 'rgba(0,0,0,0.15)';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.stroke();

            // Point dot
            ctx.fillStyle = '#111';
            ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }

    _drawCurve(ctx, w, h) {
        const fn_ = this.curveFn;
        const steps = 800;
        const dx = GRID_RANGE / steps;

        ctx.strokeStyle = '#4a9d5f';
        ctx.lineWidth = 2.5;
        ctx.beginPath();

        for (let i = 0; i < steps; i++) {
            for (let j = 0; j < steps; j++) {
                const gx = GRID_MIN + i * dx;
                const gy = GRID_MIN + j * dx;

                const v00 = fn_({ x: gx, y: gy });
                const v10 = fn_({ x: gx + dx, y: gy });
                const v01 = fn_({ x: gx, y: gy + dx });
                const v11 = fn_({ x: gx + dx, y: gy + dx });

                if (!isFinite(v00) || !isFinite(v10) || !isFinite(v01) || !isFinite(v11)) continue;

                const signs = [Math.sign(v00), Math.sign(v10), Math.sign(v01), Math.sign(v11)];
                if (signs.some(s => s === 0) || new Set(signs.filter(s => s !== 0)).size > 1) {
                    const crossings = [];
                    const corners = [
                        { x: gx, y: gy }, { x: gx + dx, y: gy },
                        { x: gx + dx, y: gy + dx }, { x: gx, y: gy + dx }
                    ];
                    const vals = [v00, v10, v11, v01];

                    for (let e = 0; e < 4; e++) {
                        const va = vals[e], vb = vals[(e + 1) % 4];
                        if (va === vb) continue;
                        const t = va / (va - vb);
                        if (t < 0 || t > 1) continue;
                        const ca = corners[e], cb = corners[(e + 1) % 4];
                        crossings.push({
                            x: ca.x + t * (cb.x - ca.x),
                            y: ca.y + t * (cb.y - ca.y)
                        });
                    }

                    for (let a = 0; a < crossings.length; a += 2) {
                        if (a + 1 >= crossings.length) break;
                        const [px1, py1] = this._toPixel(crossings[a].x, crossings[a].y);
                        const [px2, py2] = this._toPixel(crossings[a+1].x, crossings[a+1].y);
                        ctx.moveTo(px1, py1);
                        ctx.lineTo(px2, py2);
                    }
                }
            }
        }
        ctx.stroke();
    }
}
