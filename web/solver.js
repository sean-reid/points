const DEBOUNCE_MS = 30;

export class Solver {
    constructor({ onResult, onReady }) {
        this.onResult = onResult;
        this.onReady = onReady || (() => {});
        this.worker = null;
        this.ready = false;
        this.debounceTimer = null;
        this.pending = null;
        this._init();
    }

    _init() {
        this.worker = new Worker('./worker.js', { type: 'module' });
        this.worker.onmessage = (e) => {
            if (e.data.type === 'ready') {
                this.ready = true;
                this.onReady();
                if (this.pending) { this._query(this.pending); this.pending = null; }
            } else if (e.data.type === 'result') {
                this._handleResult(e.data.result);
            }
        };
        this.worker.postMessage({ type: 'init' });
    }

    _handleResult(raw) {
        if (raw === null || raw === undefined) {
            this.onResult(null);
            return;
        }
        // Parse: "equation\tcoeffs\tmonos\tscale"
        const parts = raw.split('\t');
        if (parts.length !== 4) { this.onResult(null); return; }

        const equation = parts[0];
        const coefficients = parts[1].split(',').map(Number);
        const monomials = parts[2].split(',').map(s => {
            const [i, j] = s.split(':').map(Number);
            return [i, j];
        });
        const scale = Number(parts[3]);

        this.onResult({ equation, coefficients, monomials, scale });
    }

    solve(points) {
        clearTimeout(this.debounceTimer);
        if (points.length < 1) { this.onResult(null); return; }
        this.debounceTimer = setTimeout(() => {
            if (this.ready) this._query(points);
            else this.pending = points;
        }, DEBOUNCE_MS);
    }

    _query(points) {
        const text = points.map(p => `${p.x} ${p.y}`).join('\n');
        this.worker.postMessage({ type: 'query', pointsText: text });
    }
}
