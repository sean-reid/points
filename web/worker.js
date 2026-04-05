import init, { solve } from '../pkg/points.js';

let ready = false;

self.onmessage = async function(e) {
    if (e.data.type === 'init') {
        await init();
        ready = true;
        self.postMessage({ type: 'ready' });
    } else if (e.data.type === 'query') {
        if (!ready) return;
        const result = solve(e.data.pointsText);
        self.postMessage({ type: 'result', result });
    }
};
