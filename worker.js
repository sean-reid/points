// Web Worker for running WASM search in background thread
import init, { search_equivalence } from './pkg/points.js';

let wasmInitialized = false;

self.onmessage = async function(e) {
    const { pointsText, maxSize } = e.data;
    
    try {
        // Initialize WASM if needed
        if (!wasmInitialized) {
            await init();
            wasmInitialized = true;
        }
        
        // Log callback sends messages back to main thread
        const logCallback = (msg) => {
            self.postMessage({ type: 'log', message: msg });
        };
        
        // Run the search
        const result = search_equivalence(pointsText, maxSize, logCallback);
        
        // Send result back to main thread
        self.postMessage({ 
            type: 'result', 
            result: {
                found: result.found,
                f_expr: result.f_expr,
                g_expr: result.g_expr,
                f_size: result.f_size,
                g_size: result.g_size,
                var_signature: result.var_signature,
                errors: Array.from(result.errors)
            }
        });
    } catch (error) {
        self.postMessage({ 
            type: 'error', 
            message: error.message 
        });
    }
};
