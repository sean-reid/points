import { Grid } from './grid.js';
import { Solver } from './solver.js';

const pointCountEl = document.getElementById('pointCount');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const resultContent = document.getElementById('resultContent');

const solver = new Solver({
    onReady() {},
    onResult(result) {
        if (!result) {
            resultsSection.classList.add('hidden');
            grid.clearCurve();
            return;
        }

        resultsSection.classList.remove('hidden');
        resultContent.innerHTML = `
            <div class="result-box success">
                <div class="result-expression">${result.equation}</div>
            </div>
        `;

        // Build implicit curve function h(x,y) for zero-contour rendering
        // Solver works in scaled coords, so evaluate at (x*scale, y*scale)
        const { coefficients, monomials, scale } = result;
        grid.setCurve(({x, y}) => {
            const sx = x * scale, sy = y * scale;
            let sum = 0;
            for (let k = 0; k < coefficients.length; k++) {
                sum += coefficients[k]
                    * Math.pow(sx, monomials[k][0])
                    * Math.pow(sy, monomials[k][1]);
            }
            return sum;
        });
    }
});

function onPointsChanged(points) {
    const n = points.length;
    if (n === 0) {
        pointCountEl.textContent = 'Click grid to place points';
        clearBtn.classList.add('hidden');
        resultsSection.classList.add('hidden');
        grid.clearCurve();
    } else {
        pointCountEl.textContent = `${n} point${n === 1 ? '' : 's'}`;
        clearBtn.classList.remove('hidden');
    }
    solver.solve(points);
}

const grid = new Grid(document.getElementById('gridCanvas'), onPointsChanged);
clearBtn.addEventListener('click', () => grid.clear());
