function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${alpha})`;
}

// Overlapping area/line chart — shows probability (%) of each final run total.
// Visually distinct from the innings table: smooth curves vs. a grid of numbers.
function renderScoreDistribution(canvasId, homeData, awayData, labels, homeName, awayName, homeColor, awayColor, nSims) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    homeColor = homeColor || '#0064c8';
    awayColor = awayColor || '#c83232';
    const total = nSims || 1000;
    const homePct = homeData.map(v => +(v / total * 100).toFixed(1));
    const awayPct = awayData.map(v => +(v / total * 100).toFixed(1));
    new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: awayName,
                    data: awayPct,
                    borderColor: awayColor,
                    backgroundColor: hexToRgba(awayColor, 0.12),
                    fill: true, tension: 0.35,
                    pointRadius: 2, pointHoverRadius: 5, borderWidth: 2,
                },
                {
                    label: homeName,
                    data: homePct,
                    borderColor: homeColor,
                    backgroundColor: hexToRgba(homeColor, 0.12),
                    fill: true, tension: 0.35,
                    pointRadius: 2, pointHoverRadius: 5, borderWidth: 2,
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { color: '#888', font: { size: 11 } } },
                tooltip: { callbacks: { label: c => c.dataset.label + ': ' + c.parsed.y + '% of sims' } }
            },
            scales: {
                x: { title: { display: true, text: 'Total Runs Scored', color: '#555' }, ticks: { color: '#555' }, grid: { color: '#1a1a1a' } },
                y: { title: { display: true, text: '% of Simulations', color: '#555' }, ticks: { color: '#555', callback: v => v + '%' }, grid: { color: '#1a1a1a' }, min: 0 }
            }
        }
    });
}

// Away on LEFT, Home on RIGHT — matches the score display convention throughout the page.
// awayColor/homeColor default to generic colors if not supplied.
function renderWinProbBar(elementId, homeProb, awayProb, homeName, awayName, awayColor, homeColor) {
    const el = document.getElementById(elementId);
    if (!el) return;
    awayColor = awayColor || '#c83232';
    homeColor = homeColor || '#0064c8';

    const bar = document.createElement('div');
    Object.assign(bar.style, { display: 'flex', height: '36px', borderRadius: '6px', overflow: 'hidden', fontSize: '13px', fontWeight: 'bold' });

    const awayDiv = document.createElement('div');
    Object.assign(awayDiv.style, { width: awayProb + '%', background: awayColor, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', minWidth: '44px', textShadow: '0 1px 2px rgba(0,0,0,0.5)' });
    awayDiv.textContent = awayProb + '%';

    const homeDiv = document.createElement('div');
    Object.assign(homeDiv.style, { width: homeProb + '%', background: homeColor, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', minWidth: '44px', textShadow: '0 1px 2px rgba(0,0,0,0.5)' });
    homeDiv.textContent = homeProb + '%';

    bar.appendChild(awayDiv);
    bar.appendChild(homeDiv);

    const labels = document.createElement('div');
    Object.assign(labels.style, { display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#666', marginTop: '3px' });

    const awayLabel = document.createElement('span');
    awayLabel.textContent = awayName;
    const homeLabel = document.createElement('span');
    homeLabel.textContent = homeName;

    labels.appendChild(awayLabel);
    labels.appendChild(homeLabel);

    el.appendChild(bar);
    el.appendChild(labels);
}
