function renderDistributionChart(canvasId, homeData, awayData, labels, homeName, awayName) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: homeName,
                    data: homeData,
                    backgroundColor: 'rgba(0, 100, 200, 0.6)',
                    borderColor: 'rgba(0, 100, 200, 1)',
                    borderWidth: 1,
                },
                {
                    label: awayName,
                    data: awayData,
                    backgroundColor: 'rgba(200, 50, 50, 0.6)',
                    borderColor: 'rgba(200, 50, 50, 1)',
                    borderWidth: 1,
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Run Distribution (1000 Simulations)' }
            },
            scales: {
                x: { title: { display: true, text: 'Runs Scored' } },
                y: { title: { display: true, text: 'Simulations' } }
            }
        }
    });
}

function renderWinProbBar(elementId, homeProb, awayProb, homeName, awayName) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const bar = document.createElement('div');
    Object.assign(bar.style, { display: 'flex', height: '32px', borderRadius: '6px', overflow: 'hidden', fontSize: '13px', fontWeight: 'bold' });

    const homeDiv = document.createElement('div');
    Object.assign(homeDiv.style, { width: homeProb + '%', background: '#0064c8', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', minWidth: '40px' });
    homeDiv.textContent = homeProb + '%';

    const awayDiv = document.createElement('div');
    Object.assign(awayDiv.style, { width: awayProb + '%', background: '#c83232', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', minWidth: '40px' });
    awayDiv.textContent = awayProb + '%';

    bar.appendChild(homeDiv);
    bar.appendChild(awayDiv);

    const labels = document.createElement('div');
    Object.assign(labels.style, { display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#666', marginTop: '2px' });

    const homeLabel = document.createElement('span');
    homeLabel.textContent = homeName;
    const awayLabel = document.createElement('span');
    awayLabel.textContent = awayName;

    labels.appendChild(homeLabel);
    labels.appendChild(awayLabel);

    el.appendChild(bar);
    el.appendChild(labels);
}
