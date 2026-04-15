const map = L.map("map").setView([23.25, 72.55], 10);
const markersLayer = L.layerGroup().addTo(map);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

const totalCount = document.getElementById("totalCount");
const goodCount = document.getElementById("goodCount");
const moderateCount = document.getElementById("moderateCount");
const poorCount = document.getElementById("poorCount");
const resultList = document.getElementById("resultList");
const statusText = document.getElementById("statusText");
const refreshButton = document.getElementById("refreshButton");

function getColor(status) {
  if (status === "Good") {
    return "#2e8b57";
  }
  if (status === "Moderate") {
    return "#c89a1f";
  }
  return "#c54b3f";
}

function prettyType(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function updateStats(data) {
  const counts = data.reduce(
    (accumulator, item) => {
      accumulator.total += 1;
      if (item.status === "Good") {
        accumulator.good += 1;
      } else if (item.status === "Moderate") {
        accumulator.moderate += 1;
      } else {
        accumulator.poor += 1;
      }
      return accumulator;
    },
    { total: 0, good: 0, moderate: 0, poor: 0 },
  );

  totalCount.textContent = counts.total;
  goodCount.textContent = counts.good;
  moderateCount.textContent = counts.moderate;
  poorCount.textContent = counts.poor;
}

function renderList(data) {
  if (!data.length) {
    resultList.innerHTML = '<div class="empty-state">No detections yet. Run the pipeline to populate the dashboard.</div>';
    return;
  }

  resultList.innerHTML = data
    .slice()
    .reverse()
    .map(
      (item) => `
        <article class="result-item ${item.status.toLowerCase()}">
          <div class="result-title-row">
            <span class="result-type">${prettyType(item.type)}</span>
            <span class="result-score">Score ${item.score}</span>
          </div>
          <p class="result-meta">Status: ${item.status}</p>
          <p class="result-meta">Source: ${item.source} | Frame: ${item.frame_index}</p>
          <p class="result-meta">Captured: ${new Date(item.timestamp).toLocaleString()}</p>
          <p class="result-meta">GPS: ${item.lat.toFixed(4)}, ${item.lng.toFixed(4)}</p>
        </article>
      `,
    )
    .join("");
}

function renderMap(data) {
  markersLayer.clearLayers();

  if (!data.length) {
    map.setView([23.25, 72.55], 10);
    return;
  }

  const bounds = [];
  data.forEach((item) => {
    const color = getColor(item.status);
    const marker = L.circleMarker([item.lat, item.lng], {
      radius: 8,
      color,
      fillColor: color,
      fillOpacity: 0.8,
      weight: 2,
    });

    marker.bindPopup(`
      <strong>${prettyType(item.type)}</strong><br>
      Score: ${item.score}<br>
      Status: ${item.status}<br>
      Source: ${item.source}<br>
      Timestamp: ${new Date(item.timestamp).toLocaleString()}
    `);

    marker.addTo(markersLayer);
    bounds.push([item.lat, item.lng]);
  });

  if (bounds.length === 1) {
    map.setView(bounds[0], 12);
  } else {
    map.fitBounds(bounds, { padding: [30, 30] });
  }
}

async function loadData() {
  statusText.textContent = "Loading data...";

  try {
    const response = await fetch("/data");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    updateStats(data);
    renderList(data);
    renderMap(data);
    statusText.textContent = `Showing ${data.length} detections`;
  } catch (error) {
    statusText.textContent = `Backend unavailable: ${error.message}`;
    resultList.innerHTML = '<div class="empty-state">Start the Flask backend and refresh the dashboard.</div>';
    markersLayer.clearLayers();
  }
}

refreshButton.addEventListener("click", loadData);
window.addEventListener("load", loadData);
window.setInterval(loadData, 5000);
