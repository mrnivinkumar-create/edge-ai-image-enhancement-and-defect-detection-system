document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("qualityChart");
  if (!canvas || typeof Chart === "undefined") return;
  const labels = JSON.parse(canvas.dataset.labels).map((label) => label.replaceAll("_", " "));
  const values = JSON.parse(canvas.dataset.values);
  new Chart(canvas, { type: "doughnut", data: { labels, datasets: [{ data: values, backgroundColor: ["#2369d8", "#51d4e8", "#1bb88a", "#f2a93b", "#8a63d2", "#f05252"], borderWidth: 0, hoverOffset: 8 }] }, options: { cutout: "68%", plugins: { legend: { position: "bottom", labels: { padding: 14, usePointStyle: true } } } } });
});
