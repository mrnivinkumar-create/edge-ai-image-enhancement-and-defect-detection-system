document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("dropZone");
  const input = document.getElementById("imageInput");
  const preview = document.getElementById("imagePreview");
  const prompt = document.getElementById("uploadPrompt");
  const fileName = document.getElementById("fileName");
  const form = document.getElementById("uploadForm");
  if (!dropZone || !input) return;

  const showFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    preview.src = URL.createObjectURL(file);
    preview.classList.remove("d-none");
    prompt.classList.add("d-none");
  };
  input.addEventListener("change", () => showFile(input.files[0]));
  ["dragenter", "dragover"].forEach((event) => dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((event) => dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.remove("dragging"); }));
  dropZone.addEventListener("drop", (event) => { input.files = event.dataTransfer.files; showFile(input.files[0]); });
  form?.addEventListener("submit", () => { const button = form.querySelector("button[type=submit]"); button.disabled = true; button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>ANALYZING IMAGE...'; });
});
