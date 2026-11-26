// La interfaz presenta los datos. La detección y la CNN se ejecutan en Python.
const $ = (id) => document.getElementById(id);
const percent = new Intl.NumberFormat("es", {
  style: "percent",
  maximumFractionDigits: 1,
});

let selectedFile = null;
let previewUrl = null;
let prediction = null;
let busy = false;

// PASO 1. Limpiar el resultado al cambiar de fotografía o modo de recorte.
function resetResults() {
  prediction = null;
  $("result-empty").hidden = false;
  $("result-content").hidden = true;
  $("result-status").textContent = "Sin analizar";
  $("boxes").replaceChildren();
  $("error").hidden = true;
}

function selectFile(file) {
  if (!file || busy) return;
  resetResults();

  // Esta validación ayuda al usuario; Python vuelve a comprobar el contenido real.
  if (!/^image\/(jpeg|png|webp)$/.test(file.type) || file.size > 12 * 1024 * 1024) {
    $("error").textContent = "Elige una imagen JPG, PNG o WEBP de hasta 12 MB.";
    $("error").hidden = false;
    selectedFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    $("preview").removeAttribute("src");
    $("empty-photo").hidden = false;
    $("image-stage").hidden = true;
    $("file-detail").hidden = true;
    $("analyze").disabled = true;
    $("live-status").textContent = $("error").textContent;
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $("preview").src = previewUrl;
  $("empty-photo").hidden = true;
  $("image-stage").hidden = false;
  $("file-detail").hidden = false;
  $("file-name").textContent = file.name;
  $("analyze").disabled = false;
  $("live-status").textContent = "Fotografía lista para analizar.";
}

// PASO 2. Presentar las siete puntuaciones del rostro elegido.
function showFace(index) {
  const face = prediction.faces[index];
  $("emotion").textContent = face.label;
  $("top-score").textContent = percent.format(face.scores[0].score);
  $("face-description").textContent = prediction.mode === "cropped"
    ? "Análisis del rostro recortado que seleccionaste."
    : `Rostro ${face.id} de ${prediction.faces.length} analizados.`
      + (prediction.detected_faces > prediction.faces.length
        ? " Se muestran los cinco rostros más grandes." : "");

  $("scores").replaceChildren();
  for (const score of face.scores) {
    const row = document.createElement("div");
    row.className = "score-row";
    const label = document.createElement("div");
    label.className = "score-label";
    const name = document.createElement("span");
    name.textContent = score.label;
    const value = document.createElement("span");
    value.textContent = percent.format(score.score);
    label.append(name, value);

    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("div");
    fill.className = "score-fill";
    fill.style.width = `${Math.max(0, Math.min(100, score.score * 100))}%`;
    track.append(fill);
    row.append(label, track);
    $("scores").append(row);
  }

  [...$("face-tabs").children].forEach((tab, i) => {
    tab.classList.toggle("active", i === index);
    tab.setAttribute("aria-pressed", String(i === index));
  });
  [...$("boxes").children].forEach((box, i) => {
    box.classList.toggle("active", i === index);
  });
}

// PASO 3. Dibujar los rectángulos sobre la vista previa orientada por Python.
function showResult(result) {
  prediction = result;
  $("preview").src = result.preview;
  $("result-empty").hidden = true;
  $("result-content").hidden = false;
  $("result-status").textContent = "Análisis listo";
  $("face-tabs").replaceChildren();
  $("boxes").replaceChildren();

  result.faces.forEach((face, index) => {
    if (result.faces.length > 1) {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "face-tab";
      tab.textContent = `Rostro ${face.id}`;
      tab.addEventListener("click", () => showFace(index));
      $("face-tabs").append(tab);
    }

    const box = document.createElement("div");
    box.className = "face-box";
    Object.assign(box.style, {
      left: `${face.box.x / result.width * 100}%`,
      top: `${face.box.y / result.height * 100}%`,
      width: `${face.box.width / result.width * 100}%`,
      height: `${face.box.height / result.height * 100}%`,
    });
    const number = document.createElement("span");
    number.className = "face-number";
    number.textContent = `Rostro ${face.id}`;
    box.append(number);
    $("boxes").append(box);
  });

  showFace(0);
  $("live-status").textContent =
    `Análisis terminado. Expresión estimada: ${result.faces[0].label}.`;
}

// PASO 4. Conectar el selector, el cambio de foto y arrastrar/soltar.
$("photo").addEventListener("change", (event) => selectFile(event.target.files[0]));
$("change-photo").addEventListener("click", () => {
  if (!busy) $("photo").click();
});
$("cropped").addEventListener("change", resetResults);

for (const type of ["dragenter", "dragover"]) {
  $("drop-zone").addEventListener(type, (event) => {
    event.preventDefault();
    if (!busy) $("drop-zone").classList.add("dragging");
  });
}
for (const type of ["dragleave", "drop"]) {
  $("drop-zone").addEventListener(type, (event) => {
    event.preventDefault();
    $("drop-zone").classList.remove("dragging");
  });
}
$("drop-zone").addEventListener("drop", (event) => {
  selectFile(event.dataTransfer.files[0]);
});

// PASO 5. Enviar una foto al servidor local y presentar su respuesta.
// Bloquear los controles evita mezclar una predicción pendiente con otra imagen.
$("analyze").addEventListener("click", async () => {
  if (!selectedFile || busy) return;
  resetResults();
  busy = true;
  $("analyze").disabled = true;
  $("photo").disabled = true;
  $("cropped").disabled = true;
  $("button-text").textContent = "Analizando…";
  $("result-status").textContent = "Procesando fotografía…";
  $("live-status").textContent =
    "Analizando la fotografía. La primera carga puede tardar unos segundos.";

  const form = new FormData();
  form.append("photo", selectedFile);
  form.append("cropped", String($("cropped").checked));

  try {
    const response = await fetch("/api/predict", {method: "POST", body: form});
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "No se pudo analizar la fotografía.");
    }
    showResult(result);
  } catch (error) {
    $("error").textContent = error.message === "Failed to fetch"
      ? "No se pudo conectar con la aplicación. Revisa que siga abierta en la terminal."
      : error.message;
    $("error").hidden = false;
    $("result-status").textContent = "Sin resultado";
    $("live-status").textContent = $("error").textContent;
  } finally {
    busy = false;
    $("analyze").disabled = !selectedFile;
    $("photo").disabled = false;
    $("cropped").disabled = false;
    $("button-text").textContent = "Analizar expresión";
  }
});
