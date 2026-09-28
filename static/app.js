const resultValue = document.querySelector("#result-value");
const resultCategory = document.querySelector("#result-category");
const resultMessage = document.querySelector("#result-message");
const resultDominant = document.querySelector("#result-dominant");
const resultKicker = document.querySelector("#result-kicker");
const breakdown = document.querySelector("#breakdown");
const notes = document.querySelector("#notes");

function formValues(form) {
  const payload = {};
  for (const element of form.elements) {
    if (!element.name) continue;
    const text = element.value.trim();
    if (text !== "") payload[element.name] = Number(text);
  }
  return payload;
}

function showError(message) {
  resultKicker.textContent = "Could not calculate";
  resultValue.textContent = "—";
  resultValue.style.color = "";
  resultCategory.textContent = "Check the inputs";
  resultCategory.style.background = "#efe8dc";
  resultCategory.style.color = "";
  resultMessage.textContent = message;
  resultDominant.hidden = true;
  breakdown.hidden = true;
  notes.hidden = true;
}

function showReport(report) {
  resultKicker.textContent = report.mode === "estimate" ? "Next-hour prediction" : "EPA calculation";
  resultValue.textContent = String(report.aqi);
  resultValue.style.color = report.text_color === "#ffffff" ? report.color : "";
  resultCategory.textContent = report.category;
  resultCategory.style.background = report.color;
  resultCategory.style.color = report.text_color;
  resultMessage.textContent = report.message;
  if (report.dominant && report.dominant.length) {
    resultDominant.hidden = false;
    resultDominant.textContent = `Dominant pollutant: ${report.dominant.join(", ")}`;
  } else if (report.site) {
    resultDominant.hidden = false;
    resultDominant.textContent = report.site;
  } else {
    resultDominant.hidden = true;
  }

  breakdown.replaceChildren();
  if (report.pollutants && report.pollutants.length) {
    breakdown.hidden = false;
    for (const item of report.pollutants) {
      const row = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = item.name;
      const track = document.createElement("div");
      track.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${Math.min(100, (item.aqi / 500) * 100)}%`;
      fill.style.background = report.color;
      track.append(fill);
      const value = document.createElement("span");
      value.textContent = String(item.aqi);
      row.append(name, track, value);
      breakdown.append(row);
    }
  } else {
    breakdown.hidden = true;
  }

  notes.replaceChildren();
  const extra = [...(report.notes || [])];
  if (report.extrapolated) {
    extra.push("Temperature or humidity is outside the range seen at the training station.");
  }
  if (extra.length) {
    notes.hidden = false;
    for (const text of extra) {
      const item = document.createElement("li");
      item.textContent = text;
      notes.append(item);
    }
  } else {
    notes.hidden = true;
  }
}

async function post(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed.");
  return data;
}

document.querySelector("#calculate-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = formValues(event.currentTarget);
  if (Object.keys(payload).length === 0) {
    showError("Enter at least one pollutant concentration.");
    return;
  }
  try {
    showReport(await post("/api/calculate", payload));
  } catch (error) {
    showError(error.message);
  }
});

document.querySelector("#estimate-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = formValues(event.currentTarget);
  try {
    showReport(await post("/api/estimate", payload));
  } catch (error) {
    showError(error.message);
  }
});

document.querySelector("#example-concentrations").addEventListener("click", () => {
  const form = document.querySelector("#calculate-form");
  form.elements.pm25.value = "28.4";
  form.elements.ozone_8hr.value = "0.062";
  form.elements.no2.value = "40";
  form.requestSubmit();
});

document.querySelector("#example-weather").addEventListener("click", () => {
  const form = document.querySelector("#estimate-form");
  form.elements.current_aqi.value = "41";
  form.elements.temperature_c.value = "10.2";
  form.elements.relative_humidity.value = "59.6";
  form.elements.hour.value = "8";
  form.elements.month.value = "3";
  form.requestSubmit();
});
