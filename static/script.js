const form = document.getElementById("predictForm");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const body = {};

    formData.forEach((value, key) => body[key] = Number(value));

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    });

    const data = await res.json();

    document.getElementById("result").innerText =
        data.diabetes ? "Diabetes Positive" : "Diabetes Negative";
});
