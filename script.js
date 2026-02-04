function predict() {
    const data = {
        name: document.getElementById("name").value,
        year: parseInt(document.getElementById("year").value),
        fuel: document.getElementById("fuel").value,
        seller_type: document.getElementById("seller_type").value,
        km_driven: parseInt(document.getElementById("km_driven").value),
        transmission: document.getElementById("transmission").value,
        owner: document.getElementById("owner").value,
    };

    fetch("http://127.0.0.1:8000/predict", { 
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(res => {
        if (!res.ok) {
            throw new Error("Server error: " + res.status);
        }
        return res.json();
    })
    .then(response => {
        if (response.error) {
            document.getElementById("result").innerHTML = 
                "Error: " + response.error;
        } else {
            document.getElementById("result").innerHTML =
                "Predicted Price: ₹ " + response.price + "'<br> R2_SCORE : " + response.r2;
        }
    })
    .catch(err => {
        document.getElementById("result").innerHTML = "Error: " + err;
    });
}
