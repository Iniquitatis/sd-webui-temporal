async function apiRequest(url, method, data) {
    return fetch(url, {
        method: method,
        headers: method == "POST" ? {"Content-Type": "application/json"} : undefined,
        body: method == "POST" ? JSON.stringify(data) : undefined,
    })
    .then(response => {
        console.log("RESPONSE", response);

        if (!response.ok) {
            throw new Error(`${response.status}: ${response.statusText}`);
        }

        return response.json();
    });
}

export async function getRequest(url) {
    return apiRequest(url, "GET", null);
}

export async function postRequest(url, data) {
    return apiRequest(url, "POST", data);
}
