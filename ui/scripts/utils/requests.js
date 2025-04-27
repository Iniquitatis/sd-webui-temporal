const DEBUG = false;

async function apiRequest(url, method, data = null, successCallback = null, errorCallback = null) {
    if (DEBUG) console.log(method, url, data);

    return fetch(url, {
        method: method,
        headers: method == "POST" ? {"Content-Type": "application/json"} : undefined,
        body: method == "POST" ? JSON.stringify(data ?? {}) : undefined,
    })
    .then(async (response) => {
        if (DEBUG) console.log("RESPONSE", response);

        if (!response.ok) {
            throw new Error(`${response.status} (${response.statusText}): ${await response.text()}`);
        }

        return response.json();
    })
    .then((json) => {
        if (DEBUG) console.log("DATA", json);

        if (successCallback) {
            successCallback(json);
        }

        return json;
    })
    .catch((reason) => {
        if (DEBUG) console.log("ERROR", reason);

        if (errorCallback) {
            errorCallback();
        }
    });
}

export async function getRequest(url, successCallback = null, errorCallback = null) {
    return apiRequest(url, "GET", null, successCallback, errorCallback);
}

export async function postRequest(url, data = null, successCallback = null, errorCallback = null) {
    return apiRequest(url, "POST", data, successCallback, errorCallback);
}
