// Mock document: companion service sketch, not part of the Python pipeline.
// Simulates a lightweight webhook that would receive Gmail push
// notifications (via Cloud Pub/Sub) and forward them to the Python
// ingestion service instead of relying on the CLI's polling `sync-gmail`.

package com.projectx.email.webhook;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class EmailWebhookListener {

    private static final String INGESTION_SERVICE_URL =
        "http://localhost:8000/internal/ingest-email";

    private final HttpClient httpClient;

    public EmailWebhookListener() {
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();
    }

    /**
     * Called by the Pub/Sub push endpoint whenever Gmail reports a mailbox
     * change (new message, label change, etc.) for a watched account.
     */
    public void onGmailNotification(String historyId, String emailAddress) throws Exception {
        String payload = String.format(
            "{\"historyId\": \"%s\", \"emailAddress\": \"%s\"}",
            historyId, emailAddress
        );

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(INGESTION_SERVICE_URL))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(payload))
            .build();

        HttpResponse<String> response =
            httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new RuntimeException(
                "Ingestion service rejected notification: " + response.statusCode()
            );
        }
    }
}
