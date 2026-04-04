package com.signverse.launcher.data

data class AppSettings(
    val apiServerBaseUrl: String = "http://10.0.2.2:8000",
    // Kept for future: api_gateway (speech-to-sign) is separate in this repo.
    val apiGatewayBaseUrl: String = "http://10.0.2.2:8001",
)

