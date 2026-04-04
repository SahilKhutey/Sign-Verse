package com.signverse.launcher.network

import java.io.ByteArrayOutputStream
import java.nio.charset.Charset


internal class MultipartBody(private val boundary: String) {
    private val out = ByteArrayOutputStream()
    private val utf8: Charset = Charsets.UTF_8

    fun addFile(
        fieldName: String,
        filename: String,
        contentType: String,
        bytes: ByteArray,
    ) {
        writeLine("--$boundary")
        writeLine(
            "Content-Disposition: form-data; name=\"$fieldName\"; filename=\"$filename\""
        )
        writeLine("Content-Type: $contentType")
        writeLine("")
        out.write(bytes)
        writeLine("")
    }

    fun build(): ByteArray {
        writeLine("--$boundary--")
        return out.toByteArray()
    }

    private fun writeLine(s: String) {
        out.write(s.toByteArray(utf8))
        out.write("\r\n".toByteArray(utf8))
    }
}

