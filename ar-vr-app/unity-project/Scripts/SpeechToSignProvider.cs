using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// SpeechToSignProvider — Handles microphone recording and piping to SignVerse AI.
/// </summary>
public class SpeechToSignProvider : MonoBehaviour
{
    public string apiEndpoint = "http://localhost:8000/translate/speech-to-sign";
    public SignVerseUIManager uiManager;

    private AudioClip recording;
    private bool isRecording = false;

    public void StartListening()
    {
        if (isRecording) return;
        
        Debug.Log("[SpeechToSign] Recording started...");
        recording = Microphone.Start(null, false, 10, 16000);
        isRecording = true;
        uiManager?.UpdateStatus("Listening...", uiManager.searchingColor);
    }

    public void StopListening()
    {
        if (!isRecording) return;

        int lastPos = Microphone.GetPosition(null);
        Microphone.End(null);
        isRecording = false;

        Debug.Log("[SpeechToSign] Recording stopped. Sending to AI Server...");
        StartCoroutine(UploadAudio(recording, lastPos));
    }

    private IEnumerator UploadAudio(AudioClip clip, int lengthSamples)
    {
        uiManager?.UpdateStatus("Processing Voice...", uiManager.searchingColor);

        // Convert AudioClip to WAV bytes
        byte[] wavData = SaveWav.GetWavBytes(clip, lengthSamples);
        
        WWWForm form = new WWWForm();
        form.AddBinaryData("file", wavData, "speech.wav", "audio/wav");

        using (UnityWebRequest request = UnityWebRequest.Post(apiEndpoint, form))
        {
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                // Parse: {"text": "...", "sign_tokens": [...]}
                string json = request.downloadHandler.text;
                SpeechResult result = JsonUtility.FromJson<SpeechResult>(json);
                
                if (result.sign_tokens != null && result.sign_tokens.Count > 0)
                {
                    uiManager?.avatar?.PlaySignSequence(result.sign_tokens);
                    uiManager?.renderer?.RenderTokens(result.sign_tokens);
                    uiManager?.UpdateStatus("Voice Signed!", uiManager.connectedColor);
                }
            }
            else
            {
                Debug.LogError($"[SpeechToSign] Upload failed: {request.error}");
                uiManager?.UpdateStatus("Voice Error", uiManager.disconnectedColor);
            }
        }
    }

    [System.Serializable]
    private class SpeechResult
    {
        public string text;
        public System.Collections.Generic.List<string> sign_tokens;
    }
}

/// <summary>
/// Helper to convert Unity AudioClip to WAV format.
/// </summary>
public static class SaveWav
{
    public static byte[] GetWavBytes(AudioClip clip, int lengthSamples)
    {
        using (MemoryStream stream = new MemoryStream())
        {
            using (BinaryWriter writer = new BinaryWriter(stream))
            {
                int hz = clip.frequency;
                int channels = clip.channels;
                int samples = lengthSamples;

                writer.Write(new char[4] { 'R', 'I', 'F', 'F' });
                writer.Write(36 + samples * 2);
                writer.Write(new char[4] { 'W', 'A', 'V', 'E' });
                writer.Write(new char[4] { 'f', 'm', 't', ' ' });
                writer.Write(16);
                writer.Write((ushort)1);
                writer.Write((ushort)channels);
                writer.Write(hz);
                writer.Write(hz * channels * 2);
                writer.Write((ushort)(channels * 2));
                writer.Write((ushort)16);
                writer.Write(new char[4] { 'd', 'a', 't', 'a' });
                writer.Write(samples * 2);

                float[] data = new float[samples];
                clip.GetData(data, 0);
                
                foreach (float s in data)
                {
                    writer.Write((short)(s * 32767));
                }
            }
            return stream.ToArray();
        }
    }
}
