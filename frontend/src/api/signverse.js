import axios from 'axios'

const backendBase = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001'
const inferenceBase = import.meta.env.VITE_INFERENCE_URL || 'http://localhost:8000'

const backend = axios.create({
  baseURL: backendBase,
  timeout: 15000,
})

const inference = axios.create({
  baseURL: inferenceBase,
  timeout: 20000,
})

export function getBackendHealth() {
  return backend.get('/health').then(r => r.data)
}

export function getDashboardJson() {
  return backend.get('/dashboard/json').then(r => r.data)
}

export function getInferenceHealth() {
  return inference.get('/health').then(r => r.data)
}

export function textToSign(text) {
  return inference.post('/translate/text-to-sign', { text }).then(r => r.data)
}

export function signToText(sequence) {
  return inference.post('/translate/sign-to-text', { sequence }).then(r => r.data)
}

export function speechToSign(file) {
  const form = new FormData()
  form.append('file', file)
  return inference.post('/speech-to-sign/', form).then(r => r.data)
}

export function getHistory(userId, token) {
  return backend.get(`/api/translations/history/${userId}`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  }).then(r => r.data)
}
