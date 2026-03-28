import axios from 'axios'

const backendBase = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001'
const inferenceBase = import.meta.env.VITE_INFERENCE_URL || 'http://localhost:8000'
const adminToken = import.meta.env.VITE_ADMIN_TOKEN || ''

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

function adminHeaders() {
  if (!adminToken) {
    throw new Error('Missing VITE_ADMIN_TOKEN')
  }
  return { 'x-admin-token': adminToken }
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

export function classifyGestureSequence(sequence) {
  return inference.post('/gesture/classify-sequence', { sequence }).then(r => r.data)
}

export function classifyGestureImageCnn(file, minConfidence = 0.4) {
  const form = new FormData()
  form.append('file', file)
  return inference.post(`/gesture/classify-image-cnn?min_confidence=${encodeURIComponent(minConfidence)}`, form).then(r => r.data)
}

export function classifyGestureVideoLstm(file, minConfidence = 0.4) {
  const form = new FormData()
  form.append('file', file)
  return inference.post(`/gesture/classify-video-lstm?min_confidence=${encodeURIComponent(minConfidence)}`, form).then(r => r.data)
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

export function getTextGlossPipelineStatus() {
  return backend.get('/admin/pipeline/text-gloss/status', {
    headers: adminHeaders(),
  }).then(r => r.data)
}

export function triggerTextGlossPipeline(payload = {}) {
  return backend.post('/admin/pipeline/text-gloss', payload, {
    headers: adminHeaders(),
  }).then(r => r.data)
}
