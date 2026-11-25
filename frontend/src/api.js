async function request(path, options) {
  const res = await fetch(`/api${path}`, options)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${res.status})`)
  }
  return res.status === 204 ? null : res.json()
}

export const api = {
  health: () => request('/health'),
  documents: () => request('/documents'),
  upload: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/documents', { method: 'POST', body: form })
  },
  remove: (id) => request(`/documents/${id}`, { method: 'DELETE' }),
  ask: (question) =>
    request('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    }),
}
