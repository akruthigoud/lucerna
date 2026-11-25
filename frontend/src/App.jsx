import { useEffect, useState } from 'react'
import { api } from './api'

const formatDate = (iso) =>
  new Date(iso).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })

export default function App() {
  const [docs, setDocs] = useState([])
  const [health, setHealth] = useState(null)
  const [history, setHistory] = useState([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(null) // 'upload' | 'ask' | null
  const [error, setError] = useState(null)

  const refresh = async () => {
    try {
      setDocs(await api.documents())
      setHealth(await api.health())
    } catch {
      setHealth(null)
    }
  }

  useEffect(() => { refresh() }, [])

  const run = async (kind, fn) => {
    setError(null)
    setBusy(kind)
    try {
      await fn()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(null)
    }
  }

  const onUpload = (file) =>
    file && run('upload', async () => { await api.upload(file); await refresh() })

  const onDelete = (id) => run('delete', async () => { await api.remove(id); await refresh() })

  const onAsk = (e) => {
    e.preventDefault()
    const q = question.trim()
    if (!q) return
    run('ask', async () => {
      const result = await api.ask(q)
      setHistory((h) => [{ question: q, ...result }, ...h])
      setQuestion('')
    })
  }

  return (
    <div className="shell">
      <header>
        <div className="brand">
          <span className="mark">✦</span>
          <h1>Lucerna</h1>
          <span className="tagline">document q&amp;a</span>
        </div>
        <div className="status">
          <span className={`dot ${health ? 'on' : 'off'}`} />
          {health ? `${health.llm_provider} · ${health.chunks} chunks` : 'backend offline'}
        </div>
      </header>

      <div className="layout">
        <aside>
          <h2>Documents</h2>
          <label
            className="drop"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); onUpload(e.dataTransfer.files[0]) }}
          >
            {busy === 'upload' ? 'Indexing…' : 'Drop a PDF, DOCX, TXT or MD file, or click to browse'}
            <input type="file" accept=".pdf,.docx,.txt,.md" hidden onChange={(e) => onUpload(e.target.files[0])} />
          </label>

          {docs.length === 0 ? (
            <p className="muted">No documents yet.</p>
          ) : (
            <ul className="docs">
              {docs.map((d) => (
                <li key={d.id}>
                  <div>
                    <span className="name">{d.filename}</span>
                    <span className="meta">{d.chunks} chunks · {formatDate(d.uploaded_at)}</span>
                  </div>
                  <button className="remove" title="Remove" onClick={() => onDelete(d.id)}>×</button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <main>
          <form className="ask" onSubmit={onAsk}>
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={docs.length ? 'Ask a question about your documents' : 'Upload a document first'}
              disabled={!docs.length || busy === 'ask'}
            />
            <button disabled={!docs.length || !question.trim() || busy === 'ask'}>
              {busy === 'ask' ? '…' : 'Ask'}
            </button>
          </form>

          {error && <p className="error">{error}</p>}

          {history.map((h, i) => (
            <article key={i}>
              <div>
                <p className="q">{h.question}</p>
                <p className="answer">{h.answer}</p>
                <p className="meta">{h.provider} · {h.latency_ms} ms · {h.citations.length} sources</p>
              </div>
              <div className="sources">
                {h.citations.map((c, j) => (
                  <div key={j}>
                    <span className="src">{c.filename} · chunk {c.chunk} · {c.score.toFixed(2)}</span>
                    <span className="excerpt">{c.text.length > 220 ? c.text.slice(0, 220) + '…' : c.text}</span>
                  </div>
                ))}
              </div>
            </article>
          ))}
        </main>
      </div>
    </div>
  )
}
