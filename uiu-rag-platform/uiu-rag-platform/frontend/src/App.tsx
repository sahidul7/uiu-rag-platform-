// UIU Academic RAG Platform — Complete Production Frontend
// Features: Google OAuth, file upload, streaming chat, citations, error handling

import React, { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  Send, Menu, X, Trash2, LogOut, Upload, RefreshCw,
  BookOpen, FileText, HelpCircle, BookMarked, FlaskConical,
  Loader2, Database, Layers, ChevronRight, AlertCircle,
  CheckCircle2, User as UserIcon, Shield, Plus
} from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────
interface User { id: string; email: string; full_name: string | null; is_admin: boolean }
interface Citation {
  number: number; source_ref: string; chunk_id: string
  source_filename: string; document_type: string
  page_number: number | null; slide_number: number | null
  section_heading: string | null; course: string; content_preview: string
}
interface Message {
  id: string; role: 'user' | 'assistant'; content: string
  citations: Citation[]; latency_ms?: number; isStreaming?: boolean
  error?: boolean
}

// ── Constants ─────────────────────────────────────────────────────────────────
const API = import.meta.env.VITE_API_BASE_URL || '/api/v1'
const DOC_TYPES: Record<string, { label: string; icon: React.ReactNode }> = {
  lecture_slide:      { label: 'Lecture Slides',     icon: <Layers size={12}/> },
  notes:              { label: 'Faculty Notes',       icon: <FileText size={12}/> },
  previous_questions: { label: 'Previous Questions',  icon: <HelpCircle size={12}/> },
  assignment:         { label: 'Assignments',         icon: <BookMarked size={12}/> },
  ct_questions:       { label: 'CT Questions',        icon: <HelpCircle size={12}/> },
  book:               { label: 'Books',               icon: <BookOpen size={12}/> },
  lab_manual:         { label: 'Lab Manuals',         icon: <FlaskConical size={12}/> },
}

// ── Utilities ─────────────────────────────────────────────────────────────────
const cn = (...c: (string|false|null|undefined)[]) => c.filter(Boolean).join(' ')
let _id = 0
const genId = () => `m${++_id}`

function getToken() { return localStorage.getItem('access_token') }
function setTokens(a: string, r: string) {
  localStorage.setItem('access_token', a)
  localStorage.setItem('refresh_token', r)
}
function clearTokens() {
  localStorage.removeItem('access_token')
  localStorage.removeItem('refresh_token')
}

async function apiFetch(path: string, opts: RequestInit = {}) {
  const token = getToken()
  const res = await fetch(`${API}${path}`, {
    ...opts,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...((opts.headers as Record<string, string>) || {}),
    },
  })
  if (res.status === 401) { clearTokens(); window.location.reload() }
  return res
}

// ── Google OAuth Login Screen ─────────────────────────────────────────────────
function LoginScreen({ onLogin }: { onLogin: (user: User, at: string, rt: string) => void }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Handle OAuth callback code in URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const code = params.get('code')
    if (!code) return

    setLoading(true)
    window.history.replaceState({}, '', window.location.pathname)

    fetch(`${API}/auth/google`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, redirect_uri: window.location.origin }),
    })
      .then(r => r.json())
      .then(data => {
        if (data.access_token) {
          setTokens(data.access_token, data.refresh_token)
          onLogin(data.user, data.access_token, data.refresh_token)
        } else {
          setError(data.detail || 'Login failed. Only UIU email accounts are allowed.')
        }
      })
      .catch(() => setError('Network error. Please try again.'))
      .finally(() => setLoading(false))
  }, [])

  const handleGoogleLogin = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API}/auth/google/url`)
      const data = await res.json()
      if (data.url) window.location.href = data.url
      else setError('OAuth not configured. Contact admin.')
    } catch {
      setError('Failed to connect. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#0f1117] flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-brand-600/20 border border-brand-500/30 rounded-2xl
            flex items-center justify-center mx-auto mb-4">
            <Database size={30} className="text-brand-400"/>
          </div>
          <h1 className="text-2xl font-bold text-white">UIU Academic AI</h1>
          <p className="text-slate-400 text-sm mt-1">CSE Department — Powered by RAG</p>
        </div>

        {/* Card */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-2xl p-8 space-y-6">
          <div className="text-center">
            <h2 className="text-lg font-semibold text-white mb-1">Sign in to continue</h2>
            <p className="text-slate-400 text-sm">Use your UIU Google account</p>
          </div>

          {error && (
            <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/30
              rounded-xl p-3 text-red-400 text-sm">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5"/>
              <span>{error}</span>
            </div>
          )}

          <button
            onClick={handleGoogleLogin}
            disabled={loading}
            className="w-full flex items-center justify-center gap-3 bg-white hover:bg-gray-50
              text-gray-800 font-medium rounded-xl px-6 py-3.5 transition-all
              disabled:opacity-60 disabled:cursor-not-allowed active:scale-95"
          >
            {loading ? (
              <Loader2 size={18} className="animate-spin text-gray-500"/>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
            )}
            {loading ? 'Signing in...' : 'Continue with Google'}
          </button>

          <div className="flex items-center gap-2 text-xs text-slate-500 justify-center">
            <Shield size={12}/>
            <span>Restricted to @uiu.ac.bd accounts</span>
          </div>
        </div>

        <p className="text-center text-xs text-slate-600 mt-4">
          UIU Academic AI · CSE Department · Powered by Llama 3.1 + RAG
        </p>
      </div>
    </div>
  )
}

// ── Upload Modal ──────────────────────────────────────────────────────────────
function UploadModal({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [course, setCourse] = useState('DBMS')
  const [docType, setDocType] = useState('lecture_slide')
  const [semester, setSemester] = useState('')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [done, setDone] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleUpload = async () => {
    if (!file) return
    setUploading(true); setError('')
    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('course', course)
      fd.append('document_type', docType)
      if (semester) fd.append('semester', semester)

      const token = getToken()
      const res = await fetch(`${API}/documents/upload`, {
        method: 'POST',
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body: fd,
      })
      if (!res.ok) {
        const d = await res.json()
        throw new Error(d.detail || 'Upload failed')
      }
      setDone(true)
      setTimeout(() => { onSuccess(); onClose() }, 1500)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-800 border border-slate-700 rounded-2xl w-full max-w-md p-6 space-y-5">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Upload Course Material</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={18}/></button>
        </div>

        {done ? (
          <div className="flex flex-col items-center gap-3 py-6">
            <CheckCircle2 size={40} className="text-emerald-400"/>
            <p className="text-white font-medium">Uploaded successfully!</p>
            <p className="text-slate-400 text-sm text-center">Processing in background. Will be searchable in ~1 minute.</p>
          </div>
        ) : (
          <>
            {/* File drop zone */}
            <div
              onClick={() => fileRef.current?.click()}
              className={cn(
                "border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors",
                file ? "border-brand-500/50 bg-brand-500/5" : "border-slate-600 hover:border-slate-500"
              )}
            >
              <input ref={fileRef} type="file" accept=".pdf,.docx,.pptx" className="hidden"
                onChange={e => setFile(e.target.files?.[0] || null)}/>
              <Upload size={24} className="mx-auto mb-2 text-slate-400"/>
              {file ? (
                <p className="text-sm text-brand-300 font-medium">{file.name}</p>
              ) : (
                <>
                  <p className="text-sm text-slate-300">Click to select file</p>
                  <p className="text-xs text-slate-500 mt-1">PDF, DOCX, PPTX — max 100MB</p>
                </>
              )}
            </div>

            {/* Form fields */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Course *</label>
                <input value={course} onChange={e => setCourse(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                    text-sm text-white focus:outline-none focus:border-brand-500"
                  placeholder="DBMS"/>
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Semester</label>
                <input value={semester} onChange={e => setSemester(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                    text-sm text-white focus:outline-none focus:border-brand-500"
                  placeholder="Spring 2025"/>
              </div>
            </div>

            <div>
              <label className="text-xs text-slate-400 mb-1 block">Document Type *</label>
              <select value={docType} onChange={e => setDocType(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                  text-sm text-white focus:outline-none focus:border-brand-500">
                {Object.entries(DOC_TYPES).map(([k, v]) => (
                  <option key={k} value={k}>{v.label}</option>
                ))}
              </select>
            </div>

            {error && (
              <p className="text-red-400 text-sm flex items-center gap-1.5">
                <AlertCircle size={14}/>{error}
              </p>
            )}

            <button onClick={handleUpload} disabled={!file || uploading}
              className="w-full bg-brand-600 hover:bg-brand-500 disabled:opacity-40
                disabled:cursor-not-allowed text-white font-medium rounded-xl py-3
                transition-all flex items-center justify-center gap-2">
              {uploading ? <><Loader2 size={16} className="animate-spin"/>Uploading...</> : 'Upload & Process'}
            </button>
          </>
        )}
      </div>
    </div>
  )
}

// ── Citation Panel ────────────────────────────────────────────────────────────
function CitationPanel({ citation, onClose }: { citation: Citation; onClose: () => void }) {
  return (
    <div className="fixed right-4 bottom-24 w-88 bg-slate-800 border border-slate-600
      rounded-2xl shadow-2xl z-40 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700 bg-slate-900">
        <div className="flex items-center gap-2">
          <span className="w-6 h-6 rounded bg-brand-500/20 border border-brand-500/40
            flex items-center justify-center text-xs font-bold text-brand-300">
            {citation.number}
          </span>
          <span className="text-sm font-semibold text-white">Source</span>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={15}/></button>
      </div>
      <div className="p-4 space-y-3 text-sm">
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-0.5">File</p>
          <p className="text-white font-medium text-sm">{citation.source_filename}</p>
        </div>
        <div className="flex gap-4 flex-wrap">
          {citation.page_number && <div><p className="text-xs text-slate-500">Page</p><p className="text-slate-200">{citation.page_number}</p></div>}
          {citation.slide_number && <div><p className="text-xs text-slate-500">Slide</p><p className="text-slate-200">{citation.slide_number}</p></div>}
          <div>
            <p className="text-xs text-slate-500">Type</p>
            <p className="text-slate-200 flex items-center gap-1">{DOC_TYPES[citation.document_type]?.icon}{DOC_TYPES[citation.document_type]?.label}</p>
          </div>
          <div><p className="text-xs text-slate-500">Course</p><p className="text-slate-200">{citation.course}</p></div>
        </div>
        {citation.section_heading && (
          <div><p className="text-xs text-slate-500">Section</p><p className="text-slate-300 italic">§ {citation.section_heading}</p></div>
        )}
        <div>
          <p className="text-xs text-slate-500 mb-1">Excerpt</p>
          <p className="text-xs text-slate-400 bg-slate-900 rounded-lg p-3 leading-relaxed max-h-28 overflow-y-auto">
            {citation.content_preview}
          </p>
        </div>
      </div>
    </div>
  )
}

// ── Message ───────────────────────────────────────────────────────────────────
function MessageBubble({ msg, onCitationClick }: { msg: Message; onCitationClick: (c: Citation) => void }) {
  const isUser = msg.role === 'user'

  const handleCitNum = useCallback((num: number) => {
    const c = msg.citations.find(x => x.number === num)
    if (c) onCitationClick(c)
  }, [msg.citations, onCitationClick])

  const processContent = (text: string) => text.replace(/\[Source\s+(\d+)\]/gi, '[§$1§]')

  return (
    <div className={cn('flex gap-3', isUser ? 'flex-row-reverse' : 'flex-row')}>
      <div className={cn('flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center text-xs font-bold mt-1',
        isUser ? 'bg-brand-600 text-white' : 'bg-slate-700 text-slate-200')}>
        {isUser ? <UserIcon size={14}/> : 'AI'}
      </div>

      <div className={cn('flex-1 min-w-0', isUser ? 'flex flex-col items-end' : '')}>
        <div className={cn('rounded-2xl px-4 py-3 text-sm',
          isUser ? 'bg-brand-600 text-white rounded-tr-sm max-w-lg' : 'bg-slate-800 text-slate-100 rounded-tl-sm w-full',
          msg.error && 'bg-red-500/10 border border-red-500/30')}>

          {isUser ? (
            <p className="leading-relaxed">{msg.content}</p>
          ) : msg.isStreaming && !msg.content ? (
            <div className="flex gap-1 py-1">
              {[0,1,2].map(i => <span key={i} className="typing-dot w-2 h-2 bg-slate-400 rounded-full" style={{animationDelay:`${i*0.2}s`}}/>)}
            </div>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({children}) => (
                  <p className="mb-2 last:mb-0 leading-relaxed">
                    {React.Children.map(children, child => {
                      if (typeof child !== 'string') return child
                      const parts = child.split(/\[§(\d+)§\]/)
                      return parts.map((p, i) => i % 2 === 1
                        ? <button key={i} onClick={() => handleCitNum(parseInt(p))}
                            className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-bold
                              bg-brand-500/20 text-brand-300 border border-brand-500/30 rounded
                              hover:bg-brand-500/40 transition-colors mx-0.5">{p}</button>
                        : p)
                    })}
                  </p>
                ),
                code: ({className, children, ...props}: any) => {
                  const match = /language-(\w+)/.exec(className || '')
                  return match ? (
                    <SyntaxHighlighter style={oneDark} language={match[1]} PreTag="div"
                      className="rounded-lg text-xs my-2 !bg-[#1a1f2e]">
                      {String(children).replace(/\n$/,'')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className="bg-slate-900 text-emerald-300 px-1.5 py-0.5 rounded text-xs font-mono">{children}</code>
                  )
                },
                table: ({children}) => <div className="overflow-x-auto my-2"><table className="w-full text-xs border-collapse border border-slate-700">{children}</table></div>,
                th: ({children}) => <th className="border border-slate-700 bg-slate-900 px-2 py-1 text-left">{children}</th>,
                td: ({children}) => <td className="border border-slate-700 px-2 py-1">{children}</td>,
                ul: ({children}) => <ul className="list-disc list-inside space-y-0.5 my-1">{children}</ul>,
                ol: ({children}) => <ol className="list-decimal list-inside space-y-0.5 my-1">{children}</ol>,
                h1: ({children}) => <h1 className="text-lg font-bold mt-3 mb-1 text-white">{children}</h1>,
                h2: ({children}) => <h2 className="text-base font-semibold mt-2 mb-1 text-white">{children}</h2>,
                h3: ({children}) => <h3 className="text-sm font-semibold mt-1 mb-0.5">{children}</h3>,
                blockquote: ({children}) => <blockquote className="border-l-4 border-brand-500 pl-3 my-2 text-slate-400 italic">{children}</blockquote>,
              }}
            >
              {processContent(msg.content)}
            </ReactMarkdown>
          )}
        </div>

        {/* Citations row */}
        {!isUser && msg.citations.length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-1 items-center">
            <span className="text-xs text-slate-600">Sources:</span>
            {msg.citations.map(c => (
              <button key={c.number} onClick={() => onCitationClick(c)}
                className="flex items-center gap-1 text-xs text-slate-500 hover:text-brand-300
                  bg-slate-800/50 hover:bg-slate-700 border border-slate-700/50 rounded-lg px-2 py-0.5
                  transition-colors max-w-44 truncate">
                {DOC_TYPES[c.document_type]?.icon}
                <span className="truncate">{c.source_filename}</span>
                {c.page_number && <span>p.{c.page_number}</span>}
              </button>
            ))}
            {msg.latency_ms && <span className="text-xs text-slate-700 ml-auto">{msg.latency_ms}ms</span>}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [user, setUser] = useState<User | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedCourse, setSelectedCourse] = useState<string | null>(null)
  const [selectedTypes, setSelectedTypes] = useState<string[]>([])
  const [courses, setCourses] = useState<string[]>(['DBMS'])
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null)
  const [isSyncing, setIsSyncing] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Check existing session
  useEffect(() => {
    const token = getToken()
    if (!token) return
    apiFetch('/auth/me').then(r => r.ok ? r.json() : null).then(u => { if (u) setUser(u) })
  }, [])

  // Load courses when logged in
  useEffect(() => {
    if (!user) return
    apiFetch('/documents/courses').then(r => r.json()).then(d => {
      if (d.courses?.length) setCourses(d.courses)
    }).catch(() => {})
  }, [user])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleLogin = (u: User, at: string, rt: string) => {
    setTokens(at, rt); setUser(u)
  }

  const handleLogout = () => {
    clearTokens(); setUser(null); setMessages([])
  }

  const toggleType = (t: string) =>
    setSelectedTypes(prev => prev.includes(t) ? prev.filter(x => x !== t) : [...prev, t])

  const handleSync = async () => {
    setIsSyncing(true)
    await apiFetch('/ingestion/sync/incremental', { method: 'POST' }).catch(() => {})
    setTimeout(() => setIsSyncing(false), 3000)
  }

  const sendMessage = useCallback(async () => {
    const q = input.trim()
    if (!q || loading) return
    setInput(''); setLoading(true); setActiveCitation(null)

    const userId = genId()
    const assistId = genId()

    setMessages(prev => [...prev,
      { id: userId, role: 'user', content: q, citations: [] },
      { id: assistId, role: 'assistant', content: '', citations: [], isStreaming: true }
    ])

    let full = ''; let finalCitations: Citation[] = []; let latency = 0

    try {
      const token = getToken()
      const res = await fetch(`${API}/query/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify({ question: q, course: selectedCourse, document_types: selectedTypes.length ? selectedTypes : null, stream: true }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Request failed')
      }

      const reader = res.body!.getReader()
      const dec = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const lines = dec.decode(value, { stream: true }).split('\n').filter(l => l.startsWith('data: '))
        for (const line of lines) {
          try {
            const ev = JSON.parse(line.slice(6))
            if (ev.type === 'token') {
              full += ev.content
              setMessages(prev => prev.map(m => m.id === assistId ? { ...m, content: full } : m))
            } else if (ev.type === 'citations') {
              finalCitations = ev.data || []
            } else if (ev.type === 'done') {
              latency = ev.latency_ms || 0
            }
          } catch {}
        }
      }
    } catch (e: any) {
      full = e.message === 'Rate limit exceeded.' 
        ? '⚠️ Rate limit reached. Please wait a minute before asking again.'
        : `⚠️ Error: ${e.message || 'Something went wrong. Please try again.'}`
      setMessages(prev => prev.map(m => m.id === assistId ? { ...m, content: full, error: true } : m))
    }

    setMessages(prev => prev.map(m =>
      m.id === assistId ? { ...m, content: full, citations: finalCitations, latency_ms: latency, isStreaming: false } : m
    ))
    setLoading(false)
  }, [input, loading, selectedCourse, selectedTypes])

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  const suggestions = [
    "What is normalization and what are 1NF, 2NF, 3NF?",
    "Explain B+ tree structure and search operation.",
    "What are ACID properties in transactions?",
    "How does two-phase locking work?",
    "Explain ER diagram with a real example.",
  ]

  if (!user) return <LoginScreen onLogin={handleLogin}/>

  return (
    <div className="flex h-screen bg-[#0f1117] text-slate-200 overflow-hidden">
      {/* Sidebar */}
      <div className={cn('flex-shrink-0 flex flex-col bg-slate-900 border-r border-slate-800 transition-all duration-300',
        sidebarOpen ? 'w-60' : 'w-0 overflow-hidden')}>

        {/* Logo */}
        <div className="p-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-brand-600/20 border border-brand-500/30 rounded-xl flex items-center justify-center">
              <Database size={16} className="text-brand-400"/>
            </div>
            <div>
              <p className="text-sm font-bold text-white">UIU Academic AI</p>
              <p className="text-xs text-slate-500">CSE Department</p>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-4 scrollbar-thin">
          {/* Course */}
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-1 mb-1.5">Course</p>
            <div className="space-y-0.5">
              {[null, ...courses].map(c => (
                <button key={c || 'all'} onClick={() => setSelectedCourse(c)}
                  className={cn('w-full text-left text-sm px-3 py-1.5 rounded-lg transition-colors',
                    selectedCourse === c ? 'bg-brand-600/20 text-brand-300' : 'text-slate-400 hover:text-white hover:bg-slate-800')}>
                  {c || 'All Courses'}
                </button>
              ))}
            </div>
          </div>

          {/* Doc type filter */}
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-1 mb-1.5">Filter</p>
            <div className="space-y-0.5">
              {Object.entries(DOC_TYPES).map(([k, v]) => (
                <button key={k} onClick={() => toggleType(k)}
                  className={cn('w-full flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg transition-colors',
                    selectedTypes.includes(k) ? 'text-brand-300 bg-brand-600/10' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800')}>
                  <div className={cn('w-3.5 h-3.5 rounded border flex-shrink-0 flex items-center justify-center',
                    selectedTypes.includes(k) ? 'bg-brand-600 border-brand-500' : 'border-slate-600')}>
                    {selectedTypes.includes(k) && <CheckCircle2 size={8} className="text-white"/>}
                  </div>
                  {v.icon}<span>{v.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Sidebar footer */}
        <div className="p-3 border-t border-slate-800 space-y-2">
          <button onClick={() => setUploadOpen(true)}
            className="w-full flex items-center gap-2 text-xs bg-brand-600/10 hover:bg-brand-600/20
              text-brand-300 border border-brand-600/20 rounded-lg px-3 py-2 transition-colors">
            <Plus size={13}/> Upload Material
          </button>
          {user.is_admin && (
            <button onClick={handleSync} disabled={isSyncing}
              className="w-full flex items-center gap-2 text-xs bg-slate-800 hover:bg-slate-700
                text-slate-400 border border-slate-700 rounded-lg px-3 py-2 transition-colors disabled:opacity-50">
              <RefreshCw size={13} className={isSyncing ? 'animate-spin' : ''}/>
              {isSyncing ? 'Syncing...' : 'Sync Drive'}
            </button>
          )}
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900/80 backdrop-blur flex-shrink-0">
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="text-slate-400 hover:text-white transition-colors">
              <Menu size={18}/>
            </button>
            <div>
              <h1 className="text-sm font-semibold text-white">
                {selectedCourse ? `${selectedCourse} Assistant` : 'Academic Assistant'}
              </h1>
              <p className="text-xs text-slate-500">
                {selectedTypes.length > 0 ? `${selectedTypes.length} filters · ` : ''}
                Grounded answers with citations
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {messages.length > 0 && (
              <button onClick={() => setMessages([])}
                className="text-xs text-slate-500 hover:text-red-400 flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-red-500/10 transition-colors">
                <Trash2 size={12}/> Clear
              </button>
            )}
            <div className="flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 rounded-full px-2.5 py-1">
              <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"/>
              Llama 3.1
            </div>
            <div className="flex items-center gap-1.5 pl-2 border-l border-slate-700">
              <div className="w-6 h-6 bg-brand-600/30 rounded-full flex items-center justify-center">
                <UserIcon size={12} className="text-brand-300"/>
              </div>
              <span className="text-xs text-slate-400 hidden sm:block max-w-24 truncate">
                {user.full_name || user.email}
              </span>
              <button onClick={handleLogout} className="text-slate-500 hover:text-red-400 transition-colors ml-1">
                <LogOut size={14}/>
              </button>
            </div>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-4 py-5">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-6 max-w-2xl mx-auto">
              <div className="text-center">
                <div className="w-14 h-14 bg-brand-600/20 border border-brand-500/30 rounded-2xl
                  flex items-center justify-center mx-auto mb-3">
                  <Database size={24} className="text-brand-400"/>
                </div>
                <h2 className="text-xl font-bold text-white mb-1">
                  Welcome, {user.full_name?.split(' ')[0] || 'Student'} 👋
                </h2>
                <p className="text-slate-400 text-sm max-w-sm mx-auto leading-relaxed">
                  Ask anything about your course materials. Every answer is sourced from
                  real lecture slides, notes, and books — with citations.
                </p>
              </div>
              <div className="w-full max-w-xl space-y-2">
                {suggestions.map((s, i) => (
                  <button key={i} onClick={() => { setInput(s); inputRef.current?.focus() }}
                    className="w-full text-left text-sm text-slate-300 bg-slate-800/50 hover:bg-slate-800
                      border border-slate-700/50 hover:border-slate-600 rounded-xl px-4 py-2.5
                      transition-all hover:translate-x-1 flex items-center gap-2">
                    <ChevronRight size={14} className="text-brand-400 flex-shrink-0"/>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-5 max-w-4xl mx-auto">
              {messages.map(msg => (
                <MessageBubble key={msg.id} msg={msg} onCitationClick={setActiveCitation}/>
              ))}
              <div ref={bottomRef}/>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="flex-shrink-0 px-4 pb-4 pt-2 border-t border-slate-800 bg-slate-900/60">
          <div className="max-w-4xl mx-auto">
            {(selectedCourse || selectedTypes.length > 0) && (
              <div className="flex flex-wrap gap-1.5 mb-2">
                {selectedCourse && (
                  <span className="flex items-center gap-1 text-xs bg-brand-600/15 text-brand-300
                    border border-brand-600/25 rounded-full px-2.5 py-0.5">
                    <Database size={9}/>{selectedCourse}
                    <button onClick={() => setSelectedCourse(null)} className="hover:text-white ml-0.5"><X size={10}/></button>
                  </span>
                )}
                {selectedTypes.map(t => (
                  <span key={t} className="flex items-center gap-1 text-xs bg-slate-700/50 text-slate-300
                    border border-slate-600/50 rounded-full px-2.5 py-0.5">
                    {DOC_TYPES[t]?.label}
                    <button onClick={() => toggleType(t)} className="hover:text-white ml-0.5"><X size={10}/></button>
                  </span>
                ))}
              </div>
            )}

            <div className="flex items-end gap-2 bg-slate-800 border border-slate-700
              rounded-2xl p-3 focus-within:border-brand-500/50 transition-colors">
              <textarea
                ref={inputRef} value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Ask about your course materials..."
                rows={1} disabled={loading}
                className="flex-1 bg-transparent text-slate-200 placeholder-slate-500 text-sm
                  resize-none outline-none leading-relaxed max-h-36 scrollbar-thin
                  disabled:opacity-50"
                style={{ minHeight: '24px' }}
              />
              <button onClick={sendMessage} disabled={loading || !input.trim()}
                className="flex-shrink-0 w-9 h-9 bg-brand-600 hover:bg-brand-500
                  disabled:opacity-40 disabled:cursor-not-allowed rounded-xl
                  flex items-center justify-center transition-all active:scale-95">
                {loading
                  ? <Loader2 size={15} className="text-white animate-spin"/>
                  : <Send size={15} className="text-white"/>}
              </button>
            </div>
            <p className="text-center text-xs text-slate-700 mt-1.5">
              Answers grounded in UIU course materials only · ⏎ send · ⇧⏎ newline
            </p>
          </div>
        </div>
      </div>

      {/* Modals */}
      {uploadOpen && (
        <UploadModal
          onClose={() => setUploadOpen(false)}
          onSuccess={() => apiFetch('/documents/courses').then(r => r.json()).then(d => { if (d.courses?.length) setCourses(d.courses) })}
        />
      )}
      {activeCitation && (
        <CitationPanel citation={activeCitation} onClose={() => setActiveCitation(null)}/>
      )}
    </div>
  )
}
