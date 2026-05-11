// UIU Academic RAG Platform — TypeScript Types

export interface Citation {
  number: number
  source_ref: string
  chunk_id: string
  source_filename: string
  document_type: string
  page_number: number | null
  slide_number: number | null
  section_heading: string | null
  course: string
  content_preview: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  latency_ms?: number
  timestamp: Date
  isStreaming?: boolean
}

export interface Conversation {
  id: string
  title: string
  course: string | null
  messages: Message[]
  created_at: string
}

export interface Document {
  id: string
  filename: string
  course: string
  document_type: string
  chunk_count: number
  ingestion_status: string
  semester: string | null
}

export interface QueryResponse {
  answer: string
  citations: Citation[]
  source_map: Record<string, string>
  latency_ms: number
  model: string
  was_grounded: boolean
  was_compressed: boolean
  context_count: number
}

export type DocumentType =
  | 'lecture_slide'
  | 'notes'
  | 'previous_questions'
  | 'assignment'
  | 'ct_questions'
  | 'book'
  | 'lab_manual'

export const DOCUMENT_TYPE_LABELS: Record<DocumentType, string> = {
  lecture_slide: 'Lecture Slides',
  notes: 'Faculty Notes',
  previous_questions: 'Previous Questions',
  assignment: 'Assignments',
  ct_questions: 'CT Questions',
  book: 'Books',
  lab_manual: 'Lab Manuals',
}

export const COURSES = ['DBMS', 'DSA', 'OOP', 'CN', 'OS', 'AI', 'SE']
