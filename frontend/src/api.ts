const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '/api/v1').replace(/\/$/, '')

export interface HealthResponse {
  status: 'ok'
}

export type ChatRole = 'system' | 'user' | 'assistant'

export interface ChatMessage {
  role: ChatRole
  content: string
}

export interface ChatResponse {
  message: ChatMessage
  model: string
  conversation_id: number
}

interface StreamStartEvent {
  type: 'start'
  conversation_id: number
  model: string
}

interface StreamDeltaEvent {
  type: 'delta'
  content: string
}

interface StreamDoneEvent {
  type: 'done'
  conversation_id: number
  model: string
}

interface StreamErrorEvent {
  type: 'error'
  detail: string
}

type ChatStreamEvent = StreamStartEvent | StreamDeltaEvent | StreamDoneEvent | StreamErrorEvent

export interface ChatStreamHandlers {
  onStart: (conversationId: number, model: string) => void
  onDelta: (content: string) => void
}

export interface ConversationResponse {
  id: number
  title: string
  messages: ChatMessage[]
  created_at: string
  updated_at: string
}

export interface ConversationSummary {
  id: number
  title: string
  created_at: string
  updated_at: string
}

export interface DocumentSummary {
  id: number
  filename: string
  file_type: 'pdf' | 'txt' | 'markdown'
  file_path: string
  created_at: string
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init.headers },
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw error
    }

    throw new ApiError('无法连接到后端服务，请确认服务已启动。')
  }

  if (!response.ok) {
    let detail = ''
    try {
      const payload = (await response.json()) as { detail?: unknown }
      if (typeof payload.detail === 'string') detail = payload.detail
    } catch {
      // Keep the generic HTTP error when the response is not JSON.
    }
    throw new ApiError(detail || `后端服务返回错误（HTTP ${response.status}）。`, response.status)
  }

  if (response.status === 204) return undefined as T

  try {
    return (await response.json()) as T
  } catch {
    throw new ApiError('后端服务返回了无效的数据。', response.status)
  }
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const health = await request<HealthResponse>('/health', { signal })

  if (health.status !== 'ok') {
    throw new ApiError('后端服务状态异常。')
  }

  return health
}

export function getLatestConversation(signal?: AbortSignal): Promise<ConversationResponse | null> {
  return request<ConversationResponse | null>('/conversations/latest', { signal })
}

export function listConversations(signal?: AbortSignal): Promise<ConversationSummary[]> {
  return request<ConversationSummary[]>('/conversations', { signal })
}

export function createConversation(
  title = '新对话',
  signal?: AbortSignal,
): Promise<ConversationResponse> {
  return request<ConversationResponse>('/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
    signal,
  })
}

export function getConversation(
  conversationId: number,
  signal?: AbortSignal,
): Promise<ConversationResponse> {
  return request<ConversationResponse>(`/conversations/${conversationId}`, { signal })
}

export function renameConversation(
  conversationId: number,
  title: string,
  signal?: AbortSignal,
): Promise<ConversationSummary> {
  return request<ConversationSummary>(`/conversations/${conversationId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
    signal,
  })
}

export function deleteConversation(
  conversationId: number,
  signal?: AbortSignal,
): Promise<void> {
  return request<void>(`/conversations/${conversationId}`, {
    method: 'DELETE',
    signal,
  })
}

export function listDocuments(signal?: AbortSignal): Promise<DocumentSummary[]> {
  return request<DocumentSummary[]>('/files', { signal })
}

export function uploadDocument(file: File, signal?: AbortSignal): Promise<DocumentSummary> {
  const body = new FormData()
  body.append('file', file)
  return request<DocumentSummary>('/files/upload', {
    method: 'POST',
    body,
    signal,
  })
}

export function sendChat(
  messages: ChatMessage[],
  conversationId?: number,
  signal?: AbortSignal,
  ragEnabled = false,
): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      conversation_id: conversationId,
      rag_enabled: ragEnabled,
    }),
    signal,
  })
}

export async function streamChat(
  messages: ChatMessage[],
  conversationId: number | undefined,
  handlers: ChatStreamHandlers,
  ragEnabled: boolean,
  signal?: AbortSignal,
): Promise<StreamDoneEvent> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        Accept: 'application/x-ndjson',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages,
        conversation_id: conversationId,
        rag_enabled: ragEnabled,
      }),
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('无法连接到后端服务，请确认服务已启动。')
  }

  if (!response.ok) {
    let detail = ''
    try {
      const payload = (await response.json()) as { detail?: unknown }
      if (typeof payload.detail === 'string') detail = payload.detail
    } catch {
      // Keep the generic HTTP error when the response is not JSON.
    }
    throw new ApiError(detail || `后端服务返回错误（HTTP ${response.status}）。`, response.status)
  }
  if (!response.body) throw new ApiError('浏览器未收到可读取的 AI 响应流。')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed: StreamDoneEvent | undefined

  function handleLine(line: string) {
    if (!line.trim()) return
    let event: ChatStreamEvent
    try {
      event = JSON.parse(line) as ChatStreamEvent
    } catch {
      throw new ApiError('AI 响应流包含无效数据。')
    }
    if (event.type === 'start') handlers.onStart(event.conversation_id, event.model)
    else if (event.type === 'delta') handlers.onDelta(event.content)
    else if (event.type === 'done') completed = event
    else if (event.type === 'error') throw new ApiError(event.detail)
    else throw new ApiError('AI 响应流包含未知事件。')
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    lines.forEach(handleLine)
    if (done) break
  }
  handleLine(buffer)

  if (!completed) throw new ApiError('AI 响应流意外中断。')
  return completed
}
