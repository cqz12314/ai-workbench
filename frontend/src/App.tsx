import { ChangeEvent, FormEvent, MouseEvent, useEffect, useRef, useState } from 'react'

import {
  ApiError,
  ChatMessage,
  ConversationSummary,
  DocumentSummary,
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  listDocuments,
  renameConversation,
  streamChat,
  uploadDocument,
} from './api'

const welcomeMessages: ChatMessage[] = [
  { role: 'assistant', content: '你好，我是 AI Workbench 助手。想聊点什么？' },
]

function displayMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages.length > 0 ? messages : welcomeMessages
}

function formatConversationTime(value: string): string {
  return new Intl.DateTimeFormat('zh-CN', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function App() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>(welcomeMessages)
  const [conversationId, setConversationId] = useState<number>()
  const [input, setInput] = useState('')
  const [error, setError] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isLoadingHistory, setIsLoadingHistory] = useState(true)
  const [isManaging, setIsManaging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [ragEnabled, setRagEnabled] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const conversationEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isSending])

  useEffect(() => {
    const controller = new AbortController()
    controllerRef.current = controller

    async function loadHistory() {
      try {
        const [summaries, uploadedDocuments] = await Promise.all([
          listConversations(controller.signal),
          listDocuments(controller.signal),
        ])
        setConversations(summaries)
        setDocuments(uploadedDocuments)
        if (summaries.length > 0) {
          const conversation = await getConversation(summaries[0].id, controller.signal)
          setConversationId(conversation.id)
          setMessages(displayMessages(conversation.messages))
        }
      } catch (caught) {
        if (!(caught instanceof DOMException && caught.name === 'AbortError')) {
          setError(caught instanceof Error ? caught.message : '加载聊天记录失败。')
        }
      } finally {
        setIsLoadingHistory(false)
      }
    }

    void loadHistory()
    return () => controller.abort()
  }, [])

  async function handleSelectConversation(selectedId: number) {
    if (selectedId === conversationId || isSending || isManaging) return
    const controller = new AbortController()
    controllerRef.current?.abort()
    controllerRef.current = controller
    setIsLoadingHistory(true)
    setError('')
    try {
      const conversation = await getConversation(selectedId, controller.signal)
      setConversationId(conversation.id)
      setMessages(displayMessages(conversation.messages))
    } catch (caught) {
      if (!(caught instanceof DOMException && caught.name === 'AbortError')) {
        setError(caught instanceof Error ? caught.message : '加载聊天记录失败。')
      }
    } finally {
      setIsLoadingHistory(false)
    }
  }

  async function handleNewChat() {
    if (isSending || isManaging) return
    setIsManaging(true)
    setError('')
    try {
      const conversation = await createConversation()
      setConversations((current) => [conversation, ...current])
      setConversationId(conversation.id)
      setMessages(welcomeMessages)
      setInput('')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '新建聊天失败。')
    } finally {
      setIsManaging(false)
    }
  }

  async function handleRename(
    event: MouseEvent<HTMLButtonElement>,
    conversation: ConversationSummary,
  ) {
    event.stopPropagation()
    if (isSending || isManaging) return
    const title = window.prompt('输入新的聊天标题', conversation.title)?.trim()
    if (!title || title === conversation.title) return

    setIsManaging(true)
    setError('')
    try {
      const updated = await renameConversation(conversation.id, title)
      setConversations((current) =>
        current.map((item) => (item.id === updated.id ? updated : item)),
      )
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '重命名聊天失败。')
    } finally {
      setIsManaging(false)
    }
  }

  async function handleDelete(
    event: MouseEvent<HTMLButtonElement>,
    conversation: ConversationSummary,
  ) {
    event.stopPropagation()
    if (isSending || isManaging) return
    if (!window.confirm(`确定删除“${conversation.title}”吗？此操作无法撤销。`)) return

    setIsManaging(true)
    setError('')
    try {
      await deleteConversation(conversation.id)
      const remaining = conversations.filter((item) => item.id !== conversation.id)
      setConversations(remaining)
      if (conversation.id === conversationId) {
        if (remaining.length > 0) {
          const next = await getConversation(remaining[0].id)
          setConversationId(next.id)
          setMessages(displayMessages(next.messages))
        } else {
          setConversationId(undefined)
          setMessages(welcomeMessages)
        }
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '删除聊天失败。')
    } finally {
      setIsManaging(false)
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const content = input.trim()
    if (!content || isSending || isLoadingHistory || isManaging) return

    const userMessage: ChatMessage = { role: 'user', content }
    const nextMessages = [...messages, userMessage]
    const controller = new AbortController()
    controllerRef.current = controller
    setMessages([...nextMessages, { role: 'assistant', content: '' }])
    setInput('')
    setError('')
    setIsSending(true)

    try {
      await streamChat(
        nextMessages,
        conversationId,
        {
          onStart: (startedConversationId) => setConversationId(startedConversationId),
          onDelta: (content) => {
            setMessages((current) => current.map((message, index) =>
              index === current.length - 1
                ? { ...message, content: message.content + content }
                : message,
            ))
          },
        },
        ragEnabled,
        controller.signal,
      )
      void listConversations().then(setConversations).catch(() => undefined)
    } catch (caught) {
      setMessages(nextMessages)
      if (caught instanceof DOMException && caught.name === 'AbortError') {
        setError('已停止生成，未保存未完成的 AI 回复。')
      } else if (
        caught instanceof ApiError
        && caught.status === 503
        && caught.message === 'AI model is not configured'
      ) {
        setError('AI 模型尚未配置，请先设置后端环境变量。')
      } else {
        setError(caught instanceof Error ? caught.message : 'AI 回复流意外中断。')
      }
    } finally {
      setIsSending(false)
      controllerRef.current = null
    }
  }

  function handleCancel() {
    controllerRef.current?.abort()
  }

  async function handleFileUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file || isUploading) return

    setIsUploading(true)
    setError('')
    try {
      const document = await uploadDocument(file)
      setDocuments((current) => [document, ...current])
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '上传文件失败。')
    } finally {
      setIsUploading(false)
    }
  }

  const controlsDisabled = isSending || isManaging || isLoadingHistory

  return (
    <main className="app-layout">
      <aside className="sidebar" aria-label="聊天历史">
        <div className="sidebar-header">
          <p className="sidebar-brand">AI WORKBENCH</p>
          <button className="new-chat-button" type="button" onClick={handleNewChat} disabled={controlsDisabled}>
            <span aria-hidden="true">＋</span> New Chat
          </button>
        </div>

        <nav className="conversation-list" aria-label="历史聊天列表">
          {conversations.length === 0 && !isLoadingHistory && <p className="empty-history">暂无历史聊天</p>}
          {conversations.map((conversation) => (
            <div className={`conversation-item${conversation.id === conversationId ? ' conversation-item--active' : ''}`} key={conversation.id}>
              <button className="conversation-select" type="button" onClick={() => void handleSelectConversation(conversation.id)} disabled={controlsDisabled} aria-current={conversation.id === conversationId ? 'page' : undefined}>
                <span className="conversation-title">{conversation.title}</span>
                <time dateTime={conversation.updated_at}>{formatConversationTime(conversation.updated_at)}</time>
              </button>
              <div className="conversation-actions">
                <button type="button" onClick={(event) => void handleRename(event, conversation)} disabled={controlsDisabled} aria-label={`重命名 ${conversation.title}`}>✎</button>
                <button type="button" onClick={(event) => void handleDelete(event, conversation)} disabled={controlsDisabled} aria-label={`删除 ${conversation.title}`}>×</button>
              </div>
            </div>
          ))}
        </nav>

        <section className="documents-panel" aria-labelledby="documents-heading">
          <div className="documents-header">
            <h2 id="documents-heading">Documents</h2>
            <button type="button" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>
              {isUploading ? '上传中…' : '上传'}
            </button>
            <input
              ref={fileInputRef}
              className="file-input"
              type="file"
              accept=".pdf,.txt,.md,.markdown,application/pdf,text/plain,text/markdown"
              onChange={(event) => void handleFileUpload(event)}
            />
          </div>
          <div className="document-list">
            {documents.length === 0 && <p className="empty-documents">暂无文件</p>}
            {documents.map((document) => (
              <div className="document-item" key={document.id} title={document.filename}>
                <span className={`file-type file-type--${document.file_type}`}>{document.file_type}</span>
                <span>{document.filename}</span>
              </div>
            ))}
          </div>
        </section>
      </aside>

      <section className="chat-shell">
        <header className="app-header">
          <div><p className="eyebrow">CONVERSATION</p><h1>{conversations.find((item) => item.id === conversationId)?.title ?? '新对话'}</h1></div>
          <div className="header-controls">
            <label className="rag-toggle">
              <input type="checkbox" checked={ragEnabled} onChange={(event) => setRagEnabled(event.target.checked)} disabled={isSending} />
              <span aria-hidden="true" />
              知识库
            </label>
            <span className="status"><span aria-hidden="true" /> LiteLLM</span>
          </div>
        </header>

        <section className="conversation" aria-live="polite" aria-label="聊天记录">
          {isLoadingHistory && <p className="history-status">正在加载聊天记录…</p>}
          {!isLoadingHistory && messages.map((message, index) => (
            <article className={`message message--${message.role}`} key={`${message.role}-${index}`}>
              <span className="message-role">{message.role === 'user' ? '你' : 'AI'}</span>
              <p>{message.content || (isSending && index === messages.length - 1 ? '正在思考…' : '')}</p>
            </article>
          ))}
          <div ref={conversationEndRef} />
        </section>

        <form className="composer" onSubmit={handleSubmit}>
          {error && <p className="error-message" role="alert">{error}</p>}
          <div className="composer-row">
            <textarea aria-label="消息" placeholder="输入消息，Enter 发送，Shift + Enter 换行" value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                event.currentTarget.form?.requestSubmit()
              }
            }} rows={1} maxLength={32000} disabled={controlsDisabled} />
            {isSending
              ? <button className="cancel-button" type="button" onClick={handleCancel}>停止</button>
              : <button type="submit" disabled={controlsDisabled || !input.trim()}>发送</button>}
          </div>
          <small>聊天记录保存在本地 SQLite 数据库中。</small>
        </form>
      </section>
    </main>
  )
}

export default App
