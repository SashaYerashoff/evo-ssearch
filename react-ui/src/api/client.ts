// Thin fetch wrapper for the EVA Flask backend (proxied by Vite to :5000).
// Sends session cookie + X-CSRF-Token on mutating requests.

let csrfCookieName = 'eva_csrf'

function getCookie(name: string): string {
  const escaped = name.replace(/([.$?*|{}()[\]\\/+^])/g, '\\$1')
  const m = document.cookie.match(new RegExp('(?:^|; )' + escaped + '=([^;]*)'))
  return m ? decodeURIComponent(m[1]) : ''
}

export function setCsrfCookieName(value: unknown): void {
  const candidate = String(value || '').trim()
  if (/^[A-Za-z0-9_.-]{1,128}$/.test(candidate)) csrfCookieName = candidate
}

export function getCsrfToken(): string {
  return getCookie(csrfCookieName)
}

export const AUTH_EXPIRED_EVENT = 'eva:auth-expired'
export const API_FORBIDDEN_EVENT = 'eva:api-forbidden'

export function apiErrorMessage(status: number, payload: unknown): string {
  if (payload && typeof payload === 'object') {
    const body = payload as Record<string, unknown>
    const value = body.error ?? body.message
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  if (typeof payload === 'string' && payload.trim()) return payload.trim().slice(0, 240)
  if (status === 401) return 'Authentication required'
  if (status === 403) return 'You do not have permission for this action'
  return `HTTP ${status}`
}

export function shouldAttachCsrf(method: string): boolean {
  return !['GET', 'HEAD', 'OPTIONS'].includes(String(method || 'GET').toUpperCase())
}

export class ApiError extends Error {
  status: number
  payload: any
  constructor(status: number, payload: any) {
    super(apiErrorMessage(status, payload))
    this.status = status
    this.payload = payload
  }
}

async function parse(res: Response): Promise<any> {
  const text = await res.text()
  let data: any = null
  try { data = text ? JSON.parse(text) : null } catch { data = text }
  if (!res.ok) {
    if (typeof window !== 'undefined') {
      const eventName = res.status === 401 ? AUTH_EXPIRED_EVENT : res.status === 403 ? API_FORBIDDEN_EVENT : null
      if (eventName) window.dispatchEvent(new CustomEvent(eventName, { detail: { status: res.status, payload: data } }))
    }
    throw new ApiError(res.status, data)
  }
  return data
}

async function request(
  path: string,
  opts: { method?: string; json?: unknown; form?: FormData; query?: Record<string, unknown> } = {},
): Promise<any> {
  const { method = 'GET', json, form, query } = opts
  let url = path
  if (query) {
    const qs = new URLSearchParams()
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null && String(v) !== '') qs.set(k, String(v))
    }
    const s = qs.toString()
    if (s) url += (url.includes('?') ? '&' : '?') + s
  }
  const headers: Record<string, string> = {}
  let body: BodyInit | undefined
  if (json !== undefined) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(json) }
  else if (form) { body = form }
  if (shouldAttachCsrf(method)) {
    const csrf = getCsrfToken()
    if (csrf) headers['X-CSRF-Token'] = csrf
  }
  const res = await fetch(url, { method, headers, body, credentials: 'include' })
  return parse(res)
}

export const api = {
  get: (path: string, query?: Record<string, unknown>) => request(path, { query }),
  postJson: (path: string, json: unknown) => request(path, { method: 'POST', json }),
  postForm: (path: string, form: FormData) => request(path, { method: 'POST', form }),
  patch: (path: string, json: unknown) => request(path, { method: 'PATCH', json }),
  del: (path: string) => request(path, { method: 'DELETE' }),
}
