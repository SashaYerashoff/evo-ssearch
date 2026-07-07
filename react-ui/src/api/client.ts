// Thin fetch wrapper for the EVA Flask backend (proxied by Vite to :5000).
// Sends session cookie + X-CSRF-Token on mutating requests.

function getCookie(name: string): string {
  const m = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'))
  return m ? decodeURIComponent(m[1]) : ''
}

export class ApiError extends Error {
  status: number
  payload: any
  constructor(status: number, payload: any) {
    super(payload?.error || `HTTP ${status}`)
    this.status = status
    this.payload = payload
  }
}

async function parse(res: Response): Promise<any> {
  const text = await res.text()
  let data: any = null
  try { data = text ? JSON.parse(text) : null } catch { data = text }
  if (!res.ok) throw new ApiError(res.status, data)
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
  if (method !== 'GET') {
    const csrf = getCookie('eva_csrf')
    if (csrf) headers['X-CSRF-Token'] = csrf
  }
  const res = await fetch(url, { method, headers, body, credentials: 'include' })
  return parse(res)
}

export const api = {
  get: (path: string, query?: Record<string, unknown>) => request(path, { query }),
  postJson: (path: string, json: unknown) => request(path, { method: 'POST', json }),
  postForm: (path: string, form: FormData) => request(path, { method: 'POST', form }),
  del: (path: string) => request(path, { method: 'DELETE' }),
}
