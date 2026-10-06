export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export type Issue = {
  number: number;
  title: string;
  url: string;
  labels: string[];
  difficulty: string | null;
  skills: string[];
  created_at: string;
  n_comments: number;
  matched_skills: string[];
};

export type Signals = {
  starred_by_you?: boolean;
  co_starred_with?: string[];
  matched_skills?: string[];
  matched_interests?: string[];
  retrieval?: string;
};

export type Repo = {
  id: number;
  name: string;
  url: string;
  description: string;
  summary?: string;
  language: string | null;
  stars: number;
  skills?: string[];
  created_at?: string;
  signals: Signals;
  issues: Issue[];
};

export type Recommendation = {
  repos: Repo[];
  new_projects: Repo[];
  profile: Record<string, unknown> & { languages?: string[]; target_difficulty?: string; retrieval?: string };
  timings_ms: Record<string, number | boolean>;
  data_updated?: string;
};

export type Options = { languages: string[]; interests: { id: string; label: string }[]; hours: number[] };

export async function call<T>(path: string, body?: unknown, timeoutMs = 60000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_URL}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: ctrl.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = data && data.detail;
      const msg = Array.isArray(d) ? "Please check the form and try again." : d;
      throw new Error(msg || `Request failed (${res.status})`);
    }
    return data as T;
  } finally {
    clearTimeout(timer);
  }
}
