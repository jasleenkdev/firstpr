import type { Repo } from "./api";

/** "today", "3 days ago", "3 weeks ago", "2 months ago", "a year ago" */
export function age(iso: string | undefined, now = new Date()): string {
  if (!iso) return "";
  const days = Math.max(0, Math.floor((now.getTime() - new Date(iso).getTime()) / 86400000));
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 14) return `${days} days ago`;
  if (days < 60) return `${Math.round(days / 7)} weeks ago`;
  if (days < 330) return `${Math.round(days / 30)} months ago`;
  if (days < 540) return "about a year ago";
  const years = Math.round(days / 365);
  return years === 1 ? "a year ago" : `${years} years ago`;
}

function luminance(hex: string): number {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const [r, g, b] = c.map((x) => (x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

const INK = "#16203A";
const WHITE = "#FFFFFF";

/** Background + text colour for a GitHub label: the label's own colour, text in ink or white,
 * whichever contrasts more (both pass AA for most GitHub colours; ink wins on light ones). */
export function labelStyle(color: string | undefined): { background: string; color: string } {
  if (!color || !/^#[0-9a-f]{6}$/i.test(color)) return { background: "#E1E6EB", color: INK };
  return { background: color, color: contrast(color, INK) >= contrast(color, WHITE) ? INK : WHITE };
}

const CASING: Record<string, string> = {
  javascript: "JavaScript", typescript: "TypeScript", nodejs: "Node.js", "node.js": "Node.js", python: "Python",
  rust: "Rust", go: "Go", golang: "Go", java: "Java", kotlin: "Kotlin", swift: "Swift", ruby: "Ruby", php: "PHP",
  "c++": "C++", "c#": "C#", c: "C", css: "CSS", html: "HTML", sql: "SQL", react: "React", vue: "Vue",
  pytorch: "PyTorch", tensorflow: "TensorFlow", docker: "Docker", kubernetes: "Kubernetes", linux: "Linux",
  llm: "LLMs", llms: "LLMs", ai: "AI", cli: "command-line tools", api: "APIs", macos: "macOS", ios: "iOS",
  android: "Android", nextjs: "Next.js", graphql: "GraphQL", wasm: "WebAssembly", shell: "shell scripting",
};

// topic words too generic to tell a student anything
const GENERIC = new Set(["app", "apps", "code", "tool", "tools", "library", "framework", "project", "opensource",
  "open-source", "open source", "awesome", "hacktoberfest", "good-first-issue", "help-wanted", "tui", "gui", "cross-platform"]);

export function usefulSkills(raw: string[], exclude = ""): string[] {
  const seen = new Set<string>();
  for (const r of raw) {
    const k = r.toLowerCase();
    if (k === exclude.toLowerCase() || GENERIC.has(k)) continue;
    seen.add(skillName(r));
  }
  return [...seen];
}

/** Skill tokens from the API ("machine-learning", "nodejs") as people write them. */
export function skillName(raw: string): string {
  const key = raw.toLowerCase();
  return CASING[key] ?? key.replace(/[-_]/g, " ");
}

function list(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/** One plain sentence on why a project fits, built only from the signals the API returned.
 * The language is shown next to it, so the sentence does not repeat it. `seen` counts the
 * co-starred repos already used in earlier rows: one very popular repo would otherwise give
 * every row the same sentence, so after two uses the next signal is used instead. */
export function whyItFits(repo: Repo, onboarding: boolean, seen?: Map<string, number>): string {
  const s = repo.signals || {};
  const lang = (repo.language || "").toLowerCase();
  const skills = usefulSkills(s.matched_skills || [], lang);
  const co = (s.co_starred_with || []).filter((n) => (seen?.get(n) ?? 0) < 2).slice(0, 2);
  if (co.length) {
    co.forEach((n) => seen?.set(n, (seen.get(n) ?? 0) + 1));
    return `People who starred ${list(co)}, like you, also starred this.`;
  }
  if (s.starred_by_you) return "You've already starred this, and it has issues set aside for newcomers.";
  if (onboarding && s.matched_interests && s.matched_interests.length) {
    return `It fits your interest in ${list(s.matched_interests.map((x) => x.toLowerCase()).slice(0, 2))}.`;
  }
  if (skills.length) return `It uses ${list(skills.slice(0, 3))}, which you already know.`;
  if (onboarding && repo.language) return "It's written in one of the languages you picked.";
  if ((s.co_starred_with || []).length) return "It's starred by the same people as several projects you follow.";
  return "It's close to the projects you've starred.";
}

/** Sentence for a recently created project. */
export function whyNew(repo: Repo): string {
  const skills = usefulSkills(repo.signals?.matched_skills || []).slice(0, 2);
  const started = `It started ${age(repo.created_at)}`;
  return skills.length ? `${started}, and it works with ${list(skills)}.` : `${started} and matches the languages you know.`;
}
