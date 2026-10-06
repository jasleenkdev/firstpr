"use client";

import { useEffect, useState } from "react";
import { call, type Options, type Recommendation, type Repo } from "./api";

type Mode = "github" | "new";
type ServerState = "checking" | "waking" | "ready" | "down";

const HOURS = [1, 3, 5, 8, 12];

export default function Home() {
  const [mode, setMode] = useState<Mode>("github");
  const [server, setServer] = useState<ServerState>("checking");
  const [options, setOptions] = useState<Options | null>(null);
  const [username, setUsername] = useState("");
  const [hours, setHours] = useState(3);
  const [langs, setLangs] = useState<string[]>([]);
  const [interests, setInterests] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Recommendation | null>(null);

  useEffect(() => {
    let done = false;
    const slow = setTimeout(() => !done && setServer("waking"), 2500);
    call<Options>("/options", undefined, 90000)
      .then((o) => {
        setOptions(o);
        setServer("ready");
      })
      .catch(() => setServer("down"))
      .finally(() => {
        done = true;
        clearTimeout(slow);
      });
    return () => clearTimeout(slow);
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const out =
        mode === "github"
          ? await call<Recommendation>("/recommend/github", { username: username.trim(), hours })
          : await call<Recommendation>("/recommend/onboarding", { languages: langs, interests, hours });
      setResult(out);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const toggle = (list: string[], set: (v: string[]) => void, v: string) =>
    set(list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);

  const canSubmit =
    !loading && server !== "down" && (mode === "github" ? username.trim().length > 0 : langs.length + interests.length > 0);

  return (
    <main className="wrap">
      <header className="hero">
        <h1>FirstPR</h1>
        <p>Open-source projects and beginner issues picked for you, with the reasons behind each pick.</p>
      </header>

      {server === "waking" && <p className="notice">Waking up the server, this can take up to a minute…</p>}
      {server === "down" && <p className="notice error">The server is not reachable right now. Please try again later.</p>}

      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={mode === "github"} className={mode === "github" ? "tab on" : "tab"} onClick={() => setMode("github")}>
          I have GitHub
        </button>
        <button role="tab" aria-selected={mode === "new"} className={mode === "new" ? "tab on" : "tab"} onClick={() => setMode("new")}>
          New to GitHub
        </button>
      </div>

      <form className="card form" onSubmit={submit}>
        {mode === "github" ? (
          <>
            <label htmlFor="user">GitHub username</label>
            <input
              id="user"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="your-username"
              autoComplete="off"
              autoCapitalize="none"
              spellCheck={false}
              maxLength={39}
            />
            <p className="consent">
              We read your public stars from GitHub once to make these recommendations. Your username is not stored
              or logged.
            </p>
          </>
        ) : (
          <>
            <fieldset>
              <legend>Languages you know</legend>
              <div className="chips">
                {(options?.languages ?? []).map((l) => (
                  <button type="button" key={l} className={langs.includes(l) ? "chip on" : "chip"} onClick={() => toggle(langs, setLangs, l)}>
                    {l}
                  </button>
                ))}
              </div>
            </fieldset>
            <fieldset>
              <legend>What interests you</legend>
              <div className="chips">
                {(options?.interests ?? []).map((t) => (
                  <button type="button" key={t.id} className={interests.includes(t.id) ? "chip on" : "chip"} onClick={() => toggle(interests, setInterests, t.id)}>
                    {t.label}
                  </button>
                ))}
              </div>
            </fieldset>
          </>
        )}
        <label htmlFor="hours">Hours per week you can spend</label>
        <select id="hours" value={hours} onChange={(e) => setHours(Number(e.target.value))}>
          {HOURS.map((h) => (
            <option key={h} value={h}>
              {h === 12 ? "10+" : h} hours
            </option>
          ))}
        </select>
        <button className="primary" type="submit" disabled={!canSubmit}>
          {loading ? "Finding projects…" : "Find projects"}
        </button>
      </form>

      {error && <p className="notice error">{error}</p>}
      {loading && <Skeleton />}
      {result && <Results result={result} skills={mode === "github" ? ((result.profile.languages as string[]) ?? []) : langs} />}

      <footer>
        Data: GitHub public events and repository pages.{" "}
        {result?.data_updated && <>Issues updated {new Date(result.data_updated).toLocaleDateString()}.</>}
      </footer>
    </main>
  );
}

function Skeleton() {
  return (
    <div aria-hidden>
      {[0, 1, 2].map((i) => (
        <div key={i} className="card skeleton" />
      ))}
    </div>
  );
}

function Results({ result, skills }: { result: Recommendation; skills: string[] }) {
  if (!result.repos.length) return <p className="notice">No matches yet. Try adding a language or an interest.</p>;
  return (
    <section>
      {result.degraded && <p className="notice">{result.degraded.message}</p>}
      <h2>Recommended for you</h2>
      {result.profile.retrieval === "text_profile" && (
        <p className="hint">None of your stars are in our catalog yet, so these picks match the languages and topics of your stars.</p>
      )}
      {result.repos.map((r) => (
        <RepoCard key={r.id} repo={r} skills={skills} />
      ))}
      {result.new_projects.length > 0 && (
        <>
          <h2>New projects</h2>
          <p className="hint">Recently created repositories with fresh beginner issues that match your skills.</p>
          {result.new_projects.map((r) => (
            <RepoCard key={r.id} repo={r} skills={skills} isNew />
          ))}
        </>
      )}
    </section>
  );
}

function RepoCard({ repo, skills, isNew }: { repo: Repo; skills: string[]; isNew?: boolean }) {
  const [why, setWhy] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const s = repo.signals || {};

  async function explain() {
    if (why) return setWhy(null);
    setBusy(true);
    try {
      const out = await call<{ text: string }>("/explain", {
        repo_id: repo.id,
        issue_number: repo.issues[0]?.number ?? null,
        co_starred: s.co_starred_with ?? [],
        skills: [...(s.matched_skills ?? []), ...skills].slice(0, 10),
      });
      setWhy(out.text);
    } catch {
      setWhy("Could not load the explanation right now.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <article className="card repo">
      <div className="repo-head">
        <a href={repo.url} target="_blank" rel="noreferrer" className="repo-name">
          {repo.name}
        </a>
        <span className="meta">
          {repo.language && <span className="lang">{repo.language}</span>}★ {repo.stars.toLocaleString()}
          {isNew && <span className="badge new">new</span>}
        </span>
      </div>
      <p className="summary">{repo.summary || repo.description}</p>
      <div className="signals">
        {s.starred_by_you && <span className="sig">You starred this</span>}
        {(s.co_starred_with ?? []).length > 0 && <span className="sig">Starred by people who starred {s.co_starred_with!.join(", ")}</span>}
        {(s.matched_skills ?? []).slice(0, 4).map((k) => (
          <span key={k} className="sig skill">
            {k}
          </span>
        ))}
      </div>
      <ul className="issues">
        {repo.issues.map((i) => (
          <li key={i.number}>
            <a href={i.url} target="_blank" rel="noreferrer">
              {i.title}
            </a>
            <span className="issue-meta">
              {i.difficulty && <span className={`badge ${i.difficulty}`}>{i.difficulty}</span>}
              {i.labels.slice(0, 2).map((l) => (
                <span key={l} className="label">
                  {l}
                </span>
              ))}
            </span>
          </li>
        ))}
      </ul>
      <button className="link" onClick={explain} disabled={busy}>
        {busy ? "Loading…" : why ? "Hide" : "Why this?"}
      </button>
      {why && <p className="why">{why}</p>}
    </article>
  );
}
