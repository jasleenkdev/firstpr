"use client";

import { useEffect, useId, useState } from "react";
import { ApiError, call, type Options, type Recommendation, type Repo } from "./api";
import { ContributionGrid, type GridMode } from "./components/ContributionGrid";
import { age, labelStyle, whyItFits, whyNew } from "./format";

type Path = "github" | "new";
type Server = "checking" | "waking" | "ready" | "down";
type Problem = { message: string; offerQuestions?: boolean };

const HOURS = [
  { value: 1, label: "About 1 hour" },
  { value: 3, label: "About 3 hours" },
  { value: 5, label: "About 5 hours" },
  { value: 8, label: "About 8 hours" },
  { value: 12, label: "10 hours or more" },
];

function problemFor(err: unknown, path: Path): Problem {
  const status = err instanceof ApiError ? err.status : 0;
  if (status === 404 && path === "github")
    return { message: "We couldn't find that username. Check the spelling, or answer three questions instead.", offerQuestions: true };
  if (status === 422 && path === "github")
    return { message: "That doesn't look like a GitHub username. Usernames use letters, numbers and single hyphens." };
  if (status === 422) return { message: "Pick at least one language or interest, then try again." };
  if (status === 429) return { message: "That was a lot of requests in a short time. Wait a minute, then try again." };
  if (status === 503)
    return { message: "GitHub isn't answering right now. Try again in a minute, or answer three questions instead.", offerQuestions: true };
  return { message: "We can't reach the server right now. Check your connection and try again in a minute." };
}

export default function Home() {
  const [path, setPath] = useState<Path>("github");
  const [server, setServer] = useState<Server>("checking");
  const [options, setOptions] = useState<Options | null>(null);
  const [username, setUsername] = useState("");
  const [hours, setHours] = useState(3);
  const [langs, setLangs] = useState<string[]>([]);
  const [interests, setInterests] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [result, setResult] = useState<Recommendation | null>(null);
  const [resultPath, setResultPath] = useState<Path>("github");

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
    if (path === "github" && !username.trim()) return setProblem({ message: "Enter your GitHub username first." });
    if (path === "new" && langs.length + interests.length === 0)
      return setProblem({ message: "Pick at least one language or interest first." });
    setProblem(null);
    setResult(null);
    setLoading(true);
    try {
      const out =
        path === "github"
          ? await call<Recommendation>("/recommend/github", { username: username.trim(), hours })
          : await call<Recommendation>("/recommend/onboarding", { languages: langs, interests, hours });
      setResult(out);
      setResultPath(path);
    } catch (err) {
      setProblem(problemFor(err, path));
    } finally {
      setLoading(false);
    }
  }

  function switchToQuestions() {
    setPath("new");
    setProblem(null);
    document.getElementById("questions")?.focus();
  }

  const toggle = (list: string[], set: (v: string[]) => void, v: string) =>
    set(list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);

  const gridMode: GridMode =
    loading || server === "waking" ? "loading" : result && resultPath === "github" ? "result" : "idle";
  const activity = result?.activity;
  const starDays = activity?.length ?? 0;
  const stars = (activity || []).reduce((n, a) => n + a.count, 0);
  const gridText =
    gridMode === "loading"
      ? "The squares fill in one by one while we wait."
      : gridMode === "result"
      ? stars
        ? `Each square is a day. The dark ones are the ${starDays} days in the last six months when you starred projects (${stars} stars). The yellow square, a few days from now, is your first contribution.`
        : "Each square is a day. We didn't find stars from the last six months. The yellow square, a few days from now, is your first contribution."
      : "Each square is a day of the last six months. The yellow one, a few days from now, is your first contribution.";

  return (
    <>
      <header className="site-header">
        <div className="wrap">
          <span className="wordmark">FirstPR</span>
        </div>
      </header>

      <main className="wrap">
        <section className="hero" aria-labelledby="hero-title">
          <h1 id="hero-title">Find your first contribution</h1>
          <p className="lede">
            FirstPR suggests open-source projects, and specific beginner issues inside them, that match what you
            know and how much time you have.
          </p>
          <ContributionGrid mode={gridMode} activity={activity} label={gridText} />
          <p className="grid-note">{gridText}</p>
          <p className="status" role="status" aria-live="polite">
            {server === "waking" && !loading && "Starting the server. The first visit of the day can take up to a minute."}
            {server === "down" && "We can't reach the server right now. Please try again in a few minutes."}
            {loading && "Finding projects that fit you."}
          </p>
        </section>

        <section className="ask" aria-labelledby="ask-title">
          <h2 id="ask-title">Where should we start?</h2>
          <form onSubmit={submit} noValidate>
            <fieldset className="paths">
              <legend className="visually-hidden">How should we find projects for you?</legend>
              <PathOption
                checked={path === "github"}
                onChange={() => setPath("github")}
                title="I have a GitHub account"
                hint="We'll start from the projects you've starred."
              />
              <PathOption
                checked={path === "new"}
                onChange={() => setPath("new")}
                title="I'm new to GitHub"
                hint="Answer three short questions instead."
              />
            </fieldset>

            {path === "github" ? (
              <div className="field">
                <label htmlFor="username">Your GitHub username</label>
                <input
                  id="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  autoCapitalize="none"
                  spellCheck={false}
                  maxLength={39}
                  aria-describedby="consent"
                />
                <p id="consent" className="hint">
                  We read your public stars from GitHub once to make these picks. We don't store or log your username.
                </p>
              </div>
            ) : (
              <div id="questions" tabIndex={-1} className="questions">
                <ChoiceGroup
                  legend="Which languages do you know?"
                  hint="Pick any that you've written code in, even a little."
                  options={(options?.languages ?? []).map((l) => ({ value: l, label: l }))}
                  selected={langs}
                  onToggle={(v) => toggle(langs, setLangs, v)}
                />
                <ChoiceGroup
                  legend="What would you like to work on?"
                  options={(options?.interests ?? []).map((t) => ({ value: t.id, label: t.label }))}
                  selected={interests}
                  onToggle={(v) => toggle(interests, setInterests, v)}
                />
              </div>
            )}

            <fieldset className="field hours">
              <legend>How many hours a week can you give?</legend>
              <div className="hour-options">
                {HOURS.map((h) => (
                  <label key={h.value} className="hour">
                    <input type="radio" name="hours" value={h.value} checked={hours === h.value} onChange={() => setHours(h.value)} />
                    <span>{h.label}</span>
                  </label>
                ))}
              </div>
            </fieldset>

            <button className="primary" type="submit" disabled={loading || server === "down"}>
              {loading ? "Finding projects" : "Show my recommendations"}
            </button>
          </form>

          {problem && (
            <div className="problem" role="alert">
              <p>{problem.message}</p>
              {problem.offerQuestions && (
                <button type="button" className="text-button" onClick={switchToQuestions}>
                  Answer three questions instead
                </button>
              )}
            </div>
          )}
        </section>

        {result && <Results result={result} path={resultPath} skills={resultPath === "github" ? result.profile.languages ?? [] : langs} onQuestions={switchToQuestions} />}
      </main>

      <footer className="site-footer">
        <div className="wrap">
          <p>
            Recommendations use public GitHub activity and repository pages.
            {result?.data_updated && <> Open issues were last refreshed on {new Date(result.data_updated).toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" })}.</>}
          </p>
        </div>
      </footer>
    </>
  );
}

function PathOption({ checked, onChange, title, hint }: { checked: boolean; onChange: () => void; title: string; hint: string }) {
  return (
    <label className={checked ? "path on" : "path"}>
      <input type="radio" name="path" checked={checked} onChange={onChange} />
      <span className="path-title">{title}</span>
      <span className="path-hint">{hint}</span>
    </label>
  );
}

function ChoiceGroup({
  legend,
  hint,
  options,
  selected,
  onToggle,
}: {
  legend: string;
  hint?: string;
  options: { value: string; label: string }[];
  selected: string[];
  onToggle: (v: string) => void;
}) {
  return (
    <fieldset className="field">
      <legend>{legend}</legend>
      {hint && <p className="hint">{hint}</p>}
      {options.length === 0 ? (
        <p className="hint">Loading the choices.</p>
      ) : (
        <div className="choices">
          {options.map((o) => (
            <label key={o.value} className={selected.includes(o.value) ? "choice on" : "choice"}>
              <input type="checkbox" checked={selected.includes(o.value)} onChange={() => onToggle(o.value)} />
              <span>{o.label}</span>
            </label>
          ))}
        </div>
      )}
    </fieldset>
  );
}

function Results({
  result,
  path,
  skills,
  onQuestions,
}: {
  result: Recommendation;
  path: Path;
  skills: string[];
  onQuestions: () => void;
}) {
  if (!result.repos.length)
    return (
      <section className="results" aria-labelledby="results-title">
        <h2 id="results-title">No matches yet</h2>
        <p>No projects match these choices yet. Add another language or interest and try again.</p>
      </section>
    );
  const seen = new Map<string, number>();
  return (
    <section className="results" aria-labelledby="results-title">
      <h2 id="results-title">Projects for you</h2>
      {result.degraded && (
        <div className="notice">
          <p>
            GitHub is limiting our requests right now, so these are general beginner picks rather than picks based on
            your stars. Try again later, or answer three questions instead.
          </p>
          <button type="button" className="text-button" onClick={onQuestions}>
            Answer three questions instead
          </button>
        </div>
      )}
      {result.profile.retrieval === "text_profile" && (
        <p className="intro">
          None of your stars are in our catalog yet, so these picks follow the languages and topics of what you've starred.
        </p>
      )}
      <ul className="projects">
        {result.repos.map((r) => (
          <ProjectRow key={r.id} repo={r} skills={skills} why={whyItFits(r, path === "new", seen)} />
        ))}
      </ul>
      {result.new_projects.length > 0 && (
        <>
          <h2 className="section-gap">New projects looking for first contributors</h2>
          <p className="intro">Started in the last six months, with beginner issues opened in the past two weeks.</p>
          <ul className="projects">
            {result.new_projects.map((r) => (
              <ProjectRow
                key={r.id}
                repo={r}
                skills={skills}
                why={whyNew(r)}
              />
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function ProjectRow({ repo, skills, why }: { repo: Repo; skills: string[]; why: string }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const regionId = useId();
  const s = repo.signals || {};

  async function toggleExplain() {
    if (open) return setOpen(false);
    setOpen(true);
    if (text) return;
    setBusy(true);
    try {
      const out = await call<{ text: string }>("/explain", {
        repo_id: repo.id,
        issue_number: repo.issues[0]?.number ?? null,
        co_starred: s.co_starred_with ?? [],
        skills: [...(s.matched_skills ?? []), ...skills].slice(0, 10),
      });
      setText(out.text);
    } catch {
      setText("We couldn't load an explanation right now. Try again in a minute.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <li className="project">
      <div className="project-head">
        <h3>
          <a href={repo.url} target="_blank" rel="noreferrer">
            {repo.name}
          </a>
        </h3>
        {repo.language && <span className="language">{repo.language}</span>}
      </div>
      <p className="why">{why}</p>
      {(repo.summary || repo.description) && <p className="summary">{repo.summary || repo.description}</p>}

      <ul className="thread" aria-label={`Beginner issues in ${repo.name}`}>
        {repo.issues.map((i) => (
          <li key={i.number} className="issue">
            <a href={i.url} target="_blank" rel="noreferrer">
              {i.title}
            </a>
            <p className="issue-meta">
              <span>
                Opened {age(i.created_at)}
                {i.difficulty ? `, looks ${i.difficulty}` : ""}
              </span>
              {i.labels.slice(0, 3).map((l) => (
                <span key={l} className="label" style={labelStyle(i.label_colors?.[l])}>
                  {l}
                </span>
              ))}
            </p>
          </li>
        ))}
      </ul>

      <button type="button" className="text-button" aria-expanded={open} aria-controls={regionId} onClick={toggleExplain}>
        {open ? "Hide explanation" : "Explain this pick"}
      </button>
      <div id={regionId} className={open ? "explain open" : "explain"} role="region" aria-label={`Why ${repo.name} was picked`} aria-hidden={!open}>
        <div className="explain-inner">
          <p aria-live="polite">{busy ? "Writing the explanation." : text}</p>
        </div>
      </div>
    </li>
  );
}
