//! Structural tells in the code a pull request adds.
//!
//! The prose lane catches text that reads generated. This lane catches the
//! opposite failure: a pull request that reads competent and is not. Planned
//! agent output restates itself instead of factoring, and inlines the values
//! a maintainer expects to find named. A method body pasted twice, a literal
//! repeated in four places, a private helper nothing calls: none of these
//! proves who wrote the code, and all of them are work a reviewer has to
//! undo. That is what makes them scoreable independently of authorship.
//!
//! Everything here reads production source only. Tests, fixtures, examples,
//! generated code, and data files repeat themselves and hardcode values for
//! good reasons, and judging them produces false positives instead of
//! signal. The analysis is lexical, over the added lines of a unified diff:
//! it sees what a diff shows and nothing about the rest of the repository,
//! so a helper called only from an unchanged file reads as uncalled here.
//! That bound is why these rules ship dark and get priced by the fit.

use crate::codeslop::is_comment;
use crate::engine::Fire;
use crate::hashing::fnv1a64;
use std::collections::{HashMap, HashSet};

/// Below this many added production code lines there is nothing to judge.
const MIN_CODE_LINES: usize = 20;
/// Consecutive lines compared as one unit by the clone detector.
const WINDOW: usize = 5;
/// Bound on the pairwise body comparison and on how far a body may run.
const MAX_FUNCS: usize = 60;
const MAX_BODY: usize = 60;

#[derive(Default)]
struct FileDiff<'a> {
    path: String,
    added: Vec<&'a str>,
    /// Lines the change did not write: context and removals. The file's
    /// existing idiom, carried along by the diff format itself.
    existing: Vec<&'a str>,
}

/// Added lines of a unified diff, grouped by the file they land in.
fn split_files(diff: &str) -> Vec<FileDiff<'_>> {
    let mut files: Vec<FileDiff> = Vec::new();
    let mut cur = FileDiff::default();
    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            if !cur.path.is_empty() || !cur.added.is_empty() {
                files.push(std::mem::take(&mut cur));
            }
            cur.path = git_path(rest);
        } else if let Some(p) = line.strip_prefix("+++ ") {
            let p = p.trim().trim_start_matches("b/");
            if cur.path.is_empty() && p != "/dev/null" {
                cur.path = p.to_string();
            }
        } else if line.starts_with("---") || line.starts_with("@@") {
            continue;
        } else if let Some(added) = line.strip_prefix('+') {
            cur.added.push(added);
        } else if let Some(old) = line.strip_prefix('-').or_else(|| line.strip_prefix(' ')) {
            cur.existing.push(old);
        }
    }
    if !cur.path.is_empty() || !cur.added.is_empty() {
        files.push(cur);
    }
    files
}

/// `a/src/x.rs b/src/x.rs` -> `src/x.rs`.
fn git_path(header: &str) -> String {
    match header.rfind(" b/") {
        Some(i) => header[i + 3..].trim().to_string(),
        None => header.trim().to_string(),
    }
}

const DOC_EXT: &[&str] = &[
    ".md", ".rst", ".txt", ".adoc", ".svg", ".css", ".scss", ".po",
];
const DATA_EXT: &[&str] = &[
    ".json", ".yaml", ".yml", ".toml", ".xml", ".csv", ".ini", ".cfg", ".lock", ".sql", ".snap",
    ".sum",
];
const NOT_AUTHORED: &[&str] = &[
    "vendor/",
    "node_modules/",
    "third_party/",
    "/dist/",
    "/build/",
    ".min.js",
    ".pb.go",
    "_pb2.py",
    "generated",
    "migrations/",
];
const NOT_PRODUCTION: &[&str] = &[
    "test",
    "spec",
    "fixture",
    "mock",
    "__tests__",
    "/e2e/",
    "bench",
    "example",
    "sample",
    "demo",
];

/// Source a maintainer holds to production standards. A diff with no file
/// headers at all is judged rather than skipped; truncation is common and
/// silence would be the wrong default.
fn is_production_source(path: &str) -> bool {
    if path.is_empty() {
        return true;
    }
    let p = path.to_lowercase();
    if DOC_EXT.iter().any(|e| p.ends_with(e)) || DATA_EXT.iter().any(|e| p.ends_with(e)) {
        return false;
    }
    if NOT_AUTHORED.iter().any(|m| p.contains(m)) {
        return false;
    }
    !NOT_PRODUCTION.iter().any(|m| p.contains(m))
}

/// The structural form of a line: string literals collapsed to `S`, digit
/// runs to `N`, case and spacing dropped. Two lines that differ only in
/// their names or values still normalize apart; two pasted copies collapse
/// to the same string. Single quotes are left alone because they are Rust
/// lifetimes as often as they are strings, and mis-parsing them would make
/// unrelated lines look alike.
fn normalize(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut last_space = true;
    while let Some(c) = chars.next() {
        if c == '"' || c == '`' {
            out.push('S');
            let mut escaped = false;
            for d in chars.by_ref() {
                if escaped {
                    escaped = false;
                } else if d == '\\' {
                    escaped = true;
                } else if d == c {
                    break;
                }
            }
            last_space = false;
        } else if c.is_ascii_digit() {
            if !out.ends_with('N') {
                out.push('N');
            }
            last_space = false;
        } else if c.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            for lc in c.to_lowercase() {
                out.push(lc);
            }
            last_space = false;
        }
    }
    out.trim().to_string()
}

/// Numbers worth naming and strings long enough to mean something. Zero,
/// one and the round bases are not magic in anyone's code.
const TRIVIAL_NUMS: &[&str] = &["0", "1", "2", "10", "100", "1000", "0.0", "1.0"];

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Literals on a line, tagged `s:` or `n:` so the two kinds never collide.
fn literals(line: &str) -> Vec<String> {
    let ch: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < ch.len() {
        let c = ch[i];
        // `<'a` and `&'a` are Rust lifetimes. Scanning them as strings makes
        // the rest of the line disappear into a literal that is not there.
        if c == '\'' && i > 0 && (ch[i - 1] == '<' || ch[i - 1] == '&') {
            i += 1;
            continue;
        }
        if c == '"' || c == '\'' {
            let mut j = i + 1;
            let mut buf = String::new();
            while j < ch.len() && ch[j] != c {
                if ch[j] == '\\' {
                    j += 2;
                    continue;
                }
                buf.push(ch[j]);
                j += 1;
            }
            if j >= ch.len() {
                // Unterminated: an apostrophe or a lifetime, not a literal.
                i += 1;
                continue;
            }
            if buf.chars().count() >= 4 {
                out.push(format!("s:{}", buf.to_lowercase()));
            }
            i = j + 1;
            continue;
        }
        if c.is_ascii_digit() && (i == 0 || !is_ident_char(ch[i - 1])) {
            let mut j = i;
            let mut buf = String::new();
            while j < ch.len() && (ch[j].is_ascii_alphanumeric() || ch[j] == '.' || ch[j] == '_') {
                buf.push(ch[j]);
                j += 1;
            }
            let v = buf.trim_end_matches('.').trim_end_matches('_').to_string();
            if !v.is_empty() && !TRIVIAL_NUMS.contains(&v.as_str()) {
                out.push(format!("n:{v}"));
            }
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

struct Func {
    name: String,
    exported: bool,
    body: Vec<String>,
    /// Index of the definition line within its file's added lines.
    line: usize,
    file: usize,
}

const MODIFIERS: &[&str] = &[
    "public ",
    "private ",
    "protected ",
    "internal ",
    "static ",
    "void ",
    "async ",
    "override ",
    "final ",
];

/// The name a line defines, and whether it is visible outside its module.
/// Covers the keyword languages directly and the modifier-plus-signature
/// shape used by Java, C#, C and their relatives.
fn function_def(line: &str) -> Option<(String, bool)> {
    let t = line.trim();
    if t.is_empty() || is_comment(t) {
        return None;
    }
    let low = t.to_lowercase();
    let exported = low.starts_with("pub ")
        || low.starts_with("pub(")
        || low.starts_with("export ")
        || low.starts_with("public ")
        || low.contains(" pub fn ")
        || low.contains("export function")
        || low.contains("public static");

    for kw in ["def ", "fn ", "func ", "function ", "sub "] {
        let Some(at) = keyword_at(&low, kw) else {
            continue;
        };
        let mut rest = low[at + kw.len()..].trim_start();
        // A Go method receiver sits between the keyword and the name.
        if let Some(stripped) = rest.strip_prefix('(') {
            match stripped.find(')') {
                Some(close) => rest = stripped[close + 1..].trim_start(),
                None => continue,
            }
        }
        let name: String = rest.chars().take_while(|c| is_ident_char(*c)).collect();
        if name.len() >= 2 && rest[name.len()..].trim_start().starts_with('(') {
            return Some((name, exported));
        }
    }

    // `private static Result processData(Foo a) {`
    if low.ends_with('{')
        && MODIFIERS.iter().any(|m| low.starts_with(m))
        && let Some(open) = low.find('(')
    {
        let head = &low[..open];
        let name: String = head
            .chars()
            .rev()
            .take_while(|c| is_ident_char(*c))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if name.len() >= 2 && !name.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            return Some((name, exported));
        }
    }
    None
}

/// Position of `kw` when it opens the line or follows a space.
fn keyword_at(low: &str, kw: &str) -> Option<usize> {
    if low.starts_with(kw) {
        return Some(0);
    }
    low.find(&format!(" {kw}")).map(|i| i + 1)
}

fn body_lines<'a>(lines: impl Iterator<Item = &'a str>) -> Vec<String> {
    lines
        .map(str::trim)
        .filter(|l| !l.is_empty() && !is_comment(l))
        .map(normalize)
        .filter(|l| l.len() >= 3)
        .collect()
}

fn functions(files: &[&FileDiff]) -> Vec<Func> {
    let mut out = Vec::new();
    for (fi, f) in files.iter().enumerate() {
        let starts: Vec<(usize, String, bool)> = f
            .added
            .iter()
            .enumerate()
            .filter_map(|(i, l)| function_def(l).map(|(n, e)| (i, n, e)))
            .collect();
        for (k, s) in starts.iter().enumerate() {
            let next = starts.get(k + 1).map_or(f.added.len(), |n| n.0);
            let end = next.min(s.0 + MAX_BODY).min(f.added.len());
            out.push(Func {
                name: s.1.clone(),
                exported: s.2,
                body: body_lines(f.added[s.0 + 1..end].iter().copied()),
                line: s.0,
                file: fi,
            });
            if out.len() >= MAX_FUNCS {
                return out;
            }
        }
    }
    out
}

fn jaccard(a: &[String], b: &[String]) -> f64 {
    let sa: HashSet<&String> = a.iter().collect();
    let sb: HashSet<&String> = b.iter().collect();
    let union = sa.union(&sb).count();
    if union == 0 {
        return 0.0;
    }
    sa.intersection(&sb).count() as f64 / union as f64
}

/// Copy-paste inside one pull request: five-line windows that appear more
/// than once, anywhere in the added production code.
fn dup_block(files: &[&FileDiff]) -> Option<Fire> {
    let mut seen: HashSet<u64> = HashSet::new();
    let (mut windows, mut repeats) = (0usize, 0usize);
    for f in files {
        let norm = body_lines(f.added.iter().copied());
        for w in norm.windows(WINDOW) {
            // A run of near-identical boilerplate (imports, field lists) is
            // not a duplicated block.
            if w.iter().collect::<HashSet<_>>().len() < 3 {
                continue;
            }
            windows += 1;
            if !seen.insert(fnv1a64(w.join("\n").as_bytes())) {
                repeats += 1;
            }
        }
    }
    if windows < 8 {
        return None;
    }
    let frac = repeats as f64 / windows as f64;
    (frac >= 0.15).then(|| Fire::new("CODE_DUP_BLOCK", (frac * 2.0).min(1.0)))
}

/// Two added functions with the same body. The named version of the same
/// failure: the second one should have been a call to the first.
fn dup_func(funcs: &[Func]) -> Option<Fire> {
    let bodies: Vec<&Func> = funcs.iter().filter(|f| f.body.len() >= 4).collect();
    let mut pairs = 0usize;
    for i in 0..bodies.len() {
        for j in i + 1..bodies.len() {
            if jaccard(&bodies[i].body, &bodies[j].body) >= 0.85 {
                pairs += 1;
            }
        }
    }
    (pairs > 0).then(|| Fire::new("CODE_DUP_FUNC", (pairs as f64 / 3.0).min(1.0)))
}

const CONST_DEF: &[&str] = &[
    "const ",
    "static ",
    "final ",
    "#define",
    "constexpr",
    "readonly",
    "enum ",
];

fn is_declaration(low: &str) -> bool {
    if CONST_DEF.iter().any(|k| low.contains(k)) {
        return true;
    }
    for opener in [
        "import ", "from ", "use ", "#include", "require(", "package ",
    ] {
        if low.starts_with(opener) {
            return true;
        }
    }
    low.contains("version")
}

/// A named constant is not magic. Anything before `=` that carries no
/// lowercase letters is one.
fn assigns_to_constant(line: &str) -> bool {
    let Some(eq) = line.find('=') else {
        return false;
    };
    let target = &line[..eq];
    target.chars().any(|c| c.is_ascii_uppercase()) && !target.chars().any(|c| c.is_lowercase())
}

fn magic_number(code: &[&str]) -> Option<Fire> {
    let mut hits = 0usize;
    for line in code {
        let t = line.trim();
        let low = t.to_lowercase();
        if is_declaration(&low) || assigns_to_constant(t) {
            continue;
        }
        if literals(t).iter().any(|l| l.starts_with("n:")) {
            hits += 1;
        }
    }
    let frac = hits as f64 / code.len() as f64;
    (frac >= 0.10).then(|| Fire::new("CODE_MAGIC_NUMBER", (frac * 2.0).min(1.0)))
}

/// The same literal written out in three or more places. The tell is not
/// the value, it is that nothing was named: change it once and the other
/// copies stay wrong.
fn repeated_literal(code: &[&str]) -> Option<Fire> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for line in code {
        let t = line.trim();
        if is_declaration(&t.to_lowercase()) {
            continue;
        }
        // Once per line: `f(9, 9, 9)` is one place, not three.
        for lit in literals(t).into_iter().collect::<HashSet<_>>() {
            *counts.entry(lit).or_default() += 1;
        }
    }
    let repeated = counts.values().filter(|c| **c >= 3).count();
    (repeated > 0).then(|| Fire::new("CODE_REPEAT_LITERAL", (repeated as f64 / 3.0).min(1.0)))
}

const CONFIG_MARKERS: &[&str] = &[
    "http://",
    "https://",
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "/tmp/",
    "/home/",
    "/usr/",
    "/var/",
    "c:\\",
    "timeout=",
    "timeout =",
    "retries=",
    "retries =",
    "api_key",
    "apikey",
    "password =",
    "password=",
];
const CONFIG_SOURCE: &[&str] = &[
    "getenv",
    "environ",
    "env::var",
    "process.env",
    "os.environ",
    "config.",
    "settings.",
    "viper.",
];

/// Values that belong in configuration, written into the source. Lines that
/// read the environment are excluded: that is the shape being asked for.
fn hardcoded_config(code: &[&str]) -> Option<Fire> {
    let mut hits = 0usize;
    for line in code {
        let low = line.trim().to_lowercase();
        if CONFIG_SOURCE.iter().any(|s| low.contains(s)) {
            continue;
        }
        if CONFIG_MARKERS.iter().any(|m| low.contains(m)) {
            hits += 1;
        }
    }
    (hits >= 2).then(|| Fire::new("CODE_HARDCODED_CONFIG", (hits as f64 / 4.0).min(1.0)))
}

const BROAD_CATCH: &[&str] = &[
    "except exception",
    "except baseexception",
    "except:",
    "catch (exception",
    "catch(exception",
    "catch (throwable",
    "catch (error",
    "catch (e)",
    "catch (err)",
    "catch {",
    "rescue exception",
];
const SWALLOW: &[&str] = &[
    "pass",
    "continue",
    "return",
    "return none",
    "return null",
    "return nil",
    "return false",
    "return true",
    "break",
];

/// Catch-everything handlers, and how many of them discard what they caught.
/// Uniform defensive wrapping is how generated code survives its own
/// uncertainty about which calls can fail.
fn blanket_catch(files: &[&FileDiff]) -> Option<Fire> {
    let (mut broad, mut swallowed) = (0usize, 0usize);
    for f in files {
        for (i, line) in f.added.iter().enumerate() {
            let low = line.trim().to_lowercase();
            if !BROAD_CATCH.iter().any(|m| low.contains(m)) {
                continue;
            }
            broad += 1;
            let next = f.added[i + 1..]
                .iter()
                .map(|l| l.trim().to_lowercase())
                .find(|l| !l.is_empty() && !is_comment(l));
            if let Some(n) = next {
                let n = n.trim_end_matches(';').trim();
                if SWALLOW.contains(&n)
                    || n.starts_with("log")
                    || n.starts_with("print")
                    || n.starts_with("console.")
                {
                    swallowed += 1;
                }
            }
        }
    }
    (broad >= 2).then(|| {
        Fire::new(
            "CODE_BLANKET_CATCH",
            ((broad + swallowed) as f64 / 4.0).min(1.0),
        )
    })
}

const DOC_MARKERS: &[&str] = &[
    "args:",
    "arguments:",
    "returns:",
    "raises:",
    "throws:",
    "@param",
    "@returns",
    "@return",
    ":param",
    ":returns",
    ":rtype",
];

/// Every added function carrying a full parameter table, including the
/// trivial ones. This is taste rather than defect, which is why the repo's
/// AI policy can zero it: a project that welcomes generated documentation
/// gets exactly this shape and wants it.
fn doc_scaffold(files: &[&FileDiff], funcs: &[Func]) -> Option<Fire> {
    if funcs.len() < 2 {
        return None;
    }
    let mut documented = 0usize;
    for func in funcs {
        let Some(f) = files.get(func.file) else {
            continue;
        };
        let start = func.line + 1;
        let end = (start + 4).min(f.added.len());
        let mut in_doc = false;
        let scaffolded = f.added[start..end].iter().any(|l| {
            let low = l.trim().to_lowercase();
            if low.contains("\"\"\"") || low.contains("'''") {
                in_doc = !in_doc;
            }
            let doc_line = in_doc || is_comment(&low) || low.starts_with("\"\"\"");
            doc_line && DOC_MARKERS.iter().any(|m| low.contains(m))
        });
        if scaffolded {
            documented += 1;
        }
    }
    let ratio = documented as f64 / funcs.len() as f64;
    (documented >= 2 && ratio >= 0.6).then(|| Fire::new("CODE_DOC_SCAFFOLD", ratio))
}

const HOOK_NAMES: &[&str] = &[
    "main", "init", "setup", "teardown", "new", "default", "run", "index", "handler",
];

/// Private helpers the pull request defines and never calls. A diff is a
/// keyhole view: a helper called from an unchanged file looks uncalled from
/// here. So the rule asks for its own evidence that call sites are visible
/// at all, by requiring that at least one other added function is called
/// within the diff. Wiring up three helpers and leaving two stranded is a
/// claim about the change; showing no wiring at all is just a small window.
fn dead_helper(funcs: &[Func], diff: &str) -> Option<Fire> {
    let low = diff.to_lowercase();
    let called = |name: &str| low.matches(name).count() > 1;
    let mut dead = 0usize;
    let mut wired = false;
    for f in funcs {
        if f.name.len() < 4 || HOOK_NAMES.contains(&f.name.as_str()) {
            continue;
        }
        if called(&f.name) {
            wired = true;
            continue;
        }
        if f.exported || f.name.starts_with("test") || f.name.starts_with("on") {
            continue;
        }
        dead += 1;
    }
    (wired && dead >= 2).then(|| Fire::new("CODE_DEAD_HELPER", (dead as f64 / 3.0).min(1.0)))
}

/// One measurable habit of a file, and how many samples back it.
#[derive(Default, Clone, Copy)]
struct Habit {
    rate: f64,
    n: usize,
}

impl Habit {
    fn of(yes: usize, total: usize) -> Self {
        Habit {
            rate: if total == 0 {
                0.0
            } else {
                yes as f64 / total as f64
            },
            n: total,
        }
    }

    /// How far two habits are apart, when both sides have enough of the
    /// thing to have a habit about it at all. `None` means not measurable,
    /// which is different from measured and equal.
    fn gap(self, other: Habit, min_n: usize) -> Option<f64> {
        (self.n >= min_n && other.n >= min_n).then(|| (self.rate - other.rate).abs())
    }
}

#[derive(Default)]
struct Idiom {
    camel: Habit,
    tabs: Habit,
    dquote: Habit,
    comments: Habit,
}

fn identifiers(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in line.chars() {
        if is_ident_char(c) {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// How a set of lines is written: naming, indentation, quoting, comment
/// habit. Case matters here, so this reads raw lines rather than the
/// normalized form.
fn idiom_of(lines: &[&str]) -> Idiom {
    let (mut camel, mut snake) = (0usize, 0usize);
    let (mut tabbed, mut indented) = (0usize, 0usize);
    let (mut dquote, mut quotes) = (0usize, 0usize);
    let (mut comments, mut total) = (0usize, 0usize);

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        if is_comment(line.trim()) {
            comments += 1;
            continue;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            indented += 1;
            if line.starts_with('\t') {
                tabbed += 1;
            }
        }
        for c in line.chars() {
            match c {
                '"' => {
                    quotes += 1;
                    dquote += 1;
                }
                '\'' => quotes += 1,
                _ => {}
            }
        }
        for id in identifiers(line) {
            // SCREAMING_CASE is a separate convention from either.
            if id.chars().all(|c| !c.is_lowercase()) {
                continue;
            }
            let has_underscore = id.trim_matches('_').contains('_');
            let has_hump = id
                .chars()
                .zip(id.chars().skip(1))
                .any(|(a, b)| a.is_lowercase() && b.is_uppercase());
            match (has_underscore, has_hump) {
                (true, false) => snake += 1,
                (false, true) => camel += 1,
                _ => {}
            }
        }
    }

    Idiom {
        camel: Habit::of(camel, camel + snake),
        tabs: Habit::of(tabbed, indented),
        dquote: Habit::of(dquote, quotes),
        comments: Habit::of(comments, total),
    }
}

/// Added code written in a different idiom from the file it lands in.
/// Naming, indentation, quoting and commenting are the habits a
/// contributor picks up by reading the file first, and that code written
/// elsewhere arrives without.
///
/// Most files agree with themselves on most axes, so requiring two
/// divergences requires a coincidence. The value is the widest gap on any
/// measurable axis, nudged up when more than one axis moved.
fn style_drift(files: &[&FileDiff]) -> Option<Fire> {
    let mut best = 0.0f64;
    for f in files {
        if f.added.len() < 10 || f.existing.len() < 10 {
            continue;
        }
        let new = idiom_of(&f.added);
        let old = idiom_of(&f.existing);
        let gaps: Vec<f64> = [
            new.camel.gap(old.camel, 8),
            new.tabs.gap(old.tabs, 5),
            new.dquote.gap(old.dquote, 6),
            new.comments.gap(old.comments, 12),
        ]
        .into_iter()
        .flatten()
        .collect();
        let Some(widest) = gaps.iter().cloned().fold(None, |m: Option<f64>, g| {
            Some(m.map_or(g, |m: f64| m.max(g)))
        }) else {
            continue;
        };
        let also_moved = gaps.iter().filter(|g| **g >= 0.3).count().saturating_sub(1);
        best = best.max(widest + 0.15 * also_moved as f64);
    }
    (best >= 0.5).then(|| Fire::new("CODE_STYLE_DRIFT", best.min(1.0)))
}

pub fn rules(diff: &str) -> Vec<Fire> {
    let files = split_files(diff);
    let prod: Vec<&FileDiff> = files
        .iter()
        .filter(|f| is_production_source(&f.path))
        .collect();

    let code: Vec<&str> = prod
        .iter()
        .flat_map(|f| f.added.iter().copied())
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !is_comment(t)
        })
        .collect();
    if code.len() < MIN_CODE_LINES {
        return Vec::new();
    }

    let funcs = functions(&prod);
    [
        dup_block(&prod),
        dup_func(&funcs),
        magic_number(&code),
        repeated_literal(&code),
        hardcoded_config(&code),
        blanket_catch(&prod),
        doc_scaffold(&prod, &funcs),
        dead_helper(&funcs, diff),
        style_drift(&prod),
    ]
    .into_iter()
    .flatten()
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fired(diff: &str) -> Vec<String> {
        rules(diff).into_iter().map(|f| f.rule).collect()
    }

    const PASTED: &str = "\
diff --git a/src/api.py b/src/api.py
--- a/src/api.py
+++ b/src/api.py
@@ -1,0 +1,40 @@
+def fetch_users(client):
+    response = client.get(url_for(\"users\"))
+    if response.status_code != 200:
+        logger.error(\"request failed\")
+        return []
+    payload = response.json()
+    return payload[\"items\"]
+
+def fetch_groups(client):
+    response = client.get(url_for(\"groups\"))
+    if response.status_code != 200:
+        logger.error(\"request failed\")
+        return []
+    payload = response.json()
+    return payload[\"items\"]
+
+def fetch_teams(client):
+    response = client.get(url_for(\"teams\"))
+    if response.status_code != 200:
+        logger.error(\"request failed\")
+        return []
+    payload = response.json()
+    return payload[\"items\"]
+
+def load_all(client):
+    everything = []
+    everything.extend(fetch_users(client))
+    everything.extend(fetch_groups(client))
+    everything.extend(fetch_teams(client))
+    return everything
";

    #[test]
    fn pasted_methods_fire_duplication() {
        let f = fired(PASTED);
        assert!(f.contains(&"CODE_DUP_FUNC".to_string()), "{f:?}");
        assert!(f.contains(&"CODE_DUP_BLOCK".to_string()), "{f:?}");
    }

    #[test]
    fn hardcoded_values_fire() {
        let diff = "\
diff --git a/src/client.go b/src/client.go
+++ b/src/client.go
+func dial() net.Conn {
+	c, _ := net.Dial(\"tcp\", \"127.0.0.1:9091\")
+	c.SetDeadline(time.Now().Add(45 * time.Second))
+	return c
+}
+
+func dialBackup() net.Conn {
+	c, _ := net.Dial(\"tcp\", \"127.0.0.1:9091\")
+	c.SetDeadline(time.Now().Add(45 * time.Second))
+	return c
+}
+
+func endpoint() string {
+	return \"http://127.0.0.1:9091/collect\"
+}
+
+func retryFor(n int) time.Duration {
+	if n > 45 {
+		return 45 * time.Second
+	}
+	return time.Duration(n) * time.Second
+}
+
+func report(err error) {
+	http.Post(\"http://127.0.0.1:9091/errors\", \"application/json\", nil)
+}
";
        let f = fired(diff);
        assert!(f.contains(&"CODE_HARDCODED_CONFIG".to_string()), "{f:?}");
        assert!(f.contains(&"CODE_REPEAT_LITERAL".to_string()), "{f:?}");
    }

    #[test]
    fn blanket_handlers_fire() {
        let diff = "\
diff --git a/src/jobs.py b/src/jobs.py
+++ b/src/jobs.py
+def start_job(name):
+    try:
+        queue.submit(name)
+    except Exception:
+        pass
+
+def stop_job(name):
+    try:
+        queue.cancel(name)
+    except Exception:
+        logger.warning(\"could not cancel\")
+
+def drain(names):
+    for name in names:
+        try:
+            queue.cancel(name)
+        except Exception:
+            continue
+    queue.flush()
+    queue.close()
+    return True
+
+def status(name):
+    row = registry.lookup(name)
+    if row is None:
+        return \"unknown\"
+    if row.finished_at is not None:
+        return \"done\"
+    if row.started_at is not None:
+        return \"running\"
+    return \"queued\"
";
        assert!(fired(diff).contains(&"CODE_BLANKET_CATCH".to_string()));
    }

    #[test]
    fn parameter_tables_on_everything_fire() {
        let diff = "\
diff --git a/src/util.py b/src/util.py
+++ b/src/util.py
+def add(a, b):
+    \"\"\"Add two numbers.
+
+    Args:
+        a: The first number.
+        b: The second number.
+
+    Returns:
+        The sum of a and b.
+    \"\"\"
+    return a + b
+
+def double(a):
+    \"\"\"Double a number.
+
+    Args:
+        a: The number to double.
+
+    Returns:
+        Twice the input value.
+    \"\"\"
+    return a * 2
+
+def negate(a):
+    \"\"\"Negate a number.
+
+    Args:
+        a: The number to negate.
+
+    Returns:
+        The negated value.
+    \"\"\"
+    return -a
";
        assert!(fired(diff).contains(&"CODE_DOC_SCAFFOLD".to_string()));
    }

    #[test]
    fn uncalled_private_helpers_fire() {
        let diff = "\
diff --git a/src/report.py b/src/report.py
+++ b/src/report.py
+def normalize_currency(amount):
+    return round(amount, 2)
+
+def sanitize_label(label):
+    return label.strip().lower()
+
+def build_report(rows):
+    total = 0
+    for row in rows:
+        total += row.amount
+    return {\"total\": total, \"count\": len(rows)}
+
+def emit(rows):
+    report = build_report(rows)
+    writer.write(report)
+    writer.flush()
+    return report
+
+def emit_all(batches):
+    written = []
+    for batch in batches:
+        written.append(emit(batch))
+    writer.close()
+    return written
";
        assert!(fired(diff).contains(&"CODE_DEAD_HELPER".to_string()));
    }

    #[test]
    fn real_human_code_stays_quiet() {
        let diff = "\
diff --git a/src/cluster.rs b/src/cluster.rs
--- a/src/cluster.rs
+++ b/src/cluster.rs
@@ -118,7 +118,8 @@ impl ClusterStore {
     pub fn insert(&mut self, sig: PrSignature) -> ClusterView {
         let idx = self.nodes.len();
-        let hits = self.scan_all(&sig);
+        let hits = self.candidates(sig.diff_sim);
         for hit in hits {
             self.union(idx, hit);
         }
+        self.absorb_stale(idx);
         self.view(idx)
     }

+    /// Candidate lookup over the four simhash bands.
+    fn candidates(&self, sig: u64) -> Vec<usize> {
+        let mut seen = HashSet::new();
+        for band in 0..BANDS {
+            let key = (sig >> (band * BAND_BITS)) & BAND_MASK;
+            if let Some(bucket) = self.bands[band].get(&key) {
+                seen.extend(bucket.iter().copied());
+            }
+        }
+        seen.into_iter().collect()
+    }
+
+    /// Drop members that arrived outside the retention window.
+    fn absorb_stale(&mut self, root: usize) {
+        let cutoff = self.now - self.retention;
+        let node = &mut self.nodes[root];
+        node.members.retain(|m| m.arrived >= cutoff);
+        if node.members.is_empty() {
+            node.parent = root;
+        }
+        self.bands_dirty = true;
+    }
";
        // Silence from the length floor would pass this test for the wrong
        // reason, so check the diff is long enough to be judged at all.
        let judged: usize = split_files(diff)
            .iter()
            .filter(|f| is_production_source(&f.path))
            .flat_map(|f| f.added.iter())
            .filter(|l| {
                let t = l.trim();
                !t.is_empty() && !is_comment(t)
            })
            .count();
        assert!(judged >= MIN_CODE_LINES, "{judged} lines, not judged");
        assert!(fired(diff).is_empty(), "{:?}", fired(diff));
    }

    #[test]
    fn tests_and_docs_are_not_judged() {
        let diff = PASTED.replace("src/api.py", "tests/test_api.py");
        assert!(fired(&diff).is_empty(), "{:?}", fired(&diff));
    }

    #[test]
    fn code_in_a_foreign_idiom_fires() {
        // The file is tab-indented snake_case with single quotes; the
        // addition is space-indented camelCase with double quotes.
        let diff = "\
diff --git a/src/store.py b/src/store.py
--- a/src/store.py
+++ b/src/store.py
@@ -20,6 +20,14 @@ class Store:
 \tdef load_rows(self, table_name):
 \t\tquery = 'select * from ' + table_name
 \t\tcursor = self.connection.cursor()
 \t\tcursor.execute(query)
 \t\trow_list = cursor.fetchall()
 \t\treturn row_list

 \tdef drop_rows(self, table_name):
 \t\tquery = 'delete from ' + table_name
 \t\tself.connection.cursor().execute(query)
 \t\tself.row_cache.pop(table_name, None)

 \tdef table_names(self):
 \t\tname_rows = self.load_rows('sqlite_master')
 \t\treturn [name_row[0] for name_row in name_rows]
+    def loadRowsFiltered(self, tableName, filterValue):
+        queryString = \"select * from \" + tableName + \" where col = ?\"
+        dbCursor = self.connection.cursor()
+        dbCursor.execute(queryString, [filterValue])
+        rowList = dbCursor.fetchall()
+        filteredRows = []
+        for rowItem in rowList:
+            filteredRows.append(rowItem)
+        return filteredRows
+
+    def countRows(self, tableName):
+        queryString = \"select count(*) from \" + tableName
+        dbCursor = self.connection.cursor()
+        dbCursor.execute(queryString)
+        return dbCursor.fetchone()[0]
+
+    def describeTable(self, tableName):
+        queryString = \"pragma table_info(\" + tableName + \")\"
+        dbCursor = self.connection.cursor()
+        dbCursor.execute(queryString)
+        columnRows = dbCursor.fetchall()
+        return {columnRow[1]: columnRow[2] for columnRow in columnRows}
";
        assert!(
            fired(diff).contains(&"CODE_STYLE_DRIFT".to_string()),
            "{:?}",
            fired(diff)
        );
    }

    #[test]
    fn short_diffs_abstain() {
        assert!(rules("+x = 1\n+y = 2\n").is_empty());
    }

    #[test]
    fn function_defs_across_languages() {
        assert_eq!(
            function_def("pub fn score(&self) -> f64 {"),
            Some(("score".into(), true))
        );
        assert_eq!(
            function_def("async def fetch_all(client):"),
            Some(("fetch_all".into(), false))
        );
        assert_eq!(
            function_def("func (s *Server) handleAll(w http.ResponseWriter) {"),
            Some(("handleall".into(), false))
        );
        assert_eq!(
            function_def("private static void processData(Foo a) {"),
            Some(("processdata".into(), false))
        );
        assert_eq!(function_def("let x = compute(y);"), None);
        assert_eq!(function_def("// fn score() is defined below"), None);
    }

    #[test]
    fn normalize_collapses_values_not_structure() {
        assert_eq!(
            normalize("  if (status == 404) { log(\"gone\"); }"),
            normalize("if (status == 500) { log(\"missing\"); }")
        );
        assert_ne!(
            normalize("if (a) { f(); }"),
            normalize("while (a) { f(); }")
        );
    }

    #[test]
    fn literals_skip_identifiers_and_lifetimes() {
        assert_eq!(literals("let x = 4096;"), vec!["n:4096"]);
        assert!(literals("let sha256 = 1;").is_empty());
        assert!(literals("fn f<'a>(s: &'a str) {}").is_empty());
        assert_eq!(literals("open(\"/etc/passwd\")"), vec!["s:/etc/passwd"]);
    }
}
