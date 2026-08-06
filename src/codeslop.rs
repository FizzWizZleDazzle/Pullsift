//! Content tells computed from the added code itself. Lexical versions
//! of the signals the code-detection literature found discriminating:
//! placeholder bodies, restate-the-code comments, generic identifiers.
//! A tree-sitter AST upgrade can deepen these; the research caps
//! expectations for code-content classification (cross-language AUC
//! around 0.8), so these ship dark and the fit prices them.

use crate::engine::Fire;

const PLACEHOLDERS: &[&str] = &[
    "todo!()",
    "unimplemented!()",
    "notimplementederror",
    "not implemented",
    "todo: implement",
    "placeholder",
    "your code here",
    "implementation goes here",
];

const GENERIC_WORDS: &[&str] = &[
    "data", "result", "results", "item", "items", "value", "values", "process", "handle", "temp",
    "info", "obj", "res", "output", "input", "helper", "util", "utils", "manager", "wrapper",
    "thing", "stuff", "foo", "new", "my", "get", "set", "do",
];

/// Added, non-comment code lines of a unified diff.
fn added_code_lines(diff: &str) -> Vec<&str> {
    diff.lines()
        .filter_map(|l| l.strip_prefix('+'))
        .filter(|l| !l.starts_with("++"))
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect()
}

fn is_comment(line: &str) -> bool {
    line.starts_with("//")
        || line.starts_with('#')
        || line.starts_with("/*")
        || line.starts_with('*')
        || line.starts_with("--")
}

/// Lowercased word parts of the identifiers on a code line: snake and
/// camel segments of alphanumeric runs.
fn identifier_words(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    for token in crate::hashing::tokens(line) {
        for part in token.split('_') {
            // tokens() lowercases, so camelCase is already flattened;
            // split what remains on digits.
            let part: String = part.chars().filter(|c| c.is_alphabetic()).collect();
            if part.len() >= 2 {
                out.push(part);
            }
        }
    }
    out
}

pub fn rules(diff: &str) -> Vec<Fire> {
    let lines = added_code_lines(diff);
    let mut out = Vec::new();
    if lines.len() < 8 {
        return out;
    }

    let placeholder_hits = lines
        .iter()
        .filter(|l| {
            let low = l.to_lowercase();
            PLACEHOLDERS.iter().any(|p| low.contains(p))
        })
        .count();
    if placeholder_hits >= 2 {
        out.push(Fire::new(
            "CODE_PLACEHOLDER",
            (placeholder_hits as f64 / 3.0).min(1.0),
        ));
    }

    // Restate-the-code comments: a comment whose words are mostly the
    // identifier words of the next code line ("// increment the counter"
    // above "counter += 1" style).
    let mut restatements = 0usize;
    for pair in lines.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if !is_comment(a) || is_comment(b) {
            continue;
        }
        const STOPWORDS: &[&str] = &[
            "the", "a", "an", "to", "of", "is", "this", "and", "for", "in", "on", "with", "it",
        ];
        let comment_words: Vec<String> = identifier_words(a)
            .into_iter()
            .filter(|w| !STOPWORDS.contains(&w.as_str()))
            .collect();
        if comment_words.len() < 2 {
            continue;
        }
        let code_words = identifier_words(b);
        let overlap = comment_words
            .iter()
            .filter(|w| code_words.contains(w))
            .count();
        if overlap * 10 >= comment_words.len() * 6 {
            restatements += 1;
        }
    }
    if restatements >= 2 {
        out.push(Fire::new(
            "CODE_COMMENT_RESTATE",
            (restatements as f64 / 4.0).min(1.0),
        ));
    }

    // Generic identifier vocabulary in the added code.
    let code_words: Vec<String> = lines
        .iter()
        .filter(|l| !is_comment(l))
        .flat_map(|l| identifier_words(l))
        .collect();
    if code_words.len() >= 20 {
        let generic = code_words
            .iter()
            .filter(|w| GENERIC_WORDS.contains(&w.as_str()))
            .count();
        let frac = generic as f64 / code_words.len() as f64;
        if frac >= 0.3 {
            out.push(Fire::new("CODE_GENERIC_IDENT", frac.min(1.0)));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_bodies_fire() {
        let diff = "+def process(data):\n+    raise NotImplementedError\n\
                    +def handle(item):\n+    pass  # TODO: implement\n\
                    +def run():\n+    return compute()\n\
                    +x = 1\n+y = 2\n+z = 3\n";
        let fires = rules(diff);
        assert!(fires.iter().any(|f| f.rule == "CODE_PLACEHOLDER"));
    }

    #[test]
    fn restating_comments_fire() {
        let diff = "+// increment the request counter\n+request_counter += 1;\n\
                    +// close the open connection\n+close_connection(open);\n\
                    +let a = 1;\n+let b = 2;\n+let c = 3;\n+let d = 4;\n";
        let fires = rules(diff);
        assert!(fires.iter().any(|f| f.rule == "CODE_COMMENT_RESTATE"));
    }

    #[test]
    fn real_code_stays_quiet() {
        let diff = "+fn simhash(patch: &str) -> Option<u64> {\n\
                    +    let shingles = shingle_hashes(&tokens, 4);\n\
                    +    let mut acc = [0i32; 64];\n\
                    +    for h in shingles {\n\
                    +        for bit in 0..64 {\n\
                    +            acc[bit] += if h >> bit & 1 == 1 { 1 } else { -1 };\n\
                    +        }\n\
                    +    }\n\
                    +    Some(fold(acc))\n\
                    +}\n";
        assert!(rules(diff).is_empty());
    }

    #[test]
    fn short_diffs_abstain() {
        assert!(rules("+raise NotImplementedError\n+pass # TODO: implement\n").is_empty());
    }
}
