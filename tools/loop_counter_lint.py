#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 The rocket-userspace authors
"""Finds C for-loops whose counter, declared in the loop's header, the loop body never reads.

usage: loop_counter_lint.py PATH...

Such a loop runs its body the same way on every iteration. When the body is idempotent (a
scatter that writes the same bytes, a recompute into the same buffer), the result is
identical whatever the trip count, so no test that compares outputs can see the repeat; only
its cost shows. The usual cause is a refactor that moved the iteration into the callee and
left the caller's loop in place.

Only a counter declared in the init clause is checked: its scope is the loop, so a body that
never reads it cannot tell one iteration from the next. A counter declared outside the loop
is not checked, because a search loop reads it after the loop ends.

A loop that repeats on purpose (a fixed number of shuffle stages, a retry, opening one
handle per worker, counting the steps of a header-only update) states it with a comment
containing "repeat:" on the `for` line or the line above it, naming why.

Each PATH is a C or header file, or a directory scanned for *.c and *.h. The embedded cases
run first, so a parser change that stops the lint from seeing the defect fails here rather
than passing a tree. Exits 0 with no findings, 1 with findings or a failed case.
"""
import os
import re
import sys

MARK = "repeat:"
DECL = re.compile(
    r"^\s*(?:const\s+)?(?:(?:unsigned|signed|long|short)\s+)*"
    r"(?:int|long|short|char|unsigned|size_t|ssize_t|ptrdiff_t|u?int(?:8|16|32|64)_t)\s+(.*)$",
    re.S)
NAME_EQ = re.compile(r"^\s*\**\s*([A-Za-z_]\w*)\s*=")
CONTROL = ("for", "while", "if", "switch")


def strip(src):
    """Blank comments, string and character literals and line splices, keeping every
    newline, and return (code, the 1-based lines whose comments carry the mark)."""
    out, marked, i, n, line = [], set(), 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\\" and i + 1 < n and src[i + 1] == "\n":
            out.append(" \n"); i += 2; line += 1
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            body = src[i:j]
            if MARK in body:
                marked.add(line)
            out.append("".join(ch if ch == "\n" else " " for ch in body))
            line += body.count("\n"); i = j
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            if MARK in src[i:j]:
                marked.add(line)
            out.append(" " * (j - i)); i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(c + " " * (j - i - 2) + c if j - i >= 2 else src[i:j]); i = j
        else:
            if c == "\n":
                line += 1
            out.append(c); i += 1
    return "".join(out), marked


def match_close(code, i, open_ch, close_ch):
    """Index just past the bracket that closes the one at code[i]."""
    depth = 0
    for j in range(i, len(code)):
        if code[j] == open_ch:
            depth += 1
        elif code[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
    return len(code)


def skip_ws(code, i):
    while i < len(code) and code[i].isspace():
        i += 1
    return i


def word_at(code, i, word):
    return (code.startswith(word, i) and (i + len(word) >= len(code) or
            not (code[i + len(word)].isalnum() or code[i + len(word)] == "_")))


def statement_end(code, i):
    """Index just past the statement that starts at code[i] (a block, a control statement
    with its body, or an expression statement)."""
    i = skip_ws(code, i)
    if i >= len(code):
        return i
    if code[i] == "{":
        return match_close(code, i, "{", "}")
    if word_at(code, i, "do"):
        j = statement_end(code, i + 2)
        k = code.find(";", j)
        return len(code) if k < 0 else k + 1
    for kw in CONTROL:
        if word_at(code, i, kw):
            p = skip_ws(code, i + len(kw))
            if p < len(code) and code[p] == "(":
                j = statement_end(code, match_close(code, p, "(", ")"))
                if kw == "if":
                    e = skip_ws(code, j)
                    if word_at(code, e, "else"):
                        j = statement_end(code, e + 4)
                return j
    depth = 0
    for j in range(i, len(code)):
        ch = code[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return j
        elif ch == ";" and depth == 0:
            return j + 1
    return len(code)


def counters(init):
    """The names an init clause declares: `int a = 0, b = 1`. None for `i = 0`."""
    m = DECL.match(init)
    if not m:
        return []
    parts, names = m.group(1), []
    depth, cur = 0, ""
    for ch in parts:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            names.append(cur); cur = ""
        else:
            cur += ch
    names.append(cur)
    out = []
    for part in names:
        nm = NAME_EQ.match(part)
        if nm:
            out.append(nm.group(1))
    return out


def lint(src):
    """[(line, counter, header)] for each loop whose body never names its counter."""
    code, marked = strip(src)
    found = []
    for m in re.finditer(r"\bfor\s*\(", code):
        p = m.end() - 1
        q = match_close(code, p, "(", ")")
        header = code[p + 1:q - 1]
        semi = header.find(";")
        if semi < 0:
            continue                                   # a range-for or a macro's for
        body = code[q:statement_end(code, q)]
        line = code.count("\n", 0, m.start()) + 1
        if line in marked or line - 1 in marked:
            continue
        for name in counters(header[:semi]):
            if not re.search(r"\b%s\b" % re.escape(name), body):
                found.append((line, name, " ".join(code[m.start():q].split())))
    return found


CASES = [
    ("a tile scattered once per row (the defect)", 1, """
        for (int ki = 0; ki < nKt; ki++) {
            uint8_t *slot = base + (size_t)ki * stride;
            for (int kk = 1; kk <= Ntile; kk++)
                i4_wt_scatter(slot, B, K, n0, k0, Ntile, Ktile);
        }"""),
    ("an unmarked fixed repeat", 1, """
        for (int r = 0; r < 3; r++) step(x);"""),
    ("a counter named only in a string", 1, """
        for (int i = 0; i < n; i++) puts("i");"""),
    ("a comment between the header and a nested loop's body", 0, """
        for (int h = 1; h <= Mtile; h++)                            /* broadcast; rows */
            for (int ch = 1; ch <= Ntile; ch++)
                slot[feat_idx(Mtile, ch, h)] = bias[ch - 1];"""),
    ("a macro body continued over lines", 0, """
        #define SCAT(KH, KW)                                         \\
            for (int kh = 0; kh < (KH); kh++)                        \\
                for (int kw = 0; kw < (KW); kw++)                    \\
                    d[kh * (KW) + kw] = 0;"""),
    ("the counter used only in an else branch", 0, """
        for (int i = 0; i < n; i++) if (f) a(); else b(i);"""),
    ("a marked repeat, on the line", 0, """
        for (int st = 0; st < 4; st++) {   /* repeat: four zip stages */
            zip(v);
        }"""),
    ("a marked repeat, on the line above", 0, """
        // repeat: one fd per worker
        for (int i = 0; i < nthreads; i++) open_one(p);"""),
    ("two counters, one unread", 1, """
        for (int i = 0, j = 0; i < n; i++, j++) a[i] = 0;"""),
    ("a search loop, its counter declared outside", 0, """
        for (i = 0; i < C2 && ap[i] == blank; i++) ;
        if (i == C2) unwritten++;"""),
]


def self_test():
    bad = 0
    for what, want, src in CASES:
        got = len(lint(src))
        if got != want:
            print("  FAIL   case '%s': %d finding(s), want %d" % (what, got, want))
            bad += 1
    print("  %s   %d embedded case(s)" % ("FAIL" if bad else "ok  ", len(CASES)))
    return bad == 0


def files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for nm in sorted(names):
                    if nm.endswith((".c", ".h")):
                        yield os.path.join(root, nm)
        else:
            yield p


def main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 1
    ok = self_test()
    nfiles = nfound = 0
    for path in files(argv):
        nfiles += 1
        with open(path, encoding="utf-8", errors="replace") as f:
            for line, name, header in lint(f.read()):
                nfound += 1
                print("%s:%d: counter '%s' is never read in the body of `%s`; mark an "
                      "intended repeat with a '%s' comment" % (path, line, name, header, MARK))
    print("  %s   %d file(s), %d loop(s) whose counter the body never reads"
          % ("FAIL" if nfound else "ok  ", nfiles, nfound))
    return 0 if ok and not nfound else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
