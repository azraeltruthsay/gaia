# Claude shared cross-node memory

`gaia-node-memory.age` is a plain-text document encrypted with
[`age`](https://github.com/FiloSottile/age), symmetric passphrase mode
(`age -p`). It's shared memory for Claude sessions running on Azrael's
different GAIA nodes (the arch-usb field kit, the Arch gaia-host, the Arch
Gaia360, the Windows GAIA host, and the Windows Gaia360). The passphrase is
known to Azrael and shared with each node's Claude out of band — it is never
written down in this repo.

This directory lives outside `knowledge/` on purpose: large parts of
`knowledge/` are gitignored as instance-local (`Dev_Notebook/`, logs, vector
stores, etc.), so content there doesn't actually sync between nodes. This
does.

## If you're a Claude session reading this and don't have the passphrase yet

Ask Azrael for it. Don't guess, don't brute-force, don't try to read the file
without it.

## Workflow (once you have the passphrase)

```bash
# Decrypt to stdout — review it before editing
age -d -o - claude-shared-memory/gaia-node-memory.age

# Decrypt to a scratch file, edit, then re-encrypt the same way
age -d -o /tmp/gaia-node-memory.md.plain claude-shared-memory/gaia-node-memory.age
# ...edit /tmp/gaia-node-memory.md.plain...
age -p -o claude-shared-memory/gaia-node-memory.age /tmp/gaia-node-memory.md.plain
shred -u /tmp/gaia-node-memory.md.plain   # or rm -f if shred isn't available

git add claude-shared-memory/gaia-node-memory.age
git commit -m "update shared cross-node memory"
git pull --rebase   # resolve any conflict the normal git way, then push
git push
```

This is a normal shared document, not an append-only log — if two nodes edit
it around the same time, whoever pushes second needs to pull, re-decrypt,
merge by hand, and re-encrypt. Pull and decrypt fresh before you start editing
to minimize that.

`age` has no built-in non-interactive passphrase input (by design, to avoid
leaking it via shell history, process args, or logs) — run these commands in
an interactive terminal, not through something that pipes a password in or
logs its output.
