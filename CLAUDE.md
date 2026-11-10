# CLAUDE.md

All project instructions live in AGENTS.md (shared by every coding agent) — read and follow it:

@AGENTS.md

## Claude Code specifics

- A separate Claude Code session may be running on the Jetson. Hand off Jetson-only steps there
  rather than attempting them over assumptions from the training machine.
- Plugin/tool state directories (`.omc/`, `.claude/settings.local.json`) are git-ignored; never commit them.
