# Workflow Best Practices

## Context Management
- Use `/compact` proactively at ~50% context capacity — don't wait for auto-compression
- Use `/clear` when switching between unrelated tasks mid-session
- Use `Esc Esc` or `/rewind` to undo a bad tool call instead of fixing mid-context (saves tokens)

## Planning
- Always use plan mode before non-trivial implementation
- For complex plans, consider a second review pass before executing
- Break large tasks into phase-gated plans with tests at each gate

## Token Conservation
- **Gated Reading:** When reading files larger than 250 lines, you MUST use the `offset` and `limit` arguments to fetch only the relevant line ranges (max 200 lines per read). Never read massive files in full.
- Prefer targeted Glob/Grep before launching subagents
- Only escalate to broad Explore agents when directed search proves insufficient
- If the same lookup is done across multiple sessions, cache the result in memory files

## Development Flow
- Develop in candidates/ first, test in Docker, then promote
- After code changes, sync to both production AND candidate dirs
- Commit at natural milestones — don't wait until end of session

## Debugging
- Check Docker logs first: `docker logs <service> --tail 30`
- Use `/doctor` for installation/auth/config issues
- Run terminals as background tasks for better log visibility
