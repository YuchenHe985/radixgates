# Fix Docker Compose Variable Interpolation in Shell Scripts

## Symptom

Gateway container fails to start with a JSON parse error:

```
invalid character '}' looking for beginning of value
```

The gateway's generated `config.json` is malformed, with empty strings where
variable values should appear.

## Root Cause

In `docker-compose.yml` `command` sections that contain shell scripts, Docker Compose
itself interpolates `$VAR` and `${VAR}` **before** passing the script to the container's
shell. This causes shell variables to be replaced with empty strings.

## Fix

Escape shell variables with `$$` in docker-compose command sections so Compose
passes them through literally, and the container shell evaluates them at runtime.

```yaml
# Bad — Compose replaces $M2_HOST with empty string before shell runs:
command: >
  - sh
  - -c
  - |
    if [ -n "$M2_HOST" ]; then
      echo $M2_HOST
    fi

# Good — $$ tells Compose to pass a literal $ to the shell:
command: >
  - sh
  - -c
  - |
    if [ -n "$$M2_HOST" ]; then
      echo $$M2_HOST
    fi
```

Also applies to variables used inside heredocs or string concatenation within
the command section.

## Verify

```bash
docker exec gateway cat /app/gateway/config/config.json
# Must be valid JSON with actual values, not empty strings
```
