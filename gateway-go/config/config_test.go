package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func write(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "c.json")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// A config written for the original gateway must keep loading, with reliability on by default.
func TestOriginalConfigStillLoads(t *testing.T) {
	cfg, err := Load(write(t, `{"port":8081,"max_concurrent_per_node":8,
		"sglang_instances":[{"group":"local","host":"127.0.0.1","port":30000,"max_concurrent":8}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Port != 8081 || len(cfg.SGLangInstances) != 1 {
		t.Fatalf("original fields lost: %+v", cfg)
	}
	if cfg.Reliability.MaxAttempts != 3 || !cfg.Reliability.Breaker.Enabled || !cfg.Reliability.Health.Enabled {
		t.Fatalf("reliability defaults missing: %+v", cfg.Reliability)
	}
}

func TestOverridesAndDurations(t *testing.T) {
	cfg, err := Load(write(t, `{"sglang_instances":[],
		"reliability":{"max_attempts":1,"attempt_timeout":"750ms","breaker":{"enabled":false},"health":{"enabled":false}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Reliability.AttemptTimeout.Std() != 750*time.Millisecond || cfg.Reliability.Breaker.Enabled || cfg.Reliability.MaxAttempts != 1 {
		t.Fatalf("overrides ignored: %+v", cfg.Reliability)
	}
}

func TestValidation(t *testing.T) {
	for name, js := range map[string]string{
		"attempts":      `{"reliability":{"max_attempts":0}}`,
		"load factor":   `{"routing":{"bounded_load_factor":0.5}}`,
		"numeric dur":   `{"reliability":{"attempt_timeout":5}}`,
		"breaker":       `{"reliability":{"breaker":{"failure_threshold":0}}}`,
		"queue timeout": `{"admission":{"queue_timeout":"0s"}}`,
	} {
		if _, err := Load(write(t, js)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}
