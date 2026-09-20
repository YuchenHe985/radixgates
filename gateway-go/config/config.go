package config

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// Duration is a time.Duration that (un)marshals as a string such as "1500ms".
type Duration time.Duration

func (d Duration) Std() time.Duration { return time.Duration(d) }

func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return fmt.Errorf(`duration must be a string such as "2s": %w`, err)
	}
	v, err := time.ParseDuration(s)
	if err != nil {
		return err
	}
	*d = Duration(v)
	return nil
}

func (d Duration) MarshalJSON() ([]byte, error) { return json.Marshal(time.Duration(d).String()) }

// SGLangInstance describes one SGLang inference server.
type SGLangInstance struct {
	Group         string `json:"group"` // "local" | "remote" | any label
	Host          string `json:"host"`
	Port          int    `json:"port"`
	MaxConcurrent int    `json:"max_concurrent"` // semaphore size; 0 → use global default
	// Role used in PD-disaggregated deployments:
	//   ""        – combined prefill+decode (default)
	//   "prefill" – prefill-only SGLang instance (NOT a request entry point)
	//   "decode"  – decode-only SGLang instance (entry point; triggers prefill via bootstrap)
	Role string `json:"role"`
}

// Routing tunes prefix-affinity routing.
type Routing struct {
	// BoundedLoadFactor caps how far above the mean load the preferred node may run
	// before a request spills to the next-ranked node.
	BoundedLoadFactor float64 `json:"bounded_load_factor"`
	// AffinityFloor is the minimum concurrency a node may hold before the bound
	// applies, so that low traffic keeps its prefix affinity.
	AffinityFloor int `json:"affinity_floor"`
	// AffinityWait is how long a request waits for its preferred node when that node is only at its
	// concurrency limit, before spilling to the next-ranked node. Zero spills immediately. Waiting
	// keeps the prefix cache useful when a node's cache holds fewer prompts than the traffic uses.
	AffinityWait Duration `json:"affinity_wait"`
	// PlacementSize enables placement memory: the gateway remembers which node each prompt prefix was
	// placed on (up to this many prefixes, least recently used forgotten first). A new prefix goes to the
	// node holding the fewest prefixes, so a small set of hot prefixes is spread evenly instead of
	// following the hash, and a prefix stays on its node until that node stops being available.
	// Zero uses plain rendezvous hashing.
	PlacementSize int `json:"placement_size"`
}

type Breaker struct {
	Enabled          bool     `json:"enabled"`
	FailureThreshold int      `json:"failure_threshold"`
	OpenDuration     Duration `json:"open_duration"`
	MaxOpenDuration  Duration `json:"max_open_duration"`
	HalfOpenMax      int      `json:"half_open_max"`
}

type Health struct {
	Enabled            bool     `json:"enabled"`
	Interval           Duration `json:"interval"`
	Timeout            Duration `json:"timeout"`
	Path               string   `json:"path"`
	UnhealthyThreshold int      `json:"unhealthy_threshold"`
	HealthyThreshold   int      `json:"healthy_threshold"`
}

// Reliability groups failure handling. Every field is optional in config.json;
// a config written for the original gateway keeps working and gets these defaults.
type Reliability struct {
	MaxAttempts       int      `json:"max_attempts"`        // total upstream attempts per request (1 = no retry)
	AttemptTimeout    Duration `json:"attempt_timeout"`     // wait for the first response byte of one attempt (streams)
	StreamIdleTimeout Duration `json:"stream_idle_timeout"` // abort a stream that stalls after it started
	RetryBackoff      Duration `json:"retry_backoff"`       // base of the jittered exponential backoff
	OverallTimeout    Duration `json:"overall_timeout"`     // whole request including retries and queueing
	Breaker           Breaker  `json:"breaker"`
	Health            Health   `json:"health"`
}

type Admission struct {
	MaxQueue     int      `json:"max_queue"`     // requests allowed to wait for capacity before 429
	QueueTimeout Duration `json:"queue_timeout"` // longest wait for capacity before 503
}

type Config struct {
	Port         int    `json:"port"`
	DefaultModel string `json:"default_model"`
	LogLevel     string `json:"log_level"`

	// SGLangInstances is the pool of SGLang nodes the gateway routes to.
	SGLangInstances      []SGLangInstance `json:"sglang_instances"`
	MaxConcurrentPerNode int              `json:"max_concurrent_per_node"` // default semaphore if instance doesn't set its own

	Routing     Routing     `json:"routing"`
	Reliability Reliability `json:"reliability"`
	Admission   Admission   `json:"admission"`
}

// DefaultReliability returns the production-leaning defaults.
func DefaultReliability() Reliability {
	return Reliability{
		MaxAttempts:       3,
		AttemptTimeout:    Duration(15 * time.Second),
		StreamIdleTimeout: Duration(30 * time.Second),
		RetryBackoff:      Duration(25 * time.Millisecond),
		OverallTimeout:    Duration(10 * time.Minute), // matches the original per-request timeout
		Breaker: Breaker{Enabled: true, FailureThreshold: 5, OpenDuration: Duration(5 * time.Second),
			MaxOpenDuration: Duration(60 * time.Second), HalfOpenMax: 1},
		Health: Health{Enabled: true, Interval: Duration(2 * time.Second), Timeout: Duration(time.Second),
			Path: "/health", UnhealthyThreshold: 2, HealthyThreshold: 1},
	}
}

func Load(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open config %s: %w", path, err)
	}
	defer f.Close()

	cfg := &Config{
		Port:                 8080,
		DefaultModel:         "NousResearch/Meta-Llama-3-8B-Instruct",
		LogLevel:             "info",
		MaxConcurrentPerNode: 8,
		Routing:              Routing{BoundedLoadFactor: 1.25, AffinityFloor: 4},
		Reliability:          DefaultReliability(),
		Admission:            Admission{MaxQueue: 128, QueueTimeout: Duration(10 * time.Second)},
	}
	if err := json.NewDecoder(f).Decode(cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// Validate reports the first value that would make the gateway misbehave.
func (c *Config) Validate() error {
	if c.Routing.BoundedLoadFactor < 1 {
		return fmt.Errorf("config: routing.bounded_load_factor must be >= 1")
	}
	if c.Routing.AffinityFloor < 1 {
		return fmt.Errorf("config: routing.affinity_floor must be >= 1")
	}
	if c.Routing.PlacementSize < 0 {
		return fmt.Errorf("config: routing.placement_size must not be negative")
	}
	if c.Routing.AffinityWait < 0 {
		return fmt.Errorf("config: routing.affinity_wait must not be negative")
	}
	r := c.Reliability
	if r.MaxAttempts < 1 {
		return fmt.Errorf("config: reliability.max_attempts must be >= 1")
	}
	if r.AttemptTimeout <= 0 || r.StreamIdleTimeout <= 0 || r.OverallTimeout <= 0 {
		return fmt.Errorf("config: reliability timeouts must be positive")
	}
	if r.Breaker.Enabled && (r.Breaker.FailureThreshold < 1 || r.Breaker.HalfOpenMax < 1 || r.Breaker.OpenDuration <= 0) {
		return fmt.Errorf("config: reliability.breaker needs failure_threshold>=1, half_open_max>=1, open_duration>0")
	}
	if r.Health.Enabled && (r.Health.Interval <= 0 || r.Health.Timeout <= 0 || r.Health.Path == "" ||
		r.Health.UnhealthyThreshold < 1 || r.Health.HealthyThreshold < 1) {
		return fmt.Errorf("config: reliability.health is incomplete")
	}
	if c.Admission.MaxQueue < 0 || c.Admission.QueueTimeout <= 0 {
		return fmt.Errorf("config: admission.max_queue must be >= 0 and queue_timeout > 0")
	}
	return nil
}
