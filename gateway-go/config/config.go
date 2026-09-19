package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// SGLangInstance describes one SGLang inference server.
type SGLangInstance struct {
	Group         string `json:"group"`          // "local" | "remote" | any label
	Host          string `json:"host"`
	Port          int    `json:"port"`
	MaxConcurrent int    `json:"max_concurrent"` // semaphore size; 0 → use global default
	// Role used in PD-disaggregated deployments:
	//   ""        – combined prefill+decode (default)
	//   "prefill" – prefill-only SGLang instance (NOT a request entry point)
	//   "decode"  – decode-only SGLang instance (entry point; triggers prefill via bootstrap)
	Role string `json:"role"`
}

type Config struct {
	Port         int    `json:"port"`
	DefaultModel string `json:"default_model"`
	LogLevel     string `json:"log_level"`

	// SGLangInstances is the pool of SGLang nodes the gateway routes to.
	SGLangInstances      []SGLangInstance `json:"sglang_instances"`
	MaxConcurrentPerNode int              `json:"max_concurrent_per_node"` // default semaphore if instance doesn't set its own
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
	}
	if err := json.NewDecoder(f).Decode(cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	return cfg, nil
}
